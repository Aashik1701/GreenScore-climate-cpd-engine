"""
GreenScore REST API
===================
FastAPI service exposing Climate-Adjusted PD (CPD) predictions.

Endpoints
---------
GET  /health                       Liveness + model status
GET  /scenarios                    List supported NGFS scenarios
POST /predict                      Single-loan CPD prediction
POST /predict/batch                Batch CSV prediction (upload)

Run locally:
    uvicorn api:app --reload --port 8000

Docker:
    docker run -p 8000:8000 greenscore:latest uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io
import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

import config
from cpd_engine import get_baseline_pd, add_climate_features
from physical_risk import apply_physical_risk
from transition_risk import apply_transition_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# App + Model Lifecycle
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="GreenScore CPD API",
    description=(
        "Climate-Adjusted Credit Risk Engine. "
        "Computes Baseline PD and Climate-Adjusted PD (CPD) for loan portfolios "
        "under NGFS Phase V carbon price scenarios."
    ),
    version="1.0.0",
    contact={"name": "GreenScore Project"},
    license_info={"name": "MIT"},
)

_MODEL = None


def _get_model():
    """Load model once and cache in module scope."""
    global _MODEL
    if _MODEL is None:
        model_path = os.environ.get("MODEL_PATH", "models/baseline_pd_model.pkl")
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model not found at {model_path}. "
                "Run `python cpd_engine.py` to train it first."
            )
        _MODEL = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    return _MODEL


# ─────────────────────────────────────────────────────────
# Pydantic Input / Output Schemas
# ─────────────────────────────────────────────────────────

class LoanFeatures(BaseModel):
    """Input features for a single loan CPD prediction."""

    # Required financial features
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (%)")
    annual_inc: float = Field(..., gt=0, description="Annual income in USD")
    fico_range_low: float = Field(..., ge=300, le=850, description="FICO score (lower bound)")
    int_rate: float = Field(..., ge=0, le=40, description="Interest rate (%)")
    installment: float = Field(..., gt=0, description="Monthly installment (USD)")
    emp_length: float = Field(default=3.0, ge=0, le=10, description="Employment length (years)")
    loan_amnt: float = Field(default=10000.0, gt=0, description="Loan amount (USD)")
    term_months: float = Field(default=36.0, description="Loan term in months (36 or 60)")

    # Optional geographic / loan attributes
    addr_state: Optional[str] = Field(default=None, description="US state code (e.g. CA) or Indian state name")
    purpose: Optional[str] = Field(default=None, description="Loan purpose / sector")

    # Credit bureau features (optional, default to 0)
    revol_util: float = Field(default=0.0, ge=0, description="Revolving utilisation (%)")
    revol_bal: float = Field(default=0.0, ge=0, description="Revolving balance (USD)")
    open_acc: float = Field(default=5.0, ge=0, description="Number of open credit lines")
    total_acc: float = Field(default=10.0, ge=0, description="Total credit lines ever")
    pub_rec: float = Field(default=0.0, ge=0, description="Number of public derogatory records")
    delinq_2yrs: float = Field(default=0.0, ge=0, description="Delinquencies in past 2 years")
    inq_last_6mths: float = Field(default=0.0, ge=0, description="Inquiries in last 6 months")
    acc_open_past_24mths: float = Field(default=0.0, ge=0)
    mort_acc: float = Field(default=0.0, ge=0)
    total_bc_limit: float = Field(default=0.0, ge=0)
    total_rev_hi_lim: float = Field(default=0.0, ge=0)
    mo_sin_rcnt_tl: float = Field(default=12.0, ge=0)
    mo_sin_old_rev_tl_op: float = Field(default=180.0, ge=0)
    num_actv_rev_tl: float = Field(default=0.0, ge=0)
    percent_bc_gt_75: float = Field(default=0.0, ge=0, le=100)
    bc_util: float = Field(default=0.0, ge=0)
    mths_since_recent_inq: float = Field(default=12.0, ge=0)

    # API parameters
    scenario: str = Field(
        default="orderly",
        description="NGFS carbon price scenario key",
    )
    severity_factor: float = Field(
        default=config.SEVERITY_FACTOR, ge=0.0, le=1.0,
        description="Physical risk severity factor",
    )
    transition_scaling: float = Field(
        default=config.TRANSITION_SCALING, ge=0.0, le=1.0,
        description="Transition risk scaling factor",
    )

    @field_validator("scenario")
    @classmethod
    def validate_scenario(cls, v: str) -> str:
        valid = list(config.CARBON_PRICES.keys())
        if v not in valid:
            raise ValueError(f"scenario must be one of {valid}")
        return v


class CPDResponse(BaseModel):
    """Response schema for a single-loan CPD prediction."""
    baseline_pd: float
    cpd: float
    pd_uplift_pct: float
    risk_category: str
    expected_loss: float
    scenario: str
    carbon_price: float
    lgd_assumption: float


# ─────────────────────────────────────────────────────────
# Helper: run full CPD pipeline on a DataFrame row
# ─────────────────────────────────────────────────────────

def _run_cpd_pipeline(
    df: pd.DataFrame,
    scenario: str,
    severity_factor: float,
    transition_scaling: float,
) -> pd.DataFrame:
    """Add Baseline_PD, CPD, Uplift, Risk_Category, Expected_Loss columns."""
    model = _get_model()
    baseline_pd = get_baseline_pd(model, df)

    loc_col = "addr_state" if "addr_state" in df.columns else None
    purpose_col = "purpose" if "purpose" in df.columns else None

    if loc_col:
        pd_physical = apply_physical_risk(baseline_pd, df[loc_col], severity_factor=severity_factor)
    else:
        pd_physical = baseline_pd.copy()

    if purpose_col:
        cpd = apply_transition_risk(
            pd_physical, df[purpose_col], df["annual_inc"],
            scenario=scenario, transition_scaling=transition_scaling,
        )
    else:
        cpd = pd_physical.copy()

    df = df.copy()
    df["Baseline_PD"] = baseline_pd
    df["CPD"] = cpd
    df["PD_Uplift_Pct"] = ((cpd - baseline_pd) / (baseline_pd + 1e-8)) * 100
    df["Risk_Category"] = pd.cut(
        cpd, bins=config.RISK_BINS, labels=config.RISK_LABELS, include_lowest=True,
    ).astype(str)
    ead = df.get("loan_amnt", pd.Series(10000.0, index=df.index))
    df["Expected_Loss"] = cpd * config.DEFAULT_LGD * ead
    return df


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check — returns model status and config version."""
    model_path = os.environ.get("MODEL_PATH", "models/baseline_pd_model.pkl")
    model_ok = os.path.exists(model_path)
    return {
        "status": "ok" if model_ok else "degraded",
        "model_available": model_ok,
        "model_path": model_path,
        "greenscore_version": "1.0.0",
        "scenarios": list(config.CARBON_PRICES.keys()),
    }


@app.get("/scenarios", tags=["Configuration"])
def list_scenarios():
    """Return all supported NGFS carbon price scenarios."""
    return {
        key: {
            "carbon_price_usd_per_tco2": price,
            "description": {
                "orderly": "Orderly transition — early, gradual policy action",
                "disorderly": "Disorderly transition — late, abrupt policy action",
                "hot_house": "Hot house world — minimal climate policy",
                "too_little_too_late": "Too little, too late — delayed insufficient action",
            }.get(key, ""),
        }
        for key, price in config.CARBON_PRICES.items()
    }


@app.post("/predict", response_model=CPDResponse, tags=["Prediction"])
def predict_single(loan: LoanFeatures):
    """
    Compute Climate-Adjusted PD for a **single loan**.

    Returns baseline PD, CPD, PD uplift %, risk category, and expected loss.
    """
    try:
        _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    row = loan.model_dump()
    scenario = row.pop("scenario")
    severity_factor = row.pop("severity_factor")
    transition_scaling = row.pop("transition_scaling")

    df = pd.DataFrame([row])

    # Encode sub-grade and verification proxies with defaults
    df["sub_grade"] = "C3"
    df["sub_grade_num"] = config.SUB_GRADE_ORDER.get("C3", 18)
    df["verification_status"] = "Not Verified"
    df["verification_num"] = 0
    df["term"] = "36 months"
    df["earliest_cr_line"] = "Jan-2010"
    df["credit_history_months"] = 96.0

    # Engineered features
    df["income_to_installment"] = df["annual_inc"] / (df["installment"] * 12 + 1)
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["dti_bucket"] = pd.cut(df["dti"], bins=config.DTI_BINS, labels=config.DTI_LABELS, include_lowest=True).astype(float)
    df["fico_bucket"] = pd.cut(df["fico_range_low"], bins=config.FICO_BINS, labels=config.FICO_LABELS, include_lowest=True).astype(float)
    df["monthly_payment_burden"] = df["installment"] / (df["annual_inc"] / 12 + 1)
    df["credit_utilization_ratio"] = df["revol_bal"] / (df["annual_inc"] + 1)
    df["open_to_total_acc"] = df["open_acc"] / (df["total_acc"] + 1)
    df["bc_limit_to_income"] = df["total_bc_limit"] / (df["annual_inc"] + 1)
    df["rev_limit_to_income"] = df["total_rev_hi_lim"] / (df["annual_inc"] + 1)
    df["recent_accts_ratio"] = df["acc_open_past_24mths"] / (df["total_acc"] + 1)

    df = add_climate_features(df)
    result = _run_cpd_pipeline(df, scenario, severity_factor, transition_scaling)

    r = result.iloc[0]
    return CPDResponse(
        baseline_pd=round(float(r["Baseline_PD"]), 6),
        cpd=round(float(r["CPD"]), 6),
        pd_uplift_pct=round(float(r["PD_Uplift_Pct"]), 2),
        risk_category=str(r["Risk_Category"]),
        expected_loss=round(float(r["Expected_Loss"]), 2),
        scenario=scenario,
        carbon_price=config.CARBON_PRICES[scenario],
        lgd_assumption=config.DEFAULT_LGD,
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(..., description="CSV with loan portfolio data"),
    scenario: str = Query(default="orderly", description="NGFS scenario key"),
    severity_factor: float = Query(default=config.SEVERITY_FACTOR, ge=0.0, le=1.0),
    transition_scaling: float = Query(default=config.TRANSITION_SCALING, ge=0.0, le=1.0),
    output_format: str = Query(default="csv", description="Response format: csv or json"),
):
    """
    Compute CPD for a **batch of loans** from an uploaded CSV file.

    Returns the enriched DataFrame with Baseline_PD, CPD, PD_Uplift_Pct,
    Risk_Category, and Expected_Loss columns appended.

    Accepted query params:
    - **scenario**: NGFS scenario key (default: orderly)
    - **severity_factor**: physical risk multiplier
    - **transition_scaling**: transition risk multiplier
    - **output_format**: `csv` (default) or `json`
    """
    if scenario not in config.CARBON_PRICES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid scenario '{scenario}'. Valid: {list(config.CARBON_PRICES.keys())}",
        )

    try:
        _get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Read uploaded CSV
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Validate required columns
    required = ["dti", "annual_inc", "fico_range_low", "int_rate", "installment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}",
        )

    # Basic type coercion
    for col in ["dti", "annual_inc", "fico_range_low", "installment"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )
    if "int_rate" in df.columns:
        df["int_rate"] = pd.to_numeric(
            df["int_rate"].astype(str).str.replace("%", "", regex=False), errors="coerce"
        )
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)

    n_input = len(df)
    df = df.dropna(subset=["dti", "annual_inc", "fico_range_low"])
    n_clean = len(df)

    if n_clean == 0:
        raise HTTPException(status_code=422, detail="All rows dropped after cleaning required columns.")

    try:
        df = add_climate_features(df)
        result = _run_cpd_pipeline(df, scenario, severity_factor, transition_scaling)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # ── Output ──
    output_cols = list(df.columns) + ["Baseline_PD", "CPD", "PD_Uplift_Pct", "Risk_Category", "Expected_Loss"]
    output_cols = [c for c in output_cols if c in result.columns]
    result_out = result[output_cols]

    summary = {
        "n_input": n_input,
        "n_scored": n_clean,
        "scenario": scenario,
        "carbon_price_usd": config.CARBON_PRICES[scenario],
        "avg_baseline_pd": round(float(result_out["Baseline_PD"].mean()), 6),
        "avg_cpd": round(float(result_out["CPD"].mean()), 6),
        "avg_pd_uplift_pct": round(float(result_out["PD_Uplift_Pct"].mean()), 2),
        "total_expected_loss": round(float(result_out["Expected_Loss"].sum()), 2),
        "risk_distribution": result_out["Risk_Category"].value_counts().to_dict(),
    }

    if output_format == "json":
        return JSONResponse({
            "summary": summary,
            "loans": result_out.fillna(0).to_dict(orient="records"),
        })

    # Default: CSV streaming response
    csv_buf = io.StringIO()
    result_out.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    headers = {
        "X-Summary-N-Scored": str(n_clean),
        "X-Summary-Avg-CPD": str(summary["avg_cpd"]),
        "X-Summary-Total-EL": str(summary["total_expected_loss"]),
    }
    return StreamingResponse(
        io.BytesIO(csv_buf.getvalue().encode()),
        media_type="text/csv",
        headers={**headers, "Content-Disposition": "attachment; filename=cpd_results.csv"},
    )
