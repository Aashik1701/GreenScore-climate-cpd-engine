"""
GreenScore CPD Engine — Baseline Probability of Default
========================================================
Loads LendingClub data, engineers features, trains an XGBoost
baseline PD classifier with Optuna hyperparameter tuning,
cross-validation, model comparison (LR, RF, LightGBM),
SHAP explanations, and provides prediction functions for
the dashboard pipeline.
"""

import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config
from nasa_power import get_physical_features_for_state, engineer_physical_features

# ── Logging ──
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────

def load_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load LendingClub CSV with only the columns needed for PD modelling.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    nrows : int, optional
        Number of rows to read (for faster demos).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with a binary ``default`` column.
    """
    logger.info("Loading data from %s (nrows=%s)…", path, nrows)
    df = pd.read_csv(path, usecols=config.RAW_COLS, low_memory=False, nrows=nrows)

    # Binary target: 1 = default, 0 = paid
    df['default'] = df['loan_status'].apply(
        lambda x: 1 if str(x) in config.DEFAULT_STATUSES else 0
    )
    df = df.dropna(subset=['dti', 'annual_inc', 'fico_range_low'])

    # Clean numeric columns
    df['int_rate'] = (
        df['int_rate'].astype(str).str.replace('%', '', regex=False).astype(float)
    )
    df['emp_length'] = (
        df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
    )

    # ── Phase 1.3: Clean new raw columns ──
    df['revol_util'] = (
        df['revol_util'].astype(str).str.replace('%', '', regex=False)
    )
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce').fillna(0)
    df['revol_bal'] = pd.to_numeric(df['revol_bal'], errors='coerce').fillna(0)
    df['open_acc'] = pd.to_numeric(df['open_acc'], errors='coerce').fillna(0)
    df['total_acc'] = pd.to_numeric(df['total_acc'], errors='coerce').fillna(0)
    df['pub_rec'] = pd.to_numeric(df['pub_rec'], errors='coerce').fillna(0)
    df['delinq_2yrs'] = pd.to_numeric(df['delinq_2yrs'], errors='coerce').fillna(0)
    df['inq_last_6mths'] = pd.to_numeric(df['inq_last_6mths'], errors='coerce').fillna(0)
    df['term_months'] = (
        df['term'].astype(str).str.extract(r'(\d+)').astype(float).fillna(36)
    )

    # Sub-grade: ordinal encode A1=1 ... G5=35
    df['sub_grade_num'] = df['sub_grade'].map(config.SUB_GRADE_ORDER).fillna(18)

    # Verification status: ordinal encode
    df['verification_num'] = df['verification_status'].map(config.VERIFICATION_MAP).fillna(0)

    # Credit history length in months from earliest_cr_line
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['credit_history_months'] = (
        (pd.Timestamp('2018-01-01') - df['earliest_cr_line']).dt.days / 30.44
    ).fillna(180)

    # ── Phase 1.8: Clean credit bureau depth columns ──
    for col in [
        'acc_open_past_24mths', 'mort_acc', 'total_bc_limit',
        'total_rev_hi_lim', 'mo_sin_rcnt_tl', 'mo_sin_old_rev_tl_op',
        'num_actv_rev_tl', 'percent_bc_gt_75', 'bc_util',
        'mths_since_recent_inq',
    ]:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    # ── Feature Engineering (original) ──
    df['income_to_installment'] = df['annual_inc'] / (df['installment'] * 12 + 1)
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['dti_bucket'] = pd.cut(
        df['dti'], bins=config.DTI_BINS, labels=config.DTI_LABELS, include_lowest=True
    ).astype(float)
    df['fico_bucket'] = pd.cut(
        df['fico_range_low'], bins=config.FICO_BINS, labels=config.FICO_LABELS, include_lowest=True
    ).astype(float)

    # ── Phase 1.3: New engineered features ──
    # Additional ratios
    df['monthly_payment_burden'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['credit_utilization_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['open_to_total_acc'] = df['open_acc'] / (df['total_acc'] + 1)

    # ── Phase 1.8: Bureau-depth ratios ──
    df['bc_limit_to_income'] = df['total_bc_limit'] / (df['annual_inc'] + 1)
    df['rev_limit_to_income'] = df['total_rev_hi_lim'] / (df['annual_inc'] + 1)
    df['recent_accts_ratio'] = df['acc_open_past_24mths'] / (df['total_acc'] + 1)

    logger.info(
        "Loaded %s records. Default rate: %.3f",
        f"{len(df):,}", df['default'].mean(),
    )
    return df


def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a loan DataFrame with physical-risk, transition-risk, and
    geographic features so that XGBoost can learn climate signals.

    Adds the columns listed in ``config.CLIMATE_FEATURES``.
    """
    df = df.copy()
    loc_col = 'addr_state' if 'addr_state' in df.columns else None
    purpose_col = 'purpose' if 'purpose' in df.columns else None

    # ── Physical risk features (NASA POWER) ──
    if loc_col:
        unique_states = df[loc_col].dropna().unique()
        state_features = {}
        for state in unique_states:
            state_features[state] = get_physical_features_for_state(
                state, config.US_STATE_COORDS, config.INDIA_STATE_COORDS,
            )
        for feat in config.PHYSICAL_RISK_FEATURES:
            df[feat] = df[loc_col].map(
                lambda s, f=feat: state_features.get(s, {}).get(f, 0.3)
            )
    else:
        for feat in config.PHYSICAL_RISK_FEATURES:
            default_val = 0.3 if 'score' in feat or 'index' in feat else 0.0
            df[feat] = default_val

    # ── Transition risk features (NGFS) ──
    if purpose_col:
        cleaned = df[purpose_col].str.lower().str.replace(' ', '_')
        df['sector_carbon_intensity'] = cleaned.map(
            lambda x: config.SECTOR_EMISSIONS.get(x, config.SECTOR_EMISSIONS['other'])
        )
        df['policy_exposure_score'] = cleaned.map(
            lambda x: config.SECTOR_POLICY_EXPOSURE.get(x, 0.20)
        )
        # Composite transition risk score
        df['transition_risk_score'] = (
            0.60 * df['sector_carbon_intensity'].clip(0, 1)
            + 0.40 * df['policy_exposure_score']
        ).clip(0, 1)
    else:
        df['sector_carbon_intensity'] = 0.25
        df['policy_exposure_score'] = 0.20
        df['transition_risk_score'] = 0.22

    # ── Geographic features ──
    if loc_col:
        df['coastal_proximity_km'] = df[loc_col].map(
            lambda s: config.STATE_COASTAL_PROXIMITY_KM.get(s, 500)
        )
        df['elevation_meters'] = df[loc_col].map(
            lambda s: config.STATE_ELEVATION_METERS.get(s, 300)
        )
    else:
        df['coastal_proximity_km'] = 500
        df['elevation_meters'] = 300

    logger.info(
        "Climate features added — %d physical, %d transition, %d geographic.",
        len(config.PHYSICAL_RISK_FEATURES),
        len(config.TRANSITION_RISK_FEATURES),
        len(config.GEOGRAPHIC_FEATURES),
    )
    return df


# ─────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame, include_climate: bool = True) -> pd.DataFrame:
    """Select and impute model features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must contain at least ``config.ALL_FEATURES`` columns).
    include_climate : bool
        If *True* and climate columns are present, use the full
        ``ALL_FEATURES_CLIMATE`` feature set.
    """
    if include_climate and all(f in df.columns for f in config.CLIMATE_FEATURES):
        feature_cols = config.ALL_FEATURES_CLIMATE
    else:
        feature_cols = config.ALL_FEATURES
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    return X


def train_baseline_pd(df: pd.DataFrame, save_dir: str = 'models', tune: bool = False):
    """
    Train an XGBoost baseline PD model with:
      - Optional Optuna hyperparameter tuning (``tune=True``)
      - ``scale_pos_weight`` for class-imbalance handling
      - 5-fold stratified cross-validation
      - Comparison against Logistic Regression, Random Forest, and LightGBM
      - SHAP feature importance plot
      - Saved evaluation artefacts (confusion matrix, ROC, PR curves)

    Returns
    -------
    model : XGBClassifier
        The trained XGBoost model.
    results : dict
        Dictionary with AUC scores for all models and CV statistics.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Train on financial features only — climate adjustments are applied
    # post-prediction by physical_risk.py and transition_risk.py (CPD formula).
    # Including climate features during training adds noise without signal.
    X = _prepare_features(df, include_climate=False)
    y = df['default']

    # Class imbalance ratio
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    logger.info("Class balance — neg: %s, pos: %s, scale_pos_weight: %.2f", neg, pos, scale_pos_weight)

    # ── Three-way split: 60% train / 20% calibration / 20% test ──
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )
    # Split trainval into train (75%) and calib (25%) → 60/20/20 overall
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=config.RANDOM_STATE, stratify=y_trainval,
    )

    # ── Optuna Hyperparameter Tuning ──
    if tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("Starting Optuna hyperparameter tuning (%d trials)…", config.OPTUNA_N_TRIALS)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', *config.OPTUNA_PARAM_SPACE['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *config.OPTUNA_PARAM_SPACE['learning_rate'], log=True),
                'n_estimators': trial.suggest_int('n_estimators', *config.OPTUNA_PARAM_SPACE['n_estimators'], step=50),
                'subsample': trial.suggest_float('subsample', *config.OPTUNA_PARAM_SPACE['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *config.OPTUNA_PARAM_SPACE['colsample_bytree']),
                'min_child_weight': trial.suggest_int('min_child_weight', *config.OPTUNA_PARAM_SPACE['min_child_weight']),
                'reg_alpha': trial.suggest_float('reg_alpha', *config.OPTUNA_PARAM_SPACE['reg_alpha'], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', *config.OPTUNA_PARAM_SPACE['reg_lambda'], log=True),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'auc',
                'random_state': config.RANDOM_STATE,
                'verbosity': 0,
            }
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
            scores = cross_val_score(
                XGBClassifier(**params), X_train, y_train,
                cv=skf, scoring='roc_auc', n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)

        best_params = study.best_params
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'auc',
            'random_state': config.RANDOM_STATE,
            'verbosity': 0,
        })
        logger.info("Optuna best AUC (3-fold on train): %.4f", study.best_value)
        logger.info("Optuna best params: %s", {k: v for k, v in best_params.items() if k not in ('eval_metric', 'random_state', 'verbosity', 'scale_pos_weight')})

        xgb = XGBClassifier(**best_params)
    else:
        xgb_params = {**config.XGBOOST_PARAMS, 'scale_pos_weight': scale_pos_weight}
        xgb_params.pop('early_stopping_rounds', None)
        xgb = XGBClassifier(**xgb_params)

    # ── XGBoost Training ──
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_auc_raw = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    logger.info("XGBoost AUC (raw, before calibration): %.4f", xgb_auc_raw)

    # ── Phase 12.1: Isotonic Probability Calibration ──
    # Calibrate on the held-out calibration set so the model was never
    # trained on those samples (cv='prefit' = calibrate a pre-fitted estimator).
    calibrated_model = CalibratedClassifierCV(xgb, method='isotonic', cv='prefit')
    calibrated_model.fit(X_calib, y_calib)
    xgb_auc = roc_auc_score(y_test, calibrated_model.predict_proba(X_test)[:, 1])
    logger.info("XGBoost AUC (calibrated): %.4f", xgb_auc)

    logger.info("Model trained on %d features: %s", X.shape[1], list(X.columns))

    # ── 5-Fold Stratified CV for XGBoost ──
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_params = xgb.get_params()
    cv_params.pop('early_stopping_rounds', None)
    cv_params.pop('callbacks', None)
    xgb_cv_scores = cross_val_score(
        XGBClassifier(**cv_params),
        X, y, cv=skf, scoring='roc_auc', n_jobs=-1,
    )
    logger.info("XGBoost 5-Fold CV AUC: %.4f ± %.4f", xgb_cv_scores.mean(), xgb_cv_scores.std())

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=config.RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    logger.info("Logistic Regression AUC: %.4f", lr_auc)

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    logger.info("Random Forest AUC: %.4f", rf_auc)

    # ── LightGBM ──
    lgbm_params = {**config.LIGHTGBM_PARAMS, 'scale_pos_weight': scale_pos_weight}
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
    logger.info("LightGBM AUC: %.4f", lgbm_auc)

    # ── Save evaluation plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Confusion Matrix (calibrated model, cost-optimal threshold pre-computed below)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            xgb, X_test, y_test, ax=ax_cm, cmap='Blues',
        )
        ax_cm.set_title('XGBoost — Confusion Matrix')
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig_cm)

        # ROC Curve — all models
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(calibrated_model, X_test, y_test, ax=ax_roc, name='XGBoost (calibrated)')
        RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=ax_roc, name='XGBoost (raw)', linestyle='--', alpha=0.6)
        RocCurveDisplay.from_estimator(lr, X_test, y_test, ax=ax_roc, name='Logistic Regression')
        RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax_roc, name='Random Forest')
        RocCurveDisplay.from_estimator(lgbm, X_test, y_test, ax=ax_roc, name='LightGBM')
        ax_roc.set_title('ROC Curve Comparison')
        fig_roc.tight_layout()
        fig_roc.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
        plt.close(fig_roc)

        # Precision-Recall Curve
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_estimator(calibrated_model, X_test, y_test, ax=ax_pr, name='XGBoost (calibrated)')
        PrecisionRecallDisplay.from_estimator(lr, X_test, y_test, ax=ax_pr, name='Logistic Regression')
        PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax_pr, name='Random Forest')
        PrecisionRecallDisplay.from_estimator(lgbm, X_test, y_test, ax=ax_pr, name='LightGBM')
        ax_pr.set_title('Precision-Recall Curve Comparison')
        fig_pr.tight_layout()
        fig_pr.savefig(os.path.join(save_dir, 'precision_recall_curves.png'), dpi=150)
        plt.close(fig_pr)

        logger.info("Saved evaluation plots to %s/", save_dir)
    except Exception as e:
        logger.warning("Could not save evaluation plots: %s", e)

    # ── SHAP Feature Importance (use raw XGBoost — TreeExplainer needs a tree model) ──
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=config.RANDOM_STATE)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer(shap_sample)

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved SHAP explanation plots to %s/", save_dir)
    except Exception as e:
        logger.warning("Could not save SHAP plots: %s", e)

    # ── Phase 12.2: Cost-Optimal Threshold ──
    y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob_cal)
    # Cost function: FN_COST * FN_rate + FP_COST * FP_rate
    # FN_rate = 1 - recall;  FP_rate = (1 - precision) * (pos_in_test / neg_in_test)
    fn_costs = config.FN_COST * (1 - rec[:-1])
    fp_costs = config.FP_COST * (1 - prec[:-1])
    total_cost = fn_costs + fp_costs
    best_idx = int(np.argmin(total_cost))
    optimal_threshold = float(thresholds[best_idx])
    # Also compute F1-optimal threshold for reference
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    f1_optimal_threshold = float(thresholds[int(np.argmax(f1_scores))])
    logger.info(
        "Optimal threshold (cost-based): %.4f   F1-optimal: %.4f",
        optimal_threshold, f1_optimal_threshold,
    )
    # Write to file so app.py / other scripts can load it without retraining
    threshold_path = os.path.join(save_dir, 'optimal_threshold.txt')
    with open(threshold_path, 'w') as f:
        f.write(f"cost_optimal={optimal_threshold:.6f}\n")
        f.write(f"f1_optimal={f1_optimal_threshold:.6f}\n")
        f.write(f"fn_cost={config.FN_COST}\nfp_cost={config.FP_COST}\n")

    # ── Phase 12.1: Calibration plot (reliability diagram) ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
        # Raw XGBoost
        frac_raw, mean_raw = calibration_curve(
            y_test, xgb.predict_proba(X_test)[:, 1], n_bins=15, strategy='quantile',
        )
        # Calibrated
        frac_cal, mean_cal = calibration_curve(
            y_test, y_prob_cal, n_bins=15, strategy='quantile',
        )
        ax_cal.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax_cal.plot(mean_raw, frac_raw, 's-', label=f'Raw XGBoost (AUC={xgb_auc_raw:.4f})', alpha=0.8)
        ax_cal.plot(mean_cal, frac_cal, 'o-', label=f'Calibrated (AUC={xgb_auc:.4f})', alpha=0.8)
        ax_cal.set_xlabel('Mean predicted probability')
        ax_cal.set_ylabel('Fraction of positives')
        ax_cal.set_title('Reliability Diagram — Probability Calibration')
        ax_cal.legend(loc='upper left')
        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        fig_cal.tight_layout()
        fig_cal.savefig(os.path.join(save_dir, 'calibration_curve.png'), dpi=150)
        plt.close(fig_cal)
        logger.info("Calibration plot saved to %s/calibration_curve.png", save_dir)
    except Exception as e:
        logger.warning("Could not save calibration plot: %s", e)

    # ── Classification report (using calibrated model + cost-optimal threshold) ──
    y_pred = (y_prob_cal >= optimal_threshold).astype(int)
    report = classification_report(y_test, y_pred, target_names=['Paid', 'Default'])
    logger.info("Classification Report (threshold=%.4f):\n%s", optimal_threshold, report)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Threshold: {optimal_threshold:.6f} (cost-optimal, FN:FP={config.FN_COST}:{config.FP_COST})\n\n")
        f.write(report)

    # ── Save training feature statistics for PSI drift detection ──
    feat_stats = X_train.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    feat_stats.to_csv(os.path.join(save_dir, 'training_feature_stats.csv'))
    logger.info("Training feature stats saved for drift detection.")

    # ── Save the calibrated model ──
    model_path = os.path.join(save_dir, 'baseline_pd_model.pkl')
    joblib.dump(calibrated_model, model_path)
    logger.info("Calibrated model saved to %s", model_path)

    results = {
        'xgboost_auc': xgb_auc,
        'xgboost_auc_raw': xgb_auc_raw,
        'xgboost_cv_mean': xgb_cv_scores.mean(),
        'xgboost_cv_std': xgb_cv_scores.std(),
        'logistic_regression_auc': lr_auc,
        'random_forest_auc': rf_auc,
        'lightgbm_auc': lgbm_auc,
        'scale_pos_weight': scale_pos_weight,
        'n_samples': len(df),
        'default_rate': float(y.mean()),
        'optimal_threshold': optimal_threshold,
        'f1_optimal_threshold': f1_optimal_threshold,
        'tuned': tune,
    }
    return calibrated_model, results


# ─────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────

def get_baseline_pd(model, loan_data: pd.DataFrame) -> np.ndarray:
    """
    Get baseline PD predictions for loan data.

    Automatically engineers the required features if they are missing.
    Uses climate features if the model was trained with them.
    """
    df = loan_data.copy()

    # Ensure engineered features exist
    if 'income_to_installment' not in df.columns:
        df['income_to_installment'] = df['annual_inc'] / (df['installment'] * 12 + 1)
    if 'loan_to_income' not in df.columns:
        df['loan_to_income'] = df.get('loan_amnt', 0) / (df['annual_inc'] + 1)
    if 'dti_bucket' not in df.columns:
        df['dti_bucket'] = pd.cut(
            df['dti'], bins=config.DTI_BINS, labels=config.DTI_LABELS, include_lowest=True
        ).astype(float)
    if 'fico_bucket' not in df.columns:
        df['fico_bucket'] = pd.cut(
            df['fico_range_low'], bins=config.FICO_BINS, labels=config.FICO_LABELS, include_lowest=True
        ).astype(float)

    # ── Phase 1.3: Ensure new raw features have defaults ──
    for col, default in [
        ('revol_util', 0), ('revol_bal', 0), ('open_acc', 0),
        ('total_acc', 0), ('pub_rec', 0), ('delinq_2yrs', 0),
        ('inq_last_6mths', 0), ('loan_amnt', 0), ('term_months', 36),
        ('sub_grade_num', 18), ('verification_num', 0),
        ('credit_history_months', 180),
        # Phase 1.8 — credit bureau depth defaults
        ('acc_open_past_24mths', 0), ('mort_acc', 0),
        ('total_bc_limit', 0), ('total_rev_hi_lim', 0),
        ('mo_sin_rcnt_tl', 12), ('mo_sin_old_rev_tl_op', 180),
        ('num_actv_rev_tl', 0), ('percent_bc_gt_75', 0),
        ('bc_util', 0), ('mths_since_recent_inq', 12),
    ]:
        if col not in df.columns:
            df[col] = default

    # ── Phase 1.3: Ensure new engineered features exist ──
    if 'monthly_payment_burden' not in df.columns:
        df['monthly_payment_burden'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    if 'credit_utilization_ratio' not in df.columns:
        df['credit_utilization_ratio'] = df.get('revol_bal', 0) / (df['annual_inc'] + 1)
    if 'open_to_total_acc' not in df.columns:
        df['open_to_total_acc'] = df.get('open_acc', 0) / (df.get('total_acc', 0) + 1)

    # ── Phase 1.8: Ensure bureau-depth ratios exist ──
    if 'bc_limit_to_income' not in df.columns:
        df['bc_limit_to_income'] = df.get('total_bc_limit', 0) / (df['annual_inc'] + 1)
    if 'rev_limit_to_income' not in df.columns:
        df['rev_limit_to_income'] = df.get('total_rev_hi_lim', 0) / (df['annual_inc'] + 1)
    if 'recent_accts_ratio' not in df.columns:
        df['recent_accts_ratio'] = df.get('acc_open_past_24mths', 0) / (df.get('total_acc', 0) + 1)

    # Use financial features only — climate adjustments are post-prediction
    feature_cols = config.ALL_FEATURES

    X = df[feature_cols].fillna(df[feature_cols].median())
    return model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────
# PSI Drift Detection (Phase 12.3)
# ─────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = None) -> float:
    """
    Population Stability Index (PSI) between two distributions.

    PSI < 0.10  → stable (no meaningful drift)
    0.10-0.25   → moderate drift (investigate)
    PSI > 0.25  → significant drift (retrain recommended)

    Parameters
    ----------
    expected : np.ndarray
        Feature values from the training distribution.
    actual : np.ndarray
        Feature values from the new / scored dataset.
    n_bins : int, optional
        Number of quantile bins. Defaults to ``config.PSI_BINS``.

    Returns
    -------
    float
        PSI value.
    """
    if n_bins is None:
        n_bins = config.PSI_BINS

    # Build bin edges from expected distribution (quantile-based)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(expected, percentiles))

    # Ensure at least 2 distinct edges
    if len(bin_edges) < 2:
        return 0.0

    # Extend edges to capture full range
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions; replace zeros to avoid log(0)
    expected_pct = np.where(expected_counts == 0, 1e-4, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 1e-4, actual_counts / max(len(actual), 1))

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def check_feature_drift(
    new_data: pd.DataFrame,
    stats_path: str = 'models/training_feature_stats.csv',
) -> pd.DataFrame:
    """
    Compute PSI for all model features between the training distribution
    (loaded from ``stats_path``) and ``new_data``.

    Returns a DataFrame sorted by PSI descending, with a ``status`` column:
    - ``OK``       — PSI < 0.10
    - ``MODERATE`` — 0.10 ≤ PSI < 0.25
    - ``DRIFT``    — PSI ≥ 0.25 (retrain recommended)
    """
    if not os.path.exists(stats_path):
        logger.warning("Training feature stats not found at %s. Run training first.", stats_path)
        return pd.DataFrame(columns=['feature', 'psi', 'status'])

    train_stats = pd.read_csv(stats_path, index_col=0)

    results = []
    for col in config.ALL_FEATURES:
        if col not in new_data.columns:
            continue
        if col not in train_stats.index:
            continue

        # Reconstruct approximate training distribution from percentile stats
        row = train_stats.loc[col]
        # Build synthetic reference values from stored percentile summary
        ref_vals = np.array([
            row.get('min', 0), row.get('10%', 0), row.get('25%', 0),
            row.get('50%', 0), row.get('75%', 0), row.get('90%', 0),
            row.get('max', 1),
        ], dtype=float)
        # Weight the reference values to approximate the full distribution
        weights = np.array([5, 10, 15, 20, 15, 10, 5], dtype=float)
        ref = np.repeat(ref_vals, (weights * 100 / weights.sum()).astype(int))

        actual = new_data[col].dropna().values.astype(float)
        if len(actual) == 0:
            continue

        psi = compute_psi(ref, actual)
        if psi >= config.PSI_CRITICAL:
            status = 'DRIFT'
        elif psi >= config.PSI_MODERATE:
            status = 'MODERATE'
        else:
            status = 'OK'

        results.append({'feature': col, 'psi': round(psi, 5), 'status': status})

    drift_df = pd.DataFrame(results).sort_values('psi', ascending=False).reset_index(drop=True)
    n_drift = (drift_df['status'] == 'DRIFT').sum()
    n_moderate = (drift_df['status'] == 'MODERATE').sum()
    logger.info(
        "Drift check: %d features — %d DRIFT, %d MODERATE, %d OK",
        len(drift_df), n_drift, n_moderate,
        (drift_df['status'] == 'OK').sum(),
    )
    return drift_df


# ─────────────────────────────────────────────────────────
# Cross-Dataset Validation (Phase 6.3)
# ─────────────────────────────────────────────────────────

def cross_dataset_validate(
    model_path: str = 'models/baseline_pd_model.pkl',
    save_dir: str = 'models',
) -> dict:
    """
    Validate the LendingClub-trained PD model on the Home Credit dataset
    to test cross-dataset generalisation.

    Returns dict with AUC, classification report, and sample statistics.
    """
    from dataset_adapters import adapt_home_credit

    logger.info("Loading saved model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Adapting Home Credit dataset…")
    df = adapt_home_credit()

    y_true = df['default'].values
    logger.info(
        "Home Credit: %s records, default rate %.3f",
        f"{len(df):,}", y_true.mean(),
    )

    # Predict using the LendingClub-trained model
    y_prob = get_baseline_pd(model, df)

    # AUC
    auc = roc_auc_score(y_true, y_prob)
    logger.info("Cross-dataset AUC: %.4f", auc)

    # Classification report at 0.5 threshold
    y_pred_50 = (y_prob >= 0.5).astype(int)
    report_50 = classification_report(
        y_true, y_pred_50, target_names=['Paid', 'Default'], zero_division=0,
    )

    # Also evaluate at a threshold matching the default rate (more balanced)
    threshold_balanced = np.percentile(y_prob, 100 * (1 - y_true.mean()))
    y_pred_bal = (y_prob >= threshold_balanced).astype(int)
    report_bal = classification_report(
        y_true, y_pred_bal, target_names=['Paid', 'Default'], zero_division=0,
    )

    # Save report
    report_path = os.path.join(save_dir, 'cross_dataset_validation.txt')
    with open(report_path, 'w') as f:
        f.write("GreenScore — Cross-Dataset Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training dataset:    LendingClub (US consumer loans)\n")
        f.write(f"Validation dataset:  Home Credit Default Risk\n")
        f.write(f"Validation records:  {len(df):,}\n")
        f.write(f"Validation default rate: {y_true.mean():.4f}\n\n")
        f.write(f"Cross-Dataset AUC: {auc:.4f}\n\n")
        f.write(f"Classification Report (threshold=0.50):\n{report_50}\n")
        f.write(f"Classification Report (threshold={threshold_balanced:.3f} — "
                f"matched to default rate):\n{report_bal}\n")
    logger.info("Validation report saved to %s", report_path)

    # Save ROC curve
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(
            y_true, y_prob, name='XGBoost (cross-dataset)', ax=ax,
        )
        ax.set_title(f'Cross-Dataset ROC — Home Credit (AUC={auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.legend()
        plt.tight_layout()
        fig.savefig(
            os.path.join(save_dir, 'cross_dataset_roc.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)
        logger.info("ROC curve saved to %s/cross_dataset_roc.png", save_dir)
    except Exception as e:
        logger.warning("Could not save ROC curve: %s", e)

    return {
        'auc': auc,
        'n_records': len(df),
        'default_rate': float(y_true.mean()),
        'threshold_balanced': float(threshold_balanced),
        'report_path': report_path,
    }


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    if '--validate' in sys.argv:
        # Cross-dataset validation mode
        print("Running cross-dataset validation on Home Credit…")
        results = cross_dataset_validate()
        print(f"\n═══ Cross-Dataset Validation ═══")
        print(f"  Dataset:       Home Credit Default Risk")
        print(f"  Records:       {results['n_records']:,}")
        print(f"  Default Rate:  {results['default_rate']:.4f}")
        print(f"  AUC:           {results['auc']:.4f}")
        print(f"\n  Full report:   {results['report_path']}")
    else:
        positional = [a for a in sys.argv[1:] if not a.startswith('--')]
        path = positional[0] if len(positional) > 0 else 'data/accepted_2007_to_2018Q4.csv'
        nrows = int(positional[1]) if len(positional) > 1 else 500_000
        do_tune = '--tune' in sys.argv
        print(f"Loading data from {path} ({nrows} rows)…")
        if do_tune:
            print(f"Optuna tuning enabled ({config.OPTUNA_N_TRIALS} trials)")
        df = load_data(path, nrows=nrows)
        print("Adding climate features (NASA POWER + NGFS)…")
        df = add_climate_features(df)
        model, results = train_baseline_pd(df, tune=do_tune)
        print("\n═══ Model Comparison ═══")
        print(f"  XGBoost AUC (calibrated):  {results['xgboost_auc']:.4f}")
        print(f"  XGBoost AUC (raw):         {results['xgboost_auc_raw']:.4f}")
        print(f"  XGBoost AUC (5-fold CV):   {results['xgboost_cv_mean']:.4f} ± {results['xgboost_cv_std']:.4f}")
        print(f"  Logistic Regression AUC:   {results['logistic_regression_auc']:.4f}")
        print(f"  Random Forest AUC:         {results['random_forest_auc']:.4f}")
        print(f"  LightGBM AUC:              {results['lightgbm_auc']:.4f}")
        print(f"  Scale Pos Weight:          {results['scale_pos_weight']:.2f}")
        print(f"  Cost-optimal threshold:    {results['optimal_threshold']:.4f}")
        print(f"  F1-optimal threshold:      {results['f1_optimal_threshold']:.4f}")
        if results['tuned']:
            print("  Optuna Tuning:            ✅ Applied")
        print(f"\nModel + plots + SHAP saved to models/")
