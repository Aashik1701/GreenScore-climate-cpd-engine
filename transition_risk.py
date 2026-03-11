"""
GreenScore — Transition Risk Overlay
======================================
Maps loan purpose (or direct sector assignment) to a CO₂ emission intensity
and applies a carbon-cost-driven PD uplift under NGFS Phase V scenarios.

Sources:
  - CPCB (Central Pollution Control Board) industry emission reports
  - IEA World Energy Outlook 2023 — sector benchmarks
  - NGFS Phase V Technical Report (January 2025)

Methodological Note — Purpose-to-Sector Proxy
-----------------------------------------------
LendingClub's ``purpose`` field (e.g. "car", "wedding") is a **personal-loan
category**, not the borrower's employing industry. Mapping "car loan" to
"transport sector CO₂ intensity" is therefore a *proxy*, not a direct
measurement. This is acknowledged as a limitation and is acceptable for an
MVP / academic prototype. For production use, a direct ``sector`` or
``industry`` column should be provided in the uploaded CSV; when present,
the engine will use it automatically instead of the purpose mapping.

Scaling Factor Derivation
--------------------------
The 0.4 transition scaling factor converts the carbon-cost-to-income ratio
into a PD uplift:
  - Bell & van Vuuren (2022) show ~40% profit reduction from carbon costs ≈
    ~40% PD uplift for mid-grade credits.
  - Applied conservatively because personal-loan borrowers are less directly
    exposed to carbon pricing than corporate borrowers.
  - Range in literature: 0.2 (minimal) to 0.6 (aggressive, CCC-grade).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def map_purpose_to_sector(purpose_series: pd.Series) -> pd.Series:
    """
    Map LendingClub loan purpose to emission intensity score.

    Cleans the purpose string and looks it up in ``config.SECTOR_EMISSIONS``.
    Unknown purposes fall back to the ``other`` default (0.25).
    """
    return purpose_series.str.lower().str.replace(' ', '_').map(
        lambda x: config.SECTOR_EMISSIONS.get(x, config.SECTOR_EMISSIONS['other'])
    )


def get_emission_intensity(
    df: pd.DataFrame,
    purpose_col: str = 'purpose',
    sector_col: Optional[str] = None,
) -> np.ndarray:
    """
    Get emission intensity values for each row.

    If ``sector_col`` is provided and exists in ``df``, it is used directly
    (preferred for datasets with explicit industry codes). Otherwise, the
    ``purpose_col`` is mapped via the proxy lookup.
    """
    if sector_col and sector_col in df.columns:
        logger.info("Using direct sector column '%s' for emission intensity.", sector_col)
        return df[sector_col].str.lower().str.replace(' ', '_').map(
            lambda x: config.SECTOR_EMISSIONS.get(x, config.SECTOR_EMISSIONS['other'])
        ).values
    else:
        logger.info("Using purpose → sector proxy mapping from column '%s'.", purpose_col)
        return map_purpose_to_sector(df[purpose_col]).values


def apply_transition_risk(
    pd_physical: np.ndarray,
    purpose_series: pd.Series,
    annual_income: pd.Series,
    scenario: str = 'orderly',
    transition_scaling: Optional[float] = None,
    sector_series: Optional[pd.Series] = None,
) -> np.ndarray:
    """
    Apply transition risk overlay to the physically-adjusted PD.

    Formula:
        carbon_cost_ratio = (emission_intensity × carbon_price) / (income / 1000)
        transition_uplift = carbon_cost_ratio × transition_scaling
        CPD = PD_physical × (1 + transition_uplift)

    Parameters
    ----------
    pd_physical : np.ndarray
        PD values after physical risk overlay.
    purpose_series : pd.Series
        Loan purpose column (used if sector_series is None).
    annual_income : pd.Series
        Borrower annual income.
    scenario : str
        NGFS scenario key: 'orderly', 'disorderly', or 'hot_house'.
    transition_scaling : float, optional
        Override the default transition scaling factor from config (0.4).
    sector_series : pd.Series, optional
        Direct sector/industry column (preferred over purpose proxy).
    """
    ts = transition_scaling if transition_scaling is not None else config.TRANSITION_SCALING
    pd_physical = np.asarray(pd_physical, dtype=float)
    if len(pd_physical) != len(purpose_series):
        raise ValueError(f"Length mismatch: pd_physical ({len(pd_physical)}) != purpose_series ({len(purpose_series)})")
    carbon_price = config.CARBON_PRICES.get(scenario, 100)

    # Determine emission intensity
    if sector_series is not None:
        emission_intensity = sector_series.str.lower().str.replace(' ', '_').map(
            lambda x: config.SECTOR_EMISSIONS.get(x, config.SECTOR_EMISSIONS['other'])
        ).values
    else:
        emission_intensity = map_purpose_to_sector(purpose_series).values

    income = annual_income.fillna(annual_income.median()).values

    # Carbon cost as fraction of income
    carbon_cost_ratio = (emission_intensity * carbon_price) / (income / 1000 + 1e-6)
    carbon_cost_ratio = np.clip(carbon_cost_ratio, 0, config.TRANSITION_COST_CLIP_MAX)

    transition_uplift = carbon_cost_ratio * ts
    cpd = pd_physical * (1 + transition_uplift)

    logger.info(
        "Transition risk applied — scenario=%s, carbon_price=$%d/tCO₂, "
        "scaling=%.2f, mean_uplift=%.4f",
        scenario, carbon_price, ts, (cpd - pd_physical).mean(),
    )
    return np.clip(cpd, 0, 1)
