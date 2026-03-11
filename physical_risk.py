"""
GreenScore — Physical Risk Overlay
====================================
Maps borrower location (state) to a physical hazard score (0–1)
and applies a severity-weighted PD uplift based on Bell & van Vuuren (2022).

Sources:
  - India: IMD flood zone classifications, NDMA disaster risk index
  - US:    FEMA National Risk Index, NOAA historical cyclone/hurricane tracks
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def compute_physical_risk_score(location_series: pd.Series) -> pd.Series:
    """
    Map location (state code or full name) to a physical risk score (0–1).

    Unknown states fall back to the ``OTHER`` default (0.45).
    """
    scores = location_series.map(
        lambda x: config.STATE_PHYSICAL_RISK.get(
            str(x).strip(), config.STATE_PHYSICAL_RISK['OTHER']
        )
    )
    n_fallback = (scores == config.STATE_PHYSICAL_RISK['OTHER']).sum()
    if n_fallback > 0:
        logger.debug("%d locations mapped to fallback risk score.", n_fallback)
    return scores


def apply_physical_risk(
    baseline_pd: np.ndarray,
    location_series: pd.Series,
    severity_factor: Optional[float] = None,
) -> np.ndarray:
    """
    Apply physical risk overlay to baseline PD.

    Formula (Bell & van Vuuren 2022):
        PD_physical = PD_base × (1 + risk_score × severity_factor)

    Parameters
    ----------
    baseline_pd : np.ndarray
        Baseline probability-of-default values.
    location_series : pd.Series
        Borrower state codes or names.
    severity_factor : float, optional
        Override the default severity factor from config (0.3).
    """
    sf = severity_factor if severity_factor is not None else config.SEVERITY_FACTOR
    baseline_pd = np.asarray(baseline_pd, dtype=float)
    if len(baseline_pd) != len(location_series):
        raise ValueError(f"Length mismatch: baseline_pd ({len(baseline_pd)}) != location_series ({len(location_series)})")
    pr_scores = compute_physical_risk_score(location_series).values
    adjusted_pd = baseline_pd * (1 + pr_scores * sf)
    logger.info(
        "Physical risk applied — severity_factor=%.2f, mean_uplift=%.4f",
        sf,
        (adjusted_pd - baseline_pd).mean(),
    )
    return np.clip(adjusted_pd, 0, 1)
