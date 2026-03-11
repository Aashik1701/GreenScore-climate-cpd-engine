import pandas as pd
import numpy as np

# ── State-level physical risk scores (0–1 scale) ──
# Based on flood/cyclone/heat exposure
# Sources: IMD flood zone data, NDMA cyclone tracks, disaster risk index
INDIA_STATE_RISK = {
    # Indian states
    'Andhra Pradesh': 0.75, 'Assam': 0.85, 'Bihar': 0.80,
    'Gujarat': 0.65, 'Karnataka': 0.45, 'Kerala': 0.70,
    'Maharashtra': 0.55, 'Odisha': 0.90, 'Tamil Nadu': 0.65,
    'Uttar Pradesh': 0.60, 'West Bengal': 0.85,
    'Rajasthan': 0.50, 'Madhya Pradesh': 0.45,
    'Punjab': 0.40, 'Haryana': 0.35,
    # US states (for LendingClub data)
    'CA': 0.60, 'FL': 0.80, 'TX': 0.65, 'NY': 0.45,
    'WA': 0.55, 'LA': 0.85, 'NC': 0.60, 'SC': 0.65,
    'GA': 0.55, 'AL': 0.70, 'MS': 0.75, 'NJ': 0.50,
    'PA': 0.40, 'OH': 0.35, 'IL': 0.40, 'MI': 0.38,
    'VA': 0.45, 'MD': 0.48, 'AZ': 0.42, 'CO': 0.38,
    'MN': 0.40, 'WI': 0.38, 'IN': 0.42, 'MO': 0.50,
    'TN': 0.52, 'KY': 0.48, 'OR': 0.45, 'NV': 0.35,
    'CT': 0.45, 'MA': 0.42, 'OTHER': 0.45
}

# Calibrated from Bell & van Vuuren (2022) scaling factor matrix
SEVERITY_FACTOR = 0.3


def compute_physical_risk_score(location_series: pd.Series) -> pd.Series:
    """Map location (state code or name) to physical risk score (0–1)."""
    return location_series.map(
        lambda x: INDIA_STATE_RISK.get(str(x).strip(), INDIA_STATE_RISK['OTHER'])
    )


def apply_physical_risk(baseline_pd: np.ndarray,
                         location_series: pd.Series) -> np.ndarray:
    """
    Apply physical risk overlay to baseline PD.
    Formula from Bell & van Vuuren (2022):
        PD_physical = PD_base × (1 + PR × severity_factor)
    """
    pr_scores = compute_physical_risk_score(location_series).values
    adjusted_pd = baseline_pd * (1 + pr_scores * SEVERITY_FACTOR)
    return np.clip(adjusted_pd, 0, 1)
