import pandas as pd
import numpy as np

# ── Sectoral CO2 intensity (tCO2 per $1000 revenue) ──
# Source: CPCB / NGFS sectoral data
SECTOR_EMISSIONS = {
    # Industry sectors
    'cement': 0.85, 'steel': 0.75, 'thermal_power': 0.90,
    'coal': 0.95, 'oil_gas': 0.70, 'chemicals': 0.55,
    'manufacturing': 0.45, 'transport': 0.40,
    'agriculture': 0.35, 'construction': 0.30,
    'retail': 0.15, 'services': 0.10,
    'technology': 0.08, 'healthcare': 0.12,
    'renewables': -0.10, 'other': 0.25,
    # LendingClub purpose mapping
    'debt_consolidation': 0.15, 'credit_card': 0.10,
    'home_improvement': 0.20, 'small_business': 0.30,
    'car': 0.40, 'medical': 0.12, 'moving': 0.25,
    'vacation': 0.15, 'house': 0.20, 'wedding': 0.10,
    'major_purchase': 0.20, 'educational': 0.08,
    'renewable_energy': -0.10
}

# NGFS Phase V Carbon Price Pathways (USD per tCO2)
CARBON_PRICES = {
    'orderly': 100,      # Gradual transition by 2030
    'disorderly': 250,   # Sudden policy shock
    'hot_house': 25      # No transition, minimal pricing
}


def map_purpose_to_sector(purpose_series: pd.Series) -> pd.Series:
    """Map LendingClub loan purpose to emission intensity score."""
    return purpose_series.str.lower().str.replace(' ', '_').map(
        lambda x: SECTOR_EMISSIONS.get(x, SECTOR_EMISSIONS['other'])
    )


def apply_transition_risk(pd_physical: np.ndarray,
                          purpose_series: pd.Series,
                          annual_income: pd.Series,
                          scenario: str = 'orderly') -> np.ndarray:
    """
    Apply transition risk overlay.
    
    Transition risk adjustment:
        Stressed_Profit = Profit × (1 - (Emissions × Carbon_Price) / Profit)
        ΔPD_transition = max(0, emission_cost / income × scaling)
    
    Based on NGFS Phase V pathways.
    """
    carbon_price = CARBON_PRICES.get(scenario, 100)
    emission_intensity = map_purpose_to_sector(purpose_series).values
    income = annual_income.fillna(annual_income.median()).values
    
    # Carbon cost as fraction of income
    carbon_cost_ratio = (emission_intensity * carbon_price) / (income / 1000 + 1e-6)
    carbon_cost_ratio = np.clip(carbon_cost_ratio, 0, 0.5)
    
    transition_uplift = carbon_cost_ratio * 0.4  # Calibrated scaling
    cpd = pd_physical * (1 + transition_uplift)
    return np.clip(cpd, 0, 1)
