"""
GreenScore CPD Engine — Centralised Configuration
===================================================
All constants, feature lists, risk scores, model hyperparameters,
and lookup tables are defined here. Every other module imports from
this file to ensure a single source of truth.
"""

import logging

# ─────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"

# ─────────────────────────────────────────────────────────
# Feature Engineering & Model
# ─────────────────────────────────────────────────────────
RAW_COLS = [
    'loan_status', 'loan_amnt', 'dti', 'annual_inc',
    'fico_range_low', 'int_rate', 'installment',
    'emp_length', 'home_ownership', 'purpose', 'addr_state',
]

BASE_FEATURES = [
    'dti', 'annual_inc', 'fico_range_low', 'int_rate',
    'installment', 'emp_length',
]

ENGINEERED_FEATURES = [
    'income_to_installment',   # annual_inc / (installment * 12 + 1)
    'loan_to_income',          # loan_amnt / (annual_inc + 1)
    'dti_bucket',              # binned DTI
    'fico_bucket',             # binned FICO
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

DEFAULT_STATUSES = ['Charged Off', 'Default', 'Late (31-120 days)']

# DTI bins: [0–10, 10–20, 20–30, 30+]
DTI_BINS = [0, 10, 20, 30, 100]
DTI_LABELS = [0, 1, 2, 3]

# FICO bins: [300–580 (poor), 580–670 (fair), 670–740 (good), 740–850 (excellent)]
FICO_BINS = [300, 580, 670, 740, 850]
FICO_LABELS = [0, 1, 2, 3]

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 300,
    'eval_metric': 'auc',
    'early_stopping_rounds': 15,
    'random_state': 42,
    'verbosity': 0,
}

# Cross-validation
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────
# Physical Risk Scores (0–1 scale)
# ─────────────────────────────────────────────────────────
# Sources:
#   India  — IMD flood zone maps (mausam.imd.gov.in),
#            NDMA disaster risk index (ndma.gov.in/en/media/reports)
#   US     — FEMA National Risk Index (hazards.fema.gov/nri),
#            NOAA historical cyclone/hurricane tracks
#
# Methodology: Composite score based on documented flood/cyclone/
# extreme-heat exposure classifications from the above sources,
# normalised to 0–1 where 1 = highest composite hazard exposure.

STATE_PHYSICAL_RISK = {
    # ── All 28 Indian States ──
    # Source: NDMA disaster-prone area classification + IMD flood zone data
    'Andhra Pradesh':     0.75,  # High cyclone + flood (coastal)
    'Arunachal Pradesh':  0.55,  # Moderate seismic + landslide risk
    'Assam':              0.85,  # Severe annual flooding (Brahmaputra)
    'Bihar':              0.80,  # Severe flood (Kosi/Gandak rivers)
    'Chhattisgarh':       0.40,  # Moderate — some flood zones
    'Goa':                0.50,  # Moderate coastal cyclone exposure
    'Gujarat':            0.65,  # Cyclone corridor + earthquake zone
    'Haryana':            0.35,  # Low — inland, minimal flood
    'Himachal Pradesh':   0.50,  # Landslide + flash flood risk
    'Jharkhand':          0.45,  # Moderate flood + heat exposure
    'Karnataka':          0.45,  # Moderate — some flood-prone districts
    'Kerala':             0.70,  # High — 2018/2019 catastrophic floods
    'Madhya Pradesh':     0.45,  # Moderate — heat + some flood
    'Maharashtra':        0.55,  # Moderate coastal + urban flood (Mumbai)
    'Manipur':            0.50,  # Landslide + flood
    'Meghalaya':          0.55,  # Heavy rainfall + landslide
    'Mizoram':            0.45,  # Moderate landslide risk
    'Nagaland':           0.45,  # Moderate seismic + landslide
    'Odisha':             0.90,  # Highest — severe cyclone corridor (Fani, Amphan)
    'Punjab':             0.40,  # Low-moderate — river flood pockets
    'Rajasthan':          0.50,  # Drought + heat; low flood
    'Sikkim':             0.55,  # Seismic + glacial lake outburst risk
    'Tamil Nadu':         0.65,  # Cyclone + flood (Chennai 2015)
    'Telangana':          0.55,  # Urban flood (Hyderabad) + heat
    'Tripura':            0.50,  # Moderate flood
    'Uttar Pradesh':      0.60,  # Large flood-prone area (Ganga basin)
    'Uttarakhand':        0.65,  # Flash flood + landslide (Kedarnath 2013)
    'West Bengal':        0.85,  # Severe — cyclone (Amphan, Yaas) + Sundarbans flooding

    # ── 8 Union Territories ──
    'Andaman and Nicobar Islands': 0.70,  # Tsunami + cyclone exposure
    'Chandigarh':         0.30,  # Low — inland, planned city
    'Dadra and Nagar Haveli and Daman and Diu': 0.45,  # Moderate coastal
    'Delhi':              0.40,  # Urban heat island + Yamuna flooding
    'Jammu and Kashmir':  0.60,  # Seismic + flood (2014 J&K floods)
    'Ladakh':             0.50,  # Glacial melt + flash flood
    'Lakshadweep':        0.75,  # Sea-level rise + cyclone (low-lying islands)
    'Puducherry':         0.60,  # Coastal cyclone + flood

    # ── All 50 US States + DC ──
    # Source: FEMA National Risk Index (Expected Annual Loss composite)
    'AL': 0.70,  # Alabama — tornado + hurricane corridor
    'AK': 0.45,  # Alaska — earthquake + wildfire; low hurricane
    'AZ': 0.42,  # Arizona — wildfire + extreme heat
    'AR': 0.58,  # Arkansas — tornado + river flood
    'CA': 0.60,  # California — wildfire + earthquake + drought
    'CO': 0.38,  # Colorado — wildfire + flash flood
    'CT': 0.45,  # Connecticut — nor'easter + coastal flood
    'DE': 0.48,  # Delaware — coastal flood + hurricane
    'DC': 0.42,  # District of Columbia — urban heat + Potomac flood
    'FL': 0.80,  # Florida — highest hurricane exposure
    'GA': 0.55,  # Georgia — hurricane + tornado
    'HI': 0.55,  # Hawaii — hurricane + volcanic + tsunami
    'ID': 0.35,  # Idaho — wildfire; low other risks
    'IL': 0.40,  # Illinois — tornado + river flood
    'IN': 0.42,  # Indiana — tornado + flood
    'IA': 0.45,  # Iowa — severe flood (2008/2019)
    'KS': 0.50,  # Kansas — tornado alley
    'KY': 0.48,  # Kentucky — river flood + tornado
    'LA': 0.85,  # Louisiana — hurricane (Katrina, Ida) + flood
    'ME': 0.35,  # Maine — moderate nor'easter
    'MD': 0.48,  # Maryland — coastal flood + hurricane
    'MA': 0.42,  # Massachusetts — nor'easter + coastal flood
    'MI': 0.38,  # Michigan — moderate — some flood + severe storms
    'MN': 0.40,  # Minnesota — flood + severe storms
    'MS': 0.75,  # Mississippi — hurricane + tornado + flood
    'MO': 0.50,  # Missouri — tornado + river flood
    'MT': 0.35,  # Montana — wildfire; low other
    'NE': 0.45,  # Nebraska — tornado + flood
    'NV': 0.35,  # Nevada — wildfire + extreme heat; low flood
    'NH': 0.35,  # New Hampshire — moderate nor'easter
    'NJ': 0.50,  # New Jersey — coastal flood + hurricane (Sandy)
    'NM': 0.40,  # New Mexico — wildfire + drought
    'NY': 0.45,  # New York — coastal flood + nor'easter
    'NC': 0.60,  # North Carolina — hurricane corridor
    'ND': 0.40,  # North Dakota — flood + blizzard
    'OH': 0.35,  # Ohio — moderate tornado + flood
    'OK': 0.55,  # Oklahoma — tornado alley
    'OR': 0.45,  # Oregon — wildfire + earthquake (Cascadia)
    'PA': 0.40,  # Pennsylvania — flood + severe storms
    'RI': 0.42,  # Rhode Island — coastal flood + nor'easter
    'SC': 0.65,  # South Carolina — hurricane + flood (2015, 2018)
    'SD': 0.40,  # South Dakota — tornado + flood
    'TN': 0.52,  # Tennessee — tornado + flood
    'TX': 0.65,  # Texas — hurricane + tornado + flood (Harvey 2017)
    'UT': 0.35,  # Utah — wildfire + earthquake; low flood
    'VT': 0.35,  # Vermont — flood (Irene 2011); generally low
    'VA': 0.45,  # Virginia — hurricane + flood
    'WA': 0.55,  # Washington — earthquake (Cascadia) + wildfire
    'WV': 0.42,  # West Virginia — flood + landslide
    'WI': 0.38,  # Wisconsin — moderate flood + severe storms
    'WY': 0.32,  # Wyoming — lowest composite risk

    # Default fallback
    'OTHER': 0.45,
}

# ─────────────────────────────────────────────────────────
# Physical Risk — Severity Factor
# ─────────────────────────────────────────────────────────
# Calibrated from Bell & van Vuuren (2022) Table 3 scaling
# factor matrix. For mid-grade credits (BB–B) under a ~30%
# equity shock, the observed PD scaling factor is ≈1.3,
# corresponding to an additive multiplier of 0.3. This is
# applied as: PD_physical = PD_base × (1 + risk_score × 0.3)
SEVERITY_FACTOR = 0.3

# ─────────────────────────────────────────────────────────
# Transition Risk — Sector Emission Intensity
# ─────────────────────────────────────────────────────────
# CO₂ intensity (tCO₂ per $1,000 revenue), sourced from:
#   - CPCB (Central Pollution Control Board) industry reports
#   - IEA World Energy Outlook 2023 — sector benchmarks
#   - NGFS sectoral emission data
SECTOR_EMISSIONS = {
    # Industry sectors
    'cement':           0.85,
    'steel':            0.75,
    'thermal_power':    0.90,
    'coal':             0.95,
    'oil_gas':          0.70,
    'chemicals':        0.55,
    'manufacturing':    0.45,
    'transport':        0.40,
    'agriculture':      0.35,
    'construction':     0.30,
    'retail':           0.15,
    'services':         0.10,
    'technology':       0.08,
    'healthcare':       0.12,
    'renewables':      -0.10,
    'other':            0.25,
    # LendingClub purpose → proxy sector mapping
    # NOTE: LendingClub 'purpose' is a personal-loan category, not the
    # borrower's employing industry. This mapping is an approximation that
    # assigns an emission-intensity proxy based on the economic activity most
    # closely associated with each loan purpose. For datasets with a direct
    # 'sector' or 'industry' column, those should be used instead (see the
    # sector_col override in apply_transition_risk).
    'debt_consolidation': 0.15,
    'credit_card':        0.10,
    'home_improvement':   0.20,
    'small_business':     0.30,
    'car':                0.40,
    'medical':            0.12,
    'moving':             0.25,
    'vacation':           0.15,
    'house':              0.20,
    'wedding':            0.10,
    'major_purchase':     0.20,
    'educational':        0.08,
    'renewable_energy':  -0.10,
}

# ─────────────────────────────────────────────────────────
# NGFS Phase V Carbon Price Pathways (USD per tCO₂, 2030)
# ─────────────────────────────────────────────────────────
# Source: NGFS Phase V Technical Report, January 2025
# https://www.ngfs.net/ngfs-scenarios-portal/
CARBON_PRICES = {
    'orderly':    100,   # Net Zero 2050 — gradual transition
    'disorderly': 250,   # Delayed Transition — sudden policy shock
    'hot_house':   25,   # Current Policies — minimal carbon pricing
}

# ─────────────────────────────────────────────────────────
# Transition Risk — Scaling Factor
# ─────────────────────────────────────────────────────────
# The 0.4 scaling factor converts the carbon-cost-to-income ratio
# into a PD uplift. Derivation:
#   - Bell & van Vuuren (2022) show that a ~40% profit reduction
#     from carbon costs corresponds to ≈40% PD uplift for mid-grade
#     credits. Since our carbon_cost_ratio is already expressed as a
#     fraction of income (0–0.5 range after clipping), multiplying
#     by 0.4 translates this into a proportional PD uplift of 0–20%.
#   - This is conservative relative to their extreme scenarios (which
#     show up to 4.5× scaling) because personal-loan borrowers are
#     less directly exposed to carbon pricing than corporates.
TRANSITION_SCALING = 0.4
TRANSITION_COST_CLIP_MAX = 0.5

# ─────────────────────────────────────────────────────────
# Risk Category Bins
# ─────────────────────────────────────────────────────────
RISK_BINS = [0, 0.05, 0.15, 0.30, 1.0]
RISK_LABELS = ['Low', 'Medium', 'High', 'Critical']

# ─────────────────────────────────────────────────────────
# State Coordinates (for Folium heatmap)
# ─────────────────────────────────────────────────────────
US_STATE_COORDS = {
    'AL': [32.8, -86.8],  'AK': [64.2, -152.5], 'AZ': [34.0, -111.1],
    'AR': [34.8, -92.2],  'CA': [36.7, -119.4],  'CO': [39.5, -105.8],
    'CT': [41.6, -72.7],  'DE': [39.0, -75.5],   'DC': [38.9, -77.0],
    'FL': [27.6, -81.5],  'GA': [32.2, -83.4],   'HI': [19.9, -155.6],
    'ID': [44.1, -114.7], 'IL': [40.0, -89.0],   'IN': [40.3, -86.1],
    'IA': [41.9, -93.1],  'KS': [38.5, -98.8],   'KY': [37.8, -84.3],
    'LA': [30.5, -92.0],  'ME': [45.3, -69.4],   'MD': [39.0, -76.6],
    'MA': [42.4, -71.4],  'MI': [44.3, -85.6],   'MN': [46.7, -94.7],
    'MS': [32.4, -89.6],  'MO': [38.6, -92.6],   'MT': [46.9, -110.4],
    'NE': [41.5, -100.0], 'NV': [38.8, -116.4],  'NH': [43.2, -71.6],
    'NJ': [40.0, -74.5],  'NM': [34.5, -106.0],  'NY': [42.2, -74.9],
    'NC': [35.6, -79.8],  'ND': [47.5, -100.5],  'OH': [40.4, -82.8],
    'OK': [35.5, -97.5],  'OR': [43.8, -120.6],  'PA': [41.2, -77.2],
    'RI': [41.7, -71.5],  'SC': [33.8, -81.2],   'SD': [43.9, -99.9],
    'TN': [35.5, -86.6],  'TX': [31.0, -100.0],  'UT': [39.3, -111.1],
    'VT': [44.6, -72.6],  'VA': [37.4, -79.0],   'WA': [47.4, -121.5],
    'WV': [38.6, -80.6],  'WI': [43.8, -88.8],   'WY': [43.1, -107.6],
}

INDIA_STATE_COORDS = {
    'Andhra Pradesh':     [15.9, 79.7],
    'Arunachal Pradesh':  [28.2, 94.7],
    'Assam':              [26.2, 92.9],
    'Bihar':              [25.1, 85.3],
    'Chhattisgarh':       [21.3, 81.7],
    'Goa':                [15.3, 74.0],
    'Gujarat':            [22.3, 71.2],
    'Haryana':            [29.1, 76.1],
    'Himachal Pradesh':   [31.1, 77.2],
    'Jharkhand':          [23.6, 85.3],
    'Karnataka':          [15.3, 75.7],
    'Kerala':             [10.9, 76.3],
    'Madhya Pradesh':     [23.5, 78.6],
    'Maharashtra':        [19.8, 75.3],
    'Manipur':            [24.7, 93.9],
    'Meghalaya':          [25.5, 91.4],
    'Mizoram':            [23.2, 92.9],
    'Nagaland':           [26.2, 94.6],
    'Odisha':             [20.9, 84.0],
    'Punjab':             [31.1, 75.3],
    'Rajasthan':          [27.0, 74.2],
    'Sikkim':             [27.5, 88.5],
    'Tamil Nadu':         [11.1, 78.7],
    'Telangana':          [17.7, 79.0],
    'Tripura':            [23.9, 91.9],
    'Uttar Pradesh':      [26.8, 80.9],
    'Uttarakhand':        [30.1, 79.0],
    'West Bengal':        [22.6, 87.9],
    'Andaman and Nicobar Islands': [11.7, 92.7],
    'Chandigarh':         [30.7, 76.8],
    'Dadra and Nagar Haveli and Daman and Diu': [20.4, 73.0],
    'Delhi':              [28.7, 77.1],
    'Jammu and Kashmir':  [33.8, 74.8],
    'Ladakh':             [34.2, 77.6],
    'Lakshadweep':        [10.6, 72.6],
    'Puducherry':         [11.9, 79.8],
}
