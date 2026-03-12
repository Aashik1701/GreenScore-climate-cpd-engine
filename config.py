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
    # Phase 1.3 — additional predictive columns
    'revol_util', 'revol_bal', 'open_acc', 'total_acc',
    'pub_rec', 'delinq_2yrs', 'inq_last_6mths', 'term',
    'sub_grade', 'verification_status', 'earliest_cr_line',
    # Phase 1.8 — high-correlation features for AUC improvement
    'acc_open_past_24mths', 'mort_acc', 'total_bc_limit',
    'total_rev_hi_lim', 'mo_sin_rcnt_tl', 'mo_sin_old_rev_tl_op',
    'num_actv_rev_tl', 'percent_bc_gt_75', 'bc_util',
    'mths_since_recent_inq',
]

BASE_FEATURES = [
    'dti', 'annual_inc', 'fico_range_low', 'int_rate',
    'installment', 'emp_length',
    # Phase 1.3 — additional raw features
    'revol_util', 'revol_bal', 'open_acc', 'total_acc',
    'pub_rec', 'delinq_2yrs', 'inq_last_6mths', 'loan_amnt',
    'term_months', 'sub_grade_num', 'verification_num',
    'credit_history_months',
    # Phase 1.8 — credit bureau depth features
    'acc_open_past_24mths', 'mort_acc', 'total_bc_limit',
    'total_rev_hi_lim', 'mo_sin_rcnt_tl', 'mo_sin_old_rev_tl_op',
    'num_actv_rev_tl', 'percent_bc_gt_75', 'bc_util',
    'mths_since_recent_inq',
]

ENGINEERED_FEATURES = [
    'income_to_installment',   # annual_inc / (installment * 12 + 1)
    'loan_to_income',          # loan_amnt / (annual_inc + 1)
    'dti_bucket',              # binned DTI
    'fico_bucket',             # binned FICO
    # Phase 1.3 — additional ratios (tree-friendly)
    'monthly_payment_burden',  # installment / (annual_inc / 12 + 1)
    'credit_utilization_ratio', # revol_bal / (annual_inc + 1)
    'open_to_total_acc',       # open_acc / (total_acc + 1)
    # Phase 1.8 — bureau-depth ratios
    'bc_limit_to_income',      # total_bc_limit / (annual_inc + 1)
    'rev_limit_to_income',     # total_rev_hi_lim / (annual_inc + 1)
    'recent_accts_ratio',      # acc_open_past_24mths / (total_acc + 1)
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# Climate risk features fed into the model alongside financials
PHYSICAL_RISK_FEATURES = [
    'flood_freq_score',
    'drought_severity_index',
    'temp_anomaly_5yr',
    'extreme_weather_events_count',
    'physical_risk_score',
]

TRANSITION_RISK_FEATURES = [
    'sector_carbon_intensity',
    'policy_exposure_score',
    'transition_risk_score',
]

GEOGRAPHIC_FEATURES = [
    'coastal_proximity_km',
    'elevation_meters',
]

CLIMATE_FEATURES = PHYSICAL_RISK_FEATURES + TRANSITION_RISK_FEATURES + GEOGRAPHIC_FEATURES

# Full feature set for the climate-aware model
ALL_FEATURES_CLIMATE = ALL_FEATURES + CLIMATE_FEATURES

DEFAULT_STATUSES = ['Charged Off', 'Default', 'Late (31-120 days)']

# DTI bins: [0–10, 10–20, 20–30, 30+]
DTI_BINS = [0, 10, 20, 30, 100]
DTI_LABELS = [0, 1, 2, 3]

# FICO bins: [300–580 (poor), 580–670 (fair), 670–740 (good), 740–850 (excellent)]
FICO_BINS = [300, 580, 670, 740, 850]
FICO_LABELS = [0, 1, 2, 3]

# Employment length bins: [0–1 (new), 1–3 (early), 3–7 (mid), 7+ (senior)]
EMP_LENGTH_BINS = [0, 1, 3, 7, 50]
EMP_LENGTH_LABELS = [0, 1, 2, 3]

# Sub-grade ordinal mapping: A1=1, A2=2, ..., G5=35
SUB_GRADE_ORDER = {f'{g}{n}': i * 5 + n for i, g in enumerate('ABCDEFG') for n in range(1, 6)}

# Verification status ordinal mapping
VERIFICATION_MAP = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}

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
    'orderly':          100,   # Net Zero 2050 — gradual transition
    'disorderly':       250,   # Delayed Transition — sudden policy shock
    'hot_house':         25,   # Current Policies — minimal carbon pricing
    'too_little_too_late': 175, # Delayed + insufficient action — BOTH risks HIGH
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
# Portfolio Loss Estimation (Expected Loss = PD × LGD × EAD)
# ─────────────────────────────────────────────────────────
# LGD (Loss Given Default): Basel II foundation IRB for unsecured consumer = 0.45
DEFAULT_LGD = 0.45
# Carbon price ramp-up years for time-series projection
PROJECTION_YEARS = list(range(2025, 2051))

# ─────────────────────────────────────────────────────────
# Optuna Hyperparameter Tuning
# ─────────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 75
OPTUNA_PARAM_SPACE = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 500),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_weight': (1, 10),
    'reg_alpha': (1e-8, 10.0),
    'reg_lambda': (1e-8, 10.0),
}

# ─────────────────────────────────────────────────────────
# LightGBM Hyperparameters
# ─────────────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 300,
    'num_leaves': 31,
    'random_state': 42,
    'verbosity': -1,
}

# ─────────────────────────────────────────────────────────
# Home Credit Dataset Adapter Mapping
# ─────────────────────────────────────────────────────────
HOMECREDIT_ORGANIZATION_TO_SECTOR = {
    'Transport: type 1':    'transport',
    'Transport: type 2':    'transport',
    'Transport: type 3':    'transport',
    'Transport: type 4':    'transport',
    'Construction':         'construction',
    'Industry: type 1':     'manufacturing',
    'Industry: type 2':     'manufacturing',
    'Industry: type 3':     'manufacturing',
    'Industry: type 4':     'manufacturing',
    'Industry: type 5':     'manufacturing',
    'Industry: type 6':     'manufacturing',
    'Industry: type 7':     'manufacturing',
    'Industry: type 8':     'manufacturing',
    'Industry: type 9':     'manufacturing',
    'Industry: type 10':    'manufacturing',
    'Industry: type 11':    'manufacturing',
    'Industry: type 12':    'manufacturing',
    'Industry: type 13':    'manufacturing',
    'Trade: type 1':        'retail',
    'Trade: type 2':        'retail',
    'Trade: type 3':        'retail',
    'Trade: type 4':        'retail',
    'Trade: type 5':        'retail',
    'Trade: type 6':        'retail',
    'Trade: type 7':        'retail',
    'Business Entity Type 1': 'services',
    'Business Entity Type 2': 'services',
    'Business Entity Type 3': 'services',
    'Self-employed':        'services',
    'Services':             'services',
    'Medicine':             'healthcare',
    'Bank':                 'services',
    'Insurance':            'services',
    'Government':           'services',
    'Military':             'services',
    'School':               'services',
    'Kindergarten':         'services',
    'University':           'services',
    'Police':               'services',
    'Security':             'services',
    'Security Ministries':  'services',
    'Postal':               'services',
    'Religion':             'services',
    'Electricity':          'thermal_power',
    'Emergency':            'services',
    'Hotel':                'services',
    'Restaurant':           'services',
    'Telecom':              'technology',
    'Realtor':              'construction',
    'Housing':              'construction',
    'Agriculture':          'agriculture',
    'Cleaning':             'services',
    'Culture':              'services',
    'Mobile':               'technology',
    'Advertising':          'services',
    'Legal Services':       'services',
    'XNA':                  'other',
}

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

# ─────────────────────────────────────────────────────────
# Geographic Features — Coastal Proximity & Elevation
# ─────────────────────────────────────────────────────────
# Approximate coastal proximity in km and mean elevation in metres
# for each US state / Indian state.  Values are rough centroids for
# portfolio-level analysis; a production system would geocode at the
# borrower address level.

STATE_COASTAL_PROXIMITY_KM = {
    # US states — approximate distance from state centroid to coast
    'AL': 200, 'AK': 100, 'AZ': 600, 'AR': 500, 'CA': 80, 'CO': 1400,
    'CT': 20,  'DE': 10,  'DC': 60,  'FL': 30,  'GA': 150, 'HI': 5,
    'ID': 700, 'IL': 600, 'IN': 500, 'IA': 800, 'KS': 1000, 'KY': 600,
    'LA': 50,  'ME': 30,  'MD': 30,  'MA': 20,  'MI': 200, 'MN': 800,
    'MS': 100, 'MO': 700, 'MT': 900, 'NE': 1200, 'NV': 500, 'NH': 40,
    'NJ': 15,  'NM': 800, 'NY': 40,  'NC': 100, 'ND': 1500, 'OH': 400,
    'OK': 700, 'OR': 80,  'PA': 150, 'RI': 10,  'SC': 80,  'SD': 1300,
    'TN': 500, 'TX': 150, 'UT': 900, 'VT': 200, 'VA': 100, 'WA': 80,
    'WV': 400, 'WI': 400, 'WY': 1200,
    # Indian states
    'Andhra Pradesh': 30, 'Arunachal Pradesh': 800, 'Assam': 600,
    'Bihar': 700, 'Chhattisgarh': 500, 'Goa': 5, 'Gujarat': 20,
    'Haryana': 1000, 'Himachal Pradesh': 1200, 'Jharkhand': 400,
    'Karnataka': 60, 'Kerala': 10, 'Madhya Pradesh': 600,
    'Maharashtra': 40, 'Manipur': 700, 'Meghalaya': 600,
    'Mizoram': 600, 'Nagaland': 700, 'Odisha': 30, 'Punjab': 1100,
    'Rajasthan': 500, 'Sikkim': 800, 'Tamil Nadu': 15,
    'Telangana': 300, 'Tripura': 500, 'Uttar Pradesh': 800,
    'Uttarakhand': 1000, 'West Bengal': 30,
    'Delhi': 1000, 'Chandigarh': 1100, 'Puducherry': 5,
    'Lakshadweep': 5, 'Ladakh': 1500, 'Jammu and Kashmir': 1200,
    'Andaman and Nicobar Islands': 5,
    'Dadra and Nagar Haveli and Daman and Diu': 10,
    'OTHER': 500,
}

STATE_ELEVATION_METERS = {
    # US states — approximate mean elevation
    'AL': 150, 'AK': 580, 'AZ': 1250, 'AR': 200, 'CA': 880, 'CO': 2070,
    'CT': 50,  'DE': 18,  'DC': 20,   'FL': 10,  'GA': 180, 'HI': 920,
    'ID': 1520, 'IL': 180, 'IN': 210, 'IA': 330, 'KS': 610, 'KY': 230,
    'LA': 10,  'ME': 180, 'MD': 110, 'MA': 150, 'MI': 280, 'MN': 370,
    'MS': 90,  'MO': 240, 'MT': 1040, 'NE': 790, 'NV': 1680, 'NH': 300,
    'NJ': 30,  'NM': 1740, 'NY': 300, 'NC': 210, 'ND': 580, 'OH': 260,
    'OK': 400, 'OR': 1000, 'PA': 330, 'RI': 60,  'SC': 110, 'SD': 670,
    'TN': 270, 'TX': 520, 'UT': 1860, 'VT': 300, 'VA': 290, 'WA': 520,
    'WV': 460, 'WI': 320, 'WY': 2040,
    # Indian states
    'Andhra Pradesh': 250, 'Arunachal Pradesh': 2000, 'Assam': 80,
    'Bihar': 50, 'Chhattisgarh': 500, 'Goa': 20, 'Gujarat': 100,
    'Haryana': 220, 'Himachal Pradesh': 3000, 'Jharkhand': 450,
    'Karnataka': 600, 'Kerala': 120, 'Madhya Pradesh': 450,
    'Maharashtra': 400, 'Manipur': 800, 'Meghalaya': 1000,
    'Mizoram': 900, 'Nagaland': 1200, 'Odisha': 100, 'Punjab': 230,
    'Rajasthan': 350, 'Sikkim': 2700, 'Tamil Nadu': 200,
    'Telangana': 500, 'Tripura': 80, 'Uttar Pradesh': 100,
    'Uttarakhand': 2500, 'West Bengal': 40,
    'Delhi': 215, 'Chandigarh': 330, 'Puducherry': 10,
    'Lakshadweep': 2, 'Ladakh': 4500, 'Jammu and Kashmir': 2500,
    'Andaman and Nicobar Islands': 50,
    'Dadra and Nagar Haveli and Daman and Diu': 30,
    'OTHER': 300,
}

# ─────────────────────────────────────────────────────────
# Transition Risk — Sector Policy Exposure Scores (0–1)
# ─────────────────────────────────────────────────────────
# How exposed each sector is to carbon-pricing policy changes.
# High = very exposed (e.g. coal); low = insulated (e.g. tech).
SECTOR_POLICY_EXPOSURE = {
    'cement': 0.85, 'steel': 0.80, 'thermal_power': 0.95,
    'coal': 0.98, 'oil_gas': 0.82, 'chemicals': 0.60,
    'manufacturing': 0.45, 'transport': 0.50, 'agriculture': 0.35,
    'construction': 0.30, 'retail': 0.15, 'services': 0.10,
    'technology': 0.08, 'healthcare': 0.10, 'renewables': 0.05,
    # LendingClub purpose proxies
    'debt_consolidation': 0.10, 'credit_card': 0.08,
    'home_improvement': 0.20, 'small_business': 0.30,
    'car': 0.45, 'medical': 0.10, 'moving': 0.15,
    'vacation': 0.10, 'house': 0.20, 'wedding': 0.08,
    'major_purchase': 0.18, 'educational': 0.05,
    'renewable_energy': 0.05, 'other': 0.20,
}
