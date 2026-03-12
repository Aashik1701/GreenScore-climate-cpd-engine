# GreenScore — Methodology

> Technical methodology for the Climate-Adjusted Credit Risk Engine.

---

## 1. Architecture Overview

GreenScore computes a **Climate-adjusted Probability of Default (CPD)** by combining a financial PD model with two risk overlays:

```
CPD = Baseline_PD × (1 + α_physical) × (1 + α_transition)
```

Where:
- **Baseline_PD** — XGBoost classifier trained on LendingClub financial features
- **α_physical** — Physical climate risk multiplier (flood, cyclone, drought exposure)
- **α_transition** — Transition risk multiplier (carbon cost impact under NGFS scenarios)

**Key design decision:** The model trains on financial features only. Climate adjustments are applied post-prediction as multiplicative overlays. This separation ensures:
1. The PD model captures borrower creditworthiness without noise from near-constant climate features
2. Climate risk factors can be updated independently without retraining
3. The CPD formula is interpretable and auditable

---

## 2. Baseline PD Model

### 2.1 Training Data

- **Source:** LendingClub Accepted Loans (2007–2018 Q4)
- **Records:** 500,000 (configurable)
- **Target:** Binary — 1 if `loan_status ∈ {Charged Off, Default, Late (31-120 days)}`, 0 otherwise
- **Default rate:** ~16.4%
- **Class imbalance handling:** `scale_pos_weight = (1 − default_rate) / default_rate ≈ 5.11`

### 2.2 Features (38 total)

**28 Base Features:**

| Feature | Source | Description |
|---|---|---|
| `dti` | LendingClub | Debt-to-income ratio |
| `annual_inc` | LendingClub | Annual income (USD) |
| `fico_range_low` | LendingClub | Lower bound of FICO range |
| `int_rate` | LendingClub | Interest rate (%) |
| `installment` | LendingClub | Monthly payment (USD) |
| `emp_length` | LendingClub | Employment length (years) |
| `revol_util` | LendingClub | Revolving utilisation (%) |
| `revol_bal` | LendingClub | Revolving balance (USD) |
| `open_acc` | LendingClub | Open credit accounts |
| `total_acc` | LendingClub | Total credit accounts |
| `pub_rec` | LendingClub | Public derogatory records |
| `delinq_2yrs` | LendingClub | Delinquencies in past 2 years |
| `inq_last_6mths` | LendingClub | Inquiries in last 6 months |
| `loan_amnt` | LendingClub | Loan amount (USD) |
| `term_months` | LendingClub | Loan term (months) — derived from `term` |
| `sub_grade_num` | LendingClub | Sub-grade ordinal (A1=1 … G5=35) |
| `verification_num` | LendingClub | Income verification status (0/1/2) |
| `credit_history_months` | LendingClub | Months since earliest credit line |
| `acc_open_past_24mths` | LendingClub | Accounts opened in past 24 months |
| `mort_acc` | LendingClub | Mortgage accounts (stability indicator) |
| `total_bc_limit` | LendingClub | Total bankcard credit limit |
| `total_rev_hi_lim` | LendingClub | Total revolving high credit/limit |
| `mo_sin_rcnt_tl` | LendingClub | Months since most recent account opened |
| `mo_sin_old_rev_tl_op` | LendingClub | Months since oldest revolving account |
| `num_actv_rev_tl` | LendingClub | Number of active revolving trades |
| `percent_bc_gt_75` | LendingClub | % bankcard lines > 75% utilised |
| `bc_util` | LendingClub | Bankcard utilisation ratio |
| `mths_since_recent_inq` | LendingClub | Months since most recent inquiry |

**10 Engineered Features:**

| Feature | Formula | Rationale |
|---|---|---|
| `income_to_installment` | `annual_inc / (installment × 12 + 1)` | Payment affordability ratio |
| `loan_to_income` | `loan_amnt / (annual_inc + 1)` | Leverage relative to income |
| `dti_bucket` | Binned DTI (0–10, 10–20, 20–30, 30+) | Captures non-linear DTI thresholds |
| `fico_bucket` | Binned FICO (300–580, …, 780–850) | Captures credit tier boundaries |
| `monthly_payment_burden` | `installment / (annual_inc / 12 + 1)` | Monthly cash flow stress |
| `credit_utilization_ratio` | `revol_bal / (annual_inc + 1)` | Revolving debt relative to income |
| `open_to_total_acc` | `open_acc / (total_acc + 1)` | Active credit line fraction |
| `bc_limit_to_income` | `total_bc_limit / (annual_inc + 1)` | Bankcard capacity relative to income |
| `rev_limit_to_income` | `total_rev_hi_lim / (annual_inc + 1)` | Total revolving capacity ratio |
| `recent_accts_ratio` | `acc_open_past_24mths / (total_acc + 1)` | Recent credit-seeking intensity |

### 2.3 Model Selection

| Model | Hold-out AUC | Role |
|---|---|---|
| **XGBoost** | **0.7298** | Primary model — Optuna-tuned, 75 trials |
| LightGBM | 0.7304 | Comparison benchmark |
| Random Forest | 0.7112 | Comparison benchmark |
| Logistic Regression | 0.7028 | Linear baseline |

XGBoost was selected as the primary model because:
- Near-best AUC with better stability (5-fold CV: 0.7326 ± 0.0017)
- Native SHAP support for regulatory explainability
- Robust handling of missing values

### 2.4 Hyperparameter Tuning

Optuna (TPE sampler) searches over 8 hyperparameters across 75 trials:

| Parameter | Search Range |
|---|---|
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.3 (log) |
| `n_estimators` | 100 – 1000 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `min_child_weight` | 1 – 10 |
| `gamma` | 0.0 – 5.0 |
| `reg_alpha` | 1e-8 – 10.0 (log) |

Objective: maximise 3-fold stratified cross-validation AUC.

---

## 3. Physical Risk Overlay

### 3.1 Data Sources

| Region | Source | Method |
|---|---|---|
| US (50 states + DC) | FEMA National Risk Index (NRI) | Expected Annual Loss composite, normalised 0–1 |
| India (28 states + 8 UTs) | NDMA disaster classification + IMD flood zone data | Multi-hazard exposure, normalised 0–1 |
| Climate metrics | NASA POWER API | 20-year precipitation, temperature, extreme events |

### 3.2 Scoring

Each loan is assigned a physical risk score based on the borrower's state:

```
α_physical = physical_risk_score × SEVERITY_FACTOR
```

- **SEVERITY_FACTOR = 0.3** — calibrated from Bell & van Vuuren (2022) Table 3. For mid-grade credits (BB–B) under a ~30% equity shock, the observed PD scaling factor ≈ 1.3, giving an additive multiplier of 0.3.
- **Range:** α_physical ∈ [0, 0.27] (0.9 × 0.3 for the highest-risk state, Odisha).

### 3.3 NASA POWER Enhancement

When `use_nasa=True`, the static score is blended with real-time climate metrics:
- **Precipitation** (20-year mean annual) → flood frequency proxy
- **Temperature** (20-year mean) → heat stress proxy
- **Extreme events** → reported severe weather events

These are fetched via NASA POWER API and cached in `nasa_cache.json` to avoid repeated API calls.

All 88 regions have documented sources and hazard rationale in `data/physical_risk_scores_reference.csv`.

---

## 4. Transition Risk Overlay

### 4.1 Carbon Cost Model

Each loan's purpose (or direct sector column) is mapped to a CO₂ emission intensity (tCO₂ per $1,000 revenue). The transition risk is:

```
carbon_cost_ratio = (emission_intensity × carbon_price) / annual_income
α_transition = min(carbon_cost_ratio, 0.5) × TRANSITION_SCALING
```

### 4.2 NGFS Scenarios

Carbon prices are drawn from **NGFS Phase V** (January 2025):

| Scenario | Carbon Price (USD/tCO₂) | Description |
|---|---|---|
| Orderly | $100 | Net Zero 2050 — gradual transition |
| Disorderly | $250 | Delayed Transition — sudden policy shock |
| Hot House | $25 | Current Policies — minimal carbon pricing |
| Too Little Too Late | $175 | Delayed + insufficient — both physical and transition risks high |

### 4.3 Scaling Factor

**TRANSITION_SCALING = 0.4**

Derivation:
- Bell & van Vuuren (2022) show that ~40% profit reduction from carbon costs corresponds to ≈40% PD uplift for mid-grade credits
- The carbon_cost_ratio is expressed as a fraction of income (0–0.5 range after clipping), so multiplying by 0.4 translates this into a proportional PD uplift of 0–20%
- Conservative relative to extreme scenarios (up to 4.5× scaling) because personal-loan borrowers are less directly exposed to carbon pricing than corporate borrowers

### 4.4 Known Limitation

LendingClub's `purpose` field is a personal-loan category (e.g. "car", "wedding"), not the borrower's employing industry. Mapping to sector CO₂ intensity is a **proxy**. When a dataset provides a direct `sector` column, the engine uses it automatically.

---

## 5. Cross-Dataset Validation

### 5.1 Methodology

The LendingClub-trained model is validated on the **Home Credit Default Risk** dataset to test cross-dataset generalisation. The adapter maps Home Credit columns to the GreenScore schema:

| Home Credit Column | GreenScore Column | Mapping |
|---|---|---|
| `TARGET` | `default` | Direct (1 = default) |
| `AMT_INCOME_TOTAL` | `annual_inc` | Direct |
| `AMT_CREDIT` | `loan_amnt` | Direct |
| `AMT_ANNUITY` | `installment` | Direct (monthly payment) |
| `DAYS_EMPLOYED` | `emp_length` | `abs(days) / 365.25`, capped at 40 |
| `EXT_SOURCE_*` | `fico_range_low` | Mean × 550 + 300 (scaled to FICO range) |
| `AMT_ANNUITY × 12 / income` | `dti` | Computed proxy |

Features without Home Credit equivalents (e.g. `revol_util`, `sub_grade_num`) use sensible defaults.

### 5.2 Results

| Metric | Value |
|---|---|
| Validation records | 307,511 |
| Default rate | 8.07% |
| **Cross-dataset AUC** | **0.5321** |

**Interpretation:** AUC of 0.57 is above random chance (0.50), demonstrating that some universal credit risk patterns (income, employment, payment burden) transfer across datasets. The modest AUC is expected because:
1. The datasets serve different markets (US consumer loans vs. Eastern European consumer finance)
2. 12 of 25 features use default values since Home Credit lacks direct equivalents
3. Different underlying default definitions and borrower populations

The ROC curve is saved at `models/cross_dataset_roc.png`.

---

## 6. Risk Categorisation

Final CPD scores are bucketed into risk categories:

| Category | CPD Range | Interpretation |
|---|---|---|
| Low | 0 – 5% | Minimal climate-adjusted default risk |
| Medium | 5 – 15% | Moderate risk, monitor for scenario changes |
| High | 15 – 30% | Elevated risk, may require provisioning |
| Critical | 30 – 100% | Severe risk under current climate scenario |

---

## 7. Regulatory Alignment

GreenScore aligns with the **RBI Climate Risk Disclosure Framework** across four pillars:

| Pillar | Implementation |
|---|---|
| **Governance** | Centralised config, reproducible model pipeline |
| **Strategy** | Multi-scenario analysis (4 NGFS pathways) |
| **Risk Management** | Physical + transition risk quantification, CPD formula |
| **Metrics & Targets** | AUC tracking, portfolio-level risk distribution, state-level reporting |

---

## 8. References

1. Bell, A. & van Vuuren, D. (2022). *Climate risk and financial stability: A framework for credit risk assessment under climate scenarios.* — Source for severity and scaling factor calibration.
2. FEMA National Risk Index — https://hazards.fema.gov/nri — US physical risk scores.
3. NDMA (National Disaster Management Authority) — https://ndma.gov.in — India physical risk classification.
4. IMD (India Meteorological Department) — https://mausam.imd.gov.in — Flood zone maps.
5. NASA POWER — https://power.larc.nasa.gov — Climate data API for precipitation, temperature, extreme events.
6. NGFS Phase V Technical Report (January 2025) — https://www.ngfs.net/ngfs-scenarios-portal/ — Carbon price pathways.
7. IEA World Energy Outlook 2023 — Sector emission intensity benchmarks.
8. CPCB (Central Pollution Control Board) — Industry emission reports for India sectors.
