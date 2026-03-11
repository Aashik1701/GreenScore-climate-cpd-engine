
<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
<img src="https://img.shields.io/badge/XGBoost-Classifier-006400?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
<img src="https://img.shields.io/badge/NGFS-Phase%20V-0077B6?style=for-the-badge" alt="NGFS Phase V" />
<img src="https://img.shields.io/badge/RBI-2024%20Framework-FF6F00?style=for-the-badge" alt="RBI 2024" />
<img src="https://img.shields.io/github/license/Aashik1701/GreenScore-climate-cpd-engine?style=for-the-badge" alt="License" />

<br/><br/>

# GreenScore

### Climate-Adjusted Credit Risk Engine

A **B2B SaaS platform** that integrates physical and transition climate risk into loan-level credit default analysis, aligned with **RBI's 2024 disclosure framework** and **NGFS Phase V scenarios**.

[Features](#features) · [Architecture](#architecture) · [Methodology](#methodology) · [Quick Start](#quick-start) · [Configuration](#configuration) · [Datasets](#datasets) · [References](#references) · [License](#license)

</div>

---

## Problem Statement

Traditional credit risk models derive default probability from backward-looking financial indicators — income, credit score, debt ratio — and are structurally blind to the systemic risk posed by climate change.

By 2030, physical hazard events (floods, cyclones, heatwaves) and transition policies (carbon pricing, fossil-fuel phase-outs) will materially impair borrower repayment capacity across retail and MSME portfolios. Despite this, no widely adopted tool exists for banks to quantify climate-adjusted default risk at the **individual loan level**.

**GreenScore closes this gap.** It computes a **Climate-Adjusted Probability of Default (CPD)** for every loan in a portfolio by fusing:

1. An **XGBoost baseline PD model** trained on historical loan-performance data
2. A **physical risk overlay** derived from state-level flood, cyclone, and heat-exposure scores
3. A **transition risk overlay** driven by NGFS Phase V carbon-price pathways and sector emission intensity

---

## Features

| Capability | Description |
|:---|:---|
| **Dual Climate Risk Overlay** | Simultaneously applies physical *and* transition risk at the loan level — no published study integrates both |
| **NGFS Phase V Scenario Engine** | Toggle between Orderly (\$100/tCO₂), Disorderly (\$250/tCO₂), and Hot House World (\$25/tCO₂) pathways |
| **Interactive Dashboard** | Upload a loan CSV and receive instant CPD results, distribution charts, heatmaps, and sector analysis |
| **Geographic Risk Heatmap** | Folium-powered map visualising state-level climate risk concentration with drill-down popups |
| **RBI Disclosure Report** | One-click PDF generation aligned with RBI's 2024 draft climate-related financial risk framework |
| **India + US Coverage** | State-level physical risk scores for Indian states (IMD / NDMA) and US states (FEMA / historical tracks) |
| **Export Suite** | Download loan-level CPD results as CSV or the full regulatory disclosure as PDF |

---

## Architecture

### Repository Structure

```text
GreenScore/
│
├── app.py                  # Streamlit dashboard — primary entry point
├── cpd_engine.py           # Data loading, XGBoost training, baseline PD inference
├── physical_risk.py        # Physical hazard overlay (IMD / NDMA state scores)
├── transition_risk.py      # Transition risk overlay (NGFS carbon pricing)
├── report_gen.py           # RBI disclosure PDF generator (ReportLab)
├── requirements.txt        # Pinned Python dependencies
│
├── data/                   # Loan portfolio CSVs (not committed — see Datasets)
├── models/                 # Serialised model artifacts (.pkl)
└── notebooks/              # Exploratory data analysis & model training notebooks
```

### Module Dependency Graph

```text
┌──────────────────────────┐
│        app.py            │  Streamlit UI — entry point
│     (orchestrator)       │
└───┬──────┬──────┬────────┘
    │      │      │
    ▼      │      ▼
┌─────────┐│  ┌────────────┐
│cpd_     ││  │report_     │
│engine   ││  │gen         │
└────┬────┘│  └────────────┘
     │     │
     │  ┌──┴────────────────┐
     │  │                   │
     ▼  ▼                   ▼
┌────────────┐     ┌────────────────┐
│physical_   │     │transition_     │
│risk        │     │risk            │
└────────────┘     └────────────────┘
```

| Module | Responsibility |
|:---|:---|
| `cpd_engine.py` | Loads raw CSV, engineers features, trains / loads XGBoost, returns baseline PD vector |
| `physical_risk.py` | Maps borrower location to a 0–1 physical hazard score; applies severity-weighted PD uplift |
| `transition_risk.py` | Maps loan purpose to sector emission intensity; computes carbon-cost PD uplift under a chosen NGFS scenario |
| `report_gen.py` | Renders a multi-section PDF report (ReportLab) covering portfolio overview, climate metrics, sector exposure, RBI disclosure pillars, and methodology |
| `app.py` | Orchestrates the full pipeline; provides sidebar controls, file upload, metric cards, charts, map, sector table, result export |

---

## Methodology

### CPD Formula

$$CPD = PD_{\text{baseline}} \times \bigl(1 + \alpha_{\text{physical}}\bigr) \times \bigl(1 + \alpha_{\text{transition}}\bigr)$$

| Symbol | Definition | Derivation |
|:---|:---|:---|
| $PD_{\text{baseline}}$ | XGBoost-predicted probability of default | Trained on LendingClub historical loan-performance data |
| $\alpha_{\text{physical}}$ | Physical risk score × severity factor (0.3) | IMD flood-zone classifications, NDMA cyclone-track & disaster-risk index |
| $\alpha_{\text{transition}}$ | Carbon cost ratio × scaling factor (0.4) | NGFS Phase V carbon price × CPCB sector emission intensity ÷ borrower income |

> **Theoretical basis.** The multiplicative structure follows Bell & van Vuuren (2022), who demonstrate that climate-induced equity shocks translate to PD scaling factors of **1.2×–4.5×** depending on credit grade and shock severity. Bouchet, Dayan & Contoux (2022) provide the institutional justification — finance and climate science operate as "worlds apart," and GreenScore's dashboard bridges that gap by translating forward-looking climate outputs into a familiar credit-risk interface.

### Physical Risk Overlay

Each borrower's state (India or US) is mapped to a normalised hazard score on a **0–1 scale** reflecting composite flood, cyclone, and extreme-heat exposure. The score is multiplied by a severity factor calibrated from the Bell & van Vuuren scaling matrix.

### Transition Risk Overlay

Loan purpose (or assigned sector) is mapped to a CO₂ emission intensity (tCO₂ per \$1 000 revenue). Combined with the selected NGFS carbon price and the borrower's income, this yields a carbon-cost ratio that drives the transition PD uplift.

### Risk Categorisation

Final CPD values are bucketed into four tiers:

| Tier | CPD Range | Interpretation |
|:---|:---|:---|
| **Low** | 0 – 0.05 | Minimal climate-driven default risk |
| **Medium** | 0.05 – 0.15 | Moderate exposure; monitoring recommended |
| **High** | 0.15 – 0.30 | Significant climate uplift; provisioning review advised |
| **Critical** | 0.30 – 1.00 | Severe exposure; immediate risk-mitigation action warranted |

---

## Quick Start

### Prerequisites

| Requirement | Minimum |
|:---|:---|
| Python | 3.10+ |
| RAM | 8 GB (recommended for full dataset processing) |

### 1. Clone and Install

```bash
git clone https://github.com/Aashik1701/GreenScore-climate-cpd-engine.git
cd GreenScore-climate-cpd-engine

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare Data

Download the **LendingClub Dataset** from [Kaggle](https://www.kaggle.com/datasets/panchammahto/lendingclub-dataset-full-2007-to-2018) and place the CSV in the `data/` directory:

```bash
mv ~/Downloads/accepted_2007_to_2018Q4.csv data/lending_club_sample.csv
```

### 3. Train the Baseline PD Model

```bash
python3 cpd_engine.py data/lending_club_sample.csv
```

Expected output:

```text
Loading data from data/lending_club_sample.csv...
Loaded 200,000 records. Default rate: 0.214
Baseline PD Model AUC: 0.7234
Model saved to models/baseline_pd_model.pkl (AUC: 0.7234)
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** → upload a loan portfolio CSV → explore climate-adjusted results.

---

## Configuration

### NGFS Carbon Price Scenarios

Selectable from the dashboard sidebar. Each maps to a projected 2030 carbon price:

| Scenario | Carbon Price (2030) | Policy Characterisation |
|:---|:---|:---|
| **Orderly** | \$100 / tCO₂ | Gradual, early policy action; minimal market disruption |
| **Disorderly** | \$250 / tCO₂ | Delayed, sudden policy shock; high market stress |
| **Hot House World** | \$25 / tCO₂ | No meaningful transition; severe physical risk materialisation |

### Physical Risk Scores

State-level scores on a **0–1 scale**, derived from:

- **India** — IMD flood-zone classifications, NDMA disaster-risk index
- **US** — FEMA flood zones, NOAA historical cyclone / hurricane tracks

| Region | Score | Rationale |
|:---|:---:|:---|
| Odisha | 0.90 | Highest composite cyclone + flood exposure |
| West Bengal, Assam | 0.85 | Recurrent severe flooding |
| Florida (FL), Louisiana (LA) | 0.80 – 0.85 | Atlantic hurricane corridor |
| Rajasthan, Nevada (NV) | 0.35 – 0.50 | Low coastal / flood risk |

### Sector Emission Intensity

CO₂ intensity per \$1 000 revenue (tCO₂), sourced from CPCB and IEA:

| Sector | Intensity (tCO₂ / \$1k) | Risk Tier |
|:---|:---:|:---|
| Coal | 0.95 | Critical |
| Thermal Power | 0.90 | Critical |
| Cement | 0.85 | High |
| Technology | 0.08 | Low |
| Renewables | −0.10 | Negative (benefit) |

---

## Datasets

| # | Dataset | Role | Size | Source |
|:---:|:---|:---|:---|:---|
| 1 | **LendingClub (2007–2018)** | Baseline PD model training and evaluation | ~2 GB (1.3 M loans) | [Kaggle](https://www.kaggle.com/datasets/panchammahto/lendingclub-dataset-full-2007-to-2018) |
| 2 | **Indian Bank Loan Dataset** | India-context validation and RBI alignment demo | ~50 MB | [Kaggle](https://www.kaggle.com/datasets/rudrasing/indian-banks-loan-dataset) |
| 3 | **NGFS Phase V Scenarios** | Transition risk carbon-price pathways | CSV download | [NGFS Scenarios Portal](https://www.ngfs.net/ngfs-scenarios-portal/) |
| 4 | **IMD / NDMA Reports** | Physical risk state-level hazard scores | Hardcoded from published reports | [IMD](https://mausam.imd.gov.in) · [NDMA](https://ndma.gov.in) |

### Expected Input CSV Schema

| Column | Type | Description |
|:---|:---|:---|
| `dti` | `float` | Debt-to-Income ratio |
| `annual_inc` | `float` | Annual income (USD) |
| `fico_range_low` | `int` | FICO credit score (lower bound) |
| `int_rate` | `float` | Interest rate (%) |
| `installment` | `float` | Monthly installment (USD) |
| `emp_length` | `str` | Employment length (e.g., `"10+ years"`) |
| `addr_state` | `str` | US state code or Indian state name |
| `purpose` | `str` | Loan purpose / sector (e.g., `"debt_consolidation"`) |

---

## Sample Outputs

### Loan-Level CPD Table

```text
loan_id │ State │ Sector        │ Baseline PD │ CPD 2030 │ Uplift  │ Category
────────┼───────┼───────────────┼─────────────┼──────────┼─────────┼──────────
L001    │ FL    │ Manufacturing │ 0.042       │ 0.071    │ +69 %   │ High
L002    │ NY    │ Technology    │ 0.018       │ 0.023    │ +28 %   │ Low
L003    │ LA    │ Coal          │ 0.089       │ 0.198    │ +122 %  │ Critical
```

### Portfolio Summary (Orderly Scenario)

```text
Average Baseline PD  : 0.048
Average CPD 2030     : 0.074
Portfolio PD Uplift  : +54 %
High / Critical Loans: 23 % of portfolio
```

---

## Research Contribution

1. **Multi-Modal Fusion at Loan Level** — First study to jointly compute physical hazard exposure *and* carbon transition cost into a single loan-level PD adjustment.

2. **Operationalisation of Theoretical Frameworks** — Bell & van Vuuren (2022) provide the scaling factor calibration; Bouchet et al. (2022) identify the institutional gap. GreenScore translates both into a deployable, end-to-end system.

3. **India-Specific Regulatory Alignment** — No peer-reviewed study produces a loan-level CPD engine calibrated to Indian climate-hazard zones, Indian sector emission intensity, and RBI's 2024 draft climate-related financial risk disclosure framework.

---

## References

1. Bell, A. & van Vuuren, G. (2022). *The Impact of Climate Risk on Corporate Credit Risk.* — Theoretical basis for multiplicative PD scaling factors.
2. Bouchet, V., Dayan, H. & Contoux, C. (2022). *Finance and Climate Science: Worlds Apart?* — Institutional and epistemological barriers between finance and climate communities.
3. NGFS (2025). *Phase V Climate Scenarios.* Network for Greening the Financial System. — Carbon-price pathway data.
4. RBI (2024). *Draft Disclosure Framework on Climate-related Financial Risks.* Reserve Bank of India. — Regulatory alignment framework.

---

## Tech Stack

| Layer | Technologies |
|:---|:---|
| **Language** | Python 3.10+ |
| **Machine Learning** | XGBoost, scikit-learn |
| **Dashboard** | Streamlit |
| **Geospatial** | Folium, GeoPandas, streamlit-folium |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **PDF Reporting** | ReportLab |
| **Data Processing** | Pandas, NumPy |

---

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for a climate-resilient financial system.
</p>
