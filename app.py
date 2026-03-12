"""
GreenScore — Streamlit Dashboard
==================================
Main entry point for the Climate-Adjusted Credit Risk Engine.
Provides scenario selection, dataset selector, CSV upload, CPD
computation, interactive charts, geographic heatmap, SHAP
explanations, and report generation.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

import config
from cpd_engine import get_baseline_pd
from physical_risk import apply_physical_risk
from transition_risk import apply_transition_risk
from report_gen import generate_pdf_report
from dataset_adapters import DATASET_REGISTRY, adapt_home_credit, adapt_indian_bank

logger = logging.getLogger(__name__)

# ── Page Config ──
st.set_page_config(
    page_title="GreenScore CPD Engine",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 GreenScore: Climate-Adjusted Probability of Default")
st.caption("B2B SaaS | RBI Disclosure Aligned | NGFS Phase V Scenarios")

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
REQUIRED_COLS = ['dti', 'annual_inc', 'fico_range_low', 'int_rate', 'installment', 'emp_length']
OPTIONAL_COLS = ['addr_state', 'purpose', 'loan_amnt']

with st.sidebar:
    st.header("⚙️ Scenario Settings")
    scenario = st.selectbox(
        "NGFS Carbon Price Scenario",
        list(config.CARBON_PRICES.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Orderly ≈ $100/tCO₂, Disorderly ≈ $250/tCO₂, Hot House ≈ $25/tCO₂",
    )
    carbon_price = config.CARBON_PRICES[scenario]
    st.metric("Carbon Price (2030)", f"${carbon_price}/tCO₂")

    st.markdown("---")
    st.subheader("📁 Dataset Source")
    dataset_options = {k: v['label'] for k, v in DATASET_REGISTRY.items()}
    dataset_options['custom'] = '📂 Custom Upload'
    dataset_choice = st.selectbox(
        "Select Dataset",
        list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        help="Choose a built-in dataset or upload your own CSV",
    )

    st.markdown("---")
    st.subheader("🎚️ Sensitivity Analysis")
    severity_factor = st.slider(
        "Physical Risk Severity Factor",
        min_value=0.1, max_value=0.5, value=config.SEVERITY_FACTOR, step=0.05,
        help="Higher = greater PD uplift from physical hazards. Default 0.3 per Bell & van Vuuren (2022).",
    )
    transition_scaling = st.slider(
        "Transition Risk Scaling Factor",
        min_value=0.1, max_value=0.6, value=config.TRANSITION_SCALING, step=0.05,
        help="Higher = greater PD uplift from carbon costs. Default 0.4. Range 0.2–0.6 in literature.",
    )

    st.markdown("---")
    st.markdown("**Model**: XGBoost (Optuna-tuned) + LightGBM + SHAP")
    st.markdown("**Regulatory**: RBI Climate Risk Framework 2024")
    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("- Bell & van Vuuren (2022)")
    st.markdown("- Bouchet, Dayan & Contoux (2022)")
    st.markdown("- NGFS Phase V Scenarios (2025)")


# ─────────────────────────────────────────────────────────
# Helper: Compute CPD for a given scenario + factors
# ─────────────────────────────────────────────────────────
def compute_cpd(df, model, scenario_key, sev_factor, trans_scaling):
    """Run the full CPD pipeline and return baseline_pd, cpd arrays."""
    baseline_pd = get_baseline_pd(model, df)

    loc_col = 'addr_state' if 'addr_state' in df.columns else None
    purpose_col = 'purpose' if 'purpose' in df.columns else None
    income_col = 'annual_inc'

    if loc_col:
        pd_physical = apply_physical_risk(baseline_pd, df[loc_col], severity_factor=sev_factor)
    else:
        pd_physical = baseline_pd.copy()

    if purpose_col:
        sector_col_series = df['sector'] if 'sector' in df.columns else None
        cpd = apply_transition_risk(
            pd_physical, df[purpose_col], df[income_col],
            scenario=scenario_key, transition_scaling=trans_scaling,
            sector_series=sector_col_series,
        )
    else:
        cpd = pd_physical.copy()

    return baseline_pd, cpd


# ─────────────────────────────────────────────────────────
# Data Loading — Built-in datasets or custom upload
# ─────────────────────────────────────────────────────────
df = None

if dataset_choice == 'custom':
    uploaded = st.file_uploader(
        "📂 Upload Loan Portfolio (CSV)",
        type=['csv'],
        help=f"Required: {', '.join(REQUIRED_COLS)}. Optional: {', '.join(OPTIONAL_COLS)}",
    )
    if uploaded:
        df = pd.read_csv(uploaded, low_memory=False)
else:
    ds_info = DATASET_REGISTRY[dataset_choice]
    ds_path = ds_info['path']
    if not os.path.exists(ds_path):
        st.error(f"❌ Dataset file not found: `{ds_path}`. Please download it first (see README).")
        st.stop()

    @st.cache_data(show_spinner=f"Loading {ds_info['label']}…")
    def _load_builtin(key, path, nrows=50000):
        if DATASET_REGISTRY[key]['loader'] is not None:
            return DATASET_REGISTRY[key]['loader'](path=path, nrows=nrows)
        else:
            from cpd_engine import load_data
            return load_data(path, nrows=nrows)

    df = _load_builtin(dataset_choice, ds_path)
    st.success(f"✅ Loaded **{ds_info['label']}** — {len(df):,} loans")

if df is not None:

    # ── Input Validation (custom uploads only) ──
    if dataset_choice == 'custom':
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"❌ **Missing required columns:** {', '.join(missing)}")
            st.markdown("Your CSV must contain these columns:")
            for col in REQUIRED_COLS:
                icon = "✅" if col in df.columns else "❌"
                st.markdown(f"  {icon} `{col}`")
            st.stop()

    # Clean numeric types (safe for all sources)
    for col in ['dti', 'annual_inc', 'fico_range_low', 'installment']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'int_rate' in df.columns:
        df['int_rate'] = pd.to_numeric(df['int_rate'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)

    st.success(f"✅ Loaded {len(df):,} loans")

    with st.expander("📋 Preview Raw Data"):
        st.dataframe(df.head(20), use_container_width=True)

    # ── Load Model ──
    try:
        model = joblib.load('models/baseline_pd_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model not trained yet. Run `python3 cpd_engine.py` first.")
        st.stop()

    # ── Compute CPD ──
    with st.spinner("🔄 Computing Climate-Adjusted PDs…"):
        baseline_pd, cpd = compute_cpd(df, model, scenario, severity_factor, transition_scaling)

        df['Baseline_PD'] = baseline_pd
        df['CPD_2030'] = cpd
        df['PD_Uplift_%'] = ((cpd - baseline_pd) / (baseline_pd + 1e-8)) * 100
        df['Risk_Category'] = pd.cut(
            cpd, bins=config.RISK_BINS, labels=config.RISK_LABELS, include_lowest=True,
        )

    # ═══════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════
    tab_overview, tab_compare, tab_map, tab_sector, tab_shap, tab_top20, tab_results = st.tabs([
        "📊 Overview", "🔀 Multi-Scenario", "🗺️ Heatmap", "🏭 Sectors", "🔍 SHAP", "🔝 Top 20 Risk", "📋 Results",
    ])

    # ── TAB 1: Overview ──
    with tab_overview:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Baseline PD", f"{baseline_pd.mean():.4f}")
        col2.metric("Avg CPD 2030", f"{cpd.mean():.4f}")
        col3.metric("Avg PD Uplift", f"{df['PD_Uplift_%'].mean():.1f}%")
        col4.metric("High/Critical", f"{(df['Risk_Category'].isin(['High', 'Critical'])).sum():,}")

        # Plotly histogram: Baseline vs CPD
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=baseline_pd, nbinsx=60, name='Baseline PD',
            marker_color='steelblue', opacity=0.6,
        ))
        fig_hist.add_trace(go.Histogram(
            x=cpd, nbinsx=60, name='CPD 2030',
            marker_color='crimson', opacity=0.6,
        ))
        fig_hist.update_layout(
            title='Baseline PD vs Climate-Adjusted PD',
            xaxis_title='Probability of Default', yaxis_title='Count',
            barmode='overlay', template='plotly_dark',
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Risk category donut
        risk_counts = df['Risk_Category'].value_counts().reset_index()
        risk_counts.columns = ['Category', 'Count']
        color_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}
        fig_donut = px.pie(
            risk_counts, names='Category', values='Count',
            color='Category', color_discrete_map=color_map,
            hole=0.45, title='Risk Category Distribution',
        )
        fig_donut.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── TAB 2: Multi-Scenario Comparison ──
    with tab_compare:
        st.subheader("🔀 Multi-Scenario Comparison")
        st.markdown("Side-by-side CPD analysis across all three NGFS scenarios.")

        scenario_results = {}
        for sc_name in config.CARBON_PRICES:
            _, sc_cpd = compute_cpd(df, model, sc_name, severity_factor, transition_scaling)
            scenario_results[sc_name] = sc_cpd

        # Metrics row
        cols = st.columns(3)
        sc_names = list(config.CARBON_PRICES.keys())
        for i, sc_name in enumerate(sc_names):
            sc_cpd = scenario_results[sc_name]
            sc_label = sc_name.replace('_', ' ').title()
            with cols[i]:
                st.markdown(f"**{sc_label}** (${config.CARBON_PRICES[sc_name]}/tCO₂)")
                st.metric("Avg CPD", f"{sc_cpd.mean():.4f}")
                uplift = ((sc_cpd - baseline_pd) / (baseline_pd + 1e-8) * 100).mean()
                st.metric("Avg Uplift", f"{uplift:.1f}%")
                high_crit = (pd.cut(sc_cpd, bins=config.RISK_BINS, labels=config.RISK_LABELS).isin(['High', 'Critical'])).sum()
                st.metric("High/Critical", f"{high_crit:,}")

        # Overlay histogram
        fig_compare = go.Figure()
        for sc_name in sc_names:
            fig_compare.add_trace(go.Histogram(
                x=scenario_results[sc_name], nbinsx=50,
                name=sc_name.replace('_', ' ').title(), opacity=0.5,
            ))
        fig_compare.update_layout(
            title='CPD Distribution Across NGFS Scenarios',
            xaxis_title='Climate-Adjusted PD', yaxis_title='Count',
            barmode='overlay', template='plotly_dark', height=400,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # ── TAB 3: Geographic Heatmap ──
    with tab_map:
        st.subheader("🗺️ Geographic Climate Risk Heatmap")
        loc_col = 'addr_state' if 'addr_state' in df.columns else None
        if loc_col:
            state_risk = df.groupby(loc_col)['CPD_2030'].mean().reset_index()

            # Detect if US or India data
            sample_states = df[loc_col].dropna().unique()[:10]
            is_us = any(len(str(s).strip()) == 2 for s in sample_states)

            if is_us:
                m = folium.Map(location=[39.8, -98.5], zoom_start=4)
                coord_map = config.US_STATE_COORDS
            else:
                m = folium.Map(location=[20, 77], zoom_start=5)
                coord_map = config.INDIA_STATE_COORDS

            for _, row in state_risk.iterrows():
                state = str(row[loc_col]).strip()
                cpd_val = float(row['CPD_2030'])

                if state in coord_map:
                    lat, lon = coord_map[state]
                elif state in config.US_STATE_COORDS:
                    lat, lon = config.US_STATE_COORDS[state]
                elif state in config.INDIA_STATE_COORDS:
                    lat, lon = config.INDIA_STATE_COORDS[state]
                else:
                    continue  # Skip unknown states instead of random placement

                color = 'red' if cpd_val > 0.2 else ('orange' if cpd_val > 0.1 else 'green')
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=max(5, cpd_val * 40),
                    color=color, fill=True, fill_opacity=0.7,
                    popup=f"<b>{state}</b><br>CPD: {cpd_val:.4f}",
                    tooltip=f"{state}: {cpd_val:.4f}",
                ).add_to(m)

            st_folium(m, width=800, height=500)
        else:
            st.info("No `addr_state` column found. Upload data with state codes for the heatmap.")

    # ── TAB 4: Sector Analysis ──
    with tab_sector:
        st.subheader("🏭 Sector-wise Climate Exposure")
        purpose_col = 'purpose' if 'purpose' in df.columns else None
        if purpose_col:
            sector_df = df.groupby(purpose_col).agg(
                Avg_Baseline_PD=('Baseline_PD', 'mean'),
                Avg_CPD_2030=('CPD_2030', 'mean'),
                Avg_Uplift=('PD_Uplift_%', 'mean'),
                Loan_Count=(purpose_col, 'count'),
            ).round(4).reset_index().sort_values('Avg_CPD_2030', ascending=False)

            fig_sector = px.bar(
                sector_df, x=purpose_col, y=['Avg_Baseline_PD', 'Avg_CPD_2030'],
                barmode='group', title='Sector: Baseline PD vs CPD 2030',
                color_discrete_sequence=['steelblue', 'crimson'],
            )
            fig_sector.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_sector, use_container_width=True)
            st.dataframe(sector_df, use_container_width=True)
        else:
            st.info("No `purpose` column found.")

    # ── TAB 5: SHAP Explainability ──
    with tab_shap:
        st.subheader("🔍 SHAP Feature Explanations")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) values show **how each feature "
            "contributes to individual predictions**. Red = pushes PD higher, Blue = pushes PD lower."
        )

        shap_beeswarm = os.path.join('models', 'shap_beeswarm.png')
        shap_importance = os.path.join('models', 'shap_importance.png')

        if os.path.exists(shap_beeswarm) and os.path.exists(shap_importance):
            shap_col1, shap_col2 = st.columns(2)
            with shap_col1:
                st.markdown("#### Beeswarm Plot")
                st.image(shap_beeswarm, use_container_width=True)
                st.caption(
                    "Each dot = one loan. Position on X-axis = SHAP value (impact on PD). "
                    "Color = feature value (red = high, blue = low)."
                )
            with shap_col2:
                st.markdown("#### Feature Importance (Mean |SHAP|)")
                st.image(shap_importance, use_container_width=True)
                st.caption(
                    "Average absolute SHAP value per feature. Higher = more influential on default prediction."
                )

            # Live SHAP for a single loan
            st.markdown("---")
            st.markdown("#### 🔬 Explain a Single Loan")
            loan_idx = st.number_input(
                "Select loan index to explain",
                min_value=0, max_value=len(df) - 1, value=0, step=1,
            )
            try:
                import shap
                shap_sample = df.iloc[[loan_idx]]
                X_sample = shap_sample[config.ALL_FEATURES].copy()
                for feat in config.ALL_FEATURES:
                    if feat not in X_sample.columns:
                        X_sample[feat] = 0
                X_sample = X_sample.fillna(X_sample.median())

                explainer = shap.TreeExplainer(model)
                shap_vals = explainer(X_sample)

                # Build a waterfall-style table (Streamlit-friendly)
                feat_names = config.ALL_FEATURES
                sv = shap_vals.values[0]
                base = float(explainer.expected_value)
                waterfall_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Feature Value': [f"{X_sample[f].values[0]:.4f}" for f in feat_names],
                    'SHAP Value': sv,
                    'Direction': ['↑ Increases PD' if v > 0 else '↓ Decreases PD' for v in sv],
                }).sort_values('SHAP Value', key=abs, ascending=False)

                st.markdown(f"**Base value (avg prediction):** {base:.4f}")
                st.markdown(f"**This loan's predicted PD:** {base + sv.sum():.4f}")
                st.dataframe(
                    waterfall_df.style.background_gradient(subset=['SHAP Value'], cmap='RdBu_r'),
                    use_container_width=True, hide_index=True,
                )
            except Exception as e:
                st.warning(f"Live SHAP explanation unavailable: {e}")
        else:
            st.warning(
                "⚠️ SHAP plots not found. Retrain the model with:\n\n"
                "```bash\npython3 cpd_engine.py data/accepted_2007_to_2018Q4.csv 500000 --tune\n```"
            )

    # ── TAB 6: Top 20 Riskiest Loans ──
    with tab_top20:
        st.subheader("🔝 Top 20 Riskiest Loans")
        top_cols = [c for c in ['loan_amnt', 'addr_state', 'purpose', 'annual_inc',
                                'fico_range_low', 'Baseline_PD', 'CPD_2030', 'PD_Uplift_%', 'Risk_Category']
                    if c in df.columns]
        top20 = df.nlargest(20, 'CPD_2030')[top_cols]
        st.dataframe(
            top20.style.background_gradient(subset=['CPD_2030'], cmap='Reds'),
            use_container_width=True,
        )

    # ── TAB 7: Full Results ──
    with tab_results:
        st.subheader("📋 Loan-Level CPD Results")
        display_cols = [c for c in ['loan_amnt', 'addr_state', 'purpose',
                                    'Baseline_PD', 'CPD_2030', 'PD_Uplift_%', 'Risk_Category']
                        if c in df.columns]
        st.dataframe(df[display_cols].head(500), use_container_width=True)

    # ═══════════════════════════════════════════════
    # EXPORTS
    # ═══════════════════════════════════════════════
    st.markdown("---")
    st.subheader("⬇️ Export Results")
    col_a, col_b = st.columns(2)

    with col_a:
        csv_data = df[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV", csv_data,
            "cpd_results.csv", "text/csv", use_container_width=True,
        )

    with col_b:
        # Pre-generate PDF to fix two-click bug
        pdf_bytes = generate_pdf_report(df, scenario, baseline_pd.mean(), cpd.mean())
        st.download_button(
            "📄 Download RBI Disclosure PDF", pdf_bytes,
            "rbi_climate_disclosure.pdf", "application/pdf", use_container_width=True,
        )

else:
    # ── Landing State ──
    st.info("👆 Upload a loan portfolio CSV to begin analysis")
    st.markdown("---")
    st.markdown("### 📐 CPD Formula")
    st.latex(r"CPD = PD_{\text{baseline}} \times (1 + \alpha_{\text{physical}}) \times (1 + \alpha_{\text{transition}})")

    st.markdown("### 📌 Expected CSV Columns")
    col_info = pd.DataFrame({
        'Column': REQUIRED_COLS + OPTIONAL_COLS,
        'Required': ['✅'] * len(REQUIRED_COLS) + ['Optional'] * len(OPTIONAL_COLS),
        'Description': [
            'Debt-to-Income ratio', 'Annual income ($)', 'FICO credit score',
            'Interest rate (%)', 'Monthly installment ($)', 'Employment length',
            'State code (e.g. CA, FL)', 'Loan purpose / sector', 'Loan amount ($)',
        ],
    })
    st.dataframe(col_info, hide_index=True, use_container_width=True)

    st.markdown("### 🔬 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Baseline PD**")
        st.markdown("XGBoost classifier trained on historical loan performance data.")
    with col2:
        st.markdown("**2️⃣ Physical Risk**")
        st.markdown("State-level flood, cyclone, and heat hazard scores (IMD/NDMA/FEMA).")
    with col3:
        st.markdown("**3️⃣ Transition Risk**")
        st.markdown("NGFS Phase V carbon price × sector emission intensity → carbon cost burden.")
