import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pickle
import matplotlib.pyplot as plt
import io
from cpd_engine import get_baseline_pd
from physical_risk import apply_physical_risk
from transition_risk import apply_transition_risk
from report_gen import generate_pdf_report

# ── Page Config ──
st.set_page_config(
    page_title="GreenScore CPD Engine",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GreenScore: Climate-Adjusted Probability of Default")
st.caption("B2B SaaS | RBI Disclosure Aligned | NGFS Phase V Scenarios")

# ── Sidebar Controls ──
with st.sidebar:
    st.header("⚙️ Scenario Settings")
    scenario = st.selectbox(
        "NGFS Carbon Price Scenario",
        ['orderly', 'disorderly', 'hot_house'],
        help="Orderly=~$100/tCO2, Disorderly=~$250, Hot House=~$25"
    )
    carbon_price_label = {
        'orderly': '$100/tCO2',
        'disorderly': '$250/tCO2',
        'hot_house': '$25/tCO2'
    }
    st.metric("Carbon Price (2030)", carbon_price_label[scenario])
    st.markdown("---")
    st.markdown("**Model**: XGBoost + Physical + Transition Risk")
    st.markdown("**Regulatory**: RBI Climate Risk Framework 2024")
    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("- Bell & van Vuuren (2022)")
    st.markdown("- Bouchet, Dayan & Contoux (2022)")
    st.markdown("- NGFS Phase V Scenarios (2025)")

# ── File Upload ──
uploaded = st.file_uploader(
    "📂 Upload Loan Portfolio (CSV)",
    type=['csv'],
    help="Required columns: dti, annual_inc, fico_range_low, int_rate, installment, emp_length, addr_state, purpose"
)

if uploaded:
    df = pd.read_csv(uploaded, low_memory=False)
    st.success(f"✅ Loaded {len(df):,} loans")

    with st.expander("📋 Preview Raw Data"):
        st.dataframe(df.head(20))

    # Load trained model
    try:
        with open('models/baseline_pd_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model not trained yet. Run `python cpd_engine.py` first to train the baseline PD model.")
        st.stop()

    with st.spinner("🔄 Computing Climate-Adjusted PDs..."):
        # Step 1: Baseline PD
        baseline_pd = get_baseline_pd(model, df)

        # Step 2: Physical Risk Overlay
        loc_col = 'addr_state' if 'addr_state' in df.columns else df.columns[0]
        pd_physical = apply_physical_risk(baseline_pd, df[loc_col])

        # Step 3: Transition Risk Overlay
        purpose_col = 'purpose' if 'purpose' in df.columns else 'loan_status'
        income_col = 'annual_inc' if 'annual_inc' in df.columns else df.columns[0]
        cpd = apply_transition_risk(pd_physical, df[purpose_col], df[income_col], scenario)

        # Add computed columns
        df['Baseline_PD'] = baseline_pd
        df['CPD_2030'] = cpd
        df['PD_Uplift_%'] = ((cpd - baseline_pd) / (baseline_pd + 1e-8)) * 100
        df['Risk_Category'] = pd.cut(
            cpd,
            bins=[0, 0.05, 0.15, 0.30, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

    # ══════════════════════════════════════════════
    # KEY METRICS
    # ══════════════════════════════════════════════
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Baseline PD", f"{baseline_pd.mean():.4f}")
    col2.metric("Avg CPD 2030", f"{cpd.mean():.4f}")
    col3.metric("Avg PD Uplift", f"{df['PD_Uplift_%'].mean():.1f}%")
    col4.metric("High/Critical Risk", f"{(df['Risk_Category'].isin(['High', 'Critical'])).sum():,}")

    # ══════════════════════════════════════════════
    # RISK DISTRIBUTION CHARTS
    # ══════════════════════════════════════════════
    st.subheader("📊 Portfolio Risk Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram: Baseline PD vs CPD
    axes[0].hist(baseline_pd, bins=50, alpha=0.6, label='Baseline PD', color='steelblue')
    axes[0].hist(cpd, bins=50, alpha=0.6, label='CPD 2030', color='crimson')
    axes[0].set_xlabel('Probability of Default')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Baseline PD vs Climate-Adjusted PD')
    axes[0].legend()

    # Bar chart: Risk Categories
    risk_counts = df['Risk_Category'].value_counts()
    color_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Critical': '#e74c3c'}
    bar_colors = [color_map.get(str(c), 'gray') for c in risk_counts.index]
    axes[1].bar(risk_counts.index.astype(str), risk_counts.values, color=bar_colors)
    axes[1].set_title('Risk Category Distribution')
    axes[1].set_ylabel('Number of Loans')

    plt.tight_layout()
    st.pyplot(fig)

    # ══════════════════════════════════════════════
    # GEOGRAPHIC HEATMAP
    # ══════════════════════════════════════════════
    st.subheader("🗺️ Geographic Climate Risk Heatmap")
    if loc_col in df.columns:
        state_risk = df.groupby(loc_col)['CPD_2030'].mean().reset_index()

        # Determine map center based on data (US vs India)
        sample_states = df[loc_col].dropna().unique()[:5]
        is_us_data = any(len(str(s).strip()) == 2 for s in sample_states)

        if is_us_data:
            m = folium.Map(location=[39.8, -98.5], zoom_start=4)
            # Approximate US state coordinates
            us_coords = {
                'CA': [36.7, -119.4], 'FL': [27.6, -81.5], 'TX': [31.0, -100.0],
                'NY': [42.2, -74.9], 'WA': [47.4, -121.5], 'LA': [30.5, -92.0],
                'NC': [35.6, -79.8], 'SC': [33.8, -81.2], 'GA': [32.2, -83.4],
                'AL': [32.3, -86.9], 'MS': [32.4, -89.6], 'NJ': [40.0, -74.5],
                'PA': [41.2, -77.2], 'OH': [40.4, -82.8], 'IL': [40.0, -89.0],
                'MI': [44.3, -85.6], 'VA': [37.4, -79.0], 'MD': [39.0, -76.6],
                'AZ': [34.0, -111.1], 'CO': [39.5, -105.8], 'MN': [46.7, -94.7],
                'WI': [43.8, -88.8], 'IN': [40.3, -86.1], 'MO': [38.6, -92.6],
                'TN': [35.5, -86.6], 'KY': [37.8, -84.3], 'OR': [43.8, -120.6],
                'NV': [38.8, -116.4], 'CT': [41.6, -72.7], 'MA': [42.4, -71.4],
            }
        else:
            m = folium.Map(location=[20, 77], zoom_start=4)
            us_coords = {}

        for _, row in state_risk.iterrows():
            state = str(row[loc_col]).strip()
            cpd_val = float(row['CPD_2030'])

            if state in us_coords:
                lat, lon = us_coords[state]
            else:
                lat = 20 + np.random.uniform(-8, 8)
                lon = 77 + np.random.uniform(-12, 12)

            color = 'red' if cpd_val > 0.2 else ('orange' if cpd_val > 0.1 else 'green')
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, cpd_val * 40),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{state}: CPD={cpd_val:.4f}"
            ).add_to(m)

        st_folium(m, width=700, height=450)

    # ══════════════════════════════════════════════
    # SECTOR ANALYSIS
    # ══════════════════════════════════════════════
    st.subheader("🏭 Sector-wise Climate Exposure")
    if purpose_col in df.columns:
        sector_df = df.groupby(purpose_col).agg(
            Avg_Baseline_PD=('Baseline_PD', 'mean'),
            Avg_CPD_2030=('CPD_2030', 'mean'),
            Avg_Uplift=('PD_Uplift_%', 'mean'),
            Loan_Count=(purpose_col, 'count')
        ).round(4).reset_index()
        sector_df = sector_df.sort_values('Avg_CPD_2030', ascending=False)
        st.dataframe(sector_df, use_container_width=True)

    # ══════════════════════════════════════════════
    # LOAN-LEVEL RESULTS TABLE
    # ══════════════════════════════════════════════
    st.subheader("📋 Loan-Level CPD Results")
    display_cols = [c for c in ['loan_amnt', loc_col, purpose_col,
                                'Baseline_PD', 'CPD_2030', 'PD_Uplift_%', 'Risk_Category']
                    if c in df.columns]
    st.dataframe(df[display_cols].head(200), use_container_width=True)

    # ══════════════════════════════════════════════
    # DOWNLOADS
    # ══════════════════════════════════════════════
    st.markdown("---")
    st.subheader("⬇️ Export Results")
    col_a, col_b = st.columns(2)

    with col_a:
        csv = df[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            "cpd_results.csv",
            "text/csv",
            use_container_width=True
        )

    with col_b:
        if st.button("📄 Generate RBI Disclosure PDF", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_pdf_report(df, scenario, baseline_pd.mean(), cpd.mean())
                st.download_button(
                    "📥 Download PDF Report",
                    pdf_bytes,
                    "rbi_climate_disclosure.pdf",
                    use_container_width=True
                )

else:
    # ── Landing State ──
    st.info("👆 Upload a loan portfolio CSV to begin analysis")

    st.markdown("---")
    st.markdown("### 📐 CPD Formula")
    st.latex(r"CPD = PD_{baseline} \times (1 + \alpha_{physical}) \times (1 + \alpha_{transition})")

    st.markdown("### 📌 Expected CSV Columns")
    st.markdown("""
    | Column | Description |
    |---|---|
    | `dti` | Debt-to-Income ratio |
    | `annual_inc` | Annual income |
    | `fico_range_low` | FICO credit score |
    | `int_rate` | Interest rate |
    | `installment` | Monthly installment |
    | `emp_length` | Employment length |
    | `addr_state` | State/location code |
    | `purpose` | Loan purpose/sector |
    """)

    st.markdown("### 🔬 How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Baseline PD**")
        st.markdown("XGBoost classifier trained on historical loan performance data (LendingClub).")
    with col2:
        st.markdown("**2️⃣ Physical Risk**")
        st.markdown("State-level flood, cyclone, and heat hazard scores from IMD/NDMA classifications.")
    with col3:
        st.markdown("**3️⃣ Transition Risk**")
        st.markdown("NGFS Phase V carbon price pathways × sector emission intensity → carbon cost burden.")
