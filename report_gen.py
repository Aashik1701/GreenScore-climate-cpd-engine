"""
GreenScore — PDF Report Generator
====================================
Generates an RBI Climate Risk Disclosure PDF report using ReportLab.
Aligned with RBI Draft Disclosure Framework (February 2024).

Enhancements over MVP:
  - Auto-generated executive summary
  - Embedded portfolio distribution chart
  - State-level top-10 risk table
  - Multi-scenario comparison table
"""

import io
import logging
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import config

logger = logging.getLogger(__name__)


def _make_chart_image(baseline_pd, cpd, width=5.5, height=2.8) -> io.BytesIO:
    """Render the PD overlay histogram as a PNG in memory."""
    fig, axes = plt.subplots(1, 2, figsize=(width, height))

    axes[0].hist(baseline_pd, bins=50, alpha=0.6, label='Baseline PD', color='steelblue')
    axes[0].hist(cpd, bins=50, alpha=0.6, label='CPD 2030', color='crimson')
    axes[0].set_xlabel('PD', fontsize=8)
    axes[0].set_ylabel('Count', fontsize=8)
    axes[0].set_title('Baseline vs Climate-Adjusted PD', fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].tick_params(labelsize=7)

    risk_cats = pd.cut(cpd, bins=config.RISK_BINS, labels=config.RISK_LABELS)
    counts = risk_cats.value_counts().reindex(config.RISK_LABELS, fill_value=0)
    cat_colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    axes[1].bar(counts.index, counts.values, color=cat_colors)
    axes[1].set_title('Risk Category Distribution', fontsize=9)
    axes[1].set_ylabel('Loans', fontsize=8)
    axes[1].tick_params(labelsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    df: pd.DataFrame,
    scenario: str,
    avg_baseline: float,
    avg_cpd: float,
) -> bytes:
    """
    Generate an RBI Climate Risk Disclosure PDF.

    Sections:
      0. Executive Summary
      1. Portfolio Overview
      2. Climate Risk Metrics (table)
      3. Portfolio Distribution (embedded chart)
      4. Sector Exposure (top-10 table)
      5. State-Level Risk (top-10 table)
      6. Multi-Scenario Comparison
      7. RBI Disclosure Alignment (4 pillars)
      8. Methodology
      9. Disclaimer
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=50, rightMargin=50,
                            topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle('SmallNormal', parent=styles['Normal'], fontSize=9, leading=12))
    story = []

    # ── Title ──
    story.append(Paragraph("GreenScore Climate Risk Disclosure Report", styles['Title']))
    story.append(Paragraph(
        f"RBI Climate-Related Financial Risk Framework  |  "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Scenario: {scenario.replace('_', ' ').title()}",
        styles['SmallNormal'],
    ))
    story.append(Spacer(1, 16))

    # ── 0. Executive Summary ──
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    uplift_pct = ((avg_cpd - avg_baseline) / (avg_baseline + 1e-8)) * 100
    high_crit = 0
    if 'Risk_Category' in df.columns:
        high_crit = (df['Risk_Category'].isin(['High', 'Critical'])).sum()
    high_crit_pct = high_crit / len(df) * 100 if len(df) > 0 else 0

    exec_summary = (
        f"This analysis evaluates {len(df):,} loans under the NGFS "
        f"{scenario.replace('_', ' ').title()} carbon-price pathway "
        f"(${config.CARBON_PRICES.get(scenario, 100)}/tCO₂ by 2030). "
        f"The average baseline probability of default is {avg_baseline:.4f}, "
        f"which rises to {avg_cpd:.4f} after applying physical and transition risk overlays — "
        f"an average uplift of {uplift_pct:.1f}%. "
        f"Approximately {high_crit:,} loans ({high_crit_pct:.1f}% of the portfolio) "
        f"fall into the High or Critical risk category, warranting enhanced monitoring "
        f"and potential capital provisioning adjustments."
    )
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 14))

    # ── 1. Portfolio Overview ──
    story.append(Paragraph("1. Portfolio Overview", styles['Heading2']))
    story.append(Paragraph(f"Total Loans Analyzed: {len(df):,}", styles['Normal']))
    story.append(Paragraph(
        f"NGFS Scenario: {scenario.replace('_', ' ').title()} Transition "
        f"(${config.CARBON_PRICES.get(scenario, 100)}/tCO₂)",
        styles['Normal'],
    ))
    story.append(Spacer(1, 10))

    # ── 2. Climate Risk Metrics ──
    story.append(Paragraph("2. Climate Risk Metrics", styles['Heading2']))
    data = [
        ['Metric', 'Value'],
        ['Average Baseline PD', f"{avg_baseline:.4f}"],
        ['Average Climate-Adjusted PD (CPD 2030)', f"{avg_cpd:.4f}"],
        ['Average PD Uplift', f"{uplift_pct:.1f}%"],
        ['High/Critical Risk Loans', f"{high_crit:,} ({high_crit_pct:.1f}%)"],
    ]
    t = Table(data, colWidths=[280, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e3f2fd'), colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # ── 3. Portfolio Distribution (embedded chart) ──
    story.append(Paragraph("3. Portfolio Distribution", styles['Heading2']))
    try:
        baseline_arr = df['Baseline_PD'].values if 'Baseline_PD' in df.columns else np.array([avg_baseline])
        cpd_arr = df['CPD_2030'].values if 'CPD_2030' in df.columns else np.array([avg_cpd])
        chart_buf = _make_chart_image(baseline_arr, cpd_arr)
        img = Image(chart_buf, width=5.5 * inch, height=2.8 * inch)
        story.append(img)
    except Exception as e:
        logger.warning("Could not embed chart in PDF: %s", e)
        story.append(Paragraph("(Chart generation failed.)", styles['Normal']))
    story.append(Spacer(1, 14))

    # ── 4. Sector Exposure ──
    story.append(Paragraph("4. Sector-wise Climate Exposure (Top 10)", styles['Heading2']))
    if 'purpose' in df.columns and 'CPD_2030' in df.columns:
        sector_summary = df.groupby('purpose').agg(
            Count=('purpose', 'count'),
            Avg_CPD=('CPD_2030', 'mean'),
        ).round(4).sort_values('Avg_CPD', ascending=False).head(10)

        sector_data = [['Sector / Purpose', 'Loan Count', 'Avg CPD 2030']]
        for idx, row in sector_summary.iterrows():
            sector_data.append([str(idx), f"{int(row['Count']):,}", f"{row['Avg_CPD']:.4f}"])

        st = Table(sector_data, colWidths=[200, 100, 120])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fff3e0'), colors.white]),
        ]))
        story.append(st)
    else:
        story.append(Paragraph("No sector data available.", styles['Normal']))
    story.append(Spacer(1, 14))

    # ── 5. State-Level Risk (Top 10) ──
    story.append(Paragraph("5. State-Level Climate Risk (Top 10)", styles['Heading2']))
    loc_col = 'addr_state' if 'addr_state' in df.columns else None
    if loc_col and 'CPD_2030' in df.columns:
        state_summary = df.groupby(loc_col).agg(
            Count=(loc_col, 'count'),
            Avg_CPD=('CPD_2030', 'mean'),
        ).round(4).sort_values('Avg_CPD', ascending=False).head(10)

        state_data = [['State', 'Loan Count', 'Avg CPD 2030']]
        for idx, row in state_summary.iterrows():
            state_data.append([str(idx), f"{int(row['Count']):,}", f"{row['Avg_CPD']:.4f}"])

        gt = Table(state_data, colWidths=[120, 100, 120])
        gt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e8f5e9'), colors.white]),
        ]))
        story.append(gt)
    else:
        story.append(Paragraph("No geographic data available.", styles['Normal']))
    story.append(Spacer(1, 14))

    # ── 6. Multi-Scenario Comparison ──
    story.append(Paragraph("6. Multi-Scenario Comparison", styles['Heading2']))
    story.append(Paragraph(
        "Projected climate-adjusted PD under all three NGFS Phase V carbon-price pathways:",
        styles['Normal'],
    ))
    story.append(Spacer(1, 6))

    # We only have the current scenario's full CPD, so we estimate
    # orderly/disorderly/hot_house using carbon price ratios
    from transition_risk import apply_transition_risk
    from physical_risk import apply_physical_risk

    baseline_arr = df['Baseline_PD'].values if 'Baseline_PD' in df.columns else np.full(len(df), avg_baseline)
    sc_table = [['Scenario', 'Carbon Price', 'Avg CPD 2030', 'Avg Uplift']]

    for sc_name, sc_price in config.CARBON_PRICES.items():
        if loc_col and loc_col in df.columns:
            pd_phys = apply_physical_risk(baseline_arr, df[loc_col])
        else:
            pd_phys = baseline_arr.copy()

        if 'purpose' in df.columns and 'annual_inc' in df.columns:
            sc_cpd = apply_transition_risk(pd_phys, df['purpose'], df['annual_inc'], sc_name)
        else:
            sc_cpd = pd_phys.copy()

        sc_uplift = ((sc_cpd - baseline_arr) / (baseline_arr + 1e-8) * 100).mean()
        sc_table.append([
            sc_name.replace('_', ' ').title(),
            f"${sc_price}/tCO₂",
            f"{sc_cpd.mean():.4f}",
            f"{sc_uplift:.1f}%",
        ])

    mt = Table(sc_table, colWidths=[120, 100, 120, 100])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fce4ec'), colors.white]),
    ]))
    story.append(mt)
    story.append(Spacer(1, 14))

    # ── 7. RBI Disclosure Alignment ──
    story.append(Paragraph("7. RBI Disclosure Alignment", styles['Heading2']))
    story.append(Paragraph(
        "This report aligns with the RBI Draft Disclosure Framework (February 2024):",
        styles['Normal'],
    ))
    story.append(Spacer(1, 6))
    pillars = [
        "<b>Governance:</b> Board-level oversight of climate risk integration into credit assessment.",
        "<b>Strategy:</b> Forward-looking scenario analysis using NGFS Phase V pathways.",
        "<b>Risk Management:</b> Dual-overlay methodology combining physical hazard scores and transition cost projections at loan level.",
        "<b>Metrics &amp; Targets:</b> Quantified CPD, PD uplift percentages, and portfolio risk category distribution.",
    ]
    for p in pillars:
        story.append(Paragraph(f"• {p}", styles['SmallNormal']))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 10))

    # ── 8. Methodology ──
    story.append(Paragraph("8. Methodology", styles['Heading2']))
    story.append(Paragraph(
        "CPD = Baseline_PD × (1 + Physical_Risk_Factor) × (1 + Transition_Risk_Factor). "
        "Baseline PD via XGBoost on LendingClub data. "
        "Physical risk: IMD/NDMA state-level vulnerability (severity factor = 0.3). "
        "Transition risk: NGFS Phase V carbon price × CPCB sector emissions (scaling = 0.4).",
        styles['SmallNormal'],
    ))
    story.append(Spacer(1, 10))

    # ── 9. Disclaimer ──
    story.append(Paragraph("Disclaimer", styles['Heading3']))
    story.append(Paragraph(
        "Generated by GreenScore CPD Engine for informational purposes. "
        "Climate projections are scenario-based and should not be treated as deterministic forecasts. "
        "All data sources are publicly available and independently verifiable.",
        styles['Italic'],
    ))

    doc.build(story)
    logger.info("PDF report generated (%d bytes).", buffer.tell())
    return buffer.getvalue()
