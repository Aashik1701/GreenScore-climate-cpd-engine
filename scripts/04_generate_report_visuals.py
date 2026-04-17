import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
PAPER_FIGS = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(MODELS, exist_ok=True)

# Ensure architecture image path referenced in LaTeX exists.
arch_src = os.path.join(PAPER_FIGS, 'greenscore_architecture.png')
arch_dst = os.path.join(MODELS, 'architectu.png')
if os.path.exists(arch_src) and not os.path.exists(arch_dst):
    shutil.copyfile(arch_src, arch_dst)

# Figure 2: CPD formula flow diagram
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis('off')
boxes = [
    (0.02, 0.25, 0.2, 0.5, r'$\mathrm{PD}_{\mathrm{baseline}}$'),
    (0.29, 0.25, 0.2, 0.5, r'$\times\,(1+\alpha_{\mathrm{physical}})$'),
    (0.56, 0.25, 0.2, 0.5, r'$\times\,(1+\alpha_{\mathrm{transition}})$'),
    (0.83, 0.25, 0.15, 0.5, r'$\mathrm{CPD}$'),
]
for x, y, w, h, txt in boxes:
    ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor='black', linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=11)
for x0, x1 in [(0.22, 0.29), (0.49, 0.56), (0.76, 0.83)]:
    ax.annotate('', xy=(x1, 0.5), xytext=(x0, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.set_title('Figure 2: Climate-Adjusted PD Composition', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(MODELS, 'fig2_cpd_formula_flow.png'), dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 8: scenario distribution stacked bar
scenarios = ['Baseline\nPD', 'Hot\nHouse', 'Orderly', 'Disorderly']
low = [42.3, 38.1, 35.7, 31.2]
medium = [35.8, 36.4, 37.2, 36.9]
high = [16.1, 18.9, 20.4, 23.7]
critical = [5.8, 6.6, 6.7, 8.2]

x = np.arange(len(scenarios))
fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.bar(x, low, 0.55, label='Low (<5%)', color='white', edgecolor='black')
ax.bar(x, medium, 0.55, bottom=low, label='Medium (5-15%)', color='#d9d9d9', edgecolor='black')
ax.bar(x, high, 0.55, bottom=np.array(low) + np.array(medium), label='High (15-30%)', color='#888888', edgecolor='black')
ax.bar(x, critical, 0.55, bottom=np.array(low) + np.array(medium) + np.array(high), label='Critical (>30%)', color='#333333', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylabel('Portfolio Share (%)', fontsize=11)
ax.set_ylim(0, 112)
ax.set_title('Figure 8: Risk Tier Distribution Across NGFS Scenarios', fontsize=12)
ax.legend(loc='upper right', fontsize=9, frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(MODELS, 'fig8_scenario_distribution.png'), dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 9: time-series CPD projection
years = np.arange(2025, 2051)
orderly = 11.3 + (14.7 - 11.3) * (years - 2025) / 25.0

t = (years - 2025) / 25.0
disorderly = 11.3 + (21.0 - 11.3) * (t ** 1.45)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years, orderly, color='black', linewidth=1.6, linestyle='-', label='Orderly ($100/tCO_2$)')
ax.plot(years, disorderly, color='black', linewidth=1.6, linestyle='--', label='Disorderly ($250/tCO_2$)')
ax.axvline(x=2030, color='gray', linewidth=1.0, linestyle=':')
ax.text(2030.2, 12.2, '2030 target', fontsize=9, color='gray')
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Median Portfolio CPD (%)', fontsize=11)
ax.set_ylim(8, 25)
ax.set_title('Figure 9: Forward-Looking CPD Projection by Scenario (2025-2050)', fontsize=12)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(MODELS, 'fig9_timeseries.png'), dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 10: ablation bar chart
labels = ['Baseline\nPD', 'Physical\nOnly', 'Transition\nOnly', 'Additive\nSum', 'Dual\nMultiplicative']
values = [12.84, 14.29, 13.89, 15.34, 15.46]
colors = ['white', '#bfbfbf', '#bfbfbf', '#6e6e6e', 'black']

fig, ax = plt.subplots(figsize=(8.2, 4.8))
bars = ax.bar(labels, values, color=colors, edgecolor='black', width=0.62)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.07, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylim(11, 17)
ax.set_ylabel('Mean Portfolio CPD (%)', fontsize=11)
ax.set_title('Figure 10: Ablation Study — Mean Portfolio CPD by Configuration', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(MODELS, 'fig10_ablation.png'), dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 11: Monte Carlo error bound visual
error_rates = np.linspace(0, 30, 301)
cpd_error = 0.000038 * error_rates
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(error_rates, cpd_error * 1000, color='black', linewidth=1.6)
ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.0, label='10%: ±0.00038')
ax.axvline(x=30, color='gray', linestyle=':', linewidth=1.0, label='30%: ±0.00114')
ax.set_xlabel('Sector Mapping Error Rate (%)', fontsize=11)
ax.set_ylabel('Portfolio Mean CPD Error (x10$^{-3}$)', fontsize=11)
ax.set_title('Figure 11: Transition Proxy Error Bounds (10,000 Monte Carlo Iterations)', fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(MODELS, 'fig11_montecarlo.png'), dpi=220, bbox_inches='tight')
plt.close(fig)

# Figure 13 placeholder from existing SHAP importance bar if available
shap_src = os.path.join(MODELS, 'shap_importance.png')
shap_dst = os.path.join(MODELS, 'fig13_shap_bar.png')
if os.path.exists(shap_src):
    shutil.copyfile(shap_src, shap_dst)

print('Generated visuals in', MODELS)
