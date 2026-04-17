"""
GreenScore — System Architecture Figure Generator (Task 2)
==========================================================
Generates a block diagram of the GreenScore CPD Engine.
Outputs to: figures/greenscore_architecture.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

def draw_box(ax, x, y, width, height, text, bg_color, edge_color, text_color='white', fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         fc=bg_color, ec=edge_color, lw=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            color=text_color, fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))

def generate_architecture_figure():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    c_source = '#3B82F6'   # Blue
    c_process = '#F59E0B'  # Amber
    c_model = '#10B981'    # Green
    c_output = '#8B5CF6'   # Purple
    c_edge = '#1E293B'

    # Row 1: Data Sources (y=8)
    ax.text(5, 9.2, "ROW 1: DATA SOURCES", ha='center', fontsize=12, fontweight='bold', color='#475569')
    draw_box(ax, 1, 8, 1.5, 0.8, "LendingClub\n(2007-2018 Data)", c_source, c_edge)
    draw_box(ax, 3.5, 8, 1.5, 0.8, "NASA POWER\n(Climate API)", c_source, c_edge)
    draw_box(ax, 6, 8, 1.5, 0.8, "NGFS\n(Scenarios)", c_source, c_edge)
    draw_box(ax, 8.5, 8, 1.5, 0.8, "Sector Carbon\n(Emissions Data)", c_source, c_edge)

    # Row 2: Processing (y=5.5)
    ax.text(5, 6.7, "ROW 2: PROCESSING", ha='center', fontsize=12, fontweight='bold', color='#475569')
    draw_box(ax, 1, 5.5, 2.5, 0.8, "Feature Engineering\n(Financial Ratios)", c_process, c_edge)
    draw_box(ax, 4.5, 5.5, 2, 0.8, "Hazard Scoring\n(Physical Risk)", c_process, c_edge)
    draw_box(ax, 7.5, 5.5, 2, 0.8, "Carbon Burden\n(Transition Risk)", c_process, c_edge)

    # Row 3: Core Model (y=3)
    ax.text(5, 4.2, "ROW 3: CORE MODEL", ha='center', fontsize=12, fontweight='bold', color='#475569')
    draw_box(ax, 1.5, 3, 2, 0.8, "XGBoost PD\n(Baseline Risk)", c_model, c_edge)
    draw_box(ax, 4.5, 3, 2.5, 0.8, "Physical & Transition\nRisk Overlays", c_model, c_edge)
    draw_box(ax, 8, 3, 1.5, 0.8, "SHAP\n(Explainability)", c_model, c_edge)

    # Row 4: Outputs (y=0.5)
    ax.text(5, 1.7, "ROW 4: OUTPUTS", ha='center', fontsize=12, fontweight='bold', color='#475569')
    draw_box(ax, 0.5, 0.5, 2, 0.8, "CPD + Risk Bands\n(Adjusted PD)", c_output, c_edge)
    draw_box(ax, 3.25, 0.5, 2, 0.8, "Scenario Analysis\n(Orderly/Disorderly)", c_output, c_edge)
    draw_box(ax, 6, 0.5, 2, 0.8, "Heatmap / Dashboard\n(Folium & Streamlit)", c_output, c_edge)
    draw_box(ax, 8.75, 0.5, 1, 0.8, "API\n(FastAPI)", c_output, c_edge)

    # Arrows between Row 1 and Row 2
    draw_arrow(ax, 1.75, 8, 2.25, 6.3) # LC -> Feature Eng
    draw_arrow(ax, 4.25, 8, 5.5, 6.3)  # NASA -> Hazard Score
    draw_arrow(ax, 6.75, 8, 8.5, 6.3)  # NGFS -> Carbon Burden
    draw_arrow(ax, 9.25, 8, 8.5, 6.3)  # Sector -> Carbon Burden

    # Arrows between Row 2 and Row 3
    draw_arrow(ax, 2.25, 5.5, 2.5, 3.8) # Feature Eng -> XGBoost
    draw_arrow(ax, 5.5, 5.5, 5.75, 3.8) # Hazard -> Overlay
    draw_arrow(ax, 8.5, 5.5, 5.75, 3.8) # Carbon -> Overlay
    
    # XGBoost to Overlay
    draw_arrow(ax, 3.5, 3.4, 4.5, 3.4)

    # Overlay to SHAP
    draw_arrow(ax, 7.0, 3.4, 8.0, 3.4)

    # Arrows between Row 3 and Row 4
    draw_arrow(ax, 5.75, 3, 1.5, 1.3) # Overlay -> CPD
    draw_arrow(ax, 5.75, 3, 4.25, 1.3) # Overlay -> Scenario
    draw_arrow(ax, 5.75, 3, 7.0, 1.3) # Overlay -> Heatmap
    draw_arrow(ax, 5.75, 3, 9.25, 1.3) # Overlay -> API

    plt.suptitle("GreenScore System Architecture", fontsize=16, fontweight='bold', y=0.95)
    
    # Save diagram
    out_path = os.path.join('figures', 'greenscore_architecture.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to {out_path}")

if __name__ == "__main__":
    generate_architecture_figure()
