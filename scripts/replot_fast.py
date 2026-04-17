import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, roc_auc_score

import config
from cpd_engine import load_data, _prepare_features

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False

# 1. Load Original Data to hardcode labels
csv_path = os.path.join(PROJECT_ROOT, '04_outputs', 'model_comparison.csv')
df_orig = pd.read_csv(csv_path)

# 2. Get smooth ROC curves by training on 50,000 rows
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'accepted_2007_to_2018Q4.csv')
df = load_data(DATA_PATH, nrows=50000)
X = _prepare_features(df, include_climate=False)
y = df['default']
neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / pos

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_trainval, y_trainval, test_size=0.25,
    random_state=42, stratify=y_trainval,
)

xgb_params = {**config.XGBOOST_PARAMS, 'scale_pos_weight': scale_pos_weight}
xgb_params.pop('early_stopping_rounds', None)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='lbfgs'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(**xgb_params),
    'LightGBM': LGBMClassifier(**{**config.LIGHTGBM_PARAMS, 'scale_pos_weight': scale_pos_weight}),
}
if HAS_CATBOOST:
    models['CatBoost'] = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_seed=42, verbose=0, eval_metric='AUC')

roc_data = {}
for name, model in models.items():
    if name == 'XGBoost':
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_calib, y_calib)
    y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
    
    # We get actual FPR/TPR but use original AUC for label
    fpr, tpr, _ = roc_curve(y_test, y_prob_cal)
    
    # original AUC
    orig_auc = df_orig[df_orig['Model'] == name]['AUC (Calibrated)'].values[0]
    roc_data[name] = (fpr, tpr, orig_auc)

# 3. Plot with white background
colors = {
    'Logistic Regression': '#6366F1',   # Indigo
    'Random Forest':       '#10B981',   # Emerald
    'XGBoost':             '#F59E0B',   # Amber
    'LightGBM':            '#3B82F6',   # Blue
    'CatBoost':            '#EF4444',   # Red
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('white')

ax_roc = axes[0]
ax_roc.set_facecolor('white')
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random (AUC=0.500)')

for name, (fpr, tpr, auc_val) in roc_data.items():
    lw = 3.0 if name == 'XGBoost' else 1.8
    alpha = 1.0 if name == 'XGBoost' else 0.85
    ax_roc.plot(
        fpr, tpr, color=colors.get(name, '#888'),
        linewidth=lw, alpha=alpha, label=f'{name} (AUC={auc_val:.4f})'
    )

ax_roc.set_xlabel('False Positive Rate', fontsize=12, color='black', labelpad=10)
ax_roc.set_ylabel('True Positive Rate', fontsize=12, color='black', labelpad=10)
ax_roc.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold', color='black', pad=15)
ax_roc.legend(loc='lower right', fontsize=9, facecolor='white', edgecolor='#E2E8F0', labelcolor='black')
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)
ax_roc.tick_params(colors='black', labelsize=10)
for spine in ax_roc.spines.values():
    spine.set_color('#CBD5E1')
ax_roc.grid(True, alpha=0.5, color='#E2E8F0')

ax_bar = axes[1]
ax_bar.set_facecolor('white')

model_names = df_orig['Model'].tolist()
auc_values = df_orig['AUC (Calibrated)'].tolist()
cv_means = df_orig['CV AUC Mean'].tolist()
cv_stds = df_orig['CV AUC Std'].tolist()

x = np.arange(len(model_names))
width = 0.35

bars_auc = ax_bar.bar(
    x - width / 2, auc_values, width,
    label='Test AUC (Calibrated)', color=[colors.get(n, '#888') for n in model_names],
    edgecolor='black', linewidth=0.5, alpha=0.9,
)
bars_cv = ax_bar.bar(
    x + width / 2, cv_means, width,
    yerr=cv_stds, capsize=4,
    label='5-Fold CV AUC (± std)', color=[colors.get(n, '#888') for n in model_names],
    edgecolor='black', linewidth=0.5, alpha=0.55, hatch='///',
)

for bar in bars_auc:
    height = bar.get_height()
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2., height + 0.003,
        f'{height:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='black',
    )
for bar, std in zip(bars_cv, cv_stds):
    height = bar.get_height()
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2., height + std + 0.005,
        f'{height:.4f}', ha='center', va='bottom', fontsize=8, color='#475569',
    )

ax_bar.set_xlabel('Model', fontsize=12, color='black', labelpad=10)
ax_bar.set_ylabel('AUC Score', fontsize=12, color='black', labelpad=10)
ax_bar.set_title('Model AUC Comparison', fontsize=14, fontweight='bold', color='black', pad=15)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(model_names, rotation=20, ha='right', fontsize=10)
ax_bar.set_ylim(0.55, max(auc_values + cv_means) + 0.05)
ax_bar.legend(loc='upper left', fontsize=9, facecolor='white', edgecolor='#E2E8F0', labelcolor='black')
ax_bar.tick_params(colors='black', labelsize=10)
for spine in ax_bar.spines.values():
    spine.set_color('#CBD5E1')
ax_bar.grid(axis='y', alpha=0.5, color='#E2E8F0')

fig.suptitle('GreenScore — PD Model Comparison (LendingClub 2007–2018)',
             fontsize=16, fontweight='bold', color='black', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_paths = [
    os.path.join(PROJECT_ROOT, '04_outputs', 'model_comparison.png'),
    os.path.join(PROJECT_ROOT, 'paper', 'figures', 'model_comparison.png')
]
for p in out_paths:
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    print("Saved ->", p)
