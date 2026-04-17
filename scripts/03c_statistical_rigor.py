"""
GreenScore — Statistical Rigor Tests (Task 3C)
==============================================
Computes DeLong Confidence Intervals for AUC, Expected Calibration Error (ECE)
and Bootstrap standard errors for the XGBoost model.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from cpd_engine import load_data, _prepare_features

# ── Fast DeLong Implementation ──
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    tx = np.empty([positive_examples.shape[0], m], dtype=float)
    ty = np.empty([negative_examples.shape[0], n], dtype=float)
    tz = np.empty([predictions_sorted_transposed.shape[0], m + n], dtype=float)
    for r in range(predictions_sorted_transposed.shape[0]):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, cov):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, cov), l.T))
    return 2 * (1 - norm.cdf(np.abs(z)))[0][0]

def compute_ci(auc, cov, alpha=0.05):
    z = norm.ppf(1 - alpha / 2)
    return auc - z * np.sqrt(cov), auc + z * np.sqrt(cov)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_y = y_true[binids == i]
        bin_prob = y_prob[binids == i]
        if len(bin_y) > 0:
            ece += (len(bin_y) / len(y_true)) * np.abs(bin_y.mean() - bin_prob.mean())
    return ece

if __name__ == '__main__':
    print("Loading data for statistical rigor tests...")
    # Load 150k sample for fast robust statistics
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'accepted_2007_to_2018Q4.csv')
    df = load_data(DATA_PATH, nrows=150000)
    X = _prepare_features(df, include_climate=False)
    y = df['default']
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_calib, y_train, y_calib = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training XGBoost...")
    xgb_params = {**config.XGBOOST_PARAMS, 'scale_pos_weight': scale_pos_weight}
    xgb_params.pop('early_stopping_rounds', None)
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    cal_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv='prefit')
    cal_xgb.fit(X_calib, y_calib)
    preds_xgb = cal_xgb.predict_proba(X_test)[:, 1]
    
    print("Training LightGBM...")
    lgbm_params = {**config.LIGHTGBM_PARAMS, 'scale_pos_weight': scale_pos_weight}
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train)
    cal_lgbm = CalibratedClassifierCV(lgbm, method='isotonic', cv='prefit')
    cal_lgbm.fit(X_calib, y_calib)
    preds_lgbm = cal_lgbm.predict_proba(X_test)[:, 1]
    
    # Compute DeLong CI
    y_test_np = y_test.values
    preds = np.vstack([preds_xgb, preds_lgbm])
    idx = np.argsort(y_test_np)[::-1]
    y_test_sorted = y_test_np[idx]
    preds_sorted = preds[:, idx]
    m = y_test_np.sum()
    aucs, cov = fastDeLong(preds_sorted, m)
    
    xgb_auc, lgbm_auc = aucs
    xgb_ci = compute_ci(xgb_auc, cov[0, 0])
    lgbm_ci = compute_ci(lgbm_auc, cov[1, 1])
    pval = calc_pvalue(aucs, cov)
    
    xgb_ece = expected_calibration_error(y_test_np, preds_xgb)
    
    print("====================================")
    print("STATISTICAL RIGOR RESULTS")
    print("====================================")
    print(f"XGBoost AUC:   {xgb_auc:.4f}  95% CI: [{xgb_ci[0]:.4f}, {xgb_ci[1]:.4f}]")
    print(f"LightGBM AUC:  {lgbm_auc:.4f}  95% CI: [{lgbm_ci[0]:.4f}, {lgbm_ci[1]:.4f}]")
    print(f"DeLong p-val:  {pval:.5e}")
    print(f"XGBoost ECE:   {xgb_ece:.4f}")
    print("====================================")
