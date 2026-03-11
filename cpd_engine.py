"""
GreenScore CPD Engine — Baseline Probability of Default
========================================================
Loads LendingClub data, engineers features, trains an XGBoost
baseline PD classifier with Optuna hyperparameter tuning,
cross-validation, model comparison (LR, RF, LightGBM),
SHAP explanations, and provides prediction functions for
the dashboard pipeline.
"""

import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config

# ── Logging ──
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────

def load_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load LendingClub CSV with only the columns needed for PD modelling.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    nrows : int, optional
        Number of rows to read (for faster demos).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with a binary ``default`` column.
    """
    logger.info("Loading data from %s (nrows=%s)…", path, nrows)
    df = pd.read_csv(path, usecols=config.RAW_COLS, low_memory=False, nrows=nrows)

    # Binary target: 1 = default, 0 = paid
    df['default'] = df['loan_status'].apply(
        lambda x: 1 if str(x) in config.DEFAULT_STATUSES else 0
    )
    df = df.dropna(subset=['dti', 'annual_inc', 'fico_range_low'])

    # Clean numeric columns
    df['int_rate'] = (
        df['int_rate'].astype(str).str.replace('%', '', regex=False).astype(float)
    )
    df['emp_length'] = (
        df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
    )

    # ── Feature Engineering ──
    df['income_to_installment'] = df['annual_inc'] / (df['installment'] * 12 + 1)
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['dti_bucket'] = pd.cut(
        df['dti'], bins=config.DTI_BINS, labels=config.DTI_LABELS, include_lowest=True
    ).astype(float)
    df['fico_bucket'] = pd.cut(
        df['fico_range_low'], bins=config.FICO_BINS, labels=config.FICO_LABELS, include_lowest=True
    ).astype(float)

    logger.info(
        "Loaded %s records. Default rate: %.3f",
        f"{len(df):,}", df['default'].mean(),
    )
    return df


# ─────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and impute model features."""
    X = df[config.ALL_FEATURES].copy()
    X = X.fillna(X.median())
    return X


def train_baseline_pd(df: pd.DataFrame, save_dir: str = 'models', tune: bool = False):
    """
    Train an XGBoost baseline PD model with:
      - Optional Optuna hyperparameter tuning (``tune=True``)
      - ``scale_pos_weight`` for class-imbalance handling
      - 5-fold stratified cross-validation
      - Comparison against Logistic Regression, Random Forest, and LightGBM
      - SHAP feature importance plot
      - Saved evaluation artefacts (confusion matrix, ROC, PR curves)

    Returns
    -------
    model : XGBClassifier
        The trained XGBoost model.
    results : dict
        Dictionary with AUC scores for all models and CV statistics.
    """
    os.makedirs(save_dir, exist_ok=True)
    X = _prepare_features(df)
    y = df['default']

    # Class imbalance ratio
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    logger.info("Class balance — neg: %s, pos: %s, scale_pos_weight: %.2f", neg, pos, scale_pos_weight)

    # ── Hold-out split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y,
    )

    # ── Optuna Hyperparameter Tuning ──
    if tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("Starting Optuna hyperparameter tuning (%d trials)…", config.OPTUNA_N_TRIALS)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', *config.OPTUNA_PARAM_SPACE['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *config.OPTUNA_PARAM_SPACE['learning_rate'], log=True),
                'n_estimators': trial.suggest_int('n_estimators', *config.OPTUNA_PARAM_SPACE['n_estimators'], step=50),
                'subsample': trial.suggest_float('subsample', *config.OPTUNA_PARAM_SPACE['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *config.OPTUNA_PARAM_SPACE['colsample_bytree']),
                'min_child_weight': trial.suggest_int('min_child_weight', *config.OPTUNA_PARAM_SPACE['min_child_weight']),
                'reg_alpha': trial.suggest_float('reg_alpha', *config.OPTUNA_PARAM_SPACE['reg_alpha'], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', *config.OPTUNA_PARAM_SPACE['reg_lambda'], log=True),
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'auc',
                'random_state': config.RANDOM_STATE,
                'verbosity': 0,
            }
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
            scores = cross_val_score(
                XGBClassifier(**params), X_train, y_train,
                cv=skf, scoring='roc_auc', n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)

        best_params = study.best_params
        best_params.update({
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'auc',
            'random_state': config.RANDOM_STATE,
            'verbosity': 0,
        })
        logger.info("Optuna best AUC (3-fold on train): %.4f", study.best_value)
        logger.info("Optuna best params: %s", {k: v for k, v in best_params.items() if k not in ('eval_metric', 'random_state', 'verbosity', 'scale_pos_weight')})

        xgb = XGBClassifier(**best_params)
    else:
        xgb_params = {**config.XGBOOST_PARAMS, 'scale_pos_weight': scale_pos_weight}
        xgb_params.pop('early_stopping_rounds', None)
        xgb = XGBClassifier(**xgb_params)

    # ── XGBoost Training ──
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    logger.info("XGBoost AUC: %.4f", xgb_auc)

    # ── 5-Fold Stratified CV for XGBoost ──
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_params = xgb.get_params()
    cv_params.pop('early_stopping_rounds', None)
    cv_params.pop('callbacks', None)
    xgb_cv_scores = cross_val_score(
        XGBClassifier(**cv_params),
        X, y, cv=skf, scoring='roc_auc', n_jobs=-1,
    )
    logger.info("XGBoost 5-Fold CV AUC: %.4f ± %.4f", xgb_cv_scores.mean(), xgb_cv_scores.std())

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=config.RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    logger.info("Logistic Regression AUC: %.4f", lr_auc)

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    logger.info("Random Forest AUC: %.4f", rf_auc)

    # ── LightGBM ──
    lgbm_params = {**config.LIGHTGBM_PARAMS, 'scale_pos_weight': scale_pos_weight}
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
    logger.info("LightGBM AUC: %.4f", lgbm_auc)

    # ── Save evaluation plots ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(xgb, X_test, y_test, ax=ax_cm, cmap='Blues')
        ax_cm.set_title('XGBoost — Confusion Matrix')
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig_cm)

        # ROC Curve — all models
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=ax_roc, name='XGBoost')
        RocCurveDisplay.from_estimator(lr, X_test, y_test, ax=ax_roc, name='Logistic Regression')
        RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax_roc, name='Random Forest')
        RocCurveDisplay.from_estimator(lgbm, X_test, y_test, ax=ax_roc, name='LightGBM')
        ax_roc.set_title('ROC Curve Comparison')
        fig_roc.tight_layout()
        fig_roc.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150)
        plt.close(fig_roc)

        # Precision-Recall Curve
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        PrecisionRecallDisplay.from_estimator(xgb, X_test, y_test, ax=ax_pr, name='XGBoost')
        PrecisionRecallDisplay.from_estimator(lr, X_test, y_test, ax=ax_pr, name='Logistic Regression')
        PrecisionRecallDisplay.from_estimator(rf, X_test, y_test, ax=ax_pr, name='Random Forest')
        PrecisionRecallDisplay.from_estimator(lgbm, X_test, y_test, ax=ax_pr, name='LightGBM')
        ax_pr.set_title('Precision-Recall Curve Comparison')
        fig_pr.tight_layout()
        fig_pr.savefig(os.path.join(save_dir, 'precision_recall_curves.png'), dpi=150)
        plt.close(fig_pr)

        logger.info("Saved evaluation plots to %s/", save_dir)
    except Exception as e:
        logger.warning("Could not save evaluation plots: %s", e)

    # ── SHAP Feature Importance ──
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=config.RANDOM_STATE)
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer(shap_sample)

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("Saved SHAP explanation plots to %s/", save_dir)
    except Exception as e:
        logger.warning("Could not save SHAP plots: %s", e)

    # ── Classification report ──
    y_pred = xgb.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Paid', 'Default'])
    logger.info("Classification Report:\n%s", report)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # ── Save the best model (XGBoost) ──
    model_path = os.path.join(save_dir, 'baseline_pd_model.pkl')
    joblib.dump(xgb, model_path)
    logger.info("Model saved to %s", model_path)

    results = {
        'xgboost_auc': xgb_auc,
        'xgboost_cv_mean': xgb_cv_scores.mean(),
        'xgboost_cv_std': xgb_cv_scores.std(),
        'logistic_regression_auc': lr_auc,
        'random_forest_auc': rf_auc,
        'lightgbm_auc': lgbm_auc,
        'scale_pos_weight': scale_pos_weight,
        'n_samples': len(df),
        'default_rate': y.mean(),
        'tuned': tune,
    }
    return xgb, results


# ─────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────

def get_baseline_pd(model, loan_data: pd.DataFrame) -> np.ndarray:
    """
    Get baseline PD predictions for loan data.

    Automatically engineers the required features if they are missing.
    """
    df = loan_data.copy()

    # Ensure engineered features exist
    if 'income_to_installment' not in df.columns:
        df['income_to_installment'] = df['annual_inc'] / (df['installment'] * 12 + 1)
    if 'loan_to_income' not in df.columns:
        df['loan_to_income'] = df.get('loan_amnt', 0) / (df['annual_inc'] + 1)
    if 'dti_bucket' not in df.columns:
        df['dti_bucket'] = pd.cut(
            df['dti'], bins=config.DTI_BINS, labels=config.DTI_LABELS, include_lowest=True
        ).astype(float)
    if 'fico_bucket' not in df.columns:
        df['fico_bucket'] = pd.cut(
            df['fico_range_low'], bins=config.FICO_BINS, labels=config.FICO_LABELS, include_lowest=True
        ).astype(float)

    X = df[config.ALL_FEATURES].fillna(df[config.ALL_FEATURES].median())
    return model.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/accepted_2007_to_2018Q4.csv'
    nrows = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    do_tune = '--tune' in sys.argv
    print(f"Loading data from {path} ({nrows} rows)…")
    if do_tune:
        print(f"Optuna tuning enabled ({config.OPTUNA_N_TRIALS} trials)")
    df = load_data(path, nrows=nrows)
    model, results = train_baseline_pd(df, tune=do_tune)
    print("\n═══ Model Comparison ═══")
    print(f"  XGBoost AUC (hold-out):   {results['xgboost_auc']:.4f}")
    print(f"  XGBoost AUC (5-fold CV):  {results['xgboost_cv_mean']:.4f} ± {results['xgboost_cv_std']:.4f}")
    print(f"  Logistic Regression AUC:  {results['logistic_regression_auc']:.4f}")
    print(f"  Random Forest AUC:        {results['random_forest_auc']:.4f}")
    print(f"  LightGBM AUC:             {results['lightgbm_auc']:.4f}")
    print(f"  Scale Pos Weight:         {results['scale_pos_weight']:.2f}")
    if results['tuned']:
        print("  Optuna Tuning:            ✅ Applied")
    print(f"\nModel + plots + SHAP saved to models/")
