"""
GreenScore — Dataset Adapters
===============================
Adapter functions that transform secondary datasets (Indian Bank Loan,
Home Credit Default Risk) into the schema expected by the GreenScore
CPD pipeline, enabling cross-dataset validation and India-context
demonstration.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Home Credit Default Risk Adapter
# ─────────────────────────────────────────────────────────

def adapt_home_credit(
    path: str = 'data/home-credit-default-risk/application_train.csv',
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adapt the Home Credit Default Risk dataset to GreenScore's schema.

    Mapping:
        TARGET              → default (1 = default, 0 = paid)
        AMT_INCOME_TOTAL    → annual_inc
        AMT_CREDIT          → loan_amnt
        AMT_ANNUITY / 12    → installment (monthly)
        DAYS_EMPLOYED       → emp_length (years)
        EXT_SOURCE_2 * 550 + 300 → fico_range_low (proxy: external score scaled to FICO range)
        AMT_ANNUITY * 12 / AMT_INCOME_TOTAL → dti (proxy)
        CNT_CHILDREN-based rate proxy → int_rate
        ORGANIZATION_TYPE   → purpose (mapped via sector lookup)
        REGION_RATING_CLIENT → addr_state (mapped to synthetic region codes)

    Returns a DataFrame with columns compatible with the CPD pipeline.
    """
    logger.info("Loading Home Credit dataset from %s (nrows=%s)…", path, nrows)

    use_cols = [
        'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'DAYS_BIRTH',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'CNT_CHILDREN', 'ORGANIZATION_TYPE', 'REGION_RATING_CLIENT',
        'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE',
    ]
    df = pd.read_csv(path, usecols=use_cols, nrows=nrows, low_memory=False)
    initial_count = len(df)

    # Target
    df['default'] = df['TARGET'].astype(int)
    df['loan_status'] = df['TARGET'].map({1: 'Charged Off', 0: 'Fully Paid'})

    # Income
    df['annual_inc'] = df['AMT_INCOME_TOTAL']

    # Loan amount
    df['loan_amnt'] = df['AMT_CREDIT']

    # Monthly installment
    df['installment'] = df['AMT_ANNUITY'].fillna(0)

    # Employment length (DAYS_EMPLOYED is negative; 365243 = unemployed sentinel)
    df['emp_length'] = df['DAYS_EMPLOYED'].apply(
        lambda x: 0 if x > 0 else min(abs(x) / 365.25, 40)
    )

    # FICO proxy from EXT_SOURCE_2 (0–1 external score → 300–850 FICO range)
    ext_score = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['fico_range_low'] = (ext_score * 550 + 300).clip(300, 850).fillna(580)

    # DTI proxy
    annual_payment = df['installment'] * 12
    df['dti'] = (annual_payment / (df['annual_inc'] + 1) * 100).clip(0, 100)

    # Interest rate proxy based on contract type and credit amount
    df['int_rate'] = (df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1) - 1).clip(0, 0.5) * 100
    df['int_rate'] = df['int_rate'].fillna(12.0)

    # Sector mapping from ORGANIZATION_TYPE
    df['purpose'] = df['ORGANIZATION_TYPE'].map(
        config.HOMECREDIT_ORGANIZATION_TO_SECTOR
    ).fillna('other')

    # Region → synthetic state code for physical risk
    # REGION_RATING_CLIENT: 1 = best, 2 = medium, 3 = worst
    # Map to representative US states by risk level for demonstration
    region_map = {1: 'NY', 2: 'TX', 3: 'FL'}
    df['addr_state'] = df['REGION_RATING_CLIENT'].map(region_map).fillna('NY')

    # Drop rows with critical missing values
    df = df.dropna(subset=['annual_inc', 'loan_amnt', 'fico_range_low', 'dti'])

    logger.info(
        "Home Credit adapted: %s → %s records (%.1f%% retained). Default rate: %.3f",
        f"{initial_count:,}", f"{len(df):,}",
        len(df) / initial_count * 100, df['default'].mean(),
    )

    return df


# ─────────────────────────────────────────────────────────
# Indian Bank Loan Dataset Adapter
# ─────────────────────────────────────────────────────────

def adapt_indian_bank(
    path: str = 'data/Indian Banks Loan Dataset/train_modified.csv',
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adapt the Indian Bank Loan Dataset to GreenScore's schema.

    This dataset is heavily pre-processed (one-hot encoded) with limited
    raw feature overlap. The adapter maps available features:
        Disbursed            → default (inverted: 0 = not disbursed = proxy default)
        Monthly_Income * 12  → annual_inc
        Loan_Amount_Applied  → loan_amnt
        age                  → used to derive emp_length proxy
        Existing_EMI         → installment proxy

    Missing features (dti, fico_range_low, int_rate) are synthesised from
    available data with reasonable assumptions for demonstration purposes.
    Indian state is assigned based on Source_ columns for physical risk demo.

    Returns a DataFrame with columns compatible with the CPD pipeline.
    """
    logger.info("Loading Indian Bank Loan dataset from %s (nrows=%s)…", path, nrows)
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    initial_count = len(df)

    # Target: Disbursed=0 means loan was rejected (proxy for higher risk)
    df['default'] = (1 - df['Disbursed']).astype(int)
    df['loan_status'] = df['Disbursed'].map({0: 'Charged Off', 1: 'Fully Paid'})

    # Income
    df['annual_inc'] = df['Monthly_Income'] * 12

    # Loan amount
    df['loan_amnt'] = df['Loan_Amount_Applied']

    # Installment proxy from Existing_EMI or computed from loan/tenure
    tenure_months = df['Loan_Tenure_Applied'] * 12
    df['installment'] = np.where(
        df['Existing_EMI'] > 0,
        df['Existing_EMI'],
        df['Loan_Amount_Applied'] / (tenure_months + 1),
    )

    # Employment length proxy from age (assume started working at ~22)
    df['emp_length'] = (df['age'] - 22).clip(0, 40).fillna(5)

    # DTI proxy: existing EMI burden relative to income
    df['dti'] = (df['Existing_EMI'] * 12 / (df['annual_inc'] + 1) * 100).clip(0, 100)

    # FICO proxy: India doesn't use FICO — synthesise from Var4/Var5 + income
    # Var4 appears to be a credit grade (1–6), map to FICO bands
    fico_map = {1: 750, 2: 710, 3: 670, 4: 630, 5: 590, 6: 550}
    df['fico_range_low'] = df['Var4'].map(fico_map).fillna(650)

    # Interest rate proxy
    df['int_rate'] = 12.0 + (df['Var5'] * 0.5)
    df['int_rate'] = df['int_rate'].clip(5, 30).fillna(12.0)

    # Purpose: map Source columns to sectors for transition risk
    # Source_0/1/2 are one-hot encoded channel indicators
    df['purpose'] = 'small_business'  # Default for Indian bank loans
    if 'Source_0' in df.columns:
        df.loc[df['Source_0'] == 1, 'purpose'] = 'small_business'
        df.loc[df.get('Source_1', 0) == 1, 'purpose'] = 'home_improvement'
        df.loc[df.get('Source_2', 0) == 1, 'purpose'] = 'debt_consolidation'

    # State: assign representative Indian states based on loan characteristics
    # Higher income → more likely urban (Maharashtra/Delhi), lower → rural (Bihar/UP)
    income_median = df['annual_inc'].median()
    df['addr_state'] = np.where(
        df['annual_inc'] > income_median * 1.5, 'Maharashtra',
        np.where(df['annual_inc'] > income_median, 'Tamil Nadu',
                 np.where(df['annual_inc'] > income_median * 0.5, 'Uttar Pradesh', 'Bihar'))
    )

    df = df.dropna(subset=['annual_inc', 'loan_amnt'])

    logger.info(
        "Indian Bank adapted: %s → %s records (%.1f%% retained). Default rate: %.3f",
        f"{initial_count:,}", f"{len(df):,}",
        len(df) / initial_count * 100, df['default'].mean(),
    )

    return df


# ─────────────────────────────────────────────────────────
# Unified Loader
# ─────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    'lendingclub': {
        'label': 'LendingClub (US)',
        'path': 'data/accepted_2007_to_2018Q4.csv',
        'loader': None,  # Uses cpd_engine.load_data directly
    },
    'home_credit': {
        'label': 'Home Credit Default Risk',
        'path': 'data/home-credit-default-risk/application_train.csv',
        'loader': adapt_home_credit,
    },
    'indian_bank': {
        'label': 'Indian Bank Loan Dataset',
        'path': 'data/Indian Banks Loan Dataset/train_modified.csv',
        'loader': adapt_indian_bank,
    },
}
