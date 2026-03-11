import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle

# ── Load only needed columns (handles 2GB efficiently) ──
COLS = ['loan_status','loan_amnt','dti','annual_inc',
        'fico_range_low','int_rate','installment',
        'emp_length','home_ownership','purpose','addr_state']

def load_data(path, nrows=None):
    """Load LendingClub data with only needed columns."""
    df = pd.read_csv(path, usecols=COLS, low_memory=False, nrows=nrows)
    # Target: 1 = default, 0 = paid
    df['default'] = df['loan_status'].apply(
        lambda x: 1 if str(x) in ['Charged Off','Default','Late (31-120 days)'] else 0
    )
    df = df.dropna(subset=['dti','annual_inc','fico_range_low'])
    df['int_rate'] = df['int_rate'].astype(str).str.replace('%','').astype(float)
    df['emp_length'] = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
    return df

def train_baseline_pd(df):
    """Train XGBoost baseline PD model and save to disk."""
    features = ['dti','annual_inc','fico_range_low','int_rate','installment','emp_length']
    X = df[features].fillna(df[features].median())
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        learning_rate=0.05,
        max_depth=6,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='auc',
        early_stopping_rounds=10
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f"Baseline PD Model AUC: {auc:.4f}")
    
    # Save model
    with open('models/baseline_pd_model.pkl','wb') as f:
        pickle.dump(model, f)
    return model, auc

def get_baseline_pd(model, loan_data: pd.DataFrame) -> np.ndarray:
    """Get baseline PD predictions for loan data."""
    features = ['dti','annual_inc','fico_range_low','int_rate','installment','emp_length']
    X = loan_data[features].fillna(loan_data[features].median())
    return model.predict_proba(X)[:,1]


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/lending_club_sample.csv'
    print(f"Loading data from {path}...")
    df = load_data(path, nrows=200000)
    print(f"Loaded {len(df):,} records. Default rate: {df['default'].mean():.3f}")
    model, auc = train_baseline_pd(df)
    print(f"Model saved to models/baseline_pd_model.pkl (AUC: {auc:.4f})")
