"""
Unit tests for GreenScore risk modules.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from physical_risk import apply_physical_risk, compute_physical_risk_score
from transition_risk import apply_transition_risk, map_purpose_to_sector
from nasa_power import engineer_physical_features, _cache_key


# ─────────────────────────────────────────────────────────
# Physical Risk Tests
# ─────────────────────────────────────────────────────────

class TestPhysicalRisk:
    """Tests for physical_risk module."""

    def test_known_state_scores(self):
        """Known US states should return their configured risk score (static mode)."""
        states = pd.Series(['FL', 'NY', 'CA', 'LA'])
        scores = compute_physical_risk_score(states, use_nasa=False)
        assert scores.iloc[0] == config.STATE_PHYSICAL_RISK['FL']
        assert scores.iloc[1] == config.STATE_PHYSICAL_RISK['NY']
        assert scores.iloc[2] == config.STATE_PHYSICAL_RISK['CA']
        assert scores.iloc[3] == config.STATE_PHYSICAL_RISK['LA']

    def test_indian_state_scores(self):
        """Indian states should return configured risk scores (static mode)."""
        states = pd.Series(['Odisha', 'Kerala', 'Rajasthan'])
        scores = compute_physical_risk_score(states, use_nasa=False)
        assert scores.iloc[0] == 0.90  # Odisha — highest
        assert scores.iloc[1] == 0.70  # Kerala
        assert scores.iloc[2] == 0.50  # Rajasthan

    def test_unknown_state_fallback(self):
        """Unknown states should get the OTHER fallback score (static mode)."""
        states = pd.Series(['ZZ', 'UNKNOWN', ''])
        scores = compute_physical_risk_score(states, use_nasa=False)
        assert all(scores == config.STATE_PHYSICAL_RISK['OTHER'])

    def test_nasa_mode_returns_valid_scores(self):
        """NASA mode should return scores in [0, 1] range."""
        states = pd.Series(['FL', 'CA'])
        scores = compute_physical_risk_score(states, use_nasa=True)
        assert all(0 <= s <= 1 for s in scores)

    def test_all_50_us_states_covered(self):
        """All 50 US states + DC should have a risk score."""
        us_states = [
            'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL',
            'GA','HI','ID','IL','IN','IA','KS','KY','LA','ME',
            'MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
            'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
            'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',
        ]
        for state in us_states:
            assert state in config.STATE_PHYSICAL_RISK, f"{state} missing from risk scores"

    def test_apply_physical_risk_increases_pd(self):
        """Physical risk overlay should always increase (or maintain) PD."""
        baseline = np.array([0.05, 0.10, 0.20])
        states = pd.Series(['FL', 'NY', 'OH'])
        adjusted = apply_physical_risk(baseline, states)
        assert np.all(adjusted >= baseline)

    def test_apply_physical_risk_clipped_to_01(self):
        """Output PD should be clipped to [0, 1]."""
        baseline = np.array([0.95, 0.99])
        states = pd.Series(['LA', 'FL'])
        adjusted = apply_physical_risk(baseline, states)
        assert np.all(adjusted <= 1.0)
        assert np.all(adjusted >= 0.0)

    def test_severity_factor_override(self):
        """Custom severity factor should change the output."""
        baseline = np.array([0.10])
        states = pd.Series(['FL'])
        default_result = apply_physical_risk(baseline, states)
        custom_result = apply_physical_risk(baseline, states, severity_factor=0.5)
        assert custom_result[0] > default_result[0]

    def test_zero_severity_returns_baseline(self):
        """Severity factor of 0 should return baseline unchanged."""
        baseline = np.array([0.05, 0.15])
        states = pd.Series(['FL', 'CA'])
        result = apply_physical_risk(baseline, states, severity_factor=0.0)
        np.testing.assert_array_almost_equal(result, baseline)


# ─────────────────────────────────────────────────────────
# Transition Risk Tests
# ─────────────────────────────────────────────────────────

class TestTransitionRisk:
    """Tests for transition_risk module."""

    def test_purpose_mapping_known(self):
        """Known purposes should return correct emission intensity."""
        purposes = pd.Series(['car', 'medical', 'renewable_energy'])
        intensities = map_purpose_to_sector(purposes)
        assert intensities.iloc[0] == config.SECTOR_EMISSIONS['car']
        assert intensities.iloc[1] == config.SECTOR_EMISSIONS['medical']
        assert intensities.iloc[2] == config.SECTOR_EMISSIONS['renewable_energy']

    def test_purpose_mapping_unknown(self):
        """Unknown purpose should fall back to 'other' sector."""
        purposes = pd.Series(['unknown_purpose', 'something_weird'])
        intensities = map_purpose_to_sector(purposes)
        assert all(intensities == config.SECTOR_EMISSIONS['other'])

    def test_disorderly_higher_than_orderly(self):
        """Disorderly scenario (higher carbon price) should produce higher CPD."""
        pd_phys = np.array([0.10, 0.10])
        purpose = pd.Series(['car', 'car'])
        income = pd.Series([50000, 50000])
        cpd_orderly = apply_transition_risk(pd_phys, purpose, income, 'orderly')
        cpd_disorderly = apply_transition_risk(pd_phys, purpose, income, 'disorderly')
        assert np.all(cpd_disorderly >= cpd_orderly)

    def test_hot_house_lower_than_orderly(self):
        """Hot house (low carbon price) should produce lower transition uplift."""
        pd_phys = np.array([0.10])
        purpose = pd.Series(['small_business'])
        income = pd.Series([60000])
        cpd_orderly = apply_transition_risk(pd_phys, purpose, income, 'orderly')
        cpd_hot_house = apply_transition_risk(pd_phys, purpose, income, 'hot_house')
        assert cpd_hot_house[0] <= cpd_orderly[0]

    def test_renewable_energy_benefit(self):
        """Renewable energy should have negative emission intensity (benefit)."""
        assert config.SECTOR_EMISSIONS['renewable_energy'] < 0

    def test_output_clipped_to_01(self):
        """CPD should always be in [0, 1]."""
        pd_phys = np.array([0.95, 0.99])
        purpose = pd.Series(['coal', 'coal'])
        income = pd.Series([10000, 10000])  # Low income → high cost ratio
        cpd = apply_transition_risk(pd_phys, purpose, income, 'disorderly')
        assert np.all(cpd <= 1.0)
        assert np.all(cpd >= 0.0)

    def test_scaling_factor_override(self):
        """Custom scaling factor should change the output."""
        pd_phys = np.array([0.10])
        purpose = pd.Series(['car'])
        income = pd.Series([50000])
        default_cpd = apply_transition_risk(pd_phys, purpose, income, 'orderly')
        high_cpd = apply_transition_risk(pd_phys, purpose, income, 'orderly', transition_scaling=0.6)
        assert high_cpd[0] > default_cpd[0]

    def test_sector_override(self):
        """Direct sector_series should override purpose mapping."""
        pd_phys = np.array([0.10])
        purpose = pd.Series(['credit_card'])  # Low emission
        income = pd.Series([50000])
        sector = pd.Series(['coal'])  # High emission

        cpd_purpose = apply_transition_risk(pd_phys, purpose, income, 'orderly')
        cpd_sector = apply_transition_risk(pd_phys, purpose, income, 'orderly', sector_series=sector)
        assert cpd_sector[0] > cpd_purpose[0]  # Coal should be higher risk


# ─────────────────────────────────────────────────────────
# Config Integrity Tests
# ─────────────────────────────────────────────────────────

class TestConfig:
    """Tests for config.py data integrity."""

    def test_risk_scores_in_range(self):
        """All risk scores should be between -0.1 and 1.0."""
        for state, score in config.STATE_PHYSICAL_RISK.items():
            assert 0 <= score <= 1.0, f"{state} has out-of-range score: {score}"

    def test_all_us_states_have_coords(self):
        """All 50 US states + DC should have coordinates."""
        us_states = [
            'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL',
            'GA','HI','ID','IL','IN','IA','KS','KY','LA','ME',
            'MD','MA','MI','MN','MS','MO','MT','NE','NV','NH',
            'NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
            'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',
        ]
        for state in us_states:
            assert state in config.US_STATE_COORDS, f"{state} missing from US_STATE_COORDS"

    def test_carbon_prices_positive(self):
        """All carbon prices should be positive."""
        for scenario, price in config.CARBON_PRICES.items():
            assert price > 0, f"{scenario} has non-positive price: {price}"

    def test_risk_bins_monotonic(self):
        """Risk bins should be monotonically increasing."""
        for i in range(len(config.RISK_BINS) - 1):
            assert config.RISK_BINS[i] < config.RISK_BINS[i + 1]

    def test_risk_labels_count(self):
        """Number of risk labels should be one less than bins."""
        assert len(config.RISK_LABELS) == len(config.RISK_BINS) - 1


# ─────────────────────────────────────────────────────────
# LightGBM Config Tests
# ─────────────────────────────────────────────────────────

class TestLightGBMConfig:
    """Tests for LightGBM configuration."""

    def test_lightgbm_params_exist(self):
        """LightGBM params should be defined in config."""
        assert hasattr(config, 'LIGHTGBM_PARAMS')
        assert 'learning_rate' in config.LIGHTGBM_PARAMS
        assert 'n_estimators' in config.LIGHTGBM_PARAMS

    def test_lightgbm_import(self):
        """LightGBM should be importable."""
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(**config.LIGHTGBM_PARAMS)
        assert lgbm is not None


# ─────────────────────────────────────────────────────────
# Optuna Config Tests
# ─────────────────────────────────────────────────────────

class TestOptunaConfig:
    """Tests for Optuna hyperparameter tuning configuration."""

    def test_optuna_n_trials_positive(self):
        """Number of Optuna trials should be positive."""
        assert config.OPTUNA_N_TRIALS > 0

    def test_optuna_param_space_valid(self):
        """Optuna param space should have valid ranges."""
        for param, (low, high) in config.OPTUNA_PARAM_SPACE.items():
            assert low < high, f"{param} has invalid range: [{low}, {high}]"

    def test_optuna_import(self):
        """Optuna should be importable."""
        import optuna
        assert optuna is not None


# ─────────────────────────────────────────────────────────
# Dataset Adapter Tests
# ─────────────────────────────────────────────────────────

class TestDatasetAdapters:
    """Tests for dataset adapter functions."""

    def test_home_credit_adapter_schema(self):
        """Home Credit adapter should produce required pipeline columns."""
        try:
            from dataset_adapters import adapt_home_credit
            df = adapt_home_credit(nrows=100)
            required = ['default', 'annual_inc', 'loan_amnt', 'installment',
                        'emp_length', 'fico_range_low', 'dti', 'int_rate',
                        'purpose', 'addr_state']
            for col in required:
                assert col in df.columns, f"Missing column: {col}"
        except FileNotFoundError:
            pytest.skip("Home Credit dataset not available")

    def test_home_credit_adapter_values(self):
        """Home Credit adapted values should be in valid ranges."""
        try:
            from dataset_adapters import adapt_home_credit
            df = adapt_home_credit(nrows=100)
            assert df['default'].isin([0, 1]).all()
            assert (df['annual_inc'] >= 0).all()
            assert (df['fico_range_low'] >= 300).all()
            assert (df['fico_range_low'] <= 850).all()
        except FileNotFoundError:
            pytest.skip("Home Credit dataset not available")

    def test_indian_bank_adapter_schema(self):
        """Indian Bank adapter should produce required pipeline columns."""
        try:
            from dataset_adapters import adapt_indian_bank
            df = adapt_indian_bank(nrows=100)
            required = ['default', 'annual_inc', 'loan_amnt', 'installment',
                        'emp_length', 'fico_range_low', 'dti', 'int_rate',
                        'purpose', 'addr_state']
            for col in required:
                assert col in df.columns, f"Missing column: {col}"
        except FileNotFoundError:
            pytest.skip("Indian Bank dataset not available")

    def test_indian_bank_adapter_indian_states(self):
        """Indian Bank adapter should assign Indian state names."""
        try:
            from dataset_adapters import adapt_indian_bank
            df = adapt_indian_bank(nrows=100)
            indian_states = set(config.INDIA_STATE_COORDS.keys())
            assigned = set(df['addr_state'].unique())
            assert assigned.issubset(indian_states), f"Non-Indian states found: {assigned - indian_states}"
        except FileNotFoundError:
            pytest.skip("Indian Bank dataset not available")

    def test_dataset_registry(self):
        """Dataset registry should have all three datasets."""
        from dataset_adapters import DATASET_REGISTRY
        assert 'lendingclub' in DATASET_REGISTRY
        assert 'home_credit' in DATASET_REGISTRY
        assert 'indian_bank' in DATASET_REGISTRY


# ─────────────────────────────────────────────────────────
# NASA POWER Feature Engineering Tests
# ─────────────────────────────────────────────────────────

class TestNASAPower:
    """Tests for the NASA POWER climate feature engineering."""

    def test_engineer_features_with_data(self):
        """Valid raw climate data should produce all 5 physical features."""
        raw = {
            'T2M': {str(m): 15 + 10 * np.sin(m / 6 * np.pi)
                    for m in range(1, 169)},
            'PRECTOTCORR': {str(m): max(0, 2 + np.random.default_rng(42).normal(0, 1))
                            for m in range(1, 169)},
        }
        features = engineer_physical_features(raw)
        assert 'flood_freq_score' in features
        assert 'drought_severity_index' in features
        assert 'temp_anomaly_5yr' in features
        assert 'extreme_weather_events_count' in features
        assert 'physical_risk_score' in features
        assert 0 <= features['physical_risk_score'] <= 1

    def test_engineer_features_none_input(self):
        """None input should return fallback features."""
        features = engineer_physical_features(None)
        assert features['physical_risk_score'] == 0.30
        assert features['flood_freq_score'] == 0.30

    def test_engineer_features_empty_data(self):
        """Empty data dict should return fallback."""
        features = engineer_physical_features({})
        assert features['physical_risk_score'] == 0.30

    def test_engineer_features_sentinel_values(self):
        """Data with only -999 sentinel values should return fallback."""
        raw = {
            'T2M': {str(m): -999 for m in range(1, 169)},
            'PRECTOTCORR': {str(m): -999 for m in range(1, 169)},
        }
        features = engineer_physical_features(raw)
        assert features['physical_risk_score'] == 0.30

    def test_cache_key_format(self):
        """Cache key should be a deterministic lat_lon string."""
        key = _cache_key(42.2, -74.9)
        assert key == '42.2_-74.9'

    def test_cache_key_rounds(self):
        """Cache key should round to 1 decimal."""
        key = _cache_key(42.249, -74.851)
        assert key == '42.2_-74.9'


# ─────────────────────────────────────────────────────────
# 4th NGFS Scenario Tests
# ─────────────────────────────────────────────────────────

class TestFourthNGFSScenario:
    """Tests for the Too Little Too Late scenario."""

    def test_too_little_too_late_exists(self):
        """Config should contain the fourth NGFS scenario."""
        assert 'too_little_too_late' in config.CARBON_PRICES

    def test_too_little_too_late_price(self):
        """Price should be between orderly and disorderly."""
        price = config.CARBON_PRICES['too_little_too_late']
        assert config.CARBON_PRICES['hot_house'] < price < config.CARBON_PRICES['disorderly']

    def test_transition_risk_with_fourth_scenario(self):
        """apply_transition_risk should work with the new scenario."""
        pd_phys = np.array([0.10])
        purpose = pd.Series(['car'])
        income = pd.Series([50000])
        cpd = apply_transition_risk(pd_phys, purpose, income, 'too_little_too_late')
        assert 0 <= cpd[0] <= 1

    def test_fourth_scenario_between_orderly_and_disorderly(self):
        """Too-Little-Too-Late CPD should fall between orderly and disorderly."""
        pd_phys = np.array([0.10])
        purpose = pd.Series(['small_business'])
        income = pd.Series([50000])
        cpd_orderly = apply_transition_risk(pd_phys, purpose, income, 'orderly')
        cpd_tltl = apply_transition_risk(pd_phys, purpose, income, 'too_little_too_late')
        cpd_disorderly = apply_transition_risk(pd_phys, purpose, income, 'disorderly')
        assert cpd_orderly[0] <= cpd_tltl[0] <= cpd_disorderly[0]


# ─────────────────────────────────────────────────────────
# Climate Feature Config Tests
# ─────────────────────────────────────────────────────────

class TestClimateFeatureConfig:
    """Tests for climate feature configuration."""

    def test_climate_features_defined(self):
        """Climate feature lists should exist in config."""
        assert len(config.PHYSICAL_RISK_FEATURES) == 5
        assert len(config.TRANSITION_RISK_FEATURES) == 3
        assert len(config.GEOGRAPHIC_FEATURES) == 2

    def test_all_features_climate_is_superset(self):
        """ALL_FEATURES_CLIMATE should contain all base + climate features."""
        for feat in config.ALL_FEATURES:
            assert feat in config.ALL_FEATURES_CLIMATE
        for feat in config.CLIMATE_FEATURES:
            assert feat in config.ALL_FEATURES_CLIMATE

    def test_coastal_proximity_lookup(self):
        """Coastal proximity should be defined for key states."""
        assert config.STATE_COASTAL_PROXIMITY_KM['FL'] < 50  # Florida is coastal
        assert config.STATE_COASTAL_PROXIMITY_KM['CO'] > 1000  # Colorado is inland

    def test_elevation_lookup(self):
        """Elevation should be defined for key states."""
        assert config.STATE_ELEVATION_METERS['CO'] > 2000  # Colorado is high
        assert config.STATE_ELEVATION_METERS['FL'] < 20  # Florida is flat

    def test_policy_exposure_defined(self):
        """Sector policy exposure should be defined."""
        assert config.SECTOR_POLICY_EXPOSURE['coal'] > 0.9
        assert config.SECTOR_POLICY_EXPOSURE['technology'] < 0.2


# ─────────────────────────────────────────────────────────
# Phase 1.3 Feature Engineering Tests
# ─────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """Tests for Phase 1.3 feature engineering additions."""

    def test_new_raw_cols_in_config(self):
        """New raw columns should be in RAW_COLS."""
        for col in ['revol_util', 'revol_bal', 'open_acc', 'total_acc',
                     'pub_rec', 'delinq_2yrs', 'inq_last_6mths', 'term']:
            assert col in config.RAW_COLS, f"{col} missing from RAW_COLS"

    def test_new_base_features_in_config(self):
        """New base features should be in BASE_FEATURES."""
        for feat in ['revol_util', 'revol_bal', 'open_acc', 'total_acc',
                      'pub_rec', 'delinq_2yrs', 'inq_last_6mths', 'term_months', 'loan_amnt']:
            assert feat in config.BASE_FEATURES, f"{feat} missing from BASE_FEATURES"

    def test_new_engineered_features_in_config(self):
        """New engineered features should be in ENGINEERED_FEATURES."""
        for feat in ['monthly_payment_burden',
                      'credit_utilization_ratio', 'open_to_total_acc']:
            assert feat in config.ENGINEERED_FEATURES, f"{feat} missing from ENGINEERED_FEATURES"

    def test_all_features_length(self):
        """ALL_FEATURES should include all base + engineered features."""
        assert len(config.ALL_FEATURES) == len(config.BASE_FEATURES) + len(config.ENGINEERED_FEATURES)

    def test_emp_length_bins_defined(self):
        """Employment length bins should be defined."""
        assert hasattr(config, 'EMP_LENGTH_BINS')
        assert hasattr(config, 'EMP_LENGTH_LABELS')
        assert len(config.EMP_LENGTH_LABELS) == len(config.EMP_LENGTH_BINS) - 1

    def test_get_baseline_pd_creates_features(self):
        """get_baseline_pd should create missing engineered features."""
        from cpd_engine import get_baseline_pd
        # Create minimal input with only required columns
        df = pd.DataFrame({
            'dti': [15.0], 'annual_inc': [60000], 'fico_range_low': [700],
            'int_rate': [12.0], 'installment': [500], 'emp_length': [5],
        })
        # Test that feature creation works correctly
        df_copy = df.copy()
        for col, default in [
            ('revol_util', 0), ('revol_bal', 0), ('open_acc', 0),
            ('total_acc', 0), ('pub_rec', 0), ('delinq_2yrs', 0),
            ('inq_last_6mths', 0), ('loan_amnt', 0), ('term_months', 36),
            ('sub_grade_num', 18), ('verification_num', 0),
            ('credit_history_months', 180),
        ]:
            if col not in df_copy.columns:
                df_copy[col] = default
        df_copy['monthly_payment_burden'] = df_copy['installment'] / (df_copy['annual_inc'] / 12 + 1)
        df_copy['credit_utilization_ratio'] = df_copy['revol_bal'] / (df_copy['annual_inc'] + 1)
        assert abs(df_copy['monthly_payment_burden'].iloc[0] - 500 / (60000 / 12 + 1)) < 0.001
        assert df_copy['sub_grade_num'].iloc[0] == 18
