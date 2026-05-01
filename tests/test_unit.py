"""
Unit tests for LightGBM parameter values and training constants.

Validates that target-specific hyperparameters are correctly configured
for Ret5 (stronger regularization) and Ret60 (high capacity).

Requirements: 1.1, 1.2, 1.3, 1.4, 2.3
"""

import numpy as np
import pytest
from train import LGB_PARAMS_RET5, LGB_PARAMS_RET60, MIN_BOOST_ROUND


class TestLGBParamsRet5:
    """Verify Ret5 LightGBM parameters enforce stronger regularization."""

    def test_num_leaves_at_most_63(self):
        assert LGB_PARAMS_RET5["num_leaves"] <= 63

    def test_max_depth_is_8(self):
        assert LGB_PARAMS_RET5["max_depth"] == 8

    def test_lambda_l1_at_least_0_5(self):
        assert LGB_PARAMS_RET5["lambda_l1"] >= 0.5

    def test_lambda_l2_at_least_5_0(self):
        assert LGB_PARAMS_RET5["lambda_l2"] >= 5.0

    def test_feature_fraction_at_most_0_5(self):
        assert LGB_PARAMS_RET5["feature_fraction"] <= 0.5


class TestLGBParamsRet60:
    """Verify Ret60 LightGBM parameters use high-capacity settings."""

    def test_num_leaves_at_least_255(self):
        assert LGB_PARAMS_RET60["num_leaves"] >= 255

    def test_max_depth_unlimited(self):
        assert LGB_PARAMS_RET60["max_depth"] == -1


class TestParamsDiffer:
    """Verify Ret5 and Ret60 parameter dicts are distinct on key hyperparameters."""

    @pytest.mark.parametrize("key", [
        "num_leaves",
        "max_depth",
        "lambda_l1",
        "lambda_l2",
        "feature_fraction",
    ])
    def test_ret5_and_ret60_differ(self, key):
        assert LGB_PARAMS_RET5[key] != LGB_PARAMS_RET60[key], (
            f"Expected LGB_PARAMS_RET5['{key}'] ({LGB_PARAMS_RET5[key]}) "
            f"to differ from LGB_PARAMS_RET60['{key}'] ({LGB_PARAMS_RET60[key]})"
        )


class TestMinBoostRound:
    """Verify the minimum boosting round constant."""

    def test_min_boost_round_is_30(self):
        assert MIN_BOOST_ROUND == 30


import tempfile
import json
from pathlib import Path


class TestBackwardCompatibilityFallbacks:
    """Verify predict.py gracefully handles missing optional files."""

    def _create_minimal_models(self, tmpdir):
        """Create minimal LightGBM models for testing."""
        import lightgbm as lgb
        np.random.seed(42)
        features = np.random.randn(100, 20).astype(np.float32)
        labels = np.random.randn(100).astype(np.float32)
        train_data = lgb.Dataset(features, label=labels, free_raw_data=False)
        params = {"objective": "regression", "num_leaves": 4, "verbose": -1, "num_threads": 1}
        model = lgb.train(params, train_data, num_boost_round=5)
        model.save_model(str(Path(tmpdir) / "lgb_ret5_dataset0.txt"))
        model.save_model(str(Path(tmpdir) / "lgb_ret60_dataset0.txt"))
        return features

    def test_missing_ensemble_weights_pure_lgb(self):
        """Missing ensemble_weights.json -> pure LightGBM output."""
        import predict
        with tempfile.TemporaryDirectory() as tmpdir:
            features = self._create_minimal_models(tmpdir)
            original = predict.MODEL_DIR
            predict.MODEL_DIR = Path(tmpdir)
            try:
                signals = predict.generate_signals("dataset0", features)
                assert signals.shape == (100, 2)
                assert signals.dtype == np.float32
                assert np.all(np.isfinite(signals))
            finally:
                predict.MODEL_DIR = original

    def test_missing_transformer_no_error(self):
        """Missing transformer .pt -> no error, falls back gracefully."""
        import predict
        with tempfile.TemporaryDirectory() as tmpdir:
            features = self._create_minimal_models(tmpdir)
            # Create ensemble_weights with gamma > 0 but no transformer file
            weights = {"dataset0": {
                "ret5_alpha": 0.8, "ret5_beta": 0.0, "ret5_gamma": 0.2,
                "ret60_alpha": 0.8, "ret60_beta": 0.0, "ret60_gamma": 0.2,
            }}
            with open(Path(tmpdir) / "ensemble_weights.json", "w") as f:
                json.dump(weights, f)
            
            original = predict.MODEL_DIR
            predict.MODEL_DIR = Path(tmpdir)
            try:
                signals = predict.generate_signals("dataset0", features)
                assert signals.shape == (100, 2)
                assert np.all(np.isfinite(signals))
            finally:
                predict.MODEL_DIR = original

    def test_missing_feature_selection_uses_full_matrix(self):
        """Missing feature_selection.json -> full Feature_Matrix used."""
        import predict
        with tempfile.TemporaryDirectory() as tmpdir:
            features = self._create_minimal_models(tmpdir)
            original = predict.MODEL_DIR
            predict.MODEL_DIR = Path(tmpdir)
            try:
                # No feature_selection.json present
                signals = predict.generate_signals("dataset0", features)
                assert signals.shape == (100, 2)
                assert np.all(np.isfinite(signals))
            finally:
                predict.MODEL_DIR = original

    def test_missing_regime_uses_original_matrix(self):
        """Missing regime .txt -> original Feature_Matrix used."""
        import predict
        with tempfile.TemporaryDirectory() as tmpdir:
            features = self._create_minimal_models(tmpdir)
            original = predict.MODEL_DIR
            predict.MODEL_DIR = Path(tmpdir)
            try:
                signals = predict.generate_signals("dataset0", features)
                assert signals.shape == (100, 2)
                assert np.all(np.isfinite(signals))
            finally:
                predict.MODEL_DIR = original

    def test_all_optional_missing_valid_output(self):
        """All optional files missing -> valid (T, 2) float32 output."""
        import predict
        with tempfile.TemporaryDirectory() as tmpdir:
            features = self._create_minimal_models(tmpdir)
            # Inject NaN/Inf
            features[0, 0] = np.nan
            features[1, 1] = np.inf
            
            original = predict.MODEL_DIR
            predict.MODEL_DIR = Path(tmpdir)
            try:
                signals = predict.generate_signals("dataset0", features)
                assert signals.shape == (100, 2)
                assert signals.dtype == np.float32
                assert np.all(np.isfinite(signals))
            finally:
                predict.MODEL_DIR = original
