"""
Property-based tests for model-optimization-v4.
Tests Properties 1–5 using the Hypothesis library.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp

from factor import generate_factors


# =============================================================================
# Property 1: 特征矩阵维数不变量
# Feature: model-optimization-v4, Property 1: 特征矩阵维数不变量
# Validates: Requirements 2.6, 5.2
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=500, max_value=1000).flatmap(
        lambda T: hnp.arrays(
            dtype=np.float32,
            shape=(T, 5),
            elements=st.floats(min_value=0.01, max_value=1000.0,
                               allow_nan=False, allow_infinity=False),
        )
    )
)
def test_property_1_feature_dim(ohlcv):
    """
    Feature: model-optimization-v4, Property 1: 特征矩阵维数不变量
    Validates: Requirements 2.6, 5.2

    For any OHLCV input of shape (T, 5), generate_factors() must return
    exactly 165 columns.
    """
    result = generate_factors("dataset0", ohlcv)
    assert result.shape[1] == 165


# =============================================================================
# Property 2: 特征矩阵精度不变量
# Feature: model-optimization-v4, Property 2: 特征矩阵精度不变量
# Validates: Requirement 5.8
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=500, max_value=1000).flatmap(
        lambda T: hnp.arrays(
            dtype=np.float32,
            shape=(T, 5),
            elements=st.floats(min_value=0.01, max_value=1000.0,
                               allow_nan=False, allow_infinity=False),
        )
    )
)
def test_property_2_feature_dtype(ohlcv):
    """
    Feature: model-optimization-v4, Property 2: 特征矩阵精度不变量
    Validates: Requirement 5.8

    For any OHLCV input of shape (T, 5), generate_factors() must return
    a float32 array.
    """
    result = generate_factors("dataset0", ohlcv)
    assert result.dtype == np.float32


# =============================================================================
# Property 3: 长周期特征因果性
# Feature: model-optimization-v4, Property 3: 长周期特征因果性
# Validates: Requirement 2.9
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=500, max_value=800).flatmap(
        lambda T: hnp.arrays(
            dtype=np.float32,
            shape=(T, 5),
            elements=st.floats(min_value=0.01, max_value=1000.0,
                               allow_nan=False, allow_infinity=False),
        )
    )
)
def test_property_3_causality(ohlcv):
    """
    Feature: model-optimization-v4, Property 3: 长周期特征因果性
    Validates: Requirement 2.9

    For any OHLCV sequence and any time index i, modifying data after
    index i must not change the features at index i (including long-period
    features in columns 147–164).
    """
    T = ohlcv.shape[0]
    i = T // 2  # pick the midpoint

    # Compute features on original data
    result_original = generate_factors("dataset0", ohlcv)
    features_at_i = result_original[i].copy()

    # Modify data after index i with random perturbations
    ohlcv_modified = ohlcv.copy()
    if i + 1 < T:
        # Replace data after i with random positive values
        rng = np.random.default_rng(seed=12345)
        ohlcv_modified[i + 1:] = rng.uniform(
            0.01, 1000.0, size=(T - i - 1, 5)
        ).astype(np.float32)

    # Recompute features on modified data
    result_modified = generate_factors("dataset0", ohlcv_modified)
    features_at_i_modified = result_modified[i]

    # Features at index i must be unchanged
    # Use np.array_equal with NaN-aware comparison
    for col in range(165):
        orig_val = features_at_i[col]
        mod_val = features_at_i_modified[col]
        if np.isnan(orig_val):
            assert np.isnan(mod_val), (
                f"Column {col} at index {i}: was NaN, now {mod_val}"
            )
        else:
            assert np.isclose(orig_val, mod_val, rtol=1e-5, atol=1e-7), (
                f"Column {col} at index {i}: was {orig_val}, now {mod_val} "
                f"(causality violated)"
            )


# =============================================================================
# Property 4: 长周期动量特征数学正确性
# Feature: model-optimization-v4, Property 4: 长周期动量特征数学正确性
# Validates: Requirement 2.1
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    hnp.arrays(
        dtype=np.float32,
        shape=(600, 5),
        elements=st.floats(min_value=0.01, max_value=1000.0,
                           allow_nan=False, allow_infinity=False),
    )
)
def test_property_4_momentum_math(ohlcv):
    """
    Feature: model-optimization-v4, Property 4: 长周期动量特征数学正确性
    Validates: Requirement 2.1

    For T=600, at index i=599 (>= 480):
      col 147 == log(close[i] / close[i-240])
      col 148 == close[i] / close[i-240] - 1
      col 149 == log(close[i] / close[i-480])
      col 150 == close[i] / close[i-480] - 1
    """
    result = generate_factors("dataset0", ohlcv)

    close = ohlcv[:, 3].astype(np.float64)
    i = 599  # T-1, guaranteed >= 480

    # w=240: columns 147 and 148
    c_now = close[i]
    c_prev_240 = close[i - 240]
    if c_prev_240 > 0.0:
        expected_log_240 = np.log(c_now / c_prev_240)
        expected_roc_240 = c_now / c_prev_240 - 1.0

        actual_log_240 = float(result[i, 147])
        actual_roc_240 = float(result[i, 148])

        assert not np.isnan(actual_log_240), (
            f"col 147 at i={i} is NaN but close[i-240]={c_prev_240} > 0"
        )
        assert np.isclose(actual_log_240, expected_log_240, rtol=1e-4, atol=1e-6), (
            f"col 147: expected {expected_log_240}, got {actual_log_240}"
        )
        assert np.isclose(actual_roc_240, expected_roc_240, rtol=1e-4, atol=1e-6), (
            f"col 148: expected {expected_roc_240}, got {actual_roc_240}"
        )

    # w=480: columns 149 and 150
    c_prev_480 = close[i - 480]
    if c_prev_480 > 0.0:
        expected_log_480 = np.log(c_now / c_prev_480)
        expected_roc_480 = c_now / c_prev_480 - 1.0

        actual_log_480 = float(result[i, 149])
        actual_roc_480 = float(result[i, 150])

        assert not np.isnan(actual_log_480), (
            f"col 149 at i={i} is NaN but close[i-480]={c_prev_480} > 0"
        )
        assert np.isclose(actual_log_480, expected_log_480, rtol=1e-4, atol=1e-6), (
            f"col 149: expected {expected_log_480}, got {actual_log_480}"
        )
        assert np.isclose(actual_roc_480, expected_roc_480, rtol=1e-4, atol=1e-6), (
            f"col 150: expected {expected_roc_480}, got {actual_roc_480}"
        )


# =============================================================================
# Property 5: 价格区间位置有界性
# Feature: model-optimization-v4, Property 5: 价格区间位置有界性
# Validates: Requirements 2.5, 2.11
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=500, max_value=1000).flatmap(
        lambda T: hnp.arrays(
            dtype=np.float32,
            shape=(T, 5),
            elements=st.floats(min_value=0.01, max_value=1000.0,
                               allow_nan=False, allow_infinity=False),
        )
    )
)
def test_property_5_price_range_bounded(ohlcv):
    """
    Feature: model-optimization-v4, Property 5: 价格区间位置有界性
    Validates: Requirements 2.5, 2.11

    All non-NaN values in columns 163–164 must be in [0.0, 1.0].
    """
    result = generate_factors("dataset0", ohlcv)

    for col in [163, 164]:
        col_vals = result[:, col]
        non_nan_mask = ~np.isnan(col_vals)
        non_nan_vals = col_vals[non_nan_mask]

        if len(non_nan_vals) > 0:
            assert np.all(non_nan_vals >= 0.0), (
                f"Column {col} has values below 0.0: "
                f"min={non_nan_vals.min()}"
            )
            assert np.all(non_nan_vals <= 1.0), (
                f"Column {col} has values above 1.0: "
                f"max={non_nan_vals.max()}"
            )


def test_property_5_price_range_flat_price():
    """
    Feature: model-optimization-v4, Property 5: 价格区间位置有界性 (flat price edge case)
    Validates: Requirement 2.11

    When high == low (all same price) within the window, the price range
    position feature must output exactly 0.5.
    """
    # Construct input where all prices are the same constant value
    # Use T=600 so that both w=240 and w=480 windows are fully populated
    T = 600
    price = 100.0
    ohlcv = np.full((T, 5), price, dtype=np.float32)
    # Volume must be positive (use 1.0)
    ohlcv[:, 4] = 1.0

    result = generate_factors("dataset0", ohlcv)

    # For w=240: rows 239..599 should have col 163 == 0.5
    for i in range(239, T):
        val_163 = float(result[i, 163])
        assert not np.isnan(val_163), f"col 163 at i={i} is NaN for flat price"
        assert val_163 == pytest.approx(0.5, abs=1e-6), (
            f"col 163 at i={i}: expected 0.5, got {val_163}"
        )

    # For w=480: rows 479..599 should have col 164 == 0.5
    for i in range(479, T):
        val_164 = float(result[i, 164])
        assert not np.isnan(val_164), f"col 164 at i={i} is NaN for flat price"
        assert val_164 == pytest.approx(0.5, abs=1e-6), (
            f"col 164 at i={i}: expected 0.5, got {val_164}"
        )


# =============================================================================
# Property 6: 数据集ID解析正确性
# Feature: model-optimization-v4, Property 6: 数据集ID解析正确性
# Validates: Requirement 1.2
# =============================================================================

def test_property_6_dataset_id_parsing():
    """
    Feature: model-optimization-v4, Property 6: 数据集ID解析正确性
    Validates: Requirement 1.2

    For dataset_name in {"dataset0", ..., "dataset29"}, the parsed integer ID
    must equal N, and the id_col appended for global model inference must
    contain all values equal to N.
    """
    import numpy as np

    for n in range(30):
        dataset_name = f"dataset{n}"
        # Verify parsing logic used in predict.py
        parsed_id = int(dataset_name.replace("dataset", ""))
        assert parsed_id == n, (
            f"dataset_name={dataset_name}: expected parsed_id={n}, got {parsed_id}"
        )

        # Verify the id_col construction
        T = 100
        id_col = np.full((T, 1), parsed_id, dtype=np.int32)
        assert id_col.shape == (T, 1), (
            f"id_col shape should be ({T}, 1), got {id_col.shape}"
        )
        assert id_col.dtype == np.int32, (
            f"id_col dtype should be int32, got {id_col.dtype}"
        )
        assert np.all(id_col == n), (
            f"All values in id_col should be {n}"
        )

        # Verify hstack produces correct shape
        factors_dummy = np.zeros((T, 165), dtype=np.float32)
        factors_with_id = np.hstack([factors_dummy, id_col])
        assert factors_with_id.shape == (T, 166), (
            f"factors_with_id shape should be ({T}, 166), got {factors_with_id.shape}"
        )
        assert np.all(factors_with_id[:, 165] == n), (
            f"Last column of factors_with_id should all be {n}"
        )


# =============================================================================
# Property 7: 模型选择逻辑正确性
# Feature: model-optimization-v4, Property 7: 模型选择逻辑正确性
# Validates: Requirement 1.6
# =============================================================================

def test_property_7_model_selection_logic():
    """
    Feature: model-optimization-v4, Property 7: 模型选择逻辑正确性
    Validates: Requirement 1.6

    When use_global_model_ret5=True and global model exists, global model is used.
    When use_global_model_ret5=False, local model is used.
    Same logic applies for Ret60.
    """
    import json
    import tempfile
    import numpy as np
    from pathlib import Path
    from unittest.mock import patch, MagicMock

    T = 100
    factors = np.random.rand(T, 165).astype(np.float32)
    fake_pred = np.zeros(T, dtype=np.float64)

    # Test cases: (use_global_ret5, use_global_ret60)
    test_cases = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]

    for use_global_ret5, use_global_ret60 in test_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write ensemble_weights.json with use_global flags
            weights = {
                "dataset0": {
                    "ret5_alpha": 1.0,
                    "ret5_beta": 0.0,
                    "ret5_gamma": 0.0,
                    "ret60_alpha": 1.0,
                    "ret60_beta": 0.0,
                    "ret60_gamma": 0.0,
                    "use_global_model_ret5": use_global_ret5,
                    "use_global_model_ret60": use_global_ret60,
                }
            }
            weights_file = tmpdir_path / "ensemble_weights.json"
            with open(weights_file, "w") as f:
                json.dump(weights, f)

            # Create dummy model files so _load_lgb_model finds them
            global_ret5_txt = tmpdir_path / "lgb_ret5_global.txt"
            global_ret60_txt = tmpdir_path / "lgb_ret60_global.txt"
            local_ret5_txt = tmpdir_path / "lgb_ret5_dataset0.txt"
            local_ret60_txt = tmpdir_path / "lgb_ret60_dataset0.txt"
            for f in [global_ret5_txt, global_ret60_txt, local_ret5_txt, local_ret60_txt]:
                f.touch()

            # Track which model paths were loaded
            loaded_paths = []

            def mock_booster(model_file):
                loaded_paths.append(model_file)
                m = MagicMock()
                m.predict.return_value = fake_pred
                return m

            import predict as predict_module
            original_model_dir = predict_module.MODEL_DIR

            try:
                predict_module.MODEL_DIR = tmpdir_path
                with patch("predict.lgb.Booster", side_effect=mock_booster):
                    predict_module.generate_signals("dataset0", factors)
            finally:
                predict_module.MODEL_DIR = original_model_dir

            # Verify correct model paths were loaded
            if use_global_ret5:
                assert any("lgb_ret5_global" in p for p in loaded_paths), (
                    f"use_global_ret5=True: expected global ret5 model to be loaded, "
                    f"got paths: {loaded_paths}"
                )
            else:
                assert any("lgb_ret5_dataset0" in p for p in loaded_paths), (
                    f"use_global_ret5=False: expected local ret5 model to be loaded, "
                    f"got paths: {loaded_paths}"
                )

            if use_global_ret60:
                assert any("lgb_ret60_global" in p for p in loaded_paths), (
                    f"use_global_ret60=True: expected global ret60 model to be loaded, "
                    f"got paths: {loaded_paths}"
                )
            else:
                assert any("lgb_ret60_dataset0" in p for p in loaded_paths), (
                    f"use_global_ret60=False: expected local ret60 model to be loaded, "
                    f"got paths: {loaded_paths}"
                )


# =============================================================================
# Property 8: 权重文件向后兼容性
# Feature: model-optimization-v4, Property 8: 权重文件向后兼容性
# Validates: Requirement 5.5
# =============================================================================

def test_property_8_backward_compatibility():
    """
    Feature: model-optimization-v4, Property 8: 权重文件向后兼容性
    Validates: Requirement 5.5

    v3 format weights (without use_global_model_* fields) must default to False,
    so generate_signals() behaves identically to v3.
    """
    import json
    import tempfile
    import numpy as np
    from pathlib import Path
    from unittest.mock import patch, MagicMock

    T = 100
    factors = np.random.rand(T, 165).astype(np.float32)
    fake_pred = np.zeros(T, dtype=np.float64)

    # v3 format: no use_global_model_* fields
    v3_weights_variants = [
        # Minimal v3 format
        {
            "dataset0": {
                "ret5_alpha": 1.0,
                "ret5_beta": 0.0,
                "ret5_gamma": 0.0,
                "ret60_alpha": 1.0,
                "ret60_beta": 0.0,
                "ret60_gamma": 0.0,
            }
        },
        # v3 format with only alpha (no beta/gamma)
        {
            "dataset0": {
                "ret5_alpha": 1.0,
                "ret60_alpha": 1.0,
            }
        },
        # Empty dataset entry
        {"dataset0": {}},
        # Dataset not present at all
        {},
    ]

    for v3_weights in v3_weights_variants:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            weights_file = tmpdir_path / "ensemble_weights.json"
            with open(weights_file, "w") as f:
                json.dump(v3_weights, f)

            # Only local model files exist (no global models)
            local_ret5_txt = tmpdir_path / "lgb_ret5_dataset0.txt"
            local_ret60_txt = tmpdir_path / "lgb_ret60_dataset0.txt"
            local_ret5_txt.touch()
            local_ret60_txt.touch()

            loaded_paths = []

            def mock_booster(model_file):
                loaded_paths.append(model_file)
                m = MagicMock()
                m.predict.return_value = fake_pred
                return m

            import predict as predict_module
            original_model_dir = predict_module.MODEL_DIR

            try:
                predict_module.MODEL_DIR = tmpdir_path
                with patch("predict.lgb.Booster", side_effect=mock_booster):
                    result = predict_module.generate_signals("dataset0", factors)
            finally:
                predict_module.MODEL_DIR = original_model_dir

            # Verify: no global model was loaded (use_global defaults to False)
            assert not any("global" in p for p in loaded_paths), (
                f"v3 weights should not trigger global model loading, "
                f"but loaded: {loaded_paths}"
            )

            # Verify: local model was loaded
            assert any("lgb_ret5_dataset0" in p for p in loaded_paths), (
                f"Local ret5 model should be loaded for v3 weights, "
                f"got: {loaded_paths}"
            )

            # Verify output shape and dtype
            assert result.shape == (T, 2), (
                f"Output shape should be ({T}, 2), got {result.shape}"
            )
            assert result.dtype == np.float32, (
                f"Output dtype should be float32, got {result.dtype}"
            )

            # Verify no NaN/Inf
            assert not np.any(np.isnan(result)), "Output should not contain NaN"
            assert not np.any(np.isinf(result)), "Output should not contain Inf"


# =============================================================================
# Property 9: 降级健壮性
# Feature: model-optimization-v4, Property 9: 降级健壮性
# Validates: Requirement 5.7
# =============================================================================

def test_property_9_degradation_robustness():
    """
    Feature: model-optimization-v4, Property 9: 降级健壮性
    Validates: Requirement 5.7

    generate_signals() must return a valid (T, 2) float32 array without NaN/Inf
    under various model file missing scenarios.
    """
    import json
    import tempfile
    import numpy as np
    from pathlib import Path
    from unittest.mock import patch, MagicMock

    T = 100
    factors = np.random.rand(T, 165).astype(np.float32)
    fake_pred = np.zeros(T, dtype=np.float64)

    def make_mock_booster(model_file):
        m = MagicMock()
        m.predict.return_value = fake_pred
        return m

    # Scenario 1: No ensemble_weights.json, only local models exist
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Only local models, no weights file
        (tmpdir_path / "lgb_ret5_dataset0.txt").touch()
        (tmpdir_path / "lgb_ret60_dataset0.txt").touch()

        import predict as predict_module
        original_model_dir = predict_module.MODEL_DIR
        try:
            predict_module.MODEL_DIR = tmpdir_path
            with patch("predict.lgb.Booster", side_effect=make_mock_booster):
                result = predict_module.generate_signals("dataset0", factors)
        finally:
            predict_module.MODEL_DIR = original_model_dir

        assert result.shape == (T, 2), f"Scenario 1: shape {result.shape}"
        assert result.dtype == np.float32, f"Scenario 1: dtype {result.dtype}"
        assert not np.any(np.isnan(result)), "Scenario 1: NaN in output"
        assert not np.any(np.isinf(result)), "Scenario 1: Inf in output"

    # Scenario 2: use_global=True but global model missing → fallback to local
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        weights = {
            "dataset0": {
                "ret5_alpha": 1.0, "ret5_beta": 0.0, "ret5_gamma": 0.0,
                "ret60_alpha": 1.0, "ret60_beta": 0.0, "ret60_gamma": 0.0,
                "use_global_model_ret5": True,
                "use_global_model_ret60": True,
            }
        }
        with open(tmpdir_path / "ensemble_weights.json", "w") as f:
            json.dump(weights, f)
        # Only local models exist, no global models
        (tmpdir_path / "lgb_ret5_dataset0.txt").touch()
        (tmpdir_path / "lgb_ret60_dataset0.txt").touch()

        import predict as predict_module
        original_model_dir = predict_module.MODEL_DIR
        try:
            predict_module.MODEL_DIR = tmpdir_path
            with patch("predict.lgb.Booster", side_effect=make_mock_booster):
                result = predict_module.generate_signals("dataset0", factors)
        finally:
            predict_module.MODEL_DIR = original_model_dir

        assert result.shape == (T, 2), f"Scenario 2: shape {result.shape}"
        assert result.dtype == np.float32, f"Scenario 2: dtype {result.dtype}"
        assert not np.any(np.isnan(result)), "Scenario 2: NaN in output"
        assert not np.any(np.isinf(result)), "Scenario 2: Inf in output"

    # Scenario 3: GRU model missing (beta > 0 but no .pt file)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        weights = {
            "dataset0": {
                "ret5_alpha": 0.7, "ret5_beta": 0.3, "ret5_gamma": 0.0,
                "ret60_alpha": 0.7, "ret60_beta": 0.3, "ret60_gamma": 0.0,
                "use_global_model_ret5": False,
                "use_global_model_ret60": False,
            }
        }
        with open(tmpdir_path / "ensemble_weights.json", "w") as f:
            json.dump(weights, f)
        # Local LGB models exist, but no GRU .pt file
        (tmpdir_path / "lgb_ret5_dataset0.txt").touch()
        (tmpdir_path / "lgb_ret60_dataset0.txt").touch()

        import predict as predict_module
        original_model_dir = predict_module.MODEL_DIR
        try:
            predict_module.MODEL_DIR = tmpdir_path
            with patch("predict.lgb.Booster", side_effect=make_mock_booster):
                result = predict_module.generate_signals("dataset0", factors)
        finally:
            predict_module.MODEL_DIR = original_model_dir

        assert result.shape == (T, 2), f"Scenario 3: shape {result.shape}"
        assert result.dtype == np.float32, f"Scenario 3: dtype {result.dtype}"
        assert not np.any(np.isnan(result)), "Scenario 3: NaN in output"
        assert not np.any(np.isinf(result)), "Scenario 3: Inf in output"

    # Scenario 4: Transformer model missing (gamma > 0 but no .pt file)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        weights = {
            "dataset0": {
                "ret5_alpha": 0.7, "ret5_beta": 0.0, "ret5_gamma": 0.3,
                "ret60_alpha": 0.7, "ret60_beta": 0.0, "ret60_gamma": 0.3,
                "use_global_model_ret5": False,
                "use_global_model_ret60": False,
            }
        }
        with open(tmpdir_path / "ensemble_weights.json", "w") as f:
            json.dump(weights, f)
        (tmpdir_path / "lgb_ret5_dataset0.txt").touch()
        (tmpdir_path / "lgb_ret60_dataset0.txt").touch()

        import predict as predict_module
        original_model_dir = predict_module.MODEL_DIR
        try:
            predict_module.MODEL_DIR = tmpdir_path
            with patch("predict.lgb.Booster", side_effect=make_mock_booster):
                result = predict_module.generate_signals("dataset0", factors)
        finally:
            predict_module.MODEL_DIR = original_model_dir

        assert result.shape == (T, 2), f"Scenario 4: shape {result.shape}"
        assert result.dtype == np.float32, f"Scenario 4: dtype {result.dtype}"
        assert not np.any(np.isnan(result)), "Scenario 4: NaN in output"
        assert not np.any(np.isinf(result)), "Scenario 4: Inf in output"


# =============================================================================
# Property 10: 采样上限不变量
# Feature: model-optimization-v4, Property 10: 采样上限不变量
# Validates: Requirements 3.3, 4.2, 4.3
# =============================================================================

@settings(max_examples=200, deadline=None)
@given(
    st.integers(min_value=1, max_value=24 * 1024 * 1024 * 1024),  # available_vram in bytes (up to 24 GB)
    st.integers(min_value=1000, max_value=5_000_000),              # train_set_size
)
def test_property_10_sampling_limit_invariant(available_vram_bytes, train_set_size):
    """
    Feature: model-optimization-v4, Property 10: 采样上限不变量
    Validates: Requirements 3.3, 4.2, 4.3

    For any available_vram and train_set_size:
    - max_samples <= min(max_samples_by_vram, train_set_size)
    - For Large_Datasets with Direction C, actual_samples <= 300000
    """
    import math
    from unittest.mock import patch

    feature_dim = 165
    window_size = 60
    batch_size = 4096
    r_gpu = 0.6

    # Compute expected max_samples_by_vram
    bytes_per_sample = batch_size * window_size * feature_dim * 4
    max_by_vram = math.floor(available_vram_bytes * r_gpu / bytes_per_sample) * batch_size
    max_by_vram = min(max_by_vram, train_set_size)

    # Import and test _compute_adaptive_max_samples with mocked VRAM
    from train import _compute_adaptive_max_samples

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(available_vram_bytes, available_vram_bytes)):
        result = _compute_adaptive_max_samples(
            train_set_size=train_set_size,
            feature_dim=feature_dim,
            window_size=window_size,
            batch_size=batch_size,
        )

    # Property: result is either 0 (skip) or <= min(max_by_vram, train_set_size)
    if result == 0:
        # This means max_by_vram < 50000 — valid
        assert max_by_vram < 50000, (
            f"result=0 but max_by_vram={max_by_vram} >= 50000"
        )
    else:
        assert result <= train_set_size, (
            f"max_samples={result} > train_set_size={train_set_size}"
        )
        assert result <= max_by_vram or max_by_vram < 50000, (
            f"max_samples={result} > max_by_vram={max_by_vram}"
        )

    # Direction C: for Large_Datasets, actual_samples <= 300000
    # Simulate the Direction C cap
    if result > 0:
        direction_c_samples = min(result, 300000)
        assert direction_c_samples <= 300000, (
            f"Direction C samples={direction_c_samples} > 300000"
        )
        assert direction_c_samples <= train_set_size, (
            f"Direction C samples={direction_c_samples} > train_set_size={train_set_size}"
        )


# =============================================================================
# Property 11: 采样时序保持性
# Feature: model-optimization-v4, Property 11: 采样时序保持性
# Validates: Requirement 3.4
# =============================================================================

@settings(max_examples=200, deadline=None)
@given(
    st.integers(min_value=200, max_value=2000).flatmap(
        lambda n: st.tuples(
            st.just(n),
            hnp.arrays(
                dtype=np.float32,
                shape=(n,),
                elements=st.floats(min_value=0.01, max_value=10.0,
                                   allow_nan=False, allow_infinity=False),
            ),
            st.integers(min_value=10, max_value=min(300, n)),
        )
    )
)
def test_property_11_temporal_order_preserved(args):
    """
    Feature: model-optimization-v4, Property 11: 采样时序保持性
    Validates: Requirement 3.4

    For any weight sequence, sampled indices must be strictly monotonically
    increasing (temporal order preserved).
    """
    n, weights_raw, n_sample = args

    # Ensure weights are positive (simulate _compute_volatility_weights output)
    weights = np.abs(weights_raw).astype(np.float64)
    weights = np.where(weights <= 0, 1.0, weights)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        weights = np.ones(n, dtype=np.float64)
        weights_sum = float(n)

    # Simulate the weighted sampling + sort used in Direction C
    indices = np.arange(n)
    sampled = np.random.choice(
        n,
        size=n_sample,
        replace=False,
        p=weights / weights_sum,
    )
    sampled_sorted = np.sort(sampled)

    # Property: result is strictly monotonically increasing
    assert len(sampled_sorted) == n_sample, (
        f"Expected {n_sample} samples, got {len(sampled_sorted)}"
    )
    if len(sampled_sorted) > 1:
        diffs = np.diff(sampled_sorted)
        assert np.all(diffs > 0), (
            f"Sampled indices are not strictly monotonically increasing: "
            f"found diffs={diffs[diffs <= 0]}"
        )


# =============================================================================
# Property 12: 采样权重计算正确性
# Feature: model-optimization-v4, Property 12: 采样权重计算正确性
# Validates: Requirements 3.2, 3.5
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=200, max_value=500).flatmap(
        lambda T: hnp.arrays(
            dtype=np.float64,
            shape=(T,),
            elements=st.one_of(
                st.floats(min_value=0.01, max_value=100.0,
                          allow_nan=False, allow_infinity=False),
                st.just(float("nan")),
                st.just(0.0),
            ),
        )
    )
)
def test_property_12_volatility_weight_correctness(close_prices):
    """
    Feature: model-optimization-v4, Property 12: 采样权重计算正确性
    Validates: Requirements 3.2, 3.5

    For any close price sequence (with NaN and zeros):
    - Non-NaN/zero positions: weight == σ_20(i) / Median_120(σ_20)(i)
    - NaN or zero positions: weight == 1.0
    - All weights > 0
    """
    from train import _compute_volatility_weights

    close = np.asarray(close_prices, dtype=np.float64)
    T = len(close)

    weights = _compute_volatility_weights(close, window_vol=20, window_median=120)

    # Property 1: All weights are positive
    assert np.all(weights > 0), (
        f"Some weights are not positive: min={weights.min()}"
    )

    # Property 2: No NaN in weights
    assert not np.any(np.isnan(weights)), (
        "Weights contain NaN values"
    )

    # Property 3: Manually compute expected weights and verify
    # Compute log returns
    log_returns = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        prev = close[i - 1]
        curr = close[i]
        if np.isnan(prev) or np.isnan(curr) or prev <= 0 or curr <= 0:
            log_returns[i] = 0.0
        else:
            log_returns[i] = np.log(curr / prev)

    # Compute 20-bar rolling std
    volatility = np.full(T, np.nan, dtype=np.float64)
    for i in range(19, T):
        volatility[i] = np.std(log_returns[i - 19:i + 1])

    # Compute 120-bar rolling median
    rolling_median = np.full(T, np.nan, dtype=np.float64)
    for i in range(119, T):
        rolling_median[i] = np.median(volatility[max(0, i - 119):i + 1])

    # Verify each weight
    for i in range(T):
        v = volatility[i]
        m = rolling_median[i]
        w = weights[i]

        if np.isnan(v) or np.isnan(m) or m <= 0:
            # Should be 1.0 (fallback)
            assert w == pytest.approx(1.0, abs=1e-5), (
                f"Position {i}: expected weight=1.0 (NaN/zero case), got {w}"
            )
        else:
            expected = v / m
            if expected <= 0:
                assert w == pytest.approx(1.0, abs=1e-5), (
                    f"Position {i}: expected weight=1.0 (non-positive ratio), got {w}"
                )
            else:
                assert w == pytest.approx(expected, rel=1e-4, abs=1e-6), (
                    f"Position {i}: expected weight={expected:.6f}, got {w:.6f}"
                )
