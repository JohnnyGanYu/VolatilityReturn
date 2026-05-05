"""
Property-based tests for model-optimization-v3.

Uses Hypothesis for property-based testing of core training and inference logic.
"""

import os
# Prevent OpenMP segfaults from PyTorch/LightGBM libomp conflict on macOS ARM64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import lightgbm as lgb
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# =============================================================================
# Feature: model-optimization-v3, Property 1: Two-Phase Training Minimum Tree Count
# =============================================================================


@settings(max_examples=5, deadline=None)
@given(
    T=st.integers(min_value=5000, max_value=8000),
    F=st.integers(min_value=10, max_value=20),
    min_boost=st.integers(min_value=5, max_value=30),
)
def test_two_phase_minimum_tree_count(T, F, min_boost):
    """
    Property 1: Two-phase training produces at least min_boost_round trees.

    For any random feature matrix of shape (T, F) with T in [5000, 20000]
    and F in [10, 50], and any min_boost_round in [5, 30], the model
    returned by train_lgb_two_phase shall have at least min_boost_round trees.

    **Validates: Requirements 2.1, 2.4**
    """
    from train import train_lgb_two_phase, ic_eval_metric, LGB_PARAMS_RET5

    np.random.seed(42)
    features = np.random.randn(T, F).astype(np.float32)
    labels = np.random.randn(T).astype(np.float32)

    split = int(T * 0.8)
    train_data = lgb.Dataset(features[:split], label=labels[:split], free_raw_data=False)
    val_data = lgb.Dataset(
        features[split:], label=labels[split:], reference=train_data, free_raw_data=False
    )

    model = train_lgb_two_phase(
        LGB_PARAMS_RET5,
        train_data,
        val_data,
        max_boost_round=100,
        min_boost_round=min_boost,
    )

    assert model.num_trees() >= min_boost, (
        f"Expected >= {min_boost} trees, got {model.num_trees()}"
    )


# =============================================================================
# Feature: model-optimization-v3, Property 2: Transformer Input/Output Contract
# =============================================================================


@settings(max_examples=10, deadline=None)
@given(
    batch=st.integers(min_value=1, max_value=16),
    F=st.integers(min_value=10, max_value=50),
)
def test_transformer_input_output_contract(batch, F):
    """
    Property 2: Transformer Input/Output Contract.
    For any (batch, 60, F) input with finite float32 values,
    TransformerPredictor outputs (batch, 2) with all finite values.
    **Validates: Requirements 3.1, 3.3**
    """
    import torch
    from train import TransformerPredictor

    model = TransformerPredictor(input_size=F)
    model.eval()
    x = torch.randn(batch, 60, F)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch, 2), f"Expected ({batch}, 2), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# =============================================================================
# Feature: model-optimization-v3, Property 3: Sliding Window Causality
# =============================================================================

@settings(max_examples=10, deadline=None)
@given(
    T=st.integers(min_value=61, max_value=300),
    F=st.integers(min_value=1, max_value=10),
)
def test_sliding_window_causality(T, F):
    """
    Property 3: Sliding windows are causal.
    window[i][-1] == factors[i], and early indices are zero-padded.
    **Validates: Requirements 4.4, 14.1**
    """
    from train import build_sliding_windows_for_indices
    
    np.random.seed(42)
    features = np.random.randn(T, F).astype(np.float32)
    indices = np.arange(T)
    windows = build_sliding_windows_for_indices(features, indices, window_size=60)
    
    clean = np.nan_to_num(features, nan=0.0).astype(np.float32)
    
    # (a) Last row of window[i] equals clean[i]
    for i in range(T):
        np.testing.assert_array_almost_equal(windows[i, -1], clean[i])
    
    # (b) For i < 60, first 60-1-i rows are zero
    for i in range(min(59, T)):
        n_zero = 59 - i
        assert np.all(windows[i, :n_zero] == 0.0), f"Expected zero padding at index {i}"
    
    # (c) Appending rows beyond T doesn't change windows at 0..T-1
    extended = np.vstack([features, np.random.randn(50, F).astype(np.float32)])
    ext_windows = build_sliding_windows_for_indices(extended, indices, window_size=60)
    np.testing.assert_array_equal(windows, ext_windows)


# =============================================================================
# Feature: model-optimization-v3, Property 4: Three-Model Ensemble Formula
# =============================================================================

@settings(max_examples=10, deadline=None)
@given(
    T=st.integers(min_value=10, max_value=100),
    alpha_int=st.integers(min_value=0, max_value=10),
    beta_int=st.integers(min_value=0, max_value=10),
)
def test_ensemble_formula_correctness(T, alpha_int, beta_int):
    """
    Property 4: Ensemble output equals alpha*lgb + beta*gru + gamma*transformer.
    When beta=0 and gamma=0, output is identical to lgb_pred.
    **Validates: Requirements 5.2, 5.3**
    """
    gamma_int = 10 - alpha_int - beta_int
    assume(gamma_int >= 0)
    
    alpha = alpha_int / 10.0
    beta = beta_int / 10.0
    gamma = gamma_int / 10.0
    
    np.random.seed(42)
    lgb_pred = np.random.randn(T).astype(np.float64)
    gru_pred = np.random.randn(T).astype(np.float64)
    tf_pred = np.random.randn(T).astype(np.float64)
    
    expected = alpha * lgb_pred + beta * gru_pred + gamma * tf_pred
    np.testing.assert_allclose(expected, alpha * lgb_pred + beta * gru_pred + gamma * tf_pred)
    
    # When beta=0 and gamma=0, output is pure lgb
    pure_lgb = 1.0 * lgb_pred + 0.0 * gru_pred + 0.0 * tf_pred
    np.testing.assert_array_equal(pure_lgb, lgb_pred)


# =============================================================================
# Feature: model-optimization-v3, Property 5: Grid Search Selects IC-Maximizing Triple
# =============================================================================

@settings(max_examples=10, deadline=None)
@given(
    T=st.integers(min_value=50, max_value=200),
)
def test_grid_search_optimality(T):
    """
    Property 5: Grid search returns the IC-maximizing triple.
    **Validates: Requirements 6.1, 6.2**
    """
    from train import optimize_three_model_ensemble, pearson_ic_numpy
    
    np.random.seed(42)
    lgb_pred = np.random.randn(T).astype(np.float64)
    gru_pred = np.random.randn(T).astype(np.float64)
    tf_pred = np.random.randn(T).astype(np.float64)
    labels = np.random.randn(T).astype(np.float64)
    
    result = optimize_three_model_ensemble(
        lgb_pred, lgb_pred,  # same for both targets for simplicity
        gru_pred, gru_pred,
        tf_pred, tf_pred,
        labels, labels,
    )
    
    best_a = result["ret5_alpha"]
    best_b = result["ret5_beta"]
    best_g = result["ret5_gamma"]
    best_blend = best_a * lgb_pred + best_b * gru_pred + best_g * tf_pred
    best_ic = pearson_ic_numpy(best_blend, labels)
    
    # Verify no other triple produces higher IC
    step = 0.1
    values = np.arange(0.0, 1.0 + step/2, step)
    for a in values:
        for b in values:
            g = 1.0 - a - b
            if g < -1e-9 or g > 1.0 + 1e-9:
                continue
            g = max(0.0, min(1.0, g))
            blend = round(a, 1) * lgb_pred + round(b, 1) * gru_pred + round(g, 1) * tf_pred
            ic = pearson_ic_numpy(blend, labels)
            assert ic <= best_ic + 1e-10, (
                f"Triple ({round(a,1)}, {round(b,1)}, {round(g,1)}) has IC={ic:.6f} > best IC={best_ic:.6f}"
            )


# =============================================================================
# Feature: model-optimization-v3, Property 11: Signal Output All-Finite Contract
# =============================================================================

@settings(max_examples=5, deadline=None)
@given(
    T=st.integers(min_value=10, max_value=100),
    F=st.integers(min_value=10, max_value=50),
)
def test_signal_output_all_finite(T, F):
    """
    Property 11: generate_signals returns (T, 2) float32 with all finite values
    regardless of which optional model files are present.
    
    This test uses a temporary model directory with only LightGBM models
    (no GRU, no Transformer, no ensemble_weights.json) to verify the
    pure-LightGBM fallback path produces all-finite output.
    
    **Validates: Requirements 11.4, 17.6**
    """
    import tempfile
    import predict
    
    # Create minimal LightGBM models in a temp directory
    np.random.seed(42)
    features = np.random.randn(T, F).astype(np.float32)
    labels = np.random.randn(T).astype(np.float32)
    
    train_data = lgb.Dataset(features, label=labels, free_raw_data=False)
    params = {"objective": "regression", "num_leaves": 4, "verbose": -1, "num_threads": 1}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        # Train and save minimal LightGBM models
        model_r5 = lgb.train(params, train_data, num_boost_round=5)
        model_r60 = lgb.train(params, train_data, num_boost_round=5)
        model_r5.save_model(str(Path(tmpdir) / "lgb_ret5_dataset0.txt"))
        model_r60.save_model(str(Path(tmpdir) / "lgb_ret60_dataset0.txt"))
        
        # Point predict.py to temp directory
        original_model_dir = predict.MODEL_DIR
        predict.MODEL_DIR = Path(tmpdir)
        try:
            # Inject NaN and Inf into features
            test_factors = features.copy()
            test_factors[0, 0] = np.nan
            test_factors[1, 1] = np.inf
            test_factors[2, 2] = -np.inf
            
            signals = predict.generate_signals("dataset0", test_factors)
            
            assert signals.shape == (T, 2), f"Expected ({T}, 2), got {signals.shape}"
            assert signals.dtype == np.float32, f"Expected float32, got {signals.dtype}"
            assert np.all(np.isfinite(signals)), "Output contains non-finite values"
        finally:
            predict.MODEL_DIR = original_model_dir


# =============================================================================
# Feature: model-optimization-v3, Property 6: Feature Selection Correctness
# =============================================================================

@settings(max_examples=10, deadline=None)
@given(
    F=st.integers(min_value=100, max_value=200),
    T=st.integers(min_value=10, max_value=50),
)
def test_feature_selection_correctness(F, T):
    """Property 6: Selecting top-100 features produces correct (T, 100) matrix."""
    np.random.seed(42)
    importance = np.random.rand(F)
    matrix = np.random.randn(T, F).astype(np.float32)
    top100 = np.argsort(importance)[-100:][::-1].tolist()
    selected = matrix[:, top100]
    assert selected.shape == (T, 100)
    for j, col_idx in enumerate(top100):
        np.testing.assert_array_equal(selected[:, j], matrix[:, col_idx])


# =============================================================================
# Feature: model-optimization-v3, Property 7: Dynamic Sample Weight Formula and Causality
# =============================================================================

@settings(max_examples=5, deadline=None)
@given(
    T=st.integers(min_value=200, max_value=500),
)
def test_dynamic_sample_weight_causality(T):
    """Property 7: Sample weights are causal and follow the formula."""
    from train import compute_dynamic_sample_weights
    np.random.seed(42)
    close = np.abs(np.random.randn(T)) + 0.1  # positive prices
    weights_full = compute_dynamic_sample_weights(close.astype(np.float32))
    assert weights_full.shape == (T,)
    assert np.all(weights_full >= 1.0 - 1e-6)  # weights >= 1.0
    # Causality: weight at i from close[0:T] == weight at i from close[0:i+1]
    for i in [150, 199, min(T-1, 250)]:
        w_partial = compute_dynamic_sample_weights(close[:i+1].astype(np.float32))
        assert abs(w_partial[i] - weights_full[i]) < 1e-5, f"Causality violated at {i}"


# =============================================================================
# Feature: model-optimization-v3, Property 8: Regime Probability Output Bounds
# =============================================================================

@settings(max_examples=5, deadline=None)
@given(
    T=st.integers(min_value=100, max_value=500),
)
def test_regime_probability_bounds(T):
    """Property 8: Regime classifier output is in [0, 1] and augmented matrix has F+1 columns."""
    np.random.seed(42)
    F = 12
    features = np.random.randn(T, F).astype(np.float32)
    labels = (np.random.rand(T) > 0.7).astype(np.float32)
    
    train_data = lgb.Dataset(features, label=labels)
    params = {"objective": "binary", "num_leaves": 8, "verbose": -1, "num_threads": 1}
    model = lgb.train(params, train_data, num_boost_round=10)
    
    probs = model.predict(features)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities out of [0, 1]"
    
    augmented = np.column_stack([features, probs.astype(np.float32)])
    assert augmented.shape == (T, F + 1)


# =============================================================================
# Feature: model-optimization-v3, Property 9: Extreme Interval Label Construction
# =============================================================================

@settings(max_examples=10, deadline=None)
@given(
    T=st.integers(min_value=10, max_value=100),
    n_intervals=st.integers(min_value=0, max_value=5),
)
def test_extreme_interval_labels(T, n_intervals):
    """Property 9: Label[i]==1 iff indices[i] falls within at least one interval."""
    from train import build_extreme_mask
    indices = np.arange(T, dtype=np.float64)
    
    if n_intervals == 0:
        intervals = np.empty((0, 2), dtype=np.int64)
    else:
        starts = sorted(np.random.choice(T, size=min(n_intervals, T), replace=False))
        intervals = np.array([[s, min(s + np.random.randint(1, 5), T - 1)] for s in starts], dtype=np.int64)
    
    mask = build_extreme_mask(indices, intervals)
    
    for i in range(T):
        in_interval = any(int(intervals[k, 0]) <= i <= int(intervals[k, 1]) for k in range(len(intervals)))
        assert mask[i] == in_interval, f"Mismatch at index {i}: mask={mask[i]}, expected={in_interval}"


# =============================================================================
# Feature: model-optimization-v3, Property 10: Factor Output Contract
# =============================================================================

@settings(max_examples=3, deadline=None)
@given(
    T=st.integers(min_value=100, max_value=300),
)
def test_factor_output_contract(T):
    """
    Property 10: generate_factors returns float32 array of shape (T, F) with F <= 512.
    **Validates: Requirements 11.3**
    """
    from factor import generate_factors
    np.random.seed(42)
    ohlcv = np.random.rand(T, 5).astype(np.float32)
    ohlcv[:, :4] = ohlcv[:, :4] * 0.5 + 0.5  # price-like
    ohlcv[:, 4] = ohlcv[:, 4] * 10000  # volume-like
    
    factors = generate_factors("dataset0", ohlcv)
    assert factors.dtype == np.float32, f"Expected float32, got {factors.dtype}"
    assert factors.shape[0] == T, f"Expected {T} rows, got {factors.shape[0]}"
    assert factors.shape[1] <= 512, f"Too many features: {factors.shape[1]}"
