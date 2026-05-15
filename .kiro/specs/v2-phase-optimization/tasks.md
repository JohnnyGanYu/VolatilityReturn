# Implementation Plan: Model Optimization V2

## Overview

Incremental optimization of the volatility return prediction system across four phases: (1) LightGBM training fix, (2) feature expansion, (3) PyTorch GRU ensemble, and (4) packaging + final validation. Each phase is independently verifiable with a Git commit checkpoint. All changes preserve the public interfaces of `factor.py` and `predict.py` and comply with competition constraints (≤512 features, ≤200 MB submission, 2-hour runtime).

## Tasks

- [x] 1. Phase 1: LightGBM Training Fix (highest impact)
  - [x] 1.1 Update LightGBM hyperparameters and training configuration in `train.py`
    - Set `metric` to `"None"` (string) in both `LGB_PARAMS_RET5` and `LGB_PARAMS_RET60` to disable built-in MAE early stopping
    - Remove `deterministic=True` and `force_row_wise=True` from both param dicts
    - Add `num_threads=-1` to both param dicts for multi-threaded training
    - Update `LGB_PARAMS_RET5`: `num_leaves=127`, `learning_rate=0.03`, `min_child_samples=200`
    - Update `LGB_PARAMS_RET60`: `num_leaves=255`, `learning_rate=0.02`, `min_child_samples=200`
    - Update `NUM_BOOST_ROUND_RET5=1000`, `NUM_BOOST_ROUND_RET60=1500`
    - Update `EARLY_STOPPING_ROUNDS=100`
    - Retain `seed=42` in both param dicts for reproducibility
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_

  - [x] 1.2 Verify and update the Custom IC feval function in `train.py`
    - Ensure `ic_eval_metric` returns `("ic", ic_value, True)` with `True` indicating higher-is-better
    - Ensure NaN/Inf in predictions or labels are replaced with 0.0 before computation
    - Ensure zero-variance case returns IC = 0.0
    - Ensure `feval=ic_eval_metric` is passed to `lgb.train()` and is the sole early stopping criterion (no built-in metric competing)
    - Update `lgb.log_evaluation(period=50)` for visibility during training
    - Add logging of best iteration tree count after each model trains
    - _Requirements: 1.3, 1.4, 1.5, 1.6_

  - [ ]* 1.3 Write property test for IC feval correctness
    - **Property 1: IC Feval Correctness**
    - Use Hypothesis to generate random prediction/label arrays of length N ≥ 2
    - Verify return tuple format `("ic", ic_value, True)`
    - Verify ic_value matches manual Pearson correlation computation
    - Verify zero-variance arrays return IC = 0.0
    - **Validates: Requirement 1.3**

- [-] 2. Phase 1 Checkpoint: Retrain and evaluate LightGBM models
  - Retrain all 60 LightGBM models using `python train.py`
  - Run `python evaluate_local.py` to compute IC scores
  - Verify IC improvement over baseline (Normal×Ret5 > 0.1255, Normal×Ret60 > 0.2519, Extreme×Ret5 > 0.2673, Extreme×Ret60 > 0.4316)
  - Verify previously-1-tree datasets (e.g., dataset0, dataset11, dataset28) now have >1 tree
  - Verify total model file size < 200 MB
  - Git commit and push to feature branch (not main)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 1.5, 1.6, 2.5, 5.2, 12.6, 13.1, 13.2, 13.3, 13.4, 15.1, 15.2, 15.3_

- [ ] 3. Phase 2: Feature Expansion in `factor.py`
  - [ ] 3.1 Implement `_compute_ema_ratios(close)` in `factor.py`
    - Compute `close[i] / EMA(close, span)[i]` for spans [5, 10, 20, 60, 120]
    - Use EMA formula: `EMA[i] = alpha * close[i] + (1 - alpha) * EMA[i-1]`, `alpha = 2 / (span + 1)`
    - Output NaN when EMA is zero or NaN
    - Decorate with `@njit(cache=True)`
    - Return array of shape (T, 5)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 14.1, 14.4_

  - [ ]* 3.2 Write property test for EMA ratio correctness
    - **Property 7: EMA Ratio Computation Correctness**
    - Use Hypothesis to generate close price arrays of length T ≥ 2
    - Verify EMA ratio equals `close[i] / EMA[i]` for each span
    - Verify NaN when EMA is zero or NaN
    - **Validates: Requirements 4.1, 4.2, 4.3**

  - [ ] 3.3 Implement `_compute_rolling_skew_kurt(close)` in `factor.py`
    - Compute 1-bar log returns, then rolling skewness over windows [20, 60, 120] and rolling kurtosis over windows [20, 60, 120]
    - Skewness = `m3 / std^3`, Kurtosis = `m4 / std^4`
    - Output NaN when window has fewer than 3 valid returns
    - Decorate with `@njit(cache=True)`
    - Return array of shape (T, 6)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 14.1, 14.4_

  - [ ] 3.4 Implement `_compute_close_to_open_gaps(open_, close)` in `factor.py`
    - Compute raw gap: `log(open[i] / close[i-1])` for i ≥ 1
    - Output NaN when `close[i-1]` is zero, NaN, or negative
    - Compute rolling mean of gap over windows [5, 10, 20]
    - Compute rolling std of gap over windows [5, 10, 20]
    - Decorate with `@njit(cache=True)`
    - Return array of shape (T, 7)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 14.1, 14.4_

  - [ ] 3.5 Implement `_compute_volume_weighted_returns(close, volume)` in `factor.py`
    - Compute per-bar VWR: `log_return_1bar * (volume[i] / rolling_mean_volume(w)[i])` for windows [5, 10, 20, 60]
    - Compute rolling sums of VWR over windows [5, 10, 20, 60]
    - Output NaN when volume or log return is NaN
    - Decorate with `@njit(cache=True)`
    - Return array of shape (T, 8)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 14.1, 14.4_

  - [ ] 3.6 Implement `_compute_return_autocorrelation(close)` in `factor.py`
    - Compute autocorrelation of 1-bar log returns at lag 1 over windows [20, 60]
    - Compute autocorrelation of 1-bar log returns at lag 5 over windows [20, 60]
    - Use Pearson correlation between `returns[j]` and `returns[j-lag]` within window
    - Output NaN when window has fewer than `lag + 3` valid return pairs
    - Decorate with `@njit(cache=True)`
    - Return array of shape (T, 4)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 14.1, 14.4_

  - [ ] 3.7 Implement `_compute_realized_variance(close)` in `factor.py`
    - Compute sum of squared 1-bar log returns over windows [5, 10, 20, 60]
    - Compute log(RV) for each window
    - Output NaN when window has zero valid squared returns
    - Decorate with `@njit(cache=True)`
    - Return array of shape (T, 8)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 14.1, 14.4_

  - [ ]* 3.8 Write property test for realized variance correctness
    - **Property 8: Realized Variance Correctness**
    - Use Hypothesis to generate close price arrays of length T ≥ 61
    - Verify RV equals sum of squared log returns in window
    - Verify log-RV equals log(RV)
    - **Validates: Requirements 9.1, 9.2**

  - [ ] 3.9 Wire new features into `generate_factors()` in `factor.py`
    - Call all 6 new feature functions after existing baseline features
    - Append new features via `np.column_stack` after the existing 109 columns
    - Verify total feature count is 147 (109 + 5 + 6 + 7 + 8 + 4 + 8)
    - Ensure existing 109 baseline features are unchanged in position and computation
    - Ensure output dtype is float32 and shape is (T, 147)
    - _Requirements: 10.1, 10.2, 10.3, 11.1, 11.3_

  - [ ]* 3.10 Write property tests for factor output contract and NaN robustness
    - **Property 2: Factor Output Contract** — Verify shape (T, F) with 140 ≤ F ≤ 512 and dtype float32 for any valid OHLCV input
    - **Property 3: Factor NaN Robustness** — Verify no exceptions with arbitrary NaN placements in OHLCV input
    - **Validates: Requirements 10.1, 10.2, 11.3, 14.2**

  - [ ]* 3.11 Write property test for factor causality (no look-ahead bias)
    - **Property 4: Factor Causality (No Look-Ahead Bias)**
    - Use Hypothesis to generate OHLCV arrays, verify features at index i are identical whether computed from data[0:T] or data[0:i+1]
    - **Validates: Requirements 4.4, 12.5, 14.1**

- [ ] 4. Phase 2 Checkpoint: Retrain with expanded features and evaluate
  - Retrain all 60 LightGBM models with 147 features using `python train.py`
  - Run `python evaluate_local.py` to compute IC scores
  - Verify IC improvement over Phase 1 results
  - Verify feature matrix shape is (T, 147) for all datasets
  - Verify total model file size < 200 MB
  - Git commit and push to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 10.1, 10.2, 12.1, 12.5, 13.1, 13.2, 13.3, 13.4, 15.1, 15.2_

- [ ] 5. Phase 3: PyTorch GRU Ensemble
  - [ ] 5.1 Implement GRU model definition in `train.py`
    - Define `GRUPredictor(nn.Module)` with 2-layer GRU (hidden_size=64, dropout=0.1) and Linear(64, 2) output
    - Forward method: input (batch, 60, F) → GRU → last hidden state → Linear → (batch, 2)
    - Import `torch` and `torch.nn` at top of `train.py`
    - _Requirements: 16.1, 16.2, 16.3_

  - [ ] 5.2 Implement sliding window construction utility
    - Create `build_sliding_windows` function that builds causal windows of size 60 from feature matrix
    - Window at index i contains features from `max(0, i-59)` through `i` inclusive
    - Zero-pad beginning for indices i < 60
    - Replace NaN with 0.0 before GRU input
    - Support batch construction to avoid OOM on large datasets
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

  - [ ]* 5.3 Write property test for sliding window causality and shape
    - **Property 10: Sliding Window Causality and Shape**
    - Use Hypothesis to generate feature matrices, verify output shape (T, 60, F)
    - Verify window[i] contains only data from indices max(0, i-59) through i
    - Verify zero-padding for i < 60
    - Verify window[i][-1] equals factors[i]
    - **Validates: Requirements 18.1, 18.2, 18.3, 18.4**

  - [ ] 5.4 Implement GRU training loop in `train.py`
    - Add `train_gru_model()` function that trains one GRU per dataset
    - Use same temporal 80/20 split as LightGBM
    - Train with MSE loss, Adam optimizer (lr=1e-3), max 20 epochs
    - Early stopping on validation IC with patience=5
    - Set deterministic CUDA: `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
    - Save best-epoch model as TorchScript via `torch.jit.script()` to `models/gru_{dataset_name}.pt`
    - If validation IC is negative or below 0.01, still save model but flag for alpha=1.0
    - _Requirements: 16.1, 16.4, 16.5, 16.6, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7_

  - [ ] 5.5 Implement ensemble weight optimization in `train.py`
    - Add `optimize_ensemble_weights()` function
    - Grid search over alpha values [0.0, 0.1, 0.2, ..., 1.0]
    - Evaluate `alpha * lgb_val_pred + (1 - alpha) * gru_val_pred` for each alpha
    - Select alpha maximizing validation IC independently for Ret5 and Ret60
    - Save all weights to `models/ensemble_weights.json` with structure `{"dataset0": {"ret5_alpha": 0.7, "ret60_alpha": 0.5}, ...}`
    - Log optimal alpha and validation IC for each dataset-target pair
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6_

  - [ ]* 5.6 Write property test for ensemble formula correctness
    - **Property 11: Ensemble Formula Correctness**
    - Use Hypothesis to generate lgb_pred, gru_pred arrays and alpha in [0, 1]
    - Verify output equals `alpha * lgb + (1 - alpha) * gru` element-wise
    - Verify alpha=1.0 produces identical output to lgb_pred
    - **Validates: Requirements 17.3, 17.4**

  - [ ]* 5.7 Write property test for grid search optimality
    - **Property 12: Grid Search Selects IC-Maximizing Alpha**
    - Use Hypothesis to generate prediction and label arrays
    - Verify returned alpha produces highest IC among all grid values
    - Verify Ret5 and Ret60 alphas are computed independently
    - **Validates: Requirements 20.1, 20.2, 20.3**

  - [ ] 5.8 Wire GRU training and ensemble optimization into `train_all_models()` in `train.py`
    - After LightGBM training for each dataset, call `train_gru_model()`
    - After GRU training, call `optimize_ensemble_weights()` with validation predictions
    - Accumulate ensemble weights across all datasets
    - Save `ensemble_weights.json` after all datasets are processed
    - _Requirements: 19.1, 20.4_

  - [ ] 5.9 Update `predict.py` with GRU inference and ensemble logic
    - Add `import torch` and `import json` at top
    - Implement `build_sliding_windows()` function (same logic as training)
    - Implement `batch_inference()` for memory-safe GPU inference with batch_size=65536
    - Load GRU TorchScript model with `torch.jit.load(path, map_location=device)`
    - Load ensemble weights from `ensemble_weights.json`
    - Compute `final = alpha * lgb_pred + (1 - alpha) * gru_pred` per target
    - Skip GRU inference entirely when alpha=1.0 for both targets
    - Add GPU fallback: try CUDA → try CPU → fallback to pure LightGBM
    - Set `model.eval()` and use `torch.no_grad()` context
    - Preserve function signature and return contract (T, 2) float32, all finite
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 18.1, 18.2, 18.3, 18.4, 18.5, 11.2, 11.4_

  - [ ]* 5.10 Write property test for signal output contract
    - **Property 9: Signal Output Contract**
    - Use Hypothesis to generate feature matrices with arbitrary NaN placements
    - Verify output shape (T, 2), dtype float32, all values finite
    - **Validates: Requirements 11.2, 11.4**

- [ ] 6. Phase 3 Checkpoint: Train GRU models, optimize ensemble, and evaluate
  - Train all GRU models and optimize ensemble weights using `python train.py`
  - Run `python evaluate_local.py` to compute IC scores with ensemble
  - Verify IC improvement over Phase 2 results (especially Ret60)
  - Verify `ensemble_weights.json` contains valid alphas in [0, 1] for all 30 datasets × 2 targets
  - Verify GRU .pt files are saved for all 30 datasets
  - Verify total model + GRU file size < 200 MB
  - Git commit and push to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 15.1, 15.2, 21.4_

- [ ] 7. Phase 4: Packaging and Final Validation
  - [ ] 7.1 Update `evaluate_local.py` to work with new model directory structure
    - Ensure evaluation loads models from the correct directory (including GRU .pt files and ensemble_weights.json)
    - Verify evaluation produces correct IC scores matching the ensemble predict.py logic
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [ ] 7.2 Update `package_submission.py` to include GRU artifacts
    - Add GRU .pt files (`gru_dataset{0..29}.pt`) to the required files check and archive
    - Add `ensemble_weights.json` to the required files check and archive
    - Update smoke test to verify predict.py works with ensemble (GRU + LightGBM)
    - Verify total archive size < 200 MB
    - Update file count expectations (60 LightGBM + 30 GRU + 1 JSON + 2 scripts = 93 files)
    - _Requirements: 12.1, 21.4_

  - [ ] 7.3 Run full end-to-end validation
    - Run `python package_submission.py` to create submission archive
    - Verify archive contains all required files (93 total)
    - Verify archive size < 200 MB
    - Verify smoke test passes with ensemble inference
    - Verify only approved library imports in factor.py and predict.py
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3, 12.4, 21.1, 21.4_

- [ ] 8. Final Checkpoint: Git commit and push
  - Final Git commit with all Phase 4 changes
  - Push to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 15.1, 15.2, 15.3_

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each phase has a checkpoint task to verify IC improvement before proceeding to the next phase
- Phase 1 (LightGBM fix) is expected to produce the largest IC gain since models are currently severely undertrained
- Phase 3 GRU ensemble has a safe fallback: alpha=1.0 reverts to pure LightGBM if GRU doesn't help
- The existing 109 baseline features in `factor.py` must remain unchanged in computation and column position
- The public interfaces of `factor.py` (`generate_factors`) and `predict.py` (`generate_signals`) must not change
- Property tests use Hypothesis with `max_examples=200` and `deadline=None` for numba JIT compilation tolerance
