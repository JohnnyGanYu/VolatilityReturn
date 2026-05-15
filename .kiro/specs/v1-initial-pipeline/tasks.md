# Implementation Plan: Volatility Return Prediction

## Overview

This plan implements a volatility return prediction system for the East China Cup competition. The system predicts 5-minute and 60-minute forward returns from 1-minute OHLCV data for 30 high-volatility instruments using numba-accelerated feature engineering and per-dataset LightGBM models. Implementation proceeds in order: core feature engine → prediction interface → training pipeline → local evaluation → property tests → submission packaging.

## Tasks

- [x] 1. Implement feature generation engine (`factor.py`)
  - [x] 1.1 Create `factor.py` with `generate_factors(dataset_name, data)` function skeleton and seed initialization
    - Define the single public function matching the platform interface
    - Set random seeds (numpy, random, etc.) at the start of the function body for reproducibility
    - Unpack OHLCV columns: open, high, low, close, volume from the (T, 5) input
    - Return stacked feature matrix as float32 with shape (T, F), F ≤ 512
    - _Requirements: 1.1, 1.2, 3.1, 4.1_

  - [x] 1.2 Implement numba-accelerated momentum features
    - Use `@njit(cache=True)` for all numba-compiled feature functions to minimize JIT overhead on module reload
    - Compute log returns over lookback windows: 1, 3, 5, 10, 20, 60, 120 bars (~14 features)
    - Compute rate-of-change variants
    - Fill first `w-1` rows with NaN for each lookback window `w`
    - Handle NaN in input prices gracefully (propagate NaN, no exceptions)
    - _Requirements: 6.1, 6.6, 5.1, 5.4, 1.3_

  - [x] 1.3 Implement numba-accelerated volatility features
    - Use `@njit(cache=True)` for feature functions
    - Compute rolling standard deviation of returns over windows: 5, 10, 20, 60, 120 (~10 features)
    - Compute Parkinson volatility (high-low range based) over multiple windows
    - Compute Garman-Klass volatility estimator
    - Compute ATR (Average True Range) over multiple windows
    - All computations must be causal (no look-ahead)
    - _Requirements: 6.2, 5.1, 5.4, 6.6_

  - [x] 1.4 Implement numba-accelerated volume and microstructure features
    - Use `@njit(cache=True)` for feature functions
    - Volume features: volume MA ratios, VWAP deviation, OBV (~15 features)
    - Microstructure features: spread proxy (high-low)/close, upper/lower shadow ratios, bar return skewness (~20 features)
    - Guard against zero volume and zero price with NaN fallback: `np.where(volume == 0, np.nan, ...)`
    - _Requirements: 6.3, 6.4, 5.1, 1.3_

  - [x] 1.5 Implement technical indicator features
    - Compute RSI over multiple windows (e.g., 14, 28)
    - Compute MACD (12/26/9 standard, plus variations)
    - Compute Bollinger Band width and %B over multiple windows
    - Compute Stochastic oscillator, CCI
    - Use numba `@njit(cache=True)` or numpy vectorized implementations (avoid TA-Lib if numba is faster for reload)
    - ~40 features total
    - _Requirements: 6.5, 5.1, 5.4_

  - [x] 1.6 Implement regime detection and cross-interaction features
    - Regime features: vol-of-vol, max drawdown, extreme regime flag (~15 features)
    - Cross features: momentum × volatility, volume × return interactions (~30 features)
    - All features must be causal and use `@njit(cache=True)` where applicable
    - _Requirements: 6.1, 6.2, 6.3, 9.3_

  - [ ]* 1.7 Write property tests for factor output contract and NaN robustness
    - **Property 1: Factor Output Contract** — For any valid OHLCV input of shape (T, 5), `generate_factors` returns shape (T, F) with 1 ≤ F ≤ 512 and dtype float32
    - **Validates: Requirements 1.1, 1.2**
    - **Property 2: Factor NaN Robustness** — For any OHLCV input containing arbitrary NaN placements, `generate_factors` completes without exception and returns correct shape/dtype
    - **Validates: Requirements 1.3**

  - [ ]* 1.8 Write property tests for factor determinism and causality
    - **Property 5: Factor Determinism** — Calling `generate_factors` twice on the same input produces bit-identical output
    - **Validates: Requirements 3.1, 3.3, 4.1, 4.4**
    - **Property 7: Factor Causality** — Feature at index i computed from data[0:T] equals feature at index i computed from data[0:i+1]
    - **Validates: Requirements 5.1, 5.4**
    - **Property 9: Lookback NaN Initialization** — Features requiring lookback w have NaN at indices 0 through w-2
    - **Validates: Requirements 6.6**

- [x] 2. Checkpoint — Verify factor.py
  - Ensure `factor.py` runs on real training data for at least 3 datasets (small, medium, large)
  - Verify output shape, dtype, and NaN patterns
  - Ensure all tests pass, ask the user if questions arise

- [x] 3. Implement signal generation (`predict.py`)
  - [x] 3.1 Create `predict.py` with `generate_signals(dataset_name, factors)` function
    - Set random seeds explicitly at the start of the function (numpy, random, os.environ PYTHONHASHSEED) even though LightGBM inference is deterministic — this is mandatory for strict Requirement 4 compliance
    - Load per-dataset LightGBM models from `/workspace/submission/` inside the function body (not module level)
    - Model file naming: `lgb_ret5_{dataset_name}.txt` and `lgb_ret60_{dataset_name}.txt`
    - Run inference: `model.predict(factors)` — LightGBM handles NaN natively
    - Stack predictions into (T, 2) float32 array
    - Apply `np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)` before returning to ensure all-finite output
    - _Requirements: 2.1, 2.2, 2.3, 3.2, 4.2, 11.1, 11.2, 11.3_

  - [ ]* 3.2 Write property tests for signal output contract and NaN robustness
    - **Property 3: Signal Output Contract** — For any feature matrix of shape (T, F), `generate_signals` returns shape (T, 2) with dtype float32 and all finite values
    - **Validates: Requirements 2.1, 11.1, 11.3**
    - **Property 4: Signal NaN Robustness** — For any feature matrix with arbitrary NaN, `generate_signals` completes without exception and returns all-finite (T, 2) float32
    - **Validates: Requirements 2.3, 11.2**

  - [ ]* 3.3 Write property tests for signal determinism, causality, and non-degeneracy
    - **Property 6: Signal Determinism** — Calling `generate_signals` twice on the same input produces bit-identical output
    - **Validates: Requirements 3.2, 3.4, 4.2, 4.4**
    - **Property 8: Signal Causality** — Prediction at index i from factors[0:T] equals prediction at index i from factors[0:i+1]
    - **Validates: Requirements 5.3**
    - **Property 10: Signal Non-Degeneracy** — For feature matrices with sufficient non-NaN variance, predictions are not constant-valued
    - **Validates: Requirements 8.3**

- [x] 4. Checkpoint — Verify predict.py interface
  - Test `predict.py` with mock model files and synthetic factor inputs
  - Verify output shape (T, 2), dtype float32, all-finite values
  - Ensure all tests pass, ask the user if questions arise

- [x] 5. Implement training pipeline (`train.py`)
  - [x] 5.1 Create `train.py` with data loading and label extraction
    - Load `dataset{i}_train_ohlcv.npy` files from `train_dataset/` directory
    - Extract OHLCV columns (indices 1-5) and label columns: Ret5 (index 6), Ret60 (index 7)
    - Document all hyperparameters and random seeds as constants at the top of the script
    - Set all random seeds (numpy, random, lightgbm seed param) for reproducibility
    - _Requirements: 14.1, 14.3, 7.5, 4.3_

  - [x] 5.2 Implement per-dataset model training loop with high-NaN safeguard
    - Loop over 30 datasets × 2 targets (Ret5, Ret60)
    - Call `generate_factors()` from `factor.py` to compute features for each dataset
    - Apply temporal 80/20 split (first 80% train, last 20% validation) — no shuffling to avoid look-ahead
    - Exclude rows with NaN labels from training and validation sets
    - **High-NaN safeguard for datasets 20-29**: If the number of valid (non-NaN label) samples is below a threshold (e.g., < 5000), fall back to a simple global baseline model (e.g., predict mean of valid labels, or train a minimal LightGBM with very few trees) to prevent training failure
    - Use LightGBM hyperparameters from design: `LGB_PARAMS_RET5` and `LGB_PARAMS_RET60`
    - Train with early stopping on validation IC (custom `feval` callback)
    - Save models as `lgb_ret5_dataset{i}.txt` and `lgb_ret60_dataset{i}.txt`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.3, 14.2_

  - [ ]* 5.3 Write unit tests for training pipeline
    - Test data loading and label extraction on a small synthetic dataset
    - Test temporal split logic (verify no future data leaks into training set)
    - Test high-NaN safeguard: verify fallback triggers when valid samples < threshold
    - Test model file output: verify files are created with correct naming convention
    - _Requirements: 7.1, 7.2, 7.4_

- [x] 6. Checkpoint — Train models and verify
  - Run `train.py` on the full training dataset to produce all 60 model files
  - Verify all model files are created and total size is within 200 MB
  - Ensure all tests pass, ask the user if questions arise

- [x] 7. Implement local evaluation script (`evaluate_local.py`)
  - [x] 7.1 Create `evaluate_local.py` to compute IC on training data
    - Load training data and extreme interval annotations for all 30 datasets
    - Run `generate_factors()` then `generate_signals()` for each dataset
    - Split predictions into normal and extreme subsets using extreme_intervals
    - Compute Pearson IC using the exact reference implementation from the competition (nan_to_num → pearson correlation)
    - Report 4 IC values per dataset: normal×Ret5, normal×Ret60, extreme×Ret5, extreme×Ret60
    - Report mean IC across all 30 datasets for each of the 4 categories
    - _Requirements: 8.1, 8.2, 9.1, 9.2, 10.1, 10.2_

  - [ ]* 7.2 Write unit tests for IC computation
    - Test `pearson_ic` function with known inputs (perfect correlation, zero correlation, all-NaN)
    - Test normal/extreme subset splitting logic
    - _Requirements: 10.2_

- [x] 8. Checkpoint — Evaluate model performance
  - Run `evaluate_local.py` and review IC scores
  - Verify positive mean IC on normal subsets for both Ret5 and Ret60
  - Ensure all tests pass, ask the user if questions arise

- [x] 9. Implement submission packaging
  - [x] 9.1 Create submission packaging script
    - Package `factor.py`, `predict.py`, and all model files into a `.tar.gz` archive
    - Verify the package only imports from the approved library list (torch, numpy, pandas, scikit-learn, scipy, bottleneck, numba, statsmodels, TA-Lib, lightgbm, transformers, polars)
    - Verify total package size < 200 MB
    - Include a basic smoke test: extract the archive, import both modules, run on synthetic data
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 9.2 Write integration test for end-to-end submission flow
    - Extract packaged submission
    - Run `generate_factors` → `generate_signals` on at least one real dataset
    - Verify output shapes, dtypes, and finite values
    - _Requirements: 12.1, 12.2, 13.1_

- [x] 10. Final checkpoint — Full pipeline validation
  - Run all property tests and unit tests
  - Verify end-to-end pipeline on real data produces valid predictions
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each major component
- Property tests validate the 10 correctness properties defined in the design document
- **User-requested safeguards**: (1) High-NaN fallback in `train.py` for datasets 20-29, (2) Explicit seed setting in `predict.py` even for deterministic LightGBM inference, (3) `@njit(cache=True)` on all numba feature functions
- All numba functions use `@njit(cache=True)` to minimize JIT compilation overhead during module reloads on the evaluation platform
