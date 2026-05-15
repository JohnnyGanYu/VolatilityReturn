# Implementation Plan: Model Optimization V3

## Overview

Incremental optimization of the volatility return prediction system across four phases: (1) LightGBM hyperparameter tuning with two-phase training, (2) Transformer encoder model + three-model ensemble, (3) optional enhancements (feature selection, sample weighting, regime augmentation), and (4) packaging + final validation. Each phase ends with a Git commit + push to a feature branch. All changes extend existing code in `train.py` and `predict.py` — `factor.py` is not modified.

## Tasks

- [x] 1. Phase 1: LightGBM Hyperparameter Tuning and Two-Phase Training
  - [x] 1.1 Update LightGBM hyperparameter constants in train.py
    - Replace `LGB_PARAMS_RET5` with v3 target-specific parameters: `num_leaves=63`, `max_depth=8`, `lambda_l1=0.5`, `lambda_l2=5.0`, `feature_fraction=0.5`
    - Replace `LGB_PARAMS_RET60` with v3 high-capacity parameters: `num_leaves=255`, `max_depth=-1`, keeping existing regularization
    - Add `MIN_BOOST_ROUND = 30` constant
    - Ensure Ret5 and Ret60 parameter dicts are distinct objects with different values for `num_leaves`, `max_depth`, `lambda_l1`, `lambda_l2`, `feature_fraction`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.3_

  - [x] 1.2 Implement two-phase LightGBM training function in train.py
    - Add `train_lgb_two_phase()` function that trains Phase 1 for `min_boost_round` rounds without early stopping, then Phase 2 with `init_model` set to Phase 1 output and IC-based early stopping enabled
    - Phase 1 uses `lgb.train()` with `num_boost_round=min_boost_round`, no `early_stopping` callback
    - Phase 2 uses `lgb.train()` with `init_model=phase1_model`, `num_boost_round=max_boost_round - min_boost_round`, and `early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)`
    - Handle edge case: if `max_boost_round <= min_boost_round`, return Phase 1 model directly
    - _Requirements: 2.1, 2.2, 2.5_

  - [x] 1.3 Update train_single_model to use two-phase training and target-specific params
    - Modify `train_single_model()` to accept a `target_name` parameter and select `LGB_PARAMS_RET5` or `LGB_PARAMS_RET60` accordingly
    - Replace the single-phase `lgb.train()` call with `train_lgb_two_phase()` for the normal training path
    - Keep the high-NaN safeguard fallback path unchanged
    - Update `train_all_models()` to pass the correct params and boost rounds per target
    - _Requirements: 1.3, 1.4, 2.1, 2.4, 2.5_

  - [x] 1.4 Write property test for two-phase minimum tree count
    - **Property 1: Two-Phase Training Minimum Tree Count**
    - Generate random (T, F) feature arrays with T∈[5000,20000], F∈[10,50] and random labels using Hypothesis
    - Assert `model.num_trees() >= min_boost_round` for every generated input
    - **Validates: Requirements 2.1, 2.4**

  - [x] 1.5 Write unit tests for LightGBM parameter values
    - Test that `LGB_PARAMS_RET5` has `num_leaves <= 63`, `max_depth == 8`, `lambda_l1 >= 0.5`, `lambda_l2 >= 5.0`, `feature_fraction <= 0.5`
    - Test that `LGB_PARAMS_RET60` has `num_leaves >= 255`, `max_depth == -1`
    - Test that Ret5 and Ret60 params differ on `num_leaves`, `max_depth`, `lambda_l1`, `lambda_l2`, `feature_fraction`
    - Test that `MIN_BOOST_ROUND == 30`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.3_

- [x] 2. Phase 1 Checkpoint — Retrain and evaluate LightGBM models
  - Run `python train.py` with updated hyperparameters and two-phase training (LightGBM only, skip GRU for now)
  - Run `python evaluate_local.py` and verify Ret5 IC improvement over baseline (0.2041)
  - Verify all Ret5 models have ≥ 30 trees by checking training logs
  - Git commit and push to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 2.4, 13.1, 13.5, 16.1, 16.2, 16.3_

- [x] 3. Phase 2: Transformer Encoder Model
  - [x] 3.1 Implement TransformerPredictor class in train.py
    - Add `TransformerPredictor(nn.Module)` class with: `input_proj` (Linear F→64), learnable `pos_embedding` (1, 60, 64), 4-layer `TransformerEncoder` (d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, norm_first=True), and `output_head` (Linear 64→2)
    - `forward()` takes (batch, 60, F) input, projects to d_model, adds positional encoding, runs through encoder, takes last position h[:, -1, :], and returns (batch, 2) output
    - Add Transformer hyperparameter constants: `TRANSFORMER_D_MODEL=64`, `TRANSFORMER_NHEAD=4`, `TRANSFORMER_NUM_LAYERS=4`, `TRANSFORMER_DIM_FF=256`, `TRANSFORMER_DROPOUT=0.1`, `TRANSFORMER_WINDOW_SIZE=60`, `TRANSFORMER_BATCH_SIZE=4096`, `TRANSFORMER_LR=1e-3`, `TRANSFORMER_EPOCHS=30`, `TRANSFORMER_PATIENCE=7`, `TRANSFORMER_MIN_IC_THRESHOLD=0.01`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 3.2 Write property test for Transformer input/output contract
    - **Property 2: Transformer Input/Output Contract**
    - Generate random (batch, 60, F) tensors with F∈[10,200], batch∈[1,64] using Hypothesis
    - Assert output shape is (batch, 2) and all values are finite
    - **Validates: Requirements 3.1, 3.3**

  - [x] 3.3 Implement Transformer training loop in train.py
    - Add `train_transformer_model()` function following the design: MSE loss, Adam optimizer (lr=1e-3), max 30 epochs, early stopping on mean validation IC with patience=7
    - Set deterministic seeds: `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, and enable `Deterministic_CUDA` when CUDA is available
    - Use `build_sliding_windows_for_indices()` (existing function) for window construction
    - Save best-epoch model as TorchScript via `torch.jit.trace()` on CPU-moved model, named `transformer_{dataset_name}.pt`
    - Add `_get_best_device()` helper if not already present (CUDA > MPS > CPU)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 15.1, 15.2_

  - [x] 3.4 Write property test for sliding window causality
    - **Property 3: Sliding Window Causality**
    - Generate random (T, F) arrays with T∈[1,500], F∈[1,50] and random indices using Hypothesis
    - Assert: (a) window[i][-1] == factors[i], (b) for i < W, first W-1-i rows are zero, (c) appending rows beyond T does not change windows at indices 0..T-1
    - **Validates: Requirements 4.4, 14.1**

  - [x] 3.5 Implement three-model ensemble weight optimization in train.py
    - Add `optimize_three_model_ensemble()` function: grid search over (alpha, beta, gamma) triples with step=0.1 where alpha+beta+gamma=1.0 (66 valid triples)
    - Evaluate each triple using `pearson_ic_numpy()` on blended validation predictions
    - Compute separate weights for Ret5 and Ret60 independently
    - When GRU val IC < threshold, constrain beta=0; when Transformer val IC < threshold, constrain gamma=0
    - Log optimal weights and validation IC for each dataset-target pair
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 3.6 Write property test for ensemble formula correctness
    - **Property 4: Three-Model Ensemble Formula**
    - Generate random (T,) prediction arrays and random (alpha, beta, gamma) triples summing to 1.0
    - Assert ensemble output equals `alpha * lgb + beta * gru + gamma * transformer` element-wise
    - Assert when beta=0 and gamma=0, output is identical to lgb_pred
    - **Validates: Requirements 5.2, 5.3**

  - [x] 3.7 Write property test for grid search optimality
    - **Property 5: Grid Search Selects IC-Maximizing Triple**
    - Generate random prediction arrays and labels using Hypothesis
    - Assert the returned triple produces IC >= all other valid triples
    - **Validates: Requirements 6.1, 6.2**

  - [x] 3.8 Update train_all_models to integrate Transformer training and three-model ensemble
    - After LightGBM and GRU training per dataset, call `train_transformer_model()` to train the Transformer
    - After all three models are trained, call `optimize_three_model_ensemble()` with validation predictions from all three models
    - Update `ensemble_weights.json` schema to include `ret5_alpha`, `ret5_beta`, `ret5_gamma`, `ret60_alpha`, `ret60_beta`, `ret60_gamma` per dataset
    - Save Transformer model files alongside existing LightGBM and GRU files
    - _Requirements: 3.6, 6.3, 6.4, 6.6_

  - [x] 3.9 Update predict.py for three-model ensemble inference
    - Refactor `generate_signals()` to load LightGBM (CPU), GRU (GPU), and Transformer (GPU) models
    - Load `ensemble_weights.json` with three-weight schema: alpha (LightGBM), beta (GRU), gamma (Transformer)
    - Compute `final = alpha * lgb + beta * gru + gamma * transformer` per target
    - Skip loading GRU/Transformer when their weights are 0.0 for both targets
    - Add `_run_sequence_model()` helper for shared GRU/Transformer batched inference with GPU→CPU→LightGBM fallback chain
    - Ensure `model.eval()` and `torch.no_grad()` for all neural model inference
    - Maintain all backward-compatible fallbacks: missing model files, missing ensemble_weights.json, GPU errors
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 11.2, 11.4, 12.4, 17.1, 17.4, 17.5, 17.6_

  - [x] 3.10 Write property test for signal output all-finite contract
    - **Property 11: Signal Output All-Finite Contract**
    - Generate random (T, F) feature matrices with NaN/Inf injected using Hypothesis
    - Assert `generate_signals()` returns (T, 2) float32 with all finite values regardless of which optional model files are present
    - **Validates: Requirements 11.4, 17.6**

- [x] 4. Phase 2 Checkpoint — Train all models and evaluate
  - Run full `python train.py` with LightGBM two-phase + GRU + Transformer + three-model ensemble
  - Run `python evaluate_local.py` and verify IC improvement across all four categories vs baseline
  - Verify Transformer models are saved as `transformer_dataset{0..29}.pt`
  - Verify `ensemble_weights.json` contains three-weight schema for all 30 datasets
  - Git commit and push to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.6, 16.1, 16.2, 16.3_

- [x] 5. Phase 3: Optional Enhancements
  - [x] 5.1 Implement gain-based feature importance extraction in train.py
    - Add `extract_feature_importance()` function: aggregate gain-based importance across all 30 datasets per target
    - Rank features by aggregated importance separately for Ret5 and Ret60
    - Log top-20 most important features for each target with gain values
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 5.2 Implement top-100 feature selection and LightGBM retraining
    - Select top-100 features per target from aggregated importance
    - Save `feature_selection.json` with `{"ret5_features": [...], "ret60_features": [...]}` structure
    - Retrain all 60 LightGBM models using only selected features with same two-phase training and target-specific params
    - Compare validation IC before and after feature selection, log differences
    - _Requirements: 8.1, 8.2, 8.3, 8.6_

  - [x] 5.3 Update predict.py for feature selection support
    - Load `feature_selection.json` when it exists and subset Feature_Matrix columns for Ret5 and Ret60 LightGBM inference
    - When `feature_selection.json` does not exist, use full Feature_Matrix (backward-compatible fallback)
    - _Requirements: 8.4, 8.5_

  - [x] 5.4 Write property test for feature selection correctness
    - **Property 6: Feature Selection Correctness**
    - Generate random importance arrays of length F∈[100,300] and random (T, F) matrices
    - Assert selecting top-100 indices and subsetting produces (T, 100) matrix with correct columns
    - **Validates: Requirements 7.2, 8.1, 8.4**

  - [x] 5.5 Implement dynamic sample weighting in train.py
    - Add `compute_dynamic_sample_weights()` function: `weight[i] = 1.0 + 0.5 * (volatility[i] / rolling_vol_median[i])`
    - `volatility[i]` = rolling std of 1-bar log returns over 20-bar window; `rolling_vol_median[i]` = rolling median of volatility over 120-bar window
    - All computations causal (no look-ahead); set weight to 1.0 when rolling_vol_median is 0 or NaN
    - Pass weights to `lgb.Dataset` via `weight` parameter during training only
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 14.2_

  - [x] 5.6 Write property test for dynamic sample weight formula and causality
    - **Property 7: Dynamic Sample Weight Formula and Causality**
    - Generate random positive close price arrays of length T∈[200,1000] using Hypothesis
    - Assert weight formula correctness and causality: weight at index i from close[0:T] equals weight at index i from close[0:i+1]
    - Assert weight = 1.0 when rolling_vol_median is 0 or NaN
    - **Validates: Requirements 9.1, 9.3, 9.4, 14.2**

  - [x] 5.7 Implement regime classifier training in train.py
    - Add `train_regime_classifier()` function: lightweight LightGBM binary classifier (num_leaves=31, max_depth=6, n_estimators=200)
    - Use regime-related features (columns 82-93) as input
    - Construct binary labels from extreme_intervals: label=1 for bars within any extreme interval, label=0 otherwise
    - Save as `regime_{dataset_name}.txt`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 14.3_

  - [x] 5.8 Update predict.py for regime feature augmentation
    - Load `regime_{dataset_name}.txt` when it exists, predict regime probability for every bar
    - Append regime probability as additional column to Feature_Matrix before LightGBM inference
    - When regime model file does not exist, skip augmentation and use original Feature_Matrix
    - _Requirements: 10.7, 10.8_

  - [x] 5.9 Write property test for regime probability output bounds
    - **Property 8: Regime Probability Output Bounds**
    - Generate random (T, 12) feature arrays using Hypothesis
    - Assert regime classifier output is in [0.0, 1.0] for every bar
    - Assert appending probability column produces (T, F+1) matrix
    - **Validates: Requirements 10.3, 10.7**

  - [x] 5.10 Write property test for extreme interval label construction
    - **Property 9: Extreme Interval Label Construction**
    - Generate random index arrays and extreme interval arrays using Hypothesis
    - Assert label[i] == 1 iff indices[i] falls within at least one interval, 0 otherwise
    - **Validates: Requirements 10.4**

- [x] 6. Phase 3 Checkpoint — Evaluate optional enhancements
  - For each optional enhancement implemented: run training, evaluate IC, compare to Phase 2 baseline
  - Revert any enhancement that degrades IC
  - Git commit and push to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 8.6, 13.1, 13.2, 13.3, 13.4, 16.1, 16.2, 16.3_

- [x] 7. Phase 4: Packaging and Final Validation
  - [x] 7.1 Update package_submission.py for all new artifacts
    - Add Transformer TorchScript files (`transformer_dataset{0..29}.pt`) to required file checks and archive
    - Add optional file handling: `feature_selection.json`, `regime_{dataset}.txt` files (include if present)
    - Update `ensemble_weights.json` validation for three-weight schema
    - Update smoke test to exercise three-model ensemble path
    - Update file count and size reporting
    - _Requirements: 12.2, 18.1, 18.2, 18.3, 18.4_

  - [x] 7.2 Run full end-to-end validation
    - Run `python package_submission.py` and verify archive < 200 MB
    - Run `python evaluate_local.py` on final models and verify all four IC categories exceed baseline
    - Verify `generate_factors` signature unchanged: `(dataset_name: str, data: np.ndarray) -> np.ndarray`
    - Verify `generate_signals` signature unchanged: `(dataset_name: str, factors: np.ndarray) -> np.ndarray`
    - Verify output shapes and dtypes: factors (T, F) float32 with F ≤ 512, signals (T, 2) float32 all-finite
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 13.1, 13.2, 13.3, 13.4_

  - [x] 7.3 Write property test for factor output contract
    - **Property 10: Factor Output Contract**
    - Generate random OHLCV arrays of shape (T, 5) with T∈[100,1000] using Hypothesis
    - Assert `generate_factors` returns float32 array of shape (T, F) with F ≤ 512
    - **Validates: Requirements 11.3**

  - [x] 7.4 Write unit tests for backward compatibility fallbacks
    - Test missing `transformer_{dataset}.pt` → no error, falls back to LightGBM + GRU or pure LightGBM
    - Test missing `ensemble_weights.json` → pure LightGBM output
    - Test missing `feature_selection.json` → full Feature_Matrix used
    - Test missing `regime_{dataset}.txt` → original Feature_Matrix used
    - Test all optional files missing → valid (T, 2) float32 output with all finite values
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_

- [x] 8. Final Checkpoint — Git commit and push
  - Final Git commit with all changes to feature branch
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 16.1, 16.2, 16.3_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Phase 3 tasks (5.1–5.10) are all optional enhancements — implement and evaluate incrementally, revert if IC degrades
- `factor.py` is NOT modified in any task — all 147 features remain unchanged
- `train.py` receives the most changes: new hyperparameters, two-phase training, TransformerPredictor class, training loop, three-model ensemble optimization, and optional enhancements
- `predict.py` is updated for three-model ensemble inference with full backward compatibility
- Property tests use Hypothesis library for Python PBT
- Each checkpoint includes Git commit + push to feature branch per Requirement 16
- Checkpoints verify IC improvement before proceeding to next phase
