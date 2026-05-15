# Requirements Document

## Introduction

This specification covers the v2 optimization iteration of the volatility return prediction system built for the East China Cup Mathematical Modeling Competition (华东杯数学建模比赛, Problem C). The baseline system is already working and submitted with in-sample mean IC scores of 0.1255 (Normal×Ret5), 0.2519 (Normal×Ret60), 0.2673 (Extreme×Ret5), and 0.4316 (Extreme×Ret60). Diagnosis has identified three root causes limiting performance: (1) severe model underfitting due to MAE-based early stopping conflicting with IC optimization, (2) single-threaded training due to unnecessary deterministic mode constraints, and (3) underutilized feature capacity (109 of 512 allowed features). This optimization targets all four IC metrics while preserving every competition constraint and the public interfaces of `factor.py` and `predict.py`.

With 4 days remaining for training, testing, and optimization, this iteration also introduces a **GPU-accelerated PyTorch ensemble strategy** to fully utilize the evaluation platform's resources (RTX 4090 24GB GPU, 96GB RAM, 16 CPU cores, 2-hour runtime). The current LightGBM-only approach leaves the GPU entirely idle. By training a lightweight GRU sequence model per-dataset and ensembling its predictions with LightGBM via learned weights, the system can capture temporal dependencies in OHLCV data that point-in-time tree features cannot. The ensemble is designed with a safe fallback: if the PyTorch model does not improve IC for a given dataset, the ensemble weight reverts to pure LightGBM (alpha=1.0), ensuring no regression. Requirements 1–15 cover LightGBM optimization; Requirements 16–21 cover the PyTorch ensemble addition.

## Glossary

- **Training_Pipeline**: The `train.py` script that trains per-dataset, per-target LightGBM models offline and saves model files.
- **Factor_Generator**: The `factor.py` module containing the `generate_factors()` function responsible for computing features from raw OHLCV data.
- **Signal_Generator**: The `predict.py` module containing the `generate_signals()` function responsible for producing return predictions from computed features.
- **IC_Metric**: Pearson Information Coefficient — the Pearson correlation between predicted signals and actual returns, computed per-dataset on normal and extreme subsets independently.
- **Custom_IC_Feval**: A user-defined LightGBM evaluation function that computes Pearson IC on the validation set, used as the sole early stopping criterion.
- **Early_Stopping**: LightGBM's mechanism to halt boosting when the validation metric has not improved for a specified number of rounds.
- **MAE_Metric**: Mean Absolute Error — the built-in LightGBM metric previously used for early stopping, which plateaus before IC is maximized.
- **Boosting_Round**: A single iteration of gradient boosting that adds one tree to the ensemble.
- **Feature_Matrix**: A numpy array of shape (T, F) where F ≤ 512, containing computed features for each time bar, dtype float32.
- **EMA**: Exponential Moving Average — a weighted moving average that gives more weight to recent observations, parameterized by span.
- **EMA_Ratio**: The ratio of close price to its EMA at a given span, measuring price deviation from the smoothed trend.
- **Rolling_Skewness**: The third standardized moment of returns computed over a rolling window, measuring asymmetry of the return distribution.
- **Rolling_Kurtosis**: The fourth standardized moment of returns computed over a rolling window, measuring tail heaviness of the return distribution.
- **Close_To_Open_Gap**: The log ratio log(open[i] / close[i-1]), measuring the overnight or inter-bar price jump.
- **Volume_Weighted_Return**: The product of bar return and normalized volume, capturing the directional conviction behind price moves.
- **Return_Autocorrelation**: The Pearson correlation between returns at lag 0 and returns at a specified lag within a rolling window.
- **Realized_Variance**: The sum of squared intrabar log returns over a rolling window, a non-parametric volatility estimator.
- **Baseline_IC**: The current in-sample mean IC scores: Normal×Ret5=0.1255, Normal×Ret60=0.2519, Extreme×Ret5=0.2673, Extreme×Ret60=0.4316.
- **Model_File**: Serialized LightGBM model parameters saved as `.txt` files, loaded by the Signal_Generator at inference time.
- **OHLCV_Data**: A numpy array of shape (T, 5) with columns [open, high, low, close, volume] representing 1-minute candlestick data.
- **Look_Ahead_Bias**: The prohibited use of future information (data at time i+k) when computing features or predictions at time i.
- **PyTorch_Sequence_Model**: A lightweight GRU-based recurrent neural network trained per-dataset that takes sliding windows of OHLCV-derived features as input and outputs Ret5 and Ret60 predictions, capturing temporal dependencies that LightGBM cannot.
- **GRU**: Gated Recurrent Unit — a recurrent neural network architecture that processes sequential data with gating mechanisms for learning temporal dependencies.
- **Sliding_Window**: A causal window of consecutive feature vectors (e.g., 60 bars × F features) used as input to the PyTorch_Sequence_Model, constructed without Look_Ahead_Bias.
- **TorchScript**: PyTorch's serialization format (.pt files) that saves models as portable, self-contained artifacts loadable without the original model class definition.
- **Ensemble_Weight**: A per-dataset, per-target scalar alpha in [0.0, 1.0] that controls the weighted average between LightGBM and PyTorch predictions: `final = alpha * lgb_pred + (1 - alpha) * pytorch_pred`.
- **Ensemble_Config**: A JSON file storing the learned Ensemble_Weights for all datasets and targets, loaded by the Signal_Generator at inference time.
- **GPU_Inference**: Running PyTorch model forward passes on the RTX 4090 GPU for accelerated batch prediction, with automatic CPU fallback on failure.
- **Deterministic_CUDA**: PyTorch configuration flags (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) that ensure bit-identical GPU computation across runs.

## Requirements

### Requirement 1: Fix Early Stopping to Use IC as Sole Criterion

**User Story:** As a data scientist, I want the Training_Pipeline to use Custom_IC_Feval as the only early stopping criterion instead of MAE_Metric, so that models train for more Boosting_Rounds and optimize directly for IC_Metric rather than stopping prematurely when MAE plateaus.

#### Acceptance Criteria

1. WHEN training a LightGBM model, THE Training_Pipeline SHALL set the LightGBM parameter `metric` to `"None"` (string) to disable all built-in evaluation metrics for early stopping.
2. WHEN training a LightGBM model, THE Training_Pipeline SHALL pass the Custom_IC_Feval function as the `feval` argument to `lgb.train()`, and this Custom_IC_Feval SHALL be the only metric used for early stopping decisions.
3. THE Custom_IC_Feval SHALL compute Pearson IC between predictions and labels on the validation set, return the tuple `("ic", ic_value, True)` where `True` indicates higher is better, and handle edge cases (zero variance, all-NaN) by returning an IC of 0.0.
4. WHEN early stopping triggers, THE Training_Pipeline SHALL select the model iteration with the highest validation IC, not the lowest MAE.
5. WHEN training completes for any dataset, THE Training_Pipeline SHALL log the number of Boosting_Rounds used by the best model iteration.
6. WHEN the early stopping fix is applied, THE Training_Pipeline SHALL produce models that use more than 1 tree for datasets that previously stopped at 1 tree (e.g., dataset0, dataset11, dataset28).

### Requirement 2: Increase Model Capacity

**User Story:** As a data scientist, I want the Training_Pipeline to use higher-capacity LightGBM hyperparameters, so that models can learn more complex patterns from the expanded feature set and longer training runs.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL configure the Ret5 LightGBM parameters with `num_leaves` of at least 127, `learning_rate` of at most 0.03, and `n_estimators` (max Boosting_Rounds) of at least 1000.
2. THE Training_Pipeline SHALL configure the Ret60 LightGBM parameters with `num_leaves` of at least 255, `learning_rate` of at most 0.02, and `n_estimators` (max Boosting_Rounds) of at least 1500.
3. THE Training_Pipeline SHALL set `early_stopping_rounds` to at least 100 to allow the IC_Metric sufficient rounds to demonstrate improvement before halting.
4. THE Training_Pipeline SHALL set `min_child_samples` to at least 200 to prevent overfitting with the increased tree capacity.
5. WHEN all 60 models are saved, THE total Model_File size SHALL not exceed 200 MB.

### Requirement 3: Enable Multi-Threaded Training

**User Story:** As a data scientist, I want the Training_Pipeline to use multiple CPU threads during training, so that training completes faster and allows more experimentation within the 2-hour runtime budget.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL remove the `deterministic=True` parameter from all LightGBM training configurations.
2. THE Training_Pipeline SHALL remove the `force_row_wise=True` parameter from all LightGBM training configurations.
3. THE Training_Pipeline SHALL set the LightGBM `num_threads` parameter to a value that utilizes multiple CPU cores (e.g., -1 for auto-detection, or a specific value matching the platform's 16 cores).
4. THE Training_Pipeline SHALL retain fixed random seeds (`seed` parameter) in all LightGBM configurations to maintain reproducibility at the "training script + random seed" level required by the competition.
5. THE Signal_Generator SHALL not be modified for threading, as LightGBM prediction is inherently deterministic regardless of threading settings.

### Requirement 4: Add EMA Ratio Features

**User Story:** As a data scientist, I want the Factor_Generator to compute EMA_Ratio features for multiple spans, so that the model can capture price deviation from smoothed trends at different time scales.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute EMA_Ratio features as `close[i] / EMA(close, span)[i]` for spans [5, 10, 20, 60, 120].
2. THE Factor_Generator SHALL compute each EMA using the standard formula: `EMA[i] = alpha * close[i] + (1 - alpha) * EMA[i-1]` where `alpha = 2 / (span + 1)`.
3. WHEN `EMA(close, span)[i]` is zero or NaN, THE Factor_Generator SHALL output NaN for the corresponding EMA_Ratio.
4. THE Factor_Generator SHALL compute all EMA_Ratio features using only data from indices 0 through i (no Look_Ahead_Bias).
5. THE Factor_Generator SHALL produce exactly 5 EMA_Ratio features.

### Requirement 5: Add Rolling Return Skewness and Kurtosis Features

**User Story:** As a data scientist, I want the Factor_Generator to compute Rolling_Skewness and Rolling_Kurtosis of 1-bar log returns, so that the model can detect asymmetric and fat-tailed return distributions that signal regime changes.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute Rolling_Skewness of 1-bar log returns over windows [20, 60, 120].
2. THE Factor_Generator SHALL compute Rolling_Kurtosis of 1-bar log returns over windows [20, 60, 120].
3. THE Factor_Generator SHALL compute Rolling_Skewness as the third standardized central moment: `m3 / (std^3)` where `m3` is the third central moment and `std` is the sample standard deviation over the window.
4. THE Factor_Generator SHALL compute Rolling_Kurtosis as the fourth standardized central moment: `m4 / (std^4)` where `m4` is the fourth central moment and `std` is the sample standard deviation over the window.
5. WHEN a rolling window contains fewer than 3 valid (non-NaN) return values, THE Factor_Generator SHALL output NaN for both Rolling_Skewness and Rolling_Kurtosis at that index.
6. THE Factor_Generator SHALL produce exactly 6 features (3 skewness + 3 kurtosis).

### Requirement 6: Add Close-to-Open Gap Features

**User Story:** As a data scientist, I want the Factor_Generator to compute Close_To_Open_Gap features, so that the model can capture inter-bar price jumps that often carry predictive information about short-term momentum.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute the raw Close_To_Open_Gap as `log(open[i] / close[i-1])` for each bar i ≥ 1.
2. WHEN `close[i-1]` is zero, NaN, or negative, THE Factor_Generator SHALL output NaN for the Close_To_Open_Gap at index i.
3. THE Factor_Generator SHALL compute rolling mean of Close_To_Open_Gap over windows [5, 10, 20].
4. THE Factor_Generator SHALL compute rolling standard deviation of Close_To_Open_Gap over windows [5, 10, 20].
5. THE Factor_Generator SHALL produce exactly 7 features (1 raw gap + 3 rolling means + 3 rolling stds).

### Requirement 7: Add Volume-Weighted Return Features

**User Story:** As a data scientist, I want the Factor_Generator to compute Volume_Weighted_Return features, so that the model can assess directional conviction by combining price movement with trading activity.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute the per-bar Volume_Weighted_Return as `log_return_1bar * (volume[i] / rolling_mean_volume(window)[i])` for windows [5, 10, 20, 60].
2. THE Factor_Generator SHALL compute rolling sums of Volume_Weighted_Return over windows [5, 10, 20, 60].
3. WHEN volume or log return is NaN at index i, THE Factor_Generator SHALL output NaN for the Volume_Weighted_Return at that index.
4. THE Factor_Generator SHALL produce exactly 8 features (4 per-bar VWR + 4 rolling sums).

### Requirement 8: Add Return Autocorrelation Features

**User Story:** As a data scientist, I want the Factor_Generator to compute Return_Autocorrelation features, so that the model can detect serial dependence in returns that indicates trending or mean-reverting regimes.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute Return_Autocorrelation of 1-bar log returns at lag 1 over rolling windows [20, 60].
2. THE Factor_Generator SHALL compute Return_Autocorrelation of 1-bar log returns at lag 5 over rolling windows [20, 60].
3. THE Factor_Generator SHALL compute each Return_Autocorrelation as the Pearson correlation between `returns[j]` and `returns[j-lag]` for all j within the rolling window.
4. WHEN a rolling window contains fewer than `lag + 3` valid (non-NaN) return pairs, THE Factor_Generator SHALL output NaN for the Return_Autocorrelation at that index.
5. THE Factor_Generator SHALL produce exactly 4 features (2 lags × 2 windows).

### Requirement 9: Add Realized Variance Proxy Features

**User Story:** As a data scientist, I want the Factor_Generator to compute Realized_Variance features, so that the model has a non-parametric volatility estimator that complements the existing parametric volatility features.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute Realized_Variance as the sum of squared 1-bar log returns over rolling windows [5, 10, 20, 60].
2. THE Factor_Generator SHALL compute the log of Realized_Variance (log-RV) for each window to normalize the distribution.
3. WHEN a rolling window contains zero valid (non-NaN) squared returns, THE Factor_Generator SHALL output NaN for both Realized_Variance and log-RV at that index.
4. THE Factor_Generator SHALL produce exactly 8 features (4 RV + 4 log-RV).

### Requirement 10: Feature Count Compliance

**User Story:** As a competition participant, I want the total feature count to remain within the platform limit, so that the submission is accepted by the evaluation server.

#### Acceptance Criteria

1. WHEN all new features from Requirements 4 through 9 are added to the existing 109 baseline features, THE Factor_Generator SHALL produce a Feature_Matrix with F ≤ 512 columns.
2. THE Factor_Generator SHALL produce a Feature_Matrix with at least 140 features (109 baseline + approximately 38 new features).
3. THE Factor_Generator SHALL maintain the existing 109 baseline features unchanged in their computation logic and column positions.

### Requirement 11: Preserve Public Interfaces

**User Story:** As a competition participant, I want the public interfaces of `factor.py` and `predict.py` to remain identical, so that the submission is compatible with the evaluation platform.

#### Acceptance Criteria

1. THE Factor_Generator SHALL maintain the function signature `generate_factors(dataset_name: str, data: np.ndarray) -> np.ndarray` without any changes to parameter names, types, or return type.
2. THE Signal_Generator SHALL maintain the function signature `generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray` without any changes to parameter names, types, or return type.
3. THE Factor_Generator SHALL continue to return a float32 numpy array of shape (T, F) where F ≤ 512.
4. THE Signal_Generator SHALL continue to return a float32 numpy array of shape (T, 2) with all finite values.

### Requirement 12: Maintain Competition Constraints

**User Story:** As a competition participant, I want all optimization changes to comply with competition rules, so that the submission is not disqualified.

#### Acceptance Criteria

1. THE submission package SHALL not exceed 200 MB total size including all Model_Files, `factor.py`, and `predict.py`.
2. THE Factor_Generator and Signal_Generator combined SHALL complete processing of all 30 datasets within 2 hours of wall-clock time on the evaluation platform (16 CPU cores, 96 GB RAM, RTX 4090).
3. THE Factor_Generator SHALL use only approved libraries: numpy, numba, and Python standard library.
4. THE Training_Pipeline SHALL use only approved libraries: numpy, lightgbm, numba, torch, and Python standard library.
5. THE Factor_Generator SHALL not introduce Look_Ahead_Bias in any new feature computation.
6. THE Training_Pipeline SHALL maintain fixed random seeds for reproducibility at the "training script + random seed" level.

### Requirement 13: IC Performance Improvement

**User Story:** As a competition participant, I want the optimized system to achieve higher IC_Metric scores than the Baseline_IC across all four evaluation categories, so that the competition ranking improves.

#### Acceptance Criteria

1. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Normal×Ret5 IC greater than the Baseline_IC of 0.1255.
2. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Normal×Ret60 IC greater than the Baseline_IC of 0.2519.
3. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Extreme×Ret5 IC greater than the Baseline_IC of 0.2673.
4. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Extreme×Ret60 IC greater than the Baseline_IC of 0.4316.
5. THE ensemble approach (LightGBM + PyTorch_Sequence_Model) SHALL provide additional IC improvement beyond the LightGBM-only optimizations from Requirements 1–10, especially for Ret60 predictions where temporal patterns across sliding windows are expected to be stronger than point-in-time features alone.

### Requirement 14: New Feature Causality and NaN Safety

**User Story:** As a data scientist, I want all new features to follow the same causality and NaN-handling standards as the baseline features, so that the system remains correct and robust.

#### Acceptance Criteria

1. WHEN computing any new feature at time index i, THE Factor_Generator SHALL use only OHLCV_Data from indices 0 through i (no Look_Ahead_Bias).
2. WHEN the OHLCV_Data contains NaN values, THE Factor_Generator SHALL propagate NaN through new feature computations without raising exceptions.
3. WHEN a new feature requires a lookback window of w bars, THE Factor_Generator SHALL output NaN for indices 0 through w-2.
4. THE Factor_Generator SHALL implement all new feature computations using numba `@njit(cache=True)` decorated functions for performance consistency with the baseline.

### Requirement 15: Version Control and Traceability

**User Story:** As a developer, I want every code change to be committed and pushed to GitHub, so that the optimization history is traceable and the team can review changes.

#### Acceptance Criteria

1. WHEN a code change is made to `factor.py`, `train.py`, or `predict.py`, THE developer SHALL commit the change to a Git branch with a descriptive commit message.
2. THE developer SHALL push all commits to the GitHub remote repository.
3. THE developer SHALL not commit directly to the main branch without creating a separate branch for the optimization work.

### Requirement 16: PyTorch Sequence Model for Ensemble

**User Story:** As a data scientist, I want the Training_Pipeline to train a lightweight PyTorch_Sequence_Model (GRU) per-dataset that captures temporal dependencies from sliding windows of OHLCV features, so that the ensemble can exploit sequential patterns that LightGBM's point-in-time features cannot.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train one PyTorch_Sequence_Model per dataset (30 models total), where each model outputs 2 values: a Ret5 prediction and a Ret60 prediction.
2. THE PyTorch_Sequence_Model SHALL use a 2-layer GRU architecture with hidden_size=64, taking as input a Sliding_Window of 60 bars × F features (where F is the number of features from the Feature_Matrix).
3. THE PyTorch_Sequence_Model SHALL use the final hidden state of the GRU passed through a linear layer to produce the 2-value output (Ret5 prediction, Ret60 prediction).
4. THE Training_Pipeline SHALL save each trained PyTorch_Sequence_Model as a TorchScript (.pt) file using `torch.jit.script()` or `torch.jit.trace()` for portable inference without requiring the original model class definition.
5. THE Training_Pipeline SHALL use fixed random seeds and Deterministic_CUDA operations (`torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) to ensure reproducible training.
6. THE Training_Pipeline SHALL name PyTorch model files as `gru_{dataset_name}.pt` (e.g., `gru_dataset0.pt`) and save them alongside the LightGBM Model_Files.

### Requirement 17: GPU Inference in predict.py

**User Story:** As a competition participant, I want the Signal_Generator to load both LightGBM models and PyTorch_Sequence_Models and ensemble their predictions using learned Ensemble_Weights, so that the system fully utilizes the evaluation platform's RTX 4090 GPU for improved IC.

#### Acceptance Criteria

1. THE Signal_Generator SHALL load the LightGBM models (for CPU inference) and the PyTorch_Sequence_Model (for GPU_Inference) for each dataset inside the `generate_signals()` function body.
2. THE Signal_Generator SHALL run LightGBM inference on CPU and PyTorch inference on GPU (using `map_location="cuda"` when loading TorchScript models).
3. THE Signal_Generator SHALL compute the final ensemble prediction as: `final_pred = alpha * lgb_pred + (1 - alpha) * pytorch_pred`, where alpha is the per-dataset, per-target Ensemble_Weight loaded from the Ensemble_Config JSON file.
4. WHEN the Ensemble_Config file specifies alpha=1.0 for a given dataset and target, THE Signal_Generator SHALL output pure LightGBM predictions (PyTorch model effectively disabled).
5. IF the PyTorch_Sequence_Model file does not exist for a dataset, THEN THE Signal_Generator SHALL fall back to pure LightGBM predictions without raising an error.
6. IF a GPU out-of-memory error or CUDA error occurs during PyTorch inference, THEN THE Signal_Generator SHALL catch the exception, fall back to CPU inference for the PyTorch model (or pure LightGBM if CPU inference also fails), and log a warning.
7. THE Signal_Generator SHALL set PyTorch to evaluation mode (`model.eval()`) and use `torch.no_grad()` context during inference to minimize GPU memory usage.

### Requirement 18: Sliding Window Construction in predict.py

**User Story:** As a data scientist, I want the Signal_Generator to construct causal sliding windows from the Feature_Matrix for PyTorch input, so that the GRU model receives properly formatted sequential data without Look_Ahead_Bias.

#### Acceptance Criteria

1. THE Signal_Generator SHALL construct Sliding_Windows of size 60 bars (configurable via a constant) from the Feature_Matrix for each time index i.
2. THE Sliding_Window at index i SHALL contain feature vectors from indices `max(0, i - 59)` through `i` inclusive (causal, no Look_Ahead_Bias).
3. WHEN index i is less than the window size (i < 60), THE Signal_Generator SHALL pad the beginning of the Sliding_Window with zeros to maintain a fixed window shape of (60, F).
4. THE Signal_Generator SHALL construct all Sliding_Windows as a single batched tensor of shape (T, 60, F) for efficient GPU_Inference, or process in mini-batches if the full tensor exceeds available GPU memory.
5. THE Signal_Generator SHALL convert the Sliding_Window tensor to `torch.float32` dtype before passing to the PyTorch_Sequence_Model.

### Requirement 19: Training Pipeline for PyTorch Models

**User Story:** As a data scientist, I want the Training_Pipeline to train PyTorch_Sequence_Models alongside LightGBM models using a consistent temporal split and early stopping strategy, so that both model types are trained on the same data partitions and the PyTorch models are ready for ensemble.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train PyTorch_Sequence_Models after training LightGBM models for each dataset, using the same temporal 80/20 train/validation split.
2. THE Training_Pipeline SHALL train each PyTorch_Sequence_Model using MSE loss, Adam optimizer with learning_rate=1e-3, for a maximum of 20 epochs.
3. THE Training_Pipeline SHALL implement early stopping on validation IC_Metric: training SHALL stop if validation IC does not improve for 5 consecutive epochs.
4. THE Training_Pipeline SHALL construct Sliding_Windows from the Feature_Matrix for PyTorch training data, using the same causal window construction as specified in Requirement 18.
5. THE Training_Pipeline SHALL save the best-epoch model (highest validation IC) as a TorchScript file.
6. THE Training_Pipeline SHALL support CPU-only training (GPU training on local development machines is not required), but the saved TorchScript model SHALL be loadable on both CPU and GPU for inference.
7. WHEN validation IC of the PyTorch_Sequence_Model is negative or below 0.01 for a dataset, THE Training_Pipeline SHALL still save the model but SHALL set the Ensemble_Weight alpha to 1.0 (pure LightGBM) for that dataset in the Ensemble_Config.

### Requirement 20: Ensemble Weight Optimization

**User Story:** As a data scientist, I want the Training_Pipeline to compute optimal Ensemble_Weights on the validation set via grid search, so that the ensemble automatically selects the best blend of LightGBM and PyTorch predictions per-dataset and per-target.

#### Acceptance Criteria

1. WHEN both LightGBM and PyTorch_Sequence_Model training are complete for a dataset, THE Training_Pipeline SHALL compute optimal Ensemble_Weights by grid search over alpha values [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
2. THE Training_Pipeline SHALL evaluate each alpha value using the formula `final_pred = alpha * lgb_val_pred + (1 - alpha) * pytorch_val_pred` and select the alpha that maximizes validation IC_Metric.
3. THE Training_Pipeline SHALL compute and save separate Ensemble_Weights for Ret5 and Ret60 targets independently for each dataset.
4. THE Training_Pipeline SHALL save all Ensemble_Weights as a JSON Ensemble_Config file named `ensemble_weights.json` with the structure: `{"dataset0": {"ret5_alpha": 0.7, "ret60_alpha": 0.5}, "dataset1": {...}, ...}`.
5. WHEN alpha=1.0 is the optimal weight for a dataset-target pair, THE PyTorch_Sequence_Model SHALL be effectively disabled for that pair (safe fallback to pure LightGBM).
6. THE Training_Pipeline SHALL log the optimal alpha and the corresponding validation IC for each dataset-target pair.

### Requirement 21: Resource Budget Compliance

**User Story:** As a competition participant, I want the combined LightGBM + PyTorch ensemble system to stay within all evaluation platform resource limits, so that the submission completes successfully without timeout or resource exhaustion.

#### Acceptance Criteria

1. THE combined inference pipeline (Factor_Generator + Signal_Generator with ensemble) SHALL complete processing of all 30 datasets within 2 hours of wall-clock time on the evaluation platform (16 CPU cores, 96 GB RAM, RTX 4090 24GB).
2. THE estimated total inference time SHALL be approximately 18.5 minutes for all 30 datasets: Factor_Generator ~125 seconds (numba 16-core parallel) + LightGBM inference ~10 seconds (CPU) + Sliding_Window construction and GRU GPU_Inference ~974 seconds (batched, 65536 samples per batch on RTX 4090) + overhead ~3 seconds, leaving approximately 101 minutes of margin within the 2-hour budget.
3. THE PyTorch_Sequence_Model GPU memory usage SHALL not exceed 4 GB peak (model ~50 MB + batch inference ~2 GB), well within the RTX 4090's 24 GB capacity.
4. THE total submission package size (LightGBM Model_Files ~5 MB + PyTorch TorchScript files ~30 MB + Ensemble_Config + scripts) SHALL not exceed 200 MB.
5. THE Signal_Generator SHALL use the 16 CPU cores for numba-accelerated feature computation and LightGBM inference, and the RTX 4090 GPU exclusively for PyTorch batch inference.
6. IF per-dataset inference time exceeds 10 minutes, THEN THE Signal_Generator SHALL log a warning indicating the dataset may cause the overall pipeline to exceed the 2-hour budget.
