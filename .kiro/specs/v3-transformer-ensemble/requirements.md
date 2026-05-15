# Requirements Document

## Introduction

This specification covers the v3 optimization iteration of the volatility return prediction system for the East China Cup Mathematical Modeling Competition (华东杯数学建模比赛, Problem C). The v2 system achieves in-sample mean IC scores of Normal×Ret5=0.2041, Normal×Ret60=0.3841, Extreme×Ret5=0.3315, Extreme×Ret60=0.5502 using 60 per-dataset LightGBM models, a 30-model GRU ensemble (effective for only 4/30 datasets), and 147 features. Current resource usage is 106.8 MB / 200 MB, ~6 min / 120 min, 147 / 512 features, with the RTX 4090 GPU almost idle.

Diagnosis has identified four root causes limiting further performance: (1) LightGBM Ret5 overfitting — many datasets converge to only 1–5 trees for Ret5 because each tree is too complex (num_leaves=127, unlimited depth), causing IC to peak early and drop; (2) GRU underperformance — only 4/30 datasets benefit from GRU due to CPU-only training with subsampling and an architecture too simple for this data; (3) GPU underutilization — the RTX 4090 remains almost idle, and a Transformer encoder model could leverage it for both training and inference; (4) no regime awareness — the model treats normal and extreme market conditions identically.

This iteration addresses five optimization areas in priority order:
1. **LightGBM hyperparameter tuning** (highest priority) — target-specific regularization and min_boost_round to fix Ret5 overfitting
2. **Transformer encoder model** (high priority) — GPU-accelerated sequence model for ensemble
3. **Feature selection** (optional) — gain-based top-100 feature selection per target
4. **Dynamic sample weighting** (optional) — volatility-based training weights
5. **Regime-aware feature augmentation** (optional, simplified) — train a regime classifier and add its output as an additional feature to LightGBM, rather than training separate regime-specialized models

## Glossary

- **Training_Pipeline**: The `train.py` script that trains per-dataset models offline and saves model files.
- **Factor_Generator**: The `factor.py` module containing the `generate_factors()` function responsible for computing features from raw OHLCV data.
- **Signal_Generator**: The `predict.py` module containing the `generate_signals()` function responsible for producing return predictions from computed features.
- **IC_Metric**: Pearson Information Coefficient — the Pearson correlation between predicted signals and actual returns, computed per-dataset on normal and extreme subsets independently.
- **Custom_IC_Feval**: A user-defined LightGBM evaluation function that computes Pearson IC on the validation set, used as the sole early stopping criterion.
- **Boosting_Round**: A single iteration of gradient boosting that adds one tree to the LightGBM ensemble.
- **Min_Boost_Round**: The minimum number of Boosting_Rounds that the Training_Pipeline trains before early stopping is allowed to trigger, preventing premature convergence to very few trees.
- **Two_Phase_Training**: A LightGBM training strategy where the first phase trains for Min_Boost_Round rounds without early stopping, and the second phase continues training with early stopping enabled using `init_model` to resume from the first phase's output.
- **Feature_Matrix**: A numpy array of shape (T, F) where F ≤ 512, containing computed features for each time bar, dtype float32.
- **Transformer_Encoder_Model**: A PyTorch Transformer encoder network trained per-dataset that takes sliding windows of OHLCV-derived features as input and outputs Ret5 and Ret60 predictions, capturing long-range dependencies via self-attention.
- **Sliding_Window**: A causal window of consecutive feature vectors (e.g., 60 bars × F features) used as input to sequence models, constructed without Look_Ahead_Bias.
- **TorchScript**: PyTorch's serialization format (.pt files) that saves models as portable, self-contained artifacts loadable without the original model class definition.
- **Ensemble_Weight**: Per-dataset, per-target scalars (alpha, beta, gamma) that control the weighted average among LightGBM, GRU, and Transformer predictions: `final = alpha * lgb_pred + beta * gru_pred + gamma * transformer_pred` where `alpha + beta + gamma = 1.0`.
- **Ensemble_Config**: A JSON file storing the learned Ensemble_Weights for all datasets and targets, loaded by the Signal_Generator at inference time.
- **Feature_Importance**: A gain-based ranking of features produced by LightGBM after training, used to select the most informative features and discard noisy ones.
- **Feature_Selection_Index**: A per-target list of column indices identifying the top-K most important features, saved as a JSON artifact and used during inference to subset the Feature_Matrix.
- **Dynamic_Sample_Weight**: A per-sample training weight computed as `1.0 + 0.5 * (volatility / rolling_vol_median)`, giving higher weight to high-volatility periods during LightGBM training.
- **Regime_Classifier**: A lightweight LightGBM binary classifier that predicts the probability of the current bar being in an extreme market regime, using only causal features (no future extreme_intervals labels).
- **Regime_Probability**: The output of the Regime_Classifier, a scalar in [0.0, 1.0] representing the estimated probability that the current bar is in an extreme regime, used as an additional input feature to the main LightGBM models.
- **Look_Ahead_Bias**: The prohibited use of future information (data at time i+k) when computing features or predictions at time i.
- **OHLCV_Data**: A numpy array of shape (T, 5) with columns [open, high, low, close, volume] representing 1-minute candlestick data.
- **Model_File**: Serialized model parameters saved as `.txt` (LightGBM), `.pt` (TorchScript), or `.json` (configuration) files, loaded by the Signal_Generator at inference time.
- **Baseline_IC**: The current v2 in-sample mean IC scores: Normal×Ret5=0.2041, Normal×Ret60=0.3841, Extreme×Ret5=0.3315, Extreme×Ret60=0.5502.
- **GPU_Inference**: Running PyTorch model forward passes on GPU (CUDA on evaluation platform, MPS on Apple Silicon locally) for accelerated batch prediction, with automatic CPU fallback on failure.
- **Deterministic_CUDA**: PyTorch configuration flags (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) that ensure reproducible GPU computation across runs. Applied only when CUDA is available.
- **GRU**: Gated Recurrent Unit — the existing recurrent neural network model from v2, retained in the ensemble alongside the new Transformer_Encoder_Model.

## Requirements

### Requirement 1: Target-Specific LightGBM Hyperparameters for Ret5

**User Story:** As a data scientist, I want the Training_Pipeline to use stronger regularization for Ret5 models compared to Ret60 models, so that Ret5 models train more trees with simpler splits instead of overfitting with a few complex trees.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL configure Ret5 LightGBM parameters with `num_leaves` of at most 63, `max_depth` of 8, `lambda_l1` of at least 0.5, `lambda_l2` of at least 5.0, and `feature_fraction` of at most 0.5.
2. THE Training_Pipeline SHALL configure Ret60 LightGBM parameters with `num_leaves` of at least 255, `max_depth` of -1 (unlimited), and the existing high-capacity settings from v2.
3. WHEN training a Ret5 model, THE Training_Pipeline SHALL use the Ret5-specific regularization parameters that are distinct from the Ret60 parameters.
4. WHEN training a Ret60 model, THE Training_Pipeline SHALL use the Ret60-specific high-capacity parameters that are distinct from the Ret5 parameters.

### Requirement 2: Minimum Boost Round with Two-Phase Training

**User Story:** As a data scientist, I want the Training_Pipeline to enforce a minimum number of Boosting_Rounds before early stopping can trigger, so that Ret5 models build at least 30 trees and avoid the "1-tree problem" caused by premature early stopping.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL implement Two_Phase_Training: the first phase trains for exactly Min_Boost_Round rounds (default 30) without early stopping, and the second phase continues training with early stopping enabled using `init_model` set to the first phase's output model.
2. WHEN the first phase completes, THE Training_Pipeline SHALL pass the resulting model as `init_model` to the second phase's `lgb.train()` call so that training resumes from where the first phase ended.
3. THE Training_Pipeline SHALL set Min_Boost_Round to 30 as the default value.
4. WHEN training completes for any Ret5 dataset, THE Training_Pipeline SHALL produce a model with at least Min_Boost_Round trees (30 or more).
5. THE Training_Pipeline SHALL apply Two_Phase_Training to both Ret5 and Ret60 targets, though the primary benefit is expected for Ret5.

### Requirement 3: Transformer Encoder Model Architecture

**User Story:** As a data scientist, I want the Training_Pipeline to train a Transformer encoder model per-dataset that captures long-range dependencies via self-attention, so that the ensemble can exploit sequential patterns that GRU and LightGBM cannot.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train one Transformer_Encoder_Model per dataset (30 models total), where each model outputs 2 values: a Ret5 prediction and a Ret60 prediction.
2. THE Transformer_Encoder_Model SHALL use a 4-layer TransformerEncoder architecture with `d_model=64`, `nhead=4`, and `dim_feedforward=256`.
3. THE Transformer_Encoder_Model SHALL take as input a Sliding_Window of 60 bars × F features (where F is the number of features from the Feature_Matrix), projected to `d_model=64` via a linear input layer.
4. THE Transformer_Encoder_Model SHALL use the last position's hidden state from the TransformerEncoder output, passed through a linear layer, to produce the 2-value output (Ret5 prediction, Ret60 prediction).
5. THE Transformer_Encoder_Model SHALL include a learnable positional encoding added to the input embeddings to provide sequence position information.
6. THE Training_Pipeline SHALL name Transformer model files as `transformer_{dataset_name}.pt` (e.g., `transformer_dataset0.pt`) and save them alongside the existing LightGBM and GRU Model_Files.

### Requirement 4: Transformer GPU Training

**User Story:** As a data scientist, I want the Transformer_Encoder_Model to be trained on GPU to fully utilize available GPU resources, so that training is fast enough to complete within the time budget and the model quality benefits from full-batch GPU computation.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train each Transformer_Encoder_Model on the best available GPU device: CUDA when available (evaluation platform RTX 4090), MPS when available (Apple Silicon local development), with CPU as final fallback.
2. THE Training_Pipeline SHALL train each Transformer_Encoder_Model using MSE loss and Adam optimizer with `learning_rate=1e-3` for a maximum of 30 epochs.
3. THE Training_Pipeline SHALL implement early stopping on validation IC_Metric for the Transformer: training SHALL stop if validation IC does not improve for 7 consecutive epochs.
4. THE Training_Pipeline SHALL construct Sliding_Windows from the Feature_Matrix for Transformer training data, using the same causal window construction as the existing GRU pipeline (60-bar windows, zero-padded for early indices, NaN replaced with 0.0).
5. THE Training_Pipeline SHALL save the best-epoch Transformer model (highest validation IC) as a TorchScript (.pt) file using `torch.jit.trace()` on a CPU-moved model for cross-device portability.
6. THE Training_Pipeline SHALL use fixed random seeds (`torch.manual_seed`, `torch.cuda.manual_seed_all`) for reproducible training. WHEN CUDA is available, THE Training_Pipeline SHALL additionally enable Deterministic_CUDA operations (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).

### Requirement 5: Three-Model Ensemble in predict.py

**User Story:** As a data scientist, I want the Signal_Generator to ensemble LightGBM, GRU, and Transformer predictions using learned per-dataset per-target weights, so that the system combines the strengths of all three model types.

#### Acceptance Criteria

1. THE Signal_Generator SHALL load LightGBM models (CPU inference), GRU TorchScript models (GPU inference), and Transformer TorchScript models (GPU inference) for each dataset inside the `generate_signals()` function body.
2. THE Signal_Generator SHALL compute the final ensemble prediction as: `final_pred = alpha * lgb_pred + beta * gru_pred + gamma * transformer_pred`, where alpha, beta, and gamma are per-dataset, per-target Ensemble_Weights loaded from the Ensemble_Config JSON file, and `alpha + beta + gamma = 1.0`.
3. WHEN the Ensemble_Config specifies `beta=0.0` and `gamma=0.0` for a given dataset and target, THE Signal_Generator SHALL output pure LightGBM predictions (GRU and Transformer effectively disabled).
4. IF a Transformer or GRU TorchScript file does not exist for a dataset, THEN THE Signal_Generator SHALL fall back to the available models or pure LightGBM, without raising an error.
5. IF a GPU out-of-memory error or CUDA error occurs during Transformer or GRU inference, THEN THE Signal_Generator SHALL catch the exception, attempt CPU inference, and fall back to LightGBM-only if CPU inference also fails.
6. THE Signal_Generator SHALL run Transformer and GRU inference in evaluation mode (`model.eval()`) with `torch.no_grad()` context to minimize GPU memory usage.

### Requirement 6: Three-Model Ensemble Weight Optimization

**User Story:** As a data scientist, I want the Training_Pipeline to compute optimal three-model Ensemble_Weights on the validation set, so that the ensemble automatically selects the best blend of LightGBM, GRU, and Transformer predictions per-dataset and per-target.

#### Acceptance Criteria

1. WHEN LightGBM, GRU, and Transformer training are all complete for a dataset, THE Training_Pipeline SHALL compute optimal Ensemble_Weights by grid search over (alpha, beta, gamma) triples where each value is in {0.0, 0.1, 0.2, ..., 1.0} and `alpha + beta + gamma = 1.0`.
2. THE Training_Pipeline SHALL evaluate each (alpha, beta, gamma) triple using the formula `final_pred = alpha * lgb_val_pred + beta * gru_val_pred + gamma * transformer_val_pred` and select the triple that maximizes validation IC_Metric.
3. THE Training_Pipeline SHALL compute and save separate Ensemble_Weights for Ret5 and Ret60 targets independently for each dataset.
4. THE Training_Pipeline SHALL save all Ensemble_Weights as a JSON Ensemble_Config file named `ensemble_weights.json` with the structure: `{"dataset0": {"ret5_alpha": 0.6, "ret5_beta": 0.1, "ret5_gamma": 0.3, "ret60_alpha": 0.4, "ret60_beta": 0.2, "ret60_gamma": 0.4}, ...}`.
5. WHEN the GRU model has negative or below-threshold validation IC for a dataset, THE Training_Pipeline SHALL constrain beta=0.0 for that dataset in the grid search (effectively disabling GRU).
6. THE Training_Pipeline SHALL log the optimal (alpha, beta, gamma) and the corresponding validation IC for each dataset-target pair.

### Requirement 7: Feature Importance Extraction (Optional — Area 3)

**User Story:** As a data scientist, I want the Training_Pipeline to extract gain-based Feature_Importance from trained LightGBM models, so that the most informative features can be identified for feature selection.

#### Acceptance Criteria

1. WHEN LightGBM training completes for all 30 datasets, THE Training_Pipeline SHALL extract gain-based Feature_Importance from each of the 60 trained models (30 Ret5 + 30 Ret60).
2. THE Training_Pipeline SHALL aggregate Feature_Importance across all 30 datasets for each target by summing the gain values per feature index.
3. THE Training_Pipeline SHALL rank features by aggregated importance separately for Ret5 and Ret60 targets.
4. THE Training_Pipeline SHALL log the top-20 most important features for each target with their aggregated gain values.

### Requirement 8: Top-K Feature Selection (Optional — Area 3)

**User Story:** As a data scientist, I want the Training_Pipeline to select the top-100 most important features per target and retrain LightGBM models using only those features, so that noisy features are removed and generalization improves.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL select the top-100 features by aggregated gain-based Feature_Importance for each target (Ret5 and Ret60), producing two potentially different Feature_Selection_Index lists.
2. THE Training_Pipeline SHALL retrain all 60 LightGBM models using only the selected features for each target, applying the same Two_Phase_Training and target-specific hyperparameters from Requirements 1 and 2.
3. THE Training_Pipeline SHALL save the Feature_Selection_Index lists as a JSON file named `feature_selection.json` with the structure: `{"ret5_features": [3, 7, 12, ...], "ret60_features": [1, 5, 9, ...]}`.
4. THE Signal_Generator SHALL load `feature_selection.json` and subset the Feature_Matrix to the selected columns before running LightGBM inference for each target.
5. WHEN `feature_selection.json` does not exist, THE Signal_Generator SHALL use the full Feature_Matrix for LightGBM inference (backward-compatible fallback).
6. THE Training_Pipeline SHALL compare validation IC before and after feature selection for each dataset-target pair, and log the difference.

### Requirement 9: Dynamic Sample Weighting for LightGBM Training (Optional — Area 4)

**User Story:** As a data scientist, I want the Training_Pipeline to apply Dynamic_Sample_Weights during LightGBM training that give higher weight to high-volatility periods, so that the model pays more attention to samples that correlate with extreme market conditions.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL compute Dynamic_Sample_Weight for each training sample as `1.0 + 0.5 * (volatility[i] / rolling_vol_median[i])`, where `volatility[i]` is the rolling standard deviation of 1-bar log returns over a 20-bar window at index i, and `rolling_vol_median[i]` is the rolling median of `volatility` over a 120-bar window at index i.
2. THE Training_Pipeline SHALL pass the computed Dynamic_Sample_Weights to LightGBM via the `weight` parameter of the `lgb.Dataset` constructor.
3. THE Training_Pipeline SHALL compute Dynamic_Sample_Weights using only causal data (no Look_Ahead_Bias): `volatility[i]` and `rolling_vol_median[i]` use only data from indices 0 through i.
4. WHEN `rolling_vol_median[i]` is zero or NaN, THE Training_Pipeline SHALL set the Dynamic_Sample_Weight at index i to 1.0 (default weight).
5. THE Training_Pipeline SHALL apply Dynamic_Sample_Weights only during LightGBM training, not during inference or validation IC computation.

### Requirement 10: Regime-Aware Feature Augmentation (Optional — Area 5, Simplified)

**User Story:** As a data scientist, I want the Training_Pipeline to train a Regime_Classifier and use its Regime_Probability output as an additional feature for the main LightGBM models, so that the models gain regime awareness without the complexity and storage cost of training separate regime-specialized models.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train one Regime_Classifier per dataset (30 classifiers total) using LightGBM with binary classification objective (`objective=binary`).
2. THE Regime_Classifier SHALL use a subset of existing features from the Feature_Matrix as input: rolling volatility, volume spike ratio, drawdown, and other regime-related features already computed in `factor.py`.
3. THE Regime_Classifier SHALL predict Regime_Probability: the probability that the current bar falls within an extreme_intervals range, using only features computed from data at indices 0 through i (no Look_Ahead_Bias).
4. THE Training_Pipeline SHALL construct binary labels for the Regime_Classifier from the `extreme_intervals` array: label=1 for bars within any extreme interval, label=0 otherwise. These labels are used only during offline training.
5. THE Training_Pipeline SHALL use lightweight hyperparameters for the Regime_Classifier (`num_leaves=31`, `max_depth=6`, `n_estimators=200`) to keep model files small.
6. THE Training_Pipeline SHALL save each Regime_Classifier as a LightGBM text file named `regime_{dataset_name}.txt`.
7. THE Signal_Generator SHALL load the Regime_Classifier for each dataset, compute Regime_Probability for every bar, and append the Regime_Probability as an additional column to the Feature_Matrix before running the main LightGBM inference.
8. WHEN `regime_{dataset_name}.txt` does not exist, THE Signal_Generator SHALL skip regime feature augmentation and use the original Feature_Matrix (backward-compatible fallback).

### Requirement 11: Preserve Public Interfaces

**User Story:** As a competition participant, I want the public interfaces of `factor.py` and `predict.py` to remain identical, so that the submission is compatible with the evaluation platform.

#### Acceptance Criteria

1. THE Factor_Generator SHALL maintain the function signature `generate_factors(dataset_name: str, data: np.ndarray) -> np.ndarray` without any changes to parameter names, types, or return type.
2. THE Signal_Generator SHALL maintain the function signature `generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray` without any changes to parameter names, types, or return type.
3. THE Factor_Generator SHALL continue to return a float32 numpy array of shape (T, F) where F ≤ 512.
4. THE Signal_Generator SHALL continue to return a float32 numpy array of shape (T, 2) with all finite values (no NaN, no Inf).

### Requirement 12: Resource Budget Compliance

**User Story:** As a competition participant, I want the combined optimized system to stay within all evaluation platform resource limits, so that the submission completes successfully without timeout or resource exhaustion.

#### Acceptance Criteria

1. THE combined inference pipeline (Factor_Generator + Signal_Generator with three-model ensemble and optional regime feature augmentation) SHALL complete processing of all 30 datasets within 2 hours of wall-clock time on the evaluation platform (16 CPU cores, 96 GB RAM, RTX 4090 24GB).
2. THE total submission package size (all LightGBM Model_Files + GRU TorchScript files + Transformer TorchScript files + optional Regime_Classifier files + configuration JSONs + scripts) SHALL not exceed 200 MB.
3. THE Transformer_Encoder_Model GPU memory usage SHALL not exceed 6 GB peak per dataset (model + batch inference), well within the RTX 4090's 24 GB capacity.
4. THE Signal_Generator SHALL process Transformer and GRU inference in mini-batches (batch_size ≤ 65536) to prevent GPU out-of-memory errors on large datasets.
5. IF per-dataset inference time exceeds 10 minutes, THEN THE Signal_Generator SHALL log a warning indicating the dataset may cause the overall pipeline to exceed the 2-hour budget.

### Requirement 13: IC Performance Improvement

**User Story:** As a competition participant, I want the v3 optimized system to achieve higher IC_Metric scores than the Baseline_IC across all four evaluation categories, so that the competition ranking improves.

#### Acceptance Criteria

1. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Normal×Ret5 IC greater than the Baseline_IC of 0.2041.
2. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Normal×Ret60 IC greater than the Baseline_IC of 0.3841.
3. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Extreme×Ret5 IC greater than the Baseline_IC of 0.3315.
4. WHEN evaluated on the training data using the same evaluation methodology as `evaluate_local.py`, THE optimized system SHALL achieve a mean Extreme×Ret60 IC greater than the Baseline_IC of 0.5502.
5. THE LightGBM hyperparameter tuning (Requirements 1–2) SHALL specifically improve Ret5 IC by enabling models to train more trees with stronger regularization, addressing the diagnosed Ret5 overfitting problem where 6/30 datasets had only 1–5 trees.
6. THE Transformer_Encoder_Model ensemble (Requirements 3–6) SHALL provide additional IC improvement beyond the LightGBM-only and GRU-only optimizations, especially for datasets where GRU was previously ineffective (26/30 datasets).

### Requirement 14: No Look-Ahead Bias

**User Story:** As a data scientist, I want all new model components and training procedures to be strictly causal, so that the system does not use future information during inference.

#### Acceptance Criteria

1. THE Transformer_Encoder_Model SHALL receive only causal Sliding_Windows as input: the window at index i contains only data from indices max(0, i-59) through i.
2. THE Dynamic_Sample_Weight computation SHALL use only rolling statistics computed from data at indices 0 through i for each sample i.
3. THE Regime_Classifier SHALL predict Regime_Probability using only features computed from data at indices 0 through i, without access to future extreme_intervals labels at inference time.
4. THE Feature_Selection_Index SHALL be computed from training data only, not from validation or test data.

### Requirement 15: Reproducibility

**User Story:** As a competition participant, I want all training and inference to be reproducible given the same random seeds and input data, so that results can be verified and debugged.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL set fixed random seeds (seed=42) for all random number generators: Python `random`, numpy `np.random`, PyTorch `torch.manual_seed`, and CUDA `torch.cuda.manual_seed_all`.
2. WHEN CUDA is available, THE Training_Pipeline SHALL enable Deterministic_CUDA operations for Transformer and GRU training via `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
3. THE Training_Pipeline SHALL retain fixed `seed` parameters in all LightGBM configurations.
4. THE Signal_Generator SHALL call `_set_seeds(42)` at the entry of `generate_signals()` to ensure deterministic inference across module reloads.

### Requirement 16: Version Control and Traceability

**User Story:** As a developer, I want every code change to be committed and pushed to GitHub, so that the optimization history is traceable and the team can review changes.

#### Acceptance Criteria

1. WHEN a code change is made to `factor.py`, `train.py`, or `predict.py`, THE developer SHALL commit the change to a Git branch with a descriptive commit message.
2. THE developer SHALL push all commits to the GitHub remote repository.
3. THE developer SHALL not commit directly to the main branch without creating a separate branch for the optimization work.

### Requirement 17: Backward Compatibility and Safe Fallbacks

**User Story:** As a competition participant, I want the v3 system to gracefully degrade when optional model files are missing, so that a partial submission still produces valid predictions.

#### Acceptance Criteria

1. WHEN `transformer_{dataset_name}.pt` does not exist, THE Signal_Generator SHALL fall back to the two-model ensemble (LightGBM + GRU) or pure LightGBM without raising an error.
2. WHEN `feature_selection.json` does not exist, THE Signal_Generator SHALL use the full Feature_Matrix for all model inference.
3. WHEN `regime_{dataset_name}.txt` does not exist, THE Signal_Generator SHALL skip regime feature augmentation and use the original Feature_Matrix.
4. WHEN `ensemble_weights.json` does not exist, THE Signal_Generator SHALL output pure LightGBM predictions.
5. THE Signal_Generator SHALL catch all exceptions from PyTorch model loading and inference, falling back to LightGBM-only predictions on any failure.
6. THE Signal_Generator SHALL produce a valid (T, 2) float32 output with all finite values regardless of which optional components are available.

### Requirement 18: Platform Library Compliance

**User Story:** As a competition participant, I want all code to use only libraries available on the evaluation platform, so that the submission does not fail with import errors.

#### Acceptance Criteria

1. THE Factor_Generator SHALL use only approved libraries: numpy, numba, and Python standard library.
2. THE Signal_Generator SHALL use only approved libraries: numpy, lightgbm, torch, and Python standard library (including json and pathlib).
3. THE Training_Pipeline SHALL use only approved libraries: numpy, lightgbm, torch, numba, and Python standard library.
4. THE Transformer_Encoder_Model SHALL use only `torch.nn` modules available in PyTorch 2.4.0 (TransformerEncoder, TransformerEncoderLayer, Linear, LayerNorm).
