# Requirements Document

## Introduction

This system predicts 5-minute and 60-minute forward returns from 1-minute OHLCV candlestick data for 30 high-volatility financial instruments. It is built for the East China Cup Mathematical Modeling Competition (华东杯数学建模比赛, Problem C). The system must deliver strong prediction performance under both normal and extreme market conditions, measured by Pearson IC (Information Coefficient). The deliverables are two Python scripts (`factor.py` and `predict.py`) plus model files, packaged for stateless, reproducible execution on a constrained evaluation platform.

## Glossary

- **OHLCV_Data**: A numpy array of shape (T, 5) with columns [open, high, low, close, volume] representing 1-minute candlestick data for a single financial instrument. Prices are baseline-normalized.
- **Factor_Generator**: The `factor.py` module containing the `generate_factors()` function responsible for computing features from raw OHLCV data.
- **Signal_Generator**: The `predict.py` module containing the `generate_signals()` function responsible for producing return predictions from computed features.
- **Feature_Matrix**: A numpy array of shape (T, F) where F ≤ 512, containing computed features for each time bar. dtype is float32.
- **Prediction_Matrix**: A numpy array of shape (T, 2) where column 0 is the Ret5 prediction signal and column 1 is the Ret60 prediction signal. dtype is float32.
- **Ret5**: The 5-bar forward log return defined as log(close[i+5] / close[i]).
- **Ret60**: The 60-bar forward log return defined as log(close[i+60] / close[i]).
- **Pearson_IC**: The Pearson correlation coefficient between predicted signals and actual returns, computed per-dataset on normal and extreme subsets independently.
- **Extreme_Interval**: A closed index interval [start, end] marking a period of extreme market conditions within a dataset.
- **Normal_Interval**: All time bars in a dataset that do not fall within any Extreme_Interval.
- **Dataset_Name**: An anonymous string identifier (e.g., "dataset0" through "dataset29") used to identify each instrument's data.
- **Training_Pipeline**: The offline process that trains models and saves model parameter files for later use by the Signal_Generator.
- **Model_File**: Serialized model parameters saved to disk during training and loaded during inference. Total package size must not exceed 200 MB.
- **NaN_Value**: A floating-point Not-a-Number value indicating missing or invalid data.
- **Look_Ahead_Bias**: The prohibited use of future information (data at time i+k) when computing features or predictions at time i.

## Requirements

### Requirement 1: Feature Generation Interface

**User Story:** As the evaluation platform, I want the Factor_Generator to accept OHLCV_Data and return a Feature_Matrix, so that features can be computed in a standardized way for each dataset.

#### Acceptance Criteria

1. WHEN the evaluation platform calls `generate_factors(dataset_name, data)` with a Dataset_Name string and OHLCV_Data of shape (T, 5), THE Factor_Generator SHALL return a Feature_Matrix of shape (T, F) with dtype float32.
2. THE Factor_Generator SHALL produce a Feature_Matrix where F is at most 512.
3. WHEN the OHLCV_Data contains NaN_Value entries, THE Factor_Generator SHALL handle them gracefully without raising exceptions and SHALL produce valid float32 output (NaN_Value is permitted in the Feature_Matrix for positions that cannot be computed).
4. THE Factor_Generator SHALL complete execution within a time budget that allows the full 30-dataset pipeline to finish within 2 hours combined with the Signal_Generator.

### Requirement 2: Signal Generation Interface

**User Story:** As the evaluation platform, I want the Signal_Generator to accept a Feature_Matrix and return a Prediction_Matrix, so that return predictions can be produced in a standardized way for each dataset.

#### Acceptance Criteria

1. WHEN the evaluation platform calls `generate_signals(dataset_name, factors)` with a Dataset_Name string and a Feature_Matrix of shape (T, F), THE Signal_Generator SHALL return a Prediction_Matrix of shape (T, 2) with dtype float32.
2. THE Signal_Generator SHALL load Model_File artifacts from the `/workspace/submission/` directory inside the function body, not at module level.
3. WHEN the Feature_Matrix contains NaN_Value entries, THE Signal_Generator SHALL handle them gracefully without raising exceptions and SHALL produce finite float32 predictions wherever possible.
4. THE Signal_Generator SHALL complete execution within a time budget that allows the full 30-dataset pipeline to finish within 2 hours combined with the Factor_Generator.

### Requirement 3: Stateless Execution

**User Story:** As the evaluation platform, I want each dataset call to be independent, so that module reloading between datasets does not cause errors or stale state.

#### Acceptance Criteria

1. THE Factor_Generator SHALL not rely on global variables, caches, or any state that persists across calls to `generate_factors()` for different datasets.
2. THE Signal_Generator SHALL not rely on global variables, caches, or any state that persists across calls to `generate_signals()` for different datasets.
3. WHEN the Python interpreter reloads the `factor.py` module between dataset calls, THE Factor_Generator SHALL produce identical results for the same input data and Dataset_Name.
4. WHEN the Python interpreter reloads the `predict.py` module between dataset calls, THE Signal_Generator SHALL produce identical results for the same input factors and Dataset_Name.

### Requirement 4: Strict Reproducibility

**User Story:** As the competition organizer, I want all results to be strictly reproducible, so that submitted models can be verified independently.

#### Acceptance Criteria

1. THE Factor_Generator SHALL use hard-coded random seeds for all sources of randomness (NumPy, PyTorch, LightGBM, Python random) at the start of each `generate_factors()` call.
2. THE Signal_Generator SHALL use hard-coded random seeds for all sources of randomness (NumPy, PyTorch, LightGBM, Python random) at the start of each `generate_signals()` call.
3. THE Training_Pipeline SHALL use hard-coded random seeds for all sources of randomness and SHALL set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` when using PyTorch.
4. WHEN the same submission package is executed twice on the same evaluation platform, THE Factor_Generator and THE Signal_Generator SHALL produce bit-identical output for each dataset.

### Requirement 5: No Look-Ahead Bias

**User Story:** As the competition organizer, I want the system to use only past and present information when computing features and predictions, so that results reflect realistic forecasting conditions.

#### Acceptance Criteria

1. WHEN computing features at time index i, THE Factor_Generator SHALL use only OHLCV_Data from indices 0 through i (inclusive).
2. THE Factor_Generator SHALL not use Ret5 or Ret60 label columns as input features, as these contain future information.
3. WHEN computing predictions at time index i, THE Signal_Generator SHALL use only Feature_Matrix values from indices 0 through i (inclusive).
4. THE Factor_Generator SHALL not use any rolling window, moving average, or aggregation function that incorporates data from indices greater than i when computing the feature at index i.

### Requirement 6: Feature Engineering from OHLCV Data

**User Story:** As a data scientist, I want the Factor_Generator to compute informative features from raw OHLCV data, so that the prediction model has strong input signals for both short-term and long-term return forecasting.

#### Acceptance Criteria

1. THE Factor_Generator SHALL compute momentum-based features including price returns over multiple lookback windows (e.g., 5, 10, 20, 60, 120 bars).
2. THE Factor_Generator SHALL compute volatility-based features including rolling standard deviation of returns and high-low range measures over multiple lookback windows.
3. THE Factor_Generator SHALL compute volume-based features including volume moving averages, volume ratios, and volume-price interaction measures.
4. THE Factor_Generator SHALL compute microstructure features including bid-ask spread proxies (high-low relative to close), bar-level return skewness, and intrabar price movement patterns.
5. THE Factor_Generator SHALL compute technical indicator features using available libraries (TA-Lib, numpy, scipy) such as RSI, MACD, Bollinger Band width, and ATR over multiple lookback windows.
6. WHEN a lookback window extends beyond the available data at the start of a dataset, THE Factor_Generator SHALL fill the corresponding Feature_Matrix entries with NaN_Value.

### Requirement 7: Prediction Model Training

**User Story:** As a data scientist, I want a Training_Pipeline that trains models on the provided training data, so that the Signal_Generator can load pre-trained models for inference.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train separate models for Ret5 and Ret60 prediction targets.
2. THE Training_Pipeline SHALL split training data into train and validation subsets using a temporal split (earlier data for training, later data for validation) to avoid Look_Ahead_Bias.
3. THE Training_Pipeline SHALL save all Model_File artifacts such that the total submission package size does not exceed 200 MB.
4. THE Training_Pipeline SHALL handle datasets with significant NaN_Value entries in both OHLCV_Data and labels (particularly datasets 20-29) by applying appropriate imputation or exclusion strategies during training.
5. THE Training_Pipeline SHALL produce a training script with fixed random seeds that can be submitted alongside the inference scripts for reproducibility verification.

### Requirement 8: Normal Market Condition Prediction Performance

**User Story:** As a competition participant, I want the system to achieve high Pearson_IC on Normal_Interval subsets, so that the model performs well on the majority of market data.

#### Acceptance Criteria

1. WHEN evaluated on Normal_Interval subsets, THE Signal_Generator SHALL produce Ret5 predictions that achieve a positive mean Pearson_IC across the 30 datasets.
2. WHEN evaluated on Normal_Interval subsets, THE Signal_Generator SHALL produce Ret60 predictions that achieve a positive mean Pearson_IC across the 30 datasets.
3. THE Signal_Generator SHALL produce predictions that are not constant-valued across any dataset, as constant predictions yield NaN Pearson_IC.

### Requirement 9: Extreme Market Condition Prediction Performance

**User Story:** As a competition participant, I want the system to achieve high Pearson_IC on Extreme_Interval subsets, so that the model handles volatile market regimes effectively.

#### Acceptance Criteria

1. WHEN evaluated on Extreme_Interval subsets, THE Signal_Generator SHALL produce Ret5 predictions that achieve a positive mean Pearson_IC across the 30 datasets.
2. WHEN evaluated on Extreme_Interval subsets, THE Signal_Generator SHALL produce Ret60 predictions that achieve a positive mean Pearson_IC across the 30 datasets.
3. THE Training_Pipeline SHALL incorporate extreme market condition awareness, such as training with samples from Extreme_Interval regions or applying regime-specific modeling strategies.

### Requirement 10: Per-Dataset Independent Evaluation

**User Story:** As the evaluation platform, I want IC to be computed per-dataset independently, so that scale differences between instruments do not distort results.

#### Acceptance Criteria

1. THE Signal_Generator SHALL produce predictions that are calibrated per-dataset, without assuming cross-dataset signal scale consistency.
2. WHEN the evaluation platform computes Pearson_IC, predictions and labels from different datasets SHALL not be merged or compared across datasets.
3. THE Signal_Generator SHALL produce predictions for each dataset using only the Feature_Matrix from that same dataset (no cross-dataset information sharing at inference time).

### Requirement 11: Invalid Value Handling in Predictions

**User Story:** As the evaluation platform, I want predictions to contain as few NaN/Inf values as possible, so that IC computation is not degraded by zero-replacement of invalid predictions.

#### Acceptance Criteria

1. THE Signal_Generator SHALL produce finite float32 values for all time indices where the corresponding Feature_Matrix row contains at least one finite value.
2. IF the Signal_Generator cannot produce a meaningful prediction for a time index, THEN THE Signal_Generator SHALL output 0.0 rather than NaN_Value, since the evaluation platform replaces NaN with 0 anyway.
3. THE Signal_Generator SHALL not produce Inf or -Inf values in the Prediction_Matrix.

### Requirement 12: Submission Package Compliance

**User Story:** As a competition participant, I want the submission package to meet all platform requirements, so that it executes successfully on the evaluation server.

#### Acceptance Criteria

1. THE submission package SHALL contain a `factor.py` file with exactly one function: `generate_factors(dataset_name: str, data: np.ndarray) -> np.ndarray`.
2. THE submission package SHALL contain a `predict.py` file with exactly one function: `generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray`.
3. THE submission package SHALL be a single `.tar.gz` or `.zip` file not exceeding 200 MB.
4. THE submission package SHALL only import libraries from the approved list: torch 2.4.0, numpy, pandas, scikit-learn 1.8.0, scipy, bottleneck, numba, statsmodels, TA-Lib 0.6.8, lightgbm 4.6.0, transformers, polars.
5. THE submission package SHALL not contain code that probes the platform environment, accesses unauthorized resources, or performs network requests.

### Requirement 13: Runtime Resource Compliance

**User Story:** As the evaluation platform, I want the submission to operate within hardware resource limits, so that execution completes reliably.

#### Acceptance Criteria

1. THE Factor_Generator and Signal_Generator combined SHALL complete processing of all 30 datasets within 2 hours of wall-clock time.
2. THE Signal_Generator SHALL not allocate more than 24 GB of GPU memory when using PyTorch CUDA operations.
3. THE Factor_Generator and Signal_Generator SHALL not allocate more than 96 GB of system RAM combined at any point during execution.
4. IF GPU memory is insufficient for a model inference batch, THEN THE Signal_Generator SHALL fall back to smaller batch sizes or CPU inference rather than crashing.

### Requirement 14: Training Script Submission

**User Story:** As the competition organizer, I want a training script to be submitted alongside inference scripts, so that the model training process can be independently verified.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL be implemented as a standalone Python script that reads training data from the `train_dataset/` directory.
2. THE Training_Pipeline SHALL produce the same Model_File artifacts when executed with the same random seeds on the same data.
3. THE Training_Pipeline SHALL document all hyperparameters and random seeds used in training as code constants or configuration at the top of the script.
4. WHEN executed on the evaluation platform hardware (RTX 4090, 96 GB RAM, 16 CPU cores), THE Training_Pipeline SHALL complete within a reasonable time frame.
