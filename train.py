# Version: v7 — 9-source ensemble, greedy hill climbing
#!/usr/bin/env python3
"""
Training pipeline for the Volatility Return Prediction system.

Trains per-dataset, per-target LightGBM models for Ret5 and Ret60 prediction.
Produces 60 model files: lgb_ret5_dataset{i}.txt and lgb_ret60_dataset{i}.txt

Usage:
    python train.py [--data-dir train_dataset] [--output-dir models]

All hyperparameters and random seeds are documented as constants below.
"""

import os
import sys
import time
import json
import random
import argparse
import gzip
import shutil
import math
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from pathlib import Path

# Import our feature generator
from factor import generate_factors


# =============================================================================
# Configuration constants
# =============================================================================

RANDOM_SEED = 42
NUM_DATASETS = 30
TRAIN_RATIO = 0.8                # Temporal 80/20 split
MIN_VALID_SAMPLES = 5000         # High-NaN safeguard threshold
EARLY_STOPPING_ROUNDS = 100      # Increased from 50 — IC needs more rounds to plateau
FALLBACK_NUM_BOOST_ROUND = 50    # Increased from 20 for better fallback models

# v3 Ret5: stronger regularization to fix overfitting
LGB_PARAMS_RET5 = {
    "objective": "regression",
    "metric": "None",             # Disable built-in MAE — use IC feval only
    "boosting_type": "gbdt",
    "num_leaves": 63,             # Reduced from 127 — simpler trees to prevent overfitting
    "max_depth": 8,               # Explicit depth limit (was -1)
    "learning_rate": 0.03,
    "feature_fraction": 0.5,      # Reduced from 0.7 — more feature subsampling
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 200,
    "lambda_l1": 0.5,             # Increased from 0.1 — stronger L1 regularization
    "lambda_l2": 5.0,             # Increased from 1.0 — stronger L2 regularization
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": -1,            # Multi-threaded (auto-detect cores)
}

# v3 Ret60: high capacity (smoother target, less overfitting risk)
LGB_PARAMS_RET60 = {
    "objective": "regression",
    "metric": "None",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "max_depth": -1,              # Unlimited depth
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 200,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": -1,
}

MIN_BOOST_ROUND = 30              # Minimum boosting rounds before early stopping

NUM_BOOST_ROUND_RET5 = 1000      # Increased from 500
NUM_BOOST_ROUND_RET60 = 1500     # Increased from 800

# GRU model hyperparameters — v6: h=128 for more capacity
GRU_HIDDEN_SIZE = 128            # v6: doubled from 64 for more capacity (~1MB/model)
GRU_NUM_LAYERS = 2               # Keep 2 layers (file size control)
GRU_DROPOUT = 0.1                # Keep original
GRU_WINDOW_SIZE = 60             # Sliding window: 60 bars of history
GRU_BATCH_SIZE = 16384           # H20: 4x larger batch for faster training (训练专用)
GRU_LR = 1e-3                   # Keep original (模型不变，不需要 scaling)
GRU_EPOCHS = 30                  # H20: more epochs (was 20), full data needs more
GRU_PATIENCE = 7                 # H20: more patience (was 5) for full data convergence
GRU_MIN_IC_THRESHOLD = 0.01     # Below this, set alpha=1.0 (pure LightGBM)
GRU_MAX_TRAIN_SAMPLES = None     # H20 96GB: no subsampling, use all training data
GRU_MAX_VAL_SAMPLES = None       # H20 96GB: no validation subsampling

# Transformer model hyperparameters — H20 96GB: same architecture, better training
TRANSFORMER_D_MODEL = 64         # Keep original (推理时间/模型大小不变)
TRANSFORMER_NHEAD = 4            # Keep original
TRANSFORMER_NUM_LAYERS = 4       # Keep original
TRANSFORMER_DIM_FF = 256         # Keep original
TRANSFORMER_DROPOUT = 0.1        # Keep original
TRANSFORMER_WINDOW_SIZE = 60
TRANSFORMER_BATCH_SIZE = 16384   # H20: 4x larger batch for faster training (训练专用)
TRANSFORMER_LR = 1e-3            # Keep original
TRANSFORMER_EPOCHS = 40          # H20: more epochs (was 30), full data needs more
TRANSFORMER_PATIENCE = 10        # H20: more patience (was 7) for full data convergence
TRANSFORMER_MIN_IC_THRESHOLD = 0.01
TRANSFORMER_MAX_TRAIN_SAMPLES = None     # H20 96GB: no subsampling, use all training data
TRANSFORMER_MAX_VAL_SAMPLES = None       # H20 96GB: no validation subsampling

# v6: IC-aware loss configuration
LOSS_TYPE = "ic_aware"           # "mse" or "ic_aware" — switch to "mse" to revert to v5 behavior
IC_LOSS_ALPHA = 0.5              # Weight: α * Pearson_Loss + (1-α) * ListMLE_Loss
IC_LOSS_MIN_SAMPLES = 32        # Minimum valid samples per batch for IC loss
GRAD_CLIP_MAX_NORM = 1.0        # Gradient clipping (IC loss has volatile gradients)
# When LOSS_TYPE="ic_aware", learning rates are halved automatically (see train functions)

# v6: Dual window configuration
WINDOW_RET5 = 20                 # Short window for Ret5 (captures micro-structure)
WINDOW_RET60 = 240               # Long window for Ret60 (captures trends)
BATCH_SIZE_W20 = 32768           # Larger batch for short window (less memory)
BATCH_SIZE_W240 = 4096           # Reduced: w=240 + h=128 GRU uses ~60GB at batch=16384


# =============================================================================
# Seed initialization
# =============================================================================

def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set all random seeds for strict reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# v6: IC-aware Loss Functions
# =============================================================================

def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor = None) -> torch.Tensor:
    """
    Pearson correlation loss: 1 - pearson_corr(pred, target).
    
    Directly optimizes IC (the evaluation metric).
    Returns 0.0 when valid samples < IC_LOSS_MIN_SAMPLES or target variance is zero.
    
    Args:
        pred: (batch,) or (batch, 1) predicted values
        target: (batch,) or (batch, 1) target values
        mask: (batch,) bool tensor, True = valid (non-NaN). None = all valid.
    """
    pred = pred.flatten()
    target = target.flatten()
    
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    n = pred.shape[0]
    if n < IC_LOSS_MIN_SAMPLES:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Center
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    
    # Variance check (avoid division by zero)
    pred_var = (pred_centered ** 2).sum()
    target_var = (target_centered ** 2).sum()
    
    if target_var.item() < 1e-8 or pred_var.item() < 1e-8:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Pearson correlation
    corr = (pred_centered * target_centered).sum() / (pred_var.sqrt() * target_var.sqrt())
    
    # Loss = 1 - correlation (minimize to maximize IC)
    return 1.0 - corr


def listmle_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor = None) -> torch.Tensor:
    """
    ListMLE ranking loss based on Plackett-Luce probability model.
    
    Optimizes the ranking consistency between predictions and targets.
    Uses log-sum-exp trick for numerical stability.
    Returns 0.0 when valid samples < IC_LOSS_MIN_SAMPLES.
    
    Args:
        pred: (batch,) predicted scores
        target: (batch,) true values (used for ranking order)
        mask: (batch,) bool tensor, True = valid. None = all valid.
    """
    pred = pred.flatten()
    target = target.flatten()
    
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    n = pred.shape[0]
    if n < IC_LOSS_MIN_SAMPLES:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # Sort by target descending (ground truth ranking)
    _, indices = target.sort(descending=True)
    pred_sorted = pred[indices]
    
    # ListMLE: -sum_i [pred_π(i) - log(sum_{j>=i} exp(pred_π(j)))]
    # Use log-sum-exp trick for stability
    n_items = pred_sorted.shape[0]
    
    # Compute cumulative log-sum-exp from the end
    # log_cum_sum[i] = log(sum_{j>=i} exp(pred_sorted[j]))
    # We compute this in reverse using the log-sum-exp trick
    max_val = pred_sorted.max()
    pred_shifted = pred_sorted - max_val  # shift for stability
    
    # Reverse cumulative sum of exp
    exp_pred = pred_shifted.exp()
    cum_sum_rev = exp_pred.flip(0).cumsum(0).flip(0)
    log_cum_sum = cum_sum_rev.log() + max_val
    
    # Loss = -mean(pred_sorted - log_cum_sum)
    loss = -(pred_sorted - log_cum_sum).mean()
    
    return loss


def ic_aware_loss(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor = None, alpha: float = IC_LOSS_ALPHA) -> torch.Tensor:
    """
    Combined IC-aware loss: α * Pearson_Loss + (1-α) * ListMLE_Loss.
    
    Args:
        pred: (batch,) or (batch, 1) predictions
        target: (batch,) or (batch, 1) targets
        mask: (batch,) bool, valid samples. None = all valid.
        alpha: weight between Pearson (correlation) and ListMLE (ranking)
    """
    try:
        p_loss = pearson_correlation_loss(pred, target, mask)
        l_loss = listmle_loss(pred, target, mask)
        return alpha * p_loss + (1.0 - alpha) * l_loss
    except RuntimeError as e:
        # CUDA errors (illegal memory access, etc.) — return zero loss to skip batch
        if "CUDA" in str(e) or "illegal" in str(e):
            print(f"    [WARN] IC-aware loss CUDA error, skipping batch: {e}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        raise

def save_lgb_model_gz(booster: lgb.Booster, path_gz: Path) -> None:
    """Save LightGBM model as gzip-compressed .txt.gz file.

    Saves to a temporary .txt file first, then compresses to .txt.gz,
    then removes the temporary file.
    """
    path_gz = Path(path_gz)
    path_txt_tmp = path_gz.with_suffix('')  # removes .gz → .txt
    booster.save_model(str(path_txt_tmp))
    with open(path_txt_tmp, 'rb') as f_in:
        with gzip.open(str(path_gz), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    path_txt_tmp.unlink()  # remove temp .txt


# =============================================================================
# v6: Causal extreme regime detection
# =============================================================================

def detect_extreme_regime(close_prices: np.ndarray, window: int = 60,
                          threshold_mult: float = 2.0) -> np.ndarray:
    """
    Causal extreme regime detection. Only uses data at or before time t.
    
    Algorithm:
    1. Compute close-to-close log returns
    2. Compute rolling_std(return, window) using only historical data
    3. Mark |current_return| > threshold_mult * rolling_std as extreme
    
    Args:
        close_prices: (T,) array of close prices
        window: rolling window size for std calculation (default 60)
        threshold_mult: multiplier for threshold (default 2.0)
    
    Returns:
        (T,) bool array, True = extreme regime at that time step
    """
    T = len(close_prices)
    close = np.asarray(close_prices, dtype=np.float64)
    
    # Compute log returns (causal: ret[t] = log(close[t] / close[t-1]))
    log_returns = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        if close[i - 1] > 0 and close[i] > 0:
            log_returns[i] = np.log(close[i] / close[i - 1])
    
    # Compute rolling std (causal: uses returns[t-window+1:t+1], i.e. up to and including t)
    # For t < window, we use all available data up to t
    extreme_mask = np.zeros(T, dtype=np.bool_)
    
    for t in range(1, T):
        # Use returns from max(1, t-window+1) to t (inclusive) for rolling std
        start = max(1, t - window + 1)
        window_returns = log_returns[start:t + 1]
        
        if len(window_returns) < 2:
            continue
        
        rolling_std = np.std(window_returns, ddof=0)
        
        if rolling_std > 0:
            threshold = threshold_mult * rolling_std
            if abs(log_returns[t]) > threshold:
                extreme_mask[t] = True
    
    return extreme_mask


# =============================================================================
# Task 6.1: Volatility-based sampling weights
# =============================================================================

def _compute_volatility_weights(close_prices: np.ndarray, window_vol: int = 20,
                                  window_median: int = 120) -> np.ndarray:
    """
    Compute volatility-based sampling weights for sequence model training.

    w_i = σ_20(i) / Median_120(σ_20)(i)
    NaN or zero values replaced with 1.0.
    All weights are positive.
    """
    T = len(close_prices)
    close = np.asarray(close_prices, dtype=np.float64)

    # 1-bar log returns (causal)
    log_returns = np.empty(T, dtype=np.float64)
    log_returns[0] = 0.0
    for i in range(1, T):
        prev = close[i - 1]
        curr = close[i]
        if prev > 0 and curr > 0:
            log_returns[i] = np.log(curr / prev)
        else:
            log_returns[i] = 0.0

    # 20-bar rolling std (volatility)
    volatility = np.full(T, np.nan, dtype=np.float64)
    for i in range(window_vol - 1, T):
        volatility[i] = np.std(log_returns[i - window_vol + 1:i + 1])

    # 120-bar rolling median of volatility
    rolling_median = np.full(T, np.nan, dtype=np.float64)
    for i in range(window_median - 1, T):
        rolling_median[i] = np.median(volatility[max(0, i - window_median + 1):i + 1])

    # Compute weights: w_i = σ_20(i) / Median_120(σ_20)(i)
    weights = np.ones(T, dtype=np.float32)
    for i in range(T):
        v = volatility[i]
        m = rolling_median[i]
        if np.isnan(v) or np.isnan(m) or m <= 0 or v != v:
            weights[i] = 1.0
        else:
            w = v / m
            weights[i] = float(w) if w > 0 else 1.0

    return weights


# =============================================================================
# Task 7.1: Adaptive max samples based on GPU VRAM
# =============================================================================

def _compute_adaptive_max_samples(train_set_size: int, feature_dim: int = 165,
                                    window_size: int = 60, batch_size: int = 4096) -> int:
    """
    Compute adaptive max training samples based on available GPU VRAM.

    H20 96GB mode: always return full train_set_size (no sampling).
    On-the-fly batch-wise window construction keeps memory manageable.

    Returns train_set_size unconditionally — OOM retry logic is the safety net.
    """
    # H20 96GB: use all training data, no sampling
    # The batch-wise sliding window construction (train_sequence_model_batched)
    # ensures memory stays at single-batch level regardless of total samples.
    # OOM retry logic in train_all_models handles edge cases.
    if torch.cuda.is_available():
        try:
            avail_vram, total_vram = torch.cuda.mem_get_info()
            print(f"    [VRAM] Available: {avail_vram/1024**3:.1f} GB / Total: {total_vram/1024**3:.1f} GB")
        except Exception:
            print(f"    [VRAM] Detection failed, proceeding with full data")
    return train_set_size


# =============================================================================
# Custom IC evaluation metric for early stopping
# =============================================================================

def ic_eval_metric(preds, train_data):
    """
    Custom LightGBM evaluation metric: Pearson IC.
    Returns (metric_name, metric_value, is_higher_better).
    """
    labels = train_data.get_label()
    # Clean NaN/Inf
    p = np.nan_to_num(preds.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(labels.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    p = p - p.mean()
    y = y - y.mean()
    denom = np.sqrt((p ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        ic = 0.0
    else:
        ic = float((p * y).sum() / denom)

    return "ic", ic, True


# =============================================================================
# Training logic
# =============================================================================

def load_dataset(data_dir: str, dataset_idx: int):
    """
    Load a single dataset's OHLCV data, labels, and extreme intervals.

    Returns:
        ohlcv: np.ndarray of shape (T, 5), dtype float32 — [open, high, low, close, volume]
        ret5:  np.ndarray of shape (T,), dtype float32 — 5-bar forward log return
        ret60: np.ndarray of shape (T,), dtype float32 — 60-bar forward log return
        indices: np.ndarray of shape (T,) — row index column for extreme interval matching
        extreme_intervals: np.ndarray of shape (N, 2), dtype int64 — [start, end] closed intervals
    """
    dataset_name = f"dataset{dataset_idx}"
    ohlcv_path = os.path.join(data_dir, f"{dataset_name}_train_ohlcv.npy")
    ext_path = os.path.join(data_dir, f"{dataset_name}_train_extreme_intervals.npy")

    raw = np.load(ohlcv_path)
    extreme_intervals = np.load(ext_path)

    # raw shape: (T, 8) — [index, open, high, low, close, volume, Ret5, Ret60]
    indices = raw[:, 0]
    ohlcv = raw[:, 1:6].astype(np.float32)
    ret5 = raw[:, 6].astype(np.float32)
    ret60 = raw[:, 7].astype(np.float32)
    return ohlcv, ret5, ret60, indices, extreme_intervals


def build_extreme_mask(indices: np.ndarray, extreme_intervals: np.ndarray) -> np.ndarray:
    """Build boolean mask: True for rows within any extreme interval."""
    T = indices.shape[0]
    mask = np.zeros(T, dtype=np.bool_)
    if extreme_intervals.shape[0] == 0:
        return mask
    idx = indices.astype(np.int64)
    for k in range(extreme_intervals.shape[0]):
        start = int(extreme_intervals[k, 0])
        end = int(extreme_intervals[k, 1])
        mask |= (idx >= start) & (idx <= end)
    return mask


def train_lgb_two_phase(params, train_data, val_data, max_boost_round,
                         min_boost_round=MIN_BOOST_ROUND):
    """
    Two-phase LightGBM training:
      Phase 1: Train for min_boost_round rounds WITHOUT early stopping.
      Phase 2: Continue from Phase 1 model WITH IC-based early stopping.
    """
    # Phase 1: forced minimum rounds, no early stopping
    phase1_model = lgb.train(
        params,
        train_data,
        num_boost_round=min_boost_round,
        valid_sets=[val_data],
        valid_names=["val"],
        feval=ic_eval_metric,
        callbacks=[lgb.log_evaluation(period=50)],
    )

    # Phase 2: continue with early stopping, using init_model
    remaining_rounds = max_boost_round - min_boost_round
    if remaining_rounds <= 0:
        return phase1_model

    phase2_model = lgb.train(
        params,
        train_data,
        num_boost_round=remaining_rounds,
        init_model=phase1_model,
        valid_sets=[val_data],
        valid_names=["val"],
        feval=ic_eval_metric,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    return phase2_model


def train_single_model(
    features: np.ndarray,
    labels: np.ndarray,
    params: dict,
    num_boost_round: int,
    dataset_name: str,
    target_name: str,
) -> lgb.Booster:
    """
    Train a single LightGBM model with temporal split and early stopping.

    Args:
        features: (T, F) feature matrix
        labels: (T,) target labels
        params: LightGBM parameters
        num_boost_round: max boosting rounds
        dataset_name: for logging
        target_name: "ret5" or "ret60"

    Returns:
        Trained LightGBM Booster
    """
    T = features.shape[0]

    # Build valid mask: rows where label is not NaN
    valid_mask = ~np.isnan(labels)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    print(f"    {target_name}: {n_valid} valid samples out of {T} total")

    # --- High-NaN safeguard ---
    if n_valid < MIN_VALID_SAMPLES:
        print(f"    WARNING: Only {n_valid} valid samples (< {MIN_VALID_SAMPLES}). "
              f"Using fallback minimal model.")
        X_train = features[valid_indices]
        y_train = labels[valid_indices]
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        fallback_params = {**params, "num_leaves": 8, "learning_rate": 0.1}
        model = lgb.train(
            fallback_params,
            train_data,
            num_boost_round=FALLBACK_NUM_BOOST_ROUND,
        )
        print(f"    Fallback model trained: {model.num_trees()} trees")
        return model

    # --- Normal training with temporal split ---
    split_idx = int(n_valid * TRAIN_RATIO)
    train_indices = valid_indices[:split_idx]
    val_indices = valid_indices[split_idx:]

    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_val = features[val_indices]
    y_val = labels[val_indices]

    print(f"    Train: {len(train_indices)}, Val: {len(val_indices)}")

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

    model = train_lgb_two_phase(params, train_data, val_data, num_boost_round)

    # Report validation IC
    val_preds = model.predict(X_val)
    _, val_ic, _ = ic_eval_metric(val_preds, val_data)
    print(f"    Best iteration: {model.best_iteration}, Trees: {model.num_trees()}, Val IC: {val_ic:.6f}")

    return model


# =============================================================================
# PyTorch GRU model (Phase 3)
# =============================================================================

class GRUPredictor(nn.Module):
    """Lightweight GRU for sequence-based return prediction."""

    def __init__(self, input_size: int, hidden_size: int = GRU_HIDDEN_SIZE,
                 num_layers: int = GRU_NUM_LAYERS, dropout: float = GRU_DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)  # Output: [ret5_pred, ret60_pred]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]              # (batch, hidden) — last layer's final state
        return self.fc(h_last)         # (batch, 2)


# =============================================================================
# v6: Single-target models for dual-window strategy
# =============================================================================

class GRUSingleTarget(nn.Module):
    """v6: GRU that predicts a single target (Ret5 OR Ret60), not both."""

    def __init__(self, input_size: int, hidden_size: int = GRU_HIDDEN_SIZE,
                 num_layers: int = GRU_NUM_LAYERS, dropout: float = GRU_DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)  # Single output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        h_last = h_n[-1]
        return self.fc(h_last).squeeze(-1)  # (batch,)


class TransformerSingleTarget(nn.Module):
    """v6: Transformer that predicts a single target (Ret5 OR Ret60)."""

    def __init__(self, input_size: int, d_model: int = TRANSFORMER_D_MODEL,
                 nhead: int = TRANSFORMER_NHEAD, num_layers: int = TRANSFORMER_NUM_LAYERS,
                 dim_feedforward: int = TRANSFORMER_DIM_FF, dropout: float = TRANSFORMER_DROPOUT,
                 max_seq_len: int = 60):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )
        self.output_head = nn.Linear(d_model, 1)  # Single output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :h.size(1)]
        h = self.encoder(h)
        return self.output_head(h[:, -1, :]).squeeze(-1)  # (batch,)


def build_sliding_windows_for_indices(features: np.ndarray, indices: np.ndarray,
                                       window_size: int = GRU_WINDOW_SIZE) -> np.ndarray:
    """
    Build causal sliding windows for a subset of indices.
    Window at index i = features[max(0, i-window_size+1) : i+1], zero-padded if needed.
    NaN replaced with 0.0 for GRU input.

    Args:
        features: (T, F) full feature matrix
        indices: 1D array of row indices to build windows for
        window_size: number of bars per window

    Returns:
        (len(indices), window_size, F) float32 array
    """
    T, F = features.shape
    clean = np.nan_to_num(features, nan=0.0).astype(np.float32)

    # Pad beginning with zeros
    padded = np.zeros((window_size - 1 + T, F), dtype=np.float32)
    padded[window_size - 1:] = clean

    n = len(indices)
    windows = np.empty((n, window_size, F), dtype=np.float32)
    for k in range(n):
        i = indices[k]
        start = i  # in padded coordinates (i + window_size - 1 - window_size + 1 = i)
        windows[k] = padded[start:start + window_size]

    return windows


def pearson_ic_numpy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute Pearson IC (same logic as competition reference)."""
    if len(predictions) < 2:
        return 0.0
    p = np.nan_to_num(predictions.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(labels.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = p - p.mean()
    y = y - y.mean()
    denom = np.sqrt((p ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        return 0.0
    return float((p * y).sum() / denom)


def train_gru_model(features: np.ndarray, ret5: np.ndarray, ret60: np.ndarray,
                     train_indices: np.ndarray, val_indices: np.ndarray,
                     dataset_name: str, output_dir: str):
    """
    Train a GRU model for a single dataset.
    
    H20 96GB training optimizations (architecture unchanged for inference compatibility):
    - On-the-fly batch-wise window construction (saves ~80GB RAM)
    - Val windows pre-loaded to GPU (saves CPU→GPU transfer per epoch)
    - Mixed precision (fp16) training for 2x speedup on H20
    - Larger batch (16384) and more epochs (30) for full data training

    Returns:
        (mean_val_ic, val_preds) where val_preds is (N_val, 2) numpy array.
    """
    # Set deterministic seeds
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    F = features.shape[1]
    # Use best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"    GRU device: {device}")

    model = GRUPredictor(input_size=F).to(device)
    # v6: IC-aware loss with halved learning rate and gradient clipping
    effective_lr = GRU_LR * 0.5 if LOSS_TYPE == "ic_aware" else GRU_LR
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.MSELoss()  # fallback for LOSS_TYPE="mse"

    # H20: Mixed precision training for speedup
    # DISABLED: fp16 causes NaN loss with large OHLCV values
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Pre-build val windows and load to GPU for fast validation
    print(f"    Building val sliding windows (val={len(val_indices)})...")
    val_windows_np = build_sliding_windows_for_indices(features, val_indices)
    val_labels_clean = np.nan_to_num(
        np.column_stack([ret5[val_indices], ret60[val_indices]]),
        nan=0.0
    ).astype(np.float32)

    # H20: Pre-load val windows to GPU (~17 GB for largest dataset, fits in 96 GB)
    if device.type == "cuda":
        val_windows_gpu = torch.from_numpy(val_windows_np).to(device)
        val_labels_gpu = torch.from_numpy(val_labels_clean).to(device)
        del val_windows_np  # free CPU copy
        print(f"    Val windows loaded to GPU: {val_windows_gpu.shape}")
    else:
        val_windows_gpu = None
        val_labels_gpu = None

    # Pre-compute train labels (small: N_train × 2 floats)
    train_labels = np.nan_to_num(
        np.column_stack([ret5[train_indices], ret60[train_indices]]),
        nan=0.0
    ).astype(np.float32)

    print(f"    Training with on-the-fly window construction (train={len(train_indices)})...")

    # Pre-build padded array ONCE to avoid recreating per batch
    _clean = np.nan_to_num(features, nan=0.0).astype(np.float32)
    _padded = np.zeros((GRU_WINDOW_SIZE - 1 + features.shape[0], features.shape[1]), dtype=np.float32)
    _padded[GRU_WINDOW_SIZE - 1:] = _clean
    del _clean
    print(f"    Padded array pre-built: {_padded.shape} ({_padded.nbytes/1024**3:.1f} GB)")

    best_val_ic = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(GRU_EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_indices))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_indices), GRU_BATCH_SIZE):
            end = min(start + GRU_BATCH_SIZE, len(train_indices))
            batch_idx = perm[start:end]
            actual_indices = train_indices[batch_idx]

            # Fast window construction from pre-built padded array
            n_b = len(actual_indices)
            batch_windows = np.empty((n_b, GRU_WINDOW_SIZE, features.shape[1]), dtype=np.float32)
            for k in range(n_b):
                s = actual_indices[k]
                batch_windows[k] = _padded[s:s + GRU_WINDOW_SIZE]

            x_batch = torch.from_numpy(batch_windows).to(device)
            y_batch = torch.from_numpy(train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x_batch)
                if LOSS_TYPE == "ic_aware":
                    # v6: IC-aware loss (Pearson + ListMLE)
                    # Create mask for non-NaN targets (train_labels already nan_to_num'd, so all valid)
                    # Compute loss for both outputs (Ret5 col 0, Ret60 col 1)
                    loss_r5 = ic_aware_loss(pred[:, 0], y_batch[:, 0])
                    loss_r60 = ic_aware_loss(pred[:, 1], y_batch[:, 1])
                    loss = (loss_r5 + loss_r60) / 2.0
                else:
                    loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            # v6: gradient clipping for IC-aware loss stability
            if LOSS_TYPE == "ic_aware":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation IC (fast path: data already on GPU)
        model.eval()
        with torch.no_grad():
            if val_windows_gpu is not None:
                # All val data on GPU — single or batched forward pass
                val_preds_list = []
                for vstart in range(0, len(val_indices), GRU_BATCH_SIZE):
                    vend = min(vstart + GRU_BATCH_SIZE, len(val_indices))
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = model(val_windows_gpu[vstart:vend])
                    val_preds_list.append(pred.float().cpu().numpy())
                val_preds = np.concatenate(val_preds_list, axis=0)
            else:
                val_preds_list = []
                for vstart in range(0, len(val_indices), GRU_BATCH_SIZE):
                    vend = min(vstart + GRU_BATCH_SIZE, len(val_indices))
                    x_batch = torch.from_numpy(val_windows_np[vstart:vend]).to(device)
                    pred = model(x_batch).cpu().numpy()
                    val_preds_list.append(pred)
                val_preds = np.concatenate(val_preds_list, axis=0)

        ic_ret5 = pearson_ic_numpy(val_preds[:, 0], val_labels_clean[:, 0])
        ic_ret60 = pearson_ic_numpy(val_preds[:, 1], val_labels_clean[:, 1])
        mean_ic = (ic_ret5 + ic_ret60) / 2.0

        if epoch % 5 == 0 or epoch == GRU_EPOCHS - 1:
            print(f"    Epoch {epoch+1}/{GRU_EPOCHS}: loss={epoch_loss/n_batches:.6f}, "
                  f"val_IC_r5={ic_ret5:.4f}, val_IC_r60={ic_ret60:.4f}, mean={mean_ic:.4f}")

        if mean_ic > best_val_ic:
            best_val_ic = mean_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= GRU_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Collect best-epoch validation predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    del train_labels, _padded
    
    with torch.no_grad():
        if val_windows_gpu is not None:
            final_preds_list = []
            for vstart in range(0, len(val_indices), GRU_BATCH_SIZE):
                vend = min(vstart + GRU_BATCH_SIZE, len(val_indices))
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(val_windows_gpu[vstart:vend])
                final_preds_list.append(pred.float().cpu().numpy())
            final_val_preds = np.concatenate(final_preds_list, axis=0)
            del val_windows_gpu, val_labels_gpu
        else:
            final_preds_list = []
            for vstart in range(0, len(val_indices), GRU_BATCH_SIZE):
                vend = min(vstart + GRU_BATCH_SIZE, len(val_indices))
                x_batch = torch.from_numpy(val_windows_np[vstart:vend]).to(device)
                pred = model(x_batch).cpu().numpy()
                final_preds_list.append(pred)
            final_val_preds = np.concatenate(final_preds_list, axis=0)
            del val_windows_np
    
    del val_labels_clean

    # Save as TorchScript — trace on CPU to ensure device-agnostic hidden state
    # GRU's hidden state device is determined at trace time, so we use a special approach:
    # Save as state_dict (like Transformer) to avoid device hardcoding in TorchScript
    model = model.cpu()
    model.eval()
    save_path = os.path.join(output_dir, f"gru_{dataset_name}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": F,
        "hidden_size": GRU_HIDDEN_SIZE,
        "num_layers": GRU_NUM_LAYERS,
        "dropout": GRU_DROPOUT,
        "window_size": GRU_WINDOW_SIZE,
        "model_type": "gru",
    }, save_path)
    print(f"    GRU saved: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB), best val IC: {best_val_ic:.4f}")

    # Clean up GPU resources
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_ic, final_val_preds


def _get_best_device():
    """Return best available PyTorch device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_predict_gpu(model, windows_gpu, batch_size, use_amp=False):
    """Run batched inference with data already on GPU, returns numpy array."""
    n = windows_gpu.shape[0]
    preds_list = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(windows_gpu[start:end])
            preds_list.append(pred.float().cpu().numpy())
    return np.concatenate(preds_list, axis=0)


def _batch_predict(model, windows, device, batch_size):
    """Run batched inference, returns numpy array of predictions."""
    n = len(windows)
    preds_list = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(windows[start:end]).to(device)
            pred = model(x).cpu().numpy()
            preds_list.append(pred)
    return np.concatenate(preds_list, axis=0)


def train_transformer_model(features, ret5, ret60,
                             train_indices, val_indices,
                             dataset_name, output_dir):
    """
    Train a Transformer encoder model for a single dataset.
    
    H20 96GB training optimizations (architecture unchanged for inference compatibility):
    - On-the-fly batch-wise window construction (saves ~80GB RAM)
    - Val windows pre-loaded to GPU (~17 GB, saves CPU→GPU transfer)
    - Mixed precision (fp16) training for 2x speedup on H20
    - Larger batch (16384) and more epochs (40) for full data training
    
    Returns (mean_val_ic, val_preds) where val_preds is (N_val, 2).
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    F = features.shape[1]
    device = _get_best_device()
    print(f"    Transformer device: {device}")

    model = TransformerPredictor(input_size=F).to(device)
    # Disable nested tensor to avoid CUDA deadlock on H20/CUDA 13.x
    for module in model.modules():
        if hasattr(module, 'enable_nested_tensor'):
            module.enable_nested_tensor = False
    # Force math attention backend to avoid SDPA hang on CUDA 13.x
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    # v6: IC-aware loss with halved learning rate and gradient clipping
    effective_lr = TRANSFORMER_LR * 0.5 if LOSS_TYPE == "ic_aware" else TRANSFORMER_LR
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.MSELoss()  # fallback for LOSS_TYPE="mse"

    # H20: Mixed precision training
    # DISABLED: fp16 causes NaN loss with large OHLCV values
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Pre-build val windows and load to GPU
    print(f"    Building val sliding windows (val={len(val_indices)})...")
    val_windows_np = build_sliding_windows_for_indices(features, val_indices)

    train_labels = np.nan_to_num(
        np.column_stack([ret5[train_indices], ret60[train_indices]]),
        nan=0.0
    ).astype(np.float32)
    val_labels = np.nan_to_num(
        np.column_stack([ret5[val_indices], ret60[val_indices]]),
        nan=0.0
    ).astype(np.float32)

    # H20: Pre-load val windows to GPU
    if device.type == "cuda":
        val_windows_gpu = torch.from_numpy(val_windows_np).to(device)
        del val_windows_np
        print(f"    Val windows loaded to GPU: {val_windows_gpu.shape}")
    else:
        val_windows_gpu = None

    print(f"    Training with on-the-fly window construction (train={len(train_indices)})...")

    # Pre-build padded array ONCE to avoid recreating per batch
    _clean_tf = np.nan_to_num(features, nan=0.0).astype(np.float32)
    _padded_tf = np.zeros((TRANSFORMER_WINDOW_SIZE - 1 + features.shape[0], features.shape[1]), dtype=np.float32)
    _padded_tf[TRANSFORMER_WINDOW_SIZE - 1:] = _clean_tf
    del _clean_tf
    print(f"    Padded array pre-built: {_padded_tf.shape} ({_padded_tf.nbytes/1024**3:.1f} GB)")

    best_val_ic = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(TRANSFORMER_EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_indices))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_indices), TRANSFORMER_BATCH_SIZE):
            end = min(start + TRANSFORMER_BATCH_SIZE, len(train_indices))
            batch_idx = perm[start:end]
            actual_indices = train_indices[batch_idx]

            # Fast window construction from pre-built padded array
            n_b = len(actual_indices)
            batch_windows = np.empty((n_b, TRANSFORMER_WINDOW_SIZE, features.shape[1]), dtype=np.float32)
            for k in range(n_b):
                s = actual_indices[k]
                batch_windows[k] = _padded_tf[s:s + TRANSFORMER_WINDOW_SIZE]

            x = torch.from_numpy(batch_windows).to(device)
            y = torch.from_numpy(train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                if LOSS_TYPE == "ic_aware":
                    loss_r5 = ic_aware_loss(pred[:, 0], y[:, 0])
                    loss_r60 = ic_aware_loss(pred[:, 1], y[:, 1])
                    loss = (loss_r5 + loss_r60) / 2.0
                else:
                    loss = criterion(pred, y)
            scaler.scale(loss).backward()
            if LOSS_TYPE == "ic_aware":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation IC (fast path: data on GPU)
        model.eval()
        if val_windows_gpu is not None:
            val_preds = _batch_predict_gpu(model, val_windows_gpu, TRANSFORMER_BATCH_SIZE, use_amp)
        else:
            val_preds = _batch_predict(model, val_windows_np, device, TRANSFORMER_BATCH_SIZE)
        ic_r5 = pearson_ic_numpy(val_preds[:, 0], val_labels[:, 0])
        ic_r60 = pearson_ic_numpy(val_preds[:, 1], val_labels[:, 1])
        mean_ic = (ic_r5 + ic_r60) / 2.0

        if epoch % 2 == 0 or epoch == TRANSFORMER_EPOCHS - 1:
            print(f"    Epoch {epoch+1}/{TRANSFORMER_EPOCHS}: loss={epoch_loss/max(n_batches,1):.6f}, "
                  f"val_IC_r5={ic_r5:.4f}, val_IC_r60={ic_r60:.4f}, mean={mean_ic:.4f}")

        if mean_ic > best_val_ic:
            best_val_ic = mean_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRANSFORMER_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Collect best-epoch val predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    del train_labels, _padded_tf

    if val_windows_gpu is not None:
        final_val_preds = _batch_predict_gpu(model, val_windows_gpu, TRANSFORMER_BATCH_SIZE, use_amp)
        del val_windows_gpu
    else:
        final_val_preds = _batch_predict(model, val_windows_np, device, TRANSFORMER_BATCH_SIZE)
        del val_windows_np

    del val_labels

    # Save model as state_dict + config
    model = model.cpu()
    model.eval()
    save_path = os.path.join(output_dir, f"transformer_{dataset_name}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": F,
        "window_size": TRANSFORMER_WINDOW_SIZE,
        "d_model": TRANSFORMER_D_MODEL,
        "nhead": TRANSFORMER_NHEAD,
        "num_layers": TRANSFORMER_NUM_LAYERS,
        "dim_feedforward": TRANSFORMER_DIM_FF,
        "dropout": TRANSFORMER_DROPOUT,
    }, save_path)
    print(f"    Transformer saved: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB), best val IC: {best_val_ic:.4f}")

    # Clean up
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_ic, final_val_preds


# =============================================================================
# v6: Single-target sequence model training (dual window)
# =============================================================================

def train_single_target_seq_model(
    features: np.ndarray,
    labels: np.ndarray,          # 1D array (single target: ret5 OR ret60)
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    dataset_name: str,
    output_dir: str,
    model_type: str = "gru",     # "gru" or "transformer"
    target: str = "ret5",        # "ret5" or "ret60"
    window_size: int = 20,       # 20 for ret5, 240 for ret60
    batch_size: int = 32768,     # 32768 for w=20, 16384 for w=240
):
    """
    v6: Train a single-target sequence model with configurable window size.
    
    Unlike v5 which trains one model for both Ret5+Ret60 (output_dim=2),
    v6 trains separate models per target (output_dim=1) with target-specific windows.
    
    Returns:
        (val_ic, val_preds) where val_preds is (N_val,) numpy array.
    """
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        # Sync and clear any prior CUDA errors before starting
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    np.random.seed(RANDOM_SEED)

    F = features.shape[1]
    device = _get_best_device()
    print(f"    [{model_type}_{target}] device: {device}, window={window_size}, batch={batch_size}")

    # Build model
    if model_type == "gru":
        model = GRUSingleTarget(input_size=F, hidden_size=GRU_HIDDEN_SIZE,
                                 num_layers=GRU_NUM_LAYERS, dropout=GRU_DROPOUT).to(device)
        epochs = GRU_EPOCHS
        patience = GRU_PATIENCE
        base_lr = GRU_LR
    else:
        model = TransformerSingleTarget(input_size=F, max_seq_len=window_size).to(device)
        print(f"    [{model_type}_{target}] Model: TransformerSingleTarget, output_head={model.output_head}")
        # Flash/MemEfficient SDP disabled for H20 compatibility
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        epochs = TRANSFORMER_EPOCHS
        patience = TRANSFORMER_PATIENCE
        base_lr = TRANSFORMER_LR

    effective_lr = base_lr * 0.5 if LOSS_TYPE == "ic_aware" else base_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    criterion = nn.MSELoss()

    use_amp = False  # Disabled: fp16 causes NaN with large values
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Pre-build padded array for on-the-fly window construction
    _clean = np.nan_to_num(features, nan=0.0).astype(np.float32)
    _padded = np.zeros((window_size - 1 + features.shape[0], F), dtype=np.float32)
    _padded[window_size - 1:] = _clean
    del _clean

    # Train labels (single target, 1D)
    # v6: For w=240 models, subsample training set for large datasets
    # Transformer (215K params) doesn't need 1.5M+ samples — 300K is sufficient
    MAX_TRAIN_FOR_LONG_WINDOW = 300000
    if len(train_indices) > MAX_TRAIN_FOR_LONG_WINDOW and window_size > 60:
        step = len(train_indices) // MAX_TRAIN_FOR_LONG_WINDOW
        train_indices = train_indices[::step][:MAX_TRAIN_FOR_LONG_WINDOW]
        print(f"    Train subsampled for w={window_size}: {MAX_TRAIN_FOR_LONG_WINDOW} samples")

    train_labels = np.nan_to_num(labels[train_indices], nan=0.0).astype(np.float32)

    # v6: Subsample validation set for large window models to keep training fast
    # For w=240 with large datasets, full val (500K+) is too slow for per-epoch evaluation
    MAX_VAL_FOR_TRAINING = 50000  # 50K samples is enough for IC estimation during training
    if len(val_indices) > MAX_VAL_FOR_TRAINING and window_size > 60:
        step = len(val_indices) // MAX_VAL_FOR_TRAINING
        val_indices = val_indices[::step][:MAX_VAL_FOR_TRAINING]
        print(f"    Val subsampled for training speed: {MAX_VAL_FOR_TRAINING} samples")

    val_labels = np.nan_to_num(labels[val_indices], nan=0.0).astype(np.float32)

    # v6: Determine val strategy based on memory requirements
    val_size_gb = len(val_indices) * window_size * F * 4 / (1024**3)
    if device.type == "cuda" and val_size_gb < 20.0:
        # Small enough to preload to GPU
        print(f"    Building val windows (val={len(val_indices)}, w={window_size}, {val_size_gb:.1f} GB → GPU)...")
        val_windows_np = np.empty((len(val_indices), window_size, F), dtype=np.float32)
        for k in range(len(val_indices)):
            s = val_indices[k]
            val_windows_np[k] = _padded[s:s + window_size]
        val_windows_gpu = torch.from_numpy(val_windows_np).to(device)
        del val_windows_np
        val_windows_np = None
        val_mode = "gpu"
    elif val_size_gb < 60.0:
        # Fits in RAM, preload to CPU numpy array
        print(f"    Building val windows (val={len(val_indices)}, w={window_size}, {val_size_gb:.1f} GB → CPU)...")
        val_windows_np = np.empty((len(val_indices), window_size, F), dtype=np.float32)
        for k in range(len(val_indices)):
            s = val_indices[k]
            val_windows_np[k] = _padded[s:s + window_size]
        val_windows_gpu = None
        val_mode = "cpu"
    else:
        # Too large even for RAM — use on-the-fly from _padded
        print(f"    Val windows too large ({val_size_gb:.1f} GB), using on-the-fly from padded array")
        val_windows_np = None
        val_windows_gpu = None
        val_mode = "onthefly"

    print(f"    Training ({len(train_indices)} samples, {epochs} epochs)...")

    best_val_ic = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_indices))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_indices), batch_size):
            end = min(start + batch_size, len(train_indices))
            batch_idx = perm[start:end]
            actual_indices = train_indices[batch_idx]

            # Vectorized window construction using stride_tricks (10-50x faster than for-loop)
            n_b = len(actual_indices)
            # Create a strided view of _padded: all possible windows
            strides = (_padded.strides[0], _padded.strides[0], _padded.strides[1])
            n_windows_total = _padded.shape[0] - window_size + 1
            all_windows_view = np.lib.stride_tricks.as_strided(
                _padded, shape=(n_windows_total, window_size, F), strides=strides
            )
            # Index directly into the strided view
            batch_windows = np.ascontiguousarray(all_windows_view[actual_indices])

            x_batch = torch.from_numpy(batch_windows).to(device)
            y_batch = torch.from_numpy(train_labels[batch_idx]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x_batch)  # (batch,)
                assert pred.ndim == 1, f"Expected 1D pred, got shape {pred.shape}"
                if LOSS_TYPE == "ic_aware":
                    loss = ic_aware_loss(pred, y_batch)
                else:
                    loss = criterion(pred.unsqueeze(-1), y_batch.unsqueeze(-1))
            scaler.scale(loss).backward()
            if LOSS_TYPE == "ic_aware":
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation IC
        model.eval()
        with torch.no_grad():
            if val_windows_gpu is not None:
                val_preds_list = []
                for vstart in range(0, len(val_indices), batch_size):
                    vend = min(vstart + batch_size, len(val_indices))
                    pred = model(val_windows_gpu[vstart:vend])
                    val_preds_list.append(pred.float().cpu().numpy())
                val_preds = np.concatenate(val_preds_list, axis=0)
            else:
                # Vectorized on-the-fly validation using stride_tricks
                strides_val = (_padded.strides[0], _padded.strides[0], _padded.strides[1])
                n_win_total = _padded.shape[0] - window_size + 1
                all_win_view = np.lib.stride_tricks.as_strided(
                    _padded, shape=(n_win_total, window_size, F), strides=strides_val
                )
                val_preds_list = []
                for vstart in range(0, len(val_indices), batch_size):
                    vend = min(vstart + batch_size, len(val_indices))
                    if val_windows_np is not None:
                        x = torch.from_numpy(val_windows_np[vstart:vend]).to(device)
                    else:
                        batch_w = np.ascontiguousarray(all_win_view[val_indices[vstart:vend]])
                        x = torch.from_numpy(batch_w).to(device)
                    val_preds_list.append(model(x).float().cpu().numpy())
                    del x
                val_preds = np.concatenate(val_preds_list, axis=0)

        val_ic = pearson_ic_numpy(val_preds, val_labels)

        log_interval = 5 if model_type == "gru" else 2
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"    [{model_type}_{target}] Epoch {epoch+1}/{epochs}: "
                  f"loss={epoch_loss/max(n_batches,1):.6f}, val_IC={val_ic:.4f}")

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    [{model_type}_{target}] Early stopping at epoch {epoch+1}")
                break

    # Final val predictions with best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    del train_labels, _padded

    with torch.no_grad():
        if val_windows_gpu is not None:
            final_list = []
            for vstart in range(0, len(val_indices), batch_size):
                vend = min(vstart + batch_size, len(val_indices))
                pred = model(val_windows_gpu[vstart:vend])
                final_list.append(pred.float().cpu().numpy())
            final_val_preds = np.concatenate(final_list, axis=0)
            del val_windows_gpu
        elif val_windows_np is not None:
            final_list = []
            for vstart in range(0, len(val_indices), batch_size):
                vend = min(vstart + batch_size, len(val_indices))
                x = torch.from_numpy(val_windows_np[vstart:vend]).to(device)
                final_list.append(model(x).float().cpu().numpy())
                del x
            final_val_preds = np.concatenate(final_list, axis=0)
            del val_windows_np
        else:
            final_val_preds = val_preds  # use last epoch's preds

    del val_labels

    # Save checkpoint
    model = model.cpu()
    model.eval()
    filename = f"{model_type}_{target}_{dataset_name}.pt"
    save_path = os.path.join(output_dir, filename)
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": F,
        "hidden_size": GRU_HIDDEN_SIZE if model_type == "gru" else None,
        "num_layers": GRU_NUM_LAYERS if model_type == "gru" else TRANSFORMER_NUM_LAYERS,
        "dropout": GRU_DROPOUT if model_type == "gru" else TRANSFORMER_DROPOUT,
        "d_model": TRANSFORMER_D_MODEL if model_type == "transformer" else None,
        "nhead": TRANSFORMER_NHEAD if model_type == "transformer" else None,
        "dim_feedforward": TRANSFORMER_DIM_FF if model_type == "transformer" else None,
        "window_size": window_size,
        "model_type": model_type,
        "target": target,
        "output_dim": 1,
    }, save_path)
    print(f"    [{model_type}_{target}] Saved: {save_path} "
          f"({os.path.getsize(save_path)/1024:.1f} KB), val IC: {best_val_ic:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_ic, final_val_preds


def optimize_ensemble_weights(lgb_val_ret5, lgb_val_ret60,
                                gru_val_ret5, gru_val_ret60,
                                val_ret5_labels, val_ret60_labels):
    """
    Grid search over alpha to maximize validation IC.
    final = alpha * lgb + (1 - alpha) * gru

    Returns:
        (best_alpha_ret5, best_alpha_ret60)
    """
    alphas = np.arange(0.0, 1.05, 0.1)

    best_alpha_r5, best_ic_r5 = 1.0, -np.inf
    best_alpha_r60, best_ic_r60 = 1.0, -np.inf

    for alpha in alphas:
        # Ret5
        blended = alpha * lgb_val_ret5 + (1.0 - alpha) * gru_val_ret5
        ic = pearson_ic_numpy(blended, val_ret5_labels)
        if ic > best_ic_r5:
            best_ic_r5 = ic
            best_alpha_r5 = round(float(alpha), 1)

        # Ret60
        blended = alpha * lgb_val_ret60 + (1.0 - alpha) * gru_val_ret60
        ic = pearson_ic_numpy(blended, val_ret60_labels)
        if ic > best_ic_r60:
            best_ic_r60 = ic
            best_alpha_r60 = round(float(alpha), 1)

    return best_alpha_r5, best_alpha_r60


def optimize_three_model_ensemble(lgb_val_r5, lgb_val_r60,
                                   gru_val_r5, gru_val_r60,
                                   tf_val_r5, tf_val_r60,
                                   val_r5_labels, val_r60_labels,
                                   gru_enabled=True, tf_enabled=True):
    """
    Grid search over (alpha, beta, gamma) triples where alpha+beta+gamma=1.0.
    Each value in {0.0, 0.1, 0.2, ..., 1.0}. Step size 0.1 gives 66 valid triples.

    Returns dict: {
        "ret5_alpha": float, "ret5_beta": float, "ret5_gamma": float,
        "ret60_alpha": float, "ret60_beta": float, "ret60_gamma": float,
    }
    """
    step = 0.1
    values = np.arange(0.0, 1.0 + step/2, step)

    triples = []
    for a in values:
        for b in values:
            g = 1.0 - a - b
            if g >= -1e-9 and g <= 1.0 + 1e-9:
                g = max(0.0, min(1.0, g))
                if not gru_enabled and b > 0:
                    continue
                if not tf_enabled and g > 0:
                    continue
                triples.append((round(a, 1), round(b, 1), round(g, 1)))

    best = {}
    for target, lgb_pred, gru_pred, tf_pred, labels in [
        ("ret5", lgb_val_r5, gru_val_r5, tf_val_r5, val_r5_labels),
        ("ret60", lgb_val_r60, gru_val_r60, tf_val_r60, val_r60_labels),
    ]:
        # Baseline: pure LightGBM IC
        lgb_only_ic = pearson_ic_numpy(lgb_pred, labels)

        best_ic = -np.inf
        best_triple = (1.0, 0.0, 0.0)
        for a, b, g in triples:
            blended = a * lgb_pred + b * gru_pred + g * tf_pred
            ic = pearson_ic_numpy(blended, labels)
            if ic > best_ic:
                best_ic = ic
                best_triple = (a, b, g)

        # Safety checks to prevent overfitting to validation set noise
        if best_triple != (1.0, 0.0, 0.0):
            improvement = best_ic - lgb_only_ic

            # Check 1: absolute improvement must be > 0.002
            if improvement < 0.002:
                print(f"    [{target}] Ensemble IC ({best_ic:.4f}) barely better than pure LGB ({lgb_only_ic:.4f}), using pure LGB")
                best_triple = (1.0, 0.0, 0.0)
            # Check 2: LGB alpha must be >= 0.3 (don't let seq models dominate)
            elif best_triple[0] < 0.3:
                print(f"    [{target}] LGB alpha={best_triple[0]} too low, capping at (0.3, ...)")
                # Redistribute: keep ratio of beta:gamma, scale down to fit alpha=0.3
                remaining = 0.7
                b, g = best_triple[1], best_triple[2]
                bg_sum = b + g
                if bg_sum > 0:
                    best_triple = (0.3, round(remaining * b / bg_sum, 1), round(remaining * g / bg_sum, 1))
                else:
                    best_triple = (1.0, 0.0, 0.0)

        best[f"{target}_alpha"] = best_triple[0]
        best[f"{target}_beta"] = best_triple[1]
        best[f"{target}_gamma"] = best_triple[2]

    return best


def extract_feature_importance(models_ret5, models_ret60, num_features):
    """Aggregate gain-based feature importance across all 30 datasets per target."""
    ret5_gain = np.zeros(num_features)
    ret60_gain = np.zeros(num_features)
    for model in models_ret5:
        importance = model.feature_importance(importance_type="gain")
        ret5_gain[:len(importance)] += importance
    for model in models_ret60:
        importance = model.feature_importance(importance_type="gain")
        ret60_gain[:len(importance)] += importance
    
    # Log top-20
    ret5_top20 = np.argsort(ret5_gain)[-20:][::-1]
    ret60_top20 = np.argsort(ret60_gain)[-20:][::-1]
    print("  Top-20 Ret5 features:", [(int(i), f"{ret5_gain[i]:.1f}") for i in ret5_top20])
    print("  Top-20 Ret60 features:", [(int(i), f"{ret60_gain[i]:.1f}") for i in ret60_top20])
    
    ret5_top100 = np.argsort(ret5_gain)[-100:][::-1].tolist()
    ret60_top100 = np.argsort(ret60_gain)[-100:][::-1].tolist()
    return ret5_top100, ret60_top100


def select_and_save_features(models_ret5, models_ret60, num_features, output_dir):
    """Extract feature importance, select top-100, save to feature_selection.json."""
    ret5_top100, ret60_top100 = extract_feature_importance(models_ret5, models_ret60, num_features)
    fs = {"ret5_features": ret5_top100, "ret60_features": ret60_top100}
    fs_path = os.path.join(output_dir, "feature_selection.json")
    with open(fs_path, "w") as f:
        json.dump(fs, f, indent=2)
    print(f"  Feature selection saved: {fs_path}")


# =============================================================================
# Task 5.1: Build global training dataset
# =============================================================================

def _build_global_dataset(
    datasets_features: list,   # list of (T_i, F) arrays
    datasets_labels: list,     # list of (T_i,) label arrays
    dataset_ids: list,         # list of int dataset IDs
    max_per_dataset: int = 0,  # H20 96GB: 0 = no limit, use all data
) -> tuple:
    """
    Build global training dataset by concatenating all datasets with dataset ID column.

    For each dataset:
    - Takes the first 80% as training data (temporal split, valid labels only)
    - If max_per_dataset > 0 and train rows exceed it, uniformly subsamples
    - Appends integer dataset ID column

    H20 96GB mode: max_per_dataset=0 means no subsampling, use all training data.

    Returns:
        (X_global_train, y_global_train, X_global_val, y_global_val, per_dataset_val_info)
        where per_dataset_val_info is list of (dataset_id, val_start_idx, val_end_idx) in global val array
    """
    train_X_parts = []
    train_y_parts = []
    val_X_parts = []
    val_y_parts = []
    per_dataset_val_info = []

    total_train_rows = 0
    for features, labels, ds_id in zip(datasets_features, datasets_labels, dataset_ids):
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        if n_valid < 10:
            continue

        split_idx = int(n_valid * TRAIN_RATIO)
        train_indices = valid_indices[:split_idx]
        val_indices = valid_indices[split_idx:]

        # Subsample if needed (only when max_per_dataset > 0)
        if max_per_dataset > 0 and len(train_indices) > max_per_dataset:
            step = math.ceil(len(train_indices) / max_per_dataset)
            train_indices = train_indices[::step]

        total_train_rows += len(train_indices)

    # If total > 5,000,000, trigger sampling
    needs_sampling = total_train_rows > 5_000_000

    val_offset = 0
    for features, labels, ds_id in zip(datasets_features, datasets_labels, dataset_ids):
        valid_mask = ~np.isnan(labels)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        if n_valid < 10:
            continue

        split_idx = int(n_valid * TRAIN_RATIO)
        train_indices = valid_indices[:split_idx]
        val_indices = valid_indices[split_idx:]

        # Subsample training if needed (only when max_per_dataset > 0)
        if max_per_dataset > 0 and len(train_indices) > max_per_dataset:
            step = math.ceil(len(train_indices) / max_per_dataset)
            train_indices = train_indices[::step]

        # Build feature arrays with dataset ID column appended
        X_train_i = features[train_indices]
        id_col_train = np.full((len(train_indices), 1), ds_id, dtype=np.int32)
        X_train_i = np.hstack([X_train_i, id_col_train])

        X_val_i = features[val_indices]
        id_col_val = np.full((len(val_indices), 1), ds_id, dtype=np.int32)
        X_val_i = np.hstack([X_val_i, id_col_val])

        train_X_parts.append(X_train_i)
        train_y_parts.append(labels[train_indices])
        val_X_parts.append(X_val_i)
        val_y_parts.append(labels[val_indices])

        val_start = val_offset
        val_end = val_offset + len(val_indices)
        per_dataset_val_info.append((ds_id, val_start, val_end))
        val_offset = val_end

    if not train_X_parts:
        return None, None, None, None, []

    X_global_train = np.vstack(train_X_parts)
    y_global_train = np.concatenate(train_y_parts)
    X_global_val = np.vstack(val_X_parts)
    y_global_val = np.concatenate(val_y_parts)

    print(f"  Global dataset: train={X_global_train.shape}, val={X_global_val.shape}")
    return X_global_train, y_global_train, X_global_val, y_global_val, per_dataset_val_info


# =============================================================================
# Task 5.2: Train global LightGBM model
# =============================================================================

def train_global_lgb(
    X_train: np.ndarray,   # (N, 166) — 165 features + 1 dataset ID column
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target: str,           # "ret5" or "ret60"
    output_dir: str,
) -> lgb.Booster:
    """Train global LightGBM model with dataset ID as categorical feature."""
    params = LGB_PARAMS_RET5.copy() if target == "ret5" else LGB_PARAMS_RET60.copy()
    num_boost_round = NUM_BOOST_ROUND_RET5 if target == "ret5" else NUM_BOOST_ROUND_RET60

    # Clean NaN labels
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_v = X_val[val_valid]
    y_v = y_val[val_valid]

    print(f"    Global {target}: train={len(y_tr)}, val={len(y_v)}")

    train_data = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=[165],  # dataset ID column (0-indexed)
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_v, label=y_v,
        categorical_feature=[165],
        reference=train_data,
        free_raw_data=False,
    )

    model = train_lgb_two_phase(params, train_data, val_data, num_boost_round)

    # Report validation IC
    val_preds = model.predict(X_v)
    _, val_ic, _ = ic_eval_metric(val_preds, val_data)
    print(f"    Global {target}: Trees={model.num_trees()}, Val IC={val_ic:.6f}")

    # Save as .txt.gz
    path_gz = Path(output_dir) / f"lgb_{target}_global.txt.gz"
    save_lgb_model_gz(model, path_gz)
    print(f"    Saved: {path_gz}")

    return model


# =============================================================================
# Task 7.2: Batch-wise sliding window training loop
# =============================================================================

def train_sequence_model_batched(
    model,
    features: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    ret5: np.ndarray,
    ret60: np.ndarray,
    optimizer,
    criterion,
    device,
    epochs: int,
    patience: int,
    batch_size: int,
    window_size: int,
    dataset_name: str,
    model_type: str,  # "gru" or "transformer"
) -> tuple:  # (best_val_ic, val_preds)
    """
    Train sequence model with on-the-fly batch-wise window construction.

    Instead of pre-building all windows (memory-intensive), builds windows
    for each batch on-the-fly by slicing features[idx-window_size+1:idx+1].
    Memory usage stays at single-batch level (~0.15 GB).
    """
    # Pre-build val windows once (val set is smaller, manageable)
    val_windows = build_sliding_windows_for_indices(features, val_indices, window_size)
    val_labels = np.nan_to_num(
        np.column_stack([ret5[val_indices], ret60[val_indices]]),
        nan=0.0
    ).astype(np.float32)

    best_val_ic = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_indices))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_indices), batch_size):
            end = min(start + batch_size, len(train_indices))
            batch_idx = perm[start:end]

            # Build windows on-the-fly for this batch only
            batch_windows = build_sliding_windows_for_indices(
                features, train_indices[batch_idx], window_size
            )
            x_batch = torch.from_numpy(batch_windows).to(device)

            # Labels for this batch
            batch_labels = np.nan_to_num(
                np.column_stack([ret5[train_indices[batch_idx]],
                                  ret60[train_indices[batch_idx]]]),
                nan=0.0
            ).astype(np.float32)
            y_batch = torch.from_numpy(batch_labels).to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation IC
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for start in range(0, len(val_indices), batch_size):
                end = min(start + batch_size, len(val_indices))
                x_batch = torch.from_numpy(val_windows[start:end]).to(device)
                pred = model(x_batch).cpu().numpy()
                val_preds_list.append(pred)
        val_preds = np.concatenate(val_preds_list, axis=0)

        ic_r5 = pearson_ic_numpy(val_preds[:, 0], val_labels[:, 0])
        ic_r60 = pearson_ic_numpy(val_preds[:, 1], val_labels[:, 1])
        mean_ic = (ic_r5 + ic_r60) / 2.0

        log_interval = 5 if model_type == "gru" else 2
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"    [{model_type}] Epoch {epoch+1}/{epochs}: "
                  f"loss={epoch_loss/max(n_batches,1):.6f}, "
                  f"val_IC_r5={ic_r5:.4f}, val_IC_r60={ic_r60:.4f}, mean={mean_ic:.4f}")

        if mean_ic > best_val_ic:
            best_val_ic = mean_ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    [{model_type}] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Final val predictions with best weights
    final_val_preds_list = []
    with torch.no_grad():
        for start in range(0, len(val_indices), batch_size):
            end = min(start + batch_size, len(val_indices))
            x_batch = torch.from_numpy(val_windows[start:end]).to(device)
            pred = model(x_batch).cpu().numpy()
            final_val_preds_list.append(pred)
    final_val_preds = np.concatenate(final_val_preds_list, axis=0)

    del val_windows, val_labels
    return best_val_ic, final_val_preds


def compute_dynamic_sample_weights(close_prices, window_vol=20, window_median=120):
    """Compute volatility-based sample weights. All computations causal."""
    T = len(close_prices)
    log_returns = np.diff(np.log(np.maximum(close_prices, 1e-10)))
    log_returns = np.concatenate([[0.0], log_returns])
    
    volatility = np.full(T, np.nan)
    for i in range(window_vol - 1, T):
        volatility[i] = np.std(log_returns[i - window_vol + 1:i + 1])
    
    rolling_median = np.full(T, np.nan)
    for i in range(window_median - 1, T):
        rolling_median[i] = np.median(volatility[max(0, i - window_median + 1):i + 1])
    
    weights = np.ones(T, dtype=np.float32)
    valid = (~np.isnan(volatility)) & (~np.isnan(rolling_median)) & (rolling_median > 0)
    weights[valid] = 1.0 + 0.5 * (volatility[valid] / rolling_median[valid])
    return weights


REGIME_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "verbose": -1,
    "seed": 42,
    "num_threads": -1,
}
REGIME_FEATURE_INDICES = list(range(82, 94))

def train_regime_classifier(features, indices, extreme_intervals, dataset_name, output_dir):
    """Train a lightweight binary classifier to predict extreme regime probability."""
    labels = build_extreme_mask(indices, extreme_intervals).astype(np.float32)
    X = features[:, REGIME_FEATURE_INDICES]
    T = len(labels)
    split = int(T * 0.8)
    train_data = lgb.Dataset(X[:split], label=labels[:split])
    val_data = lgb.Dataset(X[split:], label=labels[split:], reference=train_data)
    model = lgb.train(
        REGIME_PARAMS, train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    save_path = os.path.join(output_dir, f"regime_{dataset_name}.txt")
    model.save_model(save_path)
    print(f"  Regime classifier saved: {save_path}")
    return model


def _train_seq_model_with_oom_recovery(
    features, labels, train_indices, val_indices,
    dataset_name, output_dir, model_type, target,
    window_size, batch_size, max_retries=2,
):
    """
    Wrapper around train_single_target_seq_model with OOM recovery.
    On GPU OOM, halves batch_size and retries (up to max_retries times).
    Returns (val_ic, val_preds) or (None, None) if all retries fail.
    """
    import gc
    current_batch = batch_size

    for attempt in range(max_retries + 1):
        try:
            val_ic, val_preds = train_single_target_seq_model(
                features=features,
                labels=labels,
                train_indices=train_indices,
                val_indices=val_indices,
                dataset_name=dataset_name,
                output_dir=output_dir,
                model_type=model_type,
                target=target,
                window_size=window_size,
                batch_size=current_batch,
            )
            return val_ic, val_preds
        except (RuntimeError,) as e:
            err_str = str(e).lower()
            is_oom = ("out of memory" in err_str or 
                      "cublas" in err_str or 
                      "cuda error" in err_str)
            if not is_oom:
                try:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        is_oom = True
                except Exception:
                    pass

            if is_oom and attempt < max_retries:
                current_batch = current_batch // 2
                print(f"    [{model_type}_{target}] GPU OOM, halving batch to {current_batch}, "
                      f"retry {attempt+1}/{max_retries}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif is_oom:
                print(f"    [{model_type}_{target}] GPU OOM after {max_retries} retries, skipping")
                return None, None
            else:
                raise

    return None, None


def train_all_models(data_dir: str, output_dir: str) -> None:
    """
    v6: Train all models for 30 datasets.
    
    Training order (per design doc):
    1. LGB_Local (30 datasets x 2 targets)
    2. LGB_Extreme (<=30 datasets x 2 targets, skip if <1000 extreme samples)
    3. GRU_Ret5 (30 datasets, window=20, single target)
    4. GRU_Ret60 (30 datasets, window=240, single target)
    5. TF_Ret5 (30 datasets, window=20, single target)
    6. TF_Ret60 (30 datasets, window=240, single target)
    
    Each model is saved immediately after training (checkpoint-based resume).
    GPU resources are cleaned between phases.
    """
    import gc
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    ensemble_weights = {}

    # Load existing ensemble weights if available (for resume)
    weights_path = os.path.join(output_dir, "ensemble_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            ensemble_weights = json.load(f)

    # =========================================================================
    # Phase 1: LGB_Local training (all 30 datasets)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: LGB_Local Training")
    print("=" * 60)

    phase1_start = time.time()
    all_features_list = []
    all_ret5_list = []
    all_ret60_list = []
    all_ohlcv_list = []

    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        ds_start = time.time()

        # Load data
        ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(data_dir, ds_idx)

        # Generate features (with disk cache to avoid recomputation on resume)
        cache_path = os.path.join(output_dir, f"_cache_features_{dataset_name}.npy")
        if os.path.exists(cache_path):
            features = np.load(cache_path)
            print(f"\n  [{dataset_name}] Loaded: {ohlcv.shape[0]} rows, Features: {features.shape} (cached)")
        else:
            t0 = time.time()
            features = generate_factors(dataset_name, ohlcv)
            t1 = time.time()
            np.save(cache_path, features)
            print(f"\n  [{dataset_name}] Loaded: {ohlcv.shape[0]} rows, Features: {features.shape} ({t1-t0:.1f}s, cached)")

        # Store for later phases
        all_features_list.append(features)
        all_ret5_list.append(ret5)
        all_ret60_list.append(ret60)
        all_ohlcv_list.append(ohlcv)

        # Check if LGB models already exist (checkpoint resume)
        path_ret5_gz = Path(output_dir) / f"lgb_ret5_{dataset_name}.txt.gz"
        path_ret60_gz = Path(output_dir) / f"lgb_ret60_{dataset_name}.txt.gz"
        path_ret5_txt = Path(output_dir) / f"lgb_ret5_{dataset_name}.txt"
        path_ret60_txt = Path(output_dir) / f"lgb_ret60_{dataset_name}.txt"

        lgb_r5_done = path_ret5_gz.exists() or path_ret5_txt.exists()
        lgb_r60_done = path_ret60_gz.exists() or path_ret60_txt.exists()

        if lgb_r5_done and lgb_r60_done:
            print(f"  [{dataset_name}] LGB_Local already exists, skipping")
            continue

        # Train LGB Ret5
        if not lgb_r5_done:
            print(f"  [{dataset_name}] Training LGB_Local Ret5...")
            model_ret5 = train_single_model(
                features, ret5, LGB_PARAMS_RET5, NUM_BOOST_ROUND_RET5,
                dataset_name, "ret5",
            )
            save_lgb_model_gz(model_ret5, path_ret5_gz)
            del model_ret5

        # Train LGB Ret60
        if not lgb_r60_done:
            print(f"  [{dataset_name}] Training LGB_Local Ret60...")
            model_ret60 = train_single_model(
                features, ret60, LGB_PARAMS_RET60, NUM_BOOST_ROUND_RET60,
                dataset_name, "ret60",
            )
            save_lgb_model_gz(model_ret60, path_ret60_gz)
            del model_ret60

        ds_elapsed = time.time() - ds_start
        print(f"  [{dataset_name}] LGB_Local time: {ds_elapsed:.1f}s")

    phase1_elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete: {phase1_elapsed:.1f}s ({phase1_elapsed/60:.1f} min)")

    # =========================================================================
    # Phase 2: LGB_Extreme training (extreme regime samples only)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: LGB_Extreme Training")
    print("=" * 60)

    phase2_start = time.time()
    LGB_EXTREME_MAX_ROUNDS = 500
    LGB_EXTREME_MIN_SAMPLES = 1000

    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        features = all_features_list[ds_idx]
        ret5 = all_ret5_list[ds_idx]
        ret60 = all_ret60_list[ds_idx]
        ohlcv = all_ohlcv_list[ds_idx]
        close_prices = ohlcv[:, 3]  # close column

        # Check if already done (checkpoint resume)
        path_ext_r5 = Path(output_dir) / f"lgb_extreme_ret5_{dataset_name}.txt"
        path_ext_r60 = Path(output_dir) / f"lgb_extreme_ret60_{dataset_name}.txt"
        if path_ext_r5.exists() and path_ext_r60.exists():
            print(f"  [{dataset_name}] LGB_Extreme already exists, skipping")
            continue

        # Detect extreme regime (causal)
        extreme_mask = detect_extreme_regime(close_prices, window=60, threshold_mult=2.0)
        n_extreme = int(extreme_mask.sum())

        if n_extreme < LGB_EXTREME_MIN_SAMPLES:
            print(f"  [{dataset_name}] Skip LGB_Extreme: only {n_extreme} extreme samples (< {LGB_EXTREME_MIN_SAMPLES})")
            continue

        print(f"  [{dataset_name}] Extreme samples: {n_extreme} ({n_extreme/len(close_prices)*100:.1f}%)")

        # Train LGB_Extreme for Ret5
        if not path_ext_r5.exists():
            valid_mask_r5 = extreme_mask & (~np.isnan(ret5))
            valid_idx_r5 = np.where(valid_mask_r5)[0]
            if len(valid_idx_r5) >= LGB_EXTREME_MIN_SAMPLES:
                split_r5 = int(len(valid_idx_r5) * TRAIN_RATIO)
                train_idx_r5 = valid_idx_r5[:split_r5]
                val_idx_r5 = valid_idx_r5[split_r5:]

                X_train = features[train_idx_r5]
                y_train = ret5[train_idx_r5]
                X_val = features[val_idx_r5]
                y_val = ret5[val_idx_r5]

                train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

                model = train_lgb_two_phase(
                    LGB_PARAMS_RET5, train_data, val_data,
                    max_boost_round=LGB_EXTREME_MAX_ROUNDS
                )
                model.save_model(str(path_ext_r5))
                val_preds = model.predict(X_val)
                _, val_ic, _ = ic_eval_metric(val_preds, val_data)
                print(f"    [{dataset_name}] LGB_Extreme Ret5: trees={model.num_trees()}, val_IC={val_ic:.4f}")
                del model, train_data, val_data
            else:
                print(f"    [{dataset_name}] Skip LGB_Extreme Ret5: only {len(valid_idx_r5)} valid extreme samples")

        # Train LGB_Extreme for Ret60
        if not path_ext_r60.exists():
            valid_mask_r60 = extreme_mask & (~np.isnan(ret60))
            valid_idx_r60 = np.where(valid_mask_r60)[0]
            if len(valid_idx_r60) >= LGB_EXTREME_MIN_SAMPLES:
                split_r60 = int(len(valid_idx_r60) * TRAIN_RATIO)
                train_idx_r60 = valid_idx_r60[:split_r60]
                val_idx_r60 = valid_idx_r60[split_r60:]

                X_train = features[train_idx_r60]
                y_train = ret60[train_idx_r60]
                X_val = features[val_idx_r60]
                y_val = ret60[val_idx_r60]

                train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

                model = train_lgb_two_phase(
                    LGB_PARAMS_RET60, train_data, val_data,
                    max_boost_round=LGB_EXTREME_MAX_ROUNDS
                )
                model.save_model(str(path_ext_r60))
                val_preds = model.predict(X_val)
                _, val_ic, _ = ic_eval_metric(val_preds, val_data)
                print(f"    [{dataset_name}] LGB_Extreme Ret60: trees={model.num_trees()}, val_IC={val_ic:.4f}")
                del model, train_data, val_data
            else:
                print(f"    [{dataset_name}] Skip LGB_Extreme Ret60: only {len(valid_idx_r60)} valid extreme samples")

    phase2_elapsed = time.time() - phase2_start
    print(f"\nPhase 2 complete: {phase2_elapsed:.1f}s ({phase2_elapsed/60:.1f} min)")

    # =========================================================================
    # Phase 3: GRU_Ret5 (window=20, single target)
    # =========================================================================
    # Free large lists to stay within 150 GB container memory limit
    # Features will be reloaded from disk cache per dataset
    print("\n  Freeing feature lists to save memory...")
    del all_features_list, all_ret5_list, all_ret60_list
    import gc; gc.collect()
    print("  Memory freed. Features will be loaded from cache per dataset.")

    print("\n" + "=" * 60)
    print("Phase 3: GRU_Ret5 Training (window=20)")
    print("=" * 60)

    phase3_start = time.time()
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        # Load features from cache (created in Phase 1)
        cache_path = os.path.join(output_dir, f"_cache_features_{dataset_name}.npy")
        features = np.load(cache_path)
        _, ret5, ret60, _, _ = load_dataset(data_dir, ds_idx)

        # Checkpoint resume
        model_path = Path(output_dir) / f"gru_ret5_{dataset_name}.pt"
        if model_path.exists():
            print(f"  [{dataset_name}] GRU_Ret5 already exists, skipping")
            continue

        # Build valid indices
        valid_mask = ~np.isnan(ret5)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            print(f"  [{dataset_name}] Skip GRU_Ret5: only {len(valid_indices)} valid samples")
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        train_idx = valid_indices[:split_idx]
        val_idx = valid_indices[split_idx:]

        print(f"  [{dataset_name}] Training GRU_Ret5 (train={len(train_idx)}, val={len(val_idx)})...")
        ds_start = time.time()

        val_ic, _ = _train_seq_model_with_oom_recovery(
            features=features,
            labels=ret5,
            train_indices=train_idx,
            val_indices=val_idx,
            dataset_name=dataset_name,
            output_dir=output_dir,
            model_type="gru",
            target="ret5",
            window_size=WINDOW_RET5,
            batch_size=BATCH_SIZE_W20,
        )
        if val_ic is not None:
            print(f"  [{dataset_name}] GRU_Ret5 val_IC={val_ic:.4f} ({time.time()-ds_start:.1f}s)")
        else:
            print(f"  [{dataset_name}] GRU_Ret5 FAILED ({time.time()-ds_start:.1f}s)")

        # GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase3_elapsed = time.time() - phase3_start
    print(f"\nPhase 3 complete: {phase3_elapsed:.1f}s ({phase3_elapsed/60:.1f} min)")

    # =========================================================================
    # Phase 4: GRU_Ret60 (window=240, single target)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 4: GRU_Ret60 Training (window=240)")
    print("=" * 60)

    phase4_start = time.time()
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        cache_path = os.path.join(output_dir, f"_cache_features_{dataset_name}.npy")
        features = np.load(cache_path)
        _, _, ret60, _, _ = load_dataset(data_dir, ds_idx)

        # Checkpoint resume
        model_path = Path(output_dir) / f"gru_ret60_{dataset_name}.pt"
        if model_path.exists():
            print(f"  [{dataset_name}] GRU_Ret60 already exists, skipping")
            continue

        # Build valid indices
        valid_mask = ~np.isnan(ret60)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            print(f"  [{dataset_name}] Skip GRU_Ret60: only {len(valid_indices)} valid samples")
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        train_idx = valid_indices[:split_idx]
        val_idx = valid_indices[split_idx:]

        print(f"  [{dataset_name}] Training GRU_Ret60 (train={len(train_idx)}, val={len(val_idx)})...")
        ds_start = time.time()

        val_ic, _ = _train_seq_model_with_oom_recovery(
            features=features,
            labels=ret60,
            train_indices=train_idx,
            val_indices=val_idx,
            dataset_name=dataset_name,
            output_dir=output_dir,
            model_type="gru",
            target="ret60",
            window_size=WINDOW_RET60,
            batch_size=BATCH_SIZE_W240,
        )
        if val_ic is not None:
            print(f"  [{dataset_name}] GRU_Ret60 val_IC={val_ic:.4f} ({time.time()-ds_start:.1f}s)")
        else:
            print(f"  [{dataset_name}] GRU_Ret60 FAILED ({time.time()-ds_start:.1f}s)")

        # GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase4_elapsed = time.time() - phase4_start
    print(f"\nPhase 4 complete: {phase4_elapsed:.1f}s ({phase4_elapsed/60:.1f} min)")

    # =========================================================================
    # Phase 5: TF_Ret5 (window=20, single target)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 5: Transformer_Ret5 Training (window=20)")
    print("=" * 60)

    phase5_start = time.time()
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        cache_path = os.path.join(output_dir, f"_cache_features_{dataset_name}.npy")
        features = np.load(cache_path)
        _, ret5, _, _, _ = load_dataset(data_dir, ds_idx)

        # Checkpoint resume
        model_path = Path(output_dir) / f"transformer_ret5_{dataset_name}.pt"
        if model_path.exists():
            print(f"  [{dataset_name}] TF_Ret5 already exists, skipping")
            continue

        # Build valid indices
        valid_mask = ~np.isnan(ret5)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            print(f"  [{dataset_name}] Skip TF_Ret5: only {len(valid_indices)} valid samples")
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        train_idx = valid_indices[:split_idx]
        val_idx = valid_indices[split_idx:]

        print(f"  [{dataset_name}] Training TF_Ret5 (train={len(train_idx)}, val={len(val_idx)})...")
        ds_start = time.time()

        val_ic, _ = _train_seq_model_with_oom_recovery(
            features=features,
            labels=ret5,
            train_indices=train_idx,
            val_indices=val_idx,
            dataset_name=dataset_name,
            output_dir=output_dir,
            model_type="transformer",
            target="ret5",
            window_size=WINDOW_RET5,
            batch_size=BATCH_SIZE_W20,
        )
        if val_ic is not None:
            print(f"  [{dataset_name}] TF_Ret5 val_IC={val_ic:.4f} ({time.time()-ds_start:.1f}s)")
        else:
            print(f"  [{dataset_name}] TF_Ret5 FAILED ({time.time()-ds_start:.1f}s)")

        # GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase5_elapsed = time.time() - phase5_start
    print(f"\nPhase 5 complete: {phase5_elapsed:.1f}s ({phase5_elapsed/60:.1f} min)")

    # =========================================================================
    # Phase 6: TF_Ret60 (window=240, single target)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 6: Transformer_Ret60 Training (window=240)")
    print("=" * 60)

    phase6_start = time.time()
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        cache_path = os.path.join(output_dir, f"_cache_features_{dataset_name}.npy")
        features = np.load(cache_path)
        _, _, ret60, _, _ = load_dataset(data_dir, ds_idx)

        # Checkpoint resume
        model_path = Path(output_dir) / f"transformer_ret60_{dataset_name}.pt"
        if model_path.exists():
            print(f"  [{dataset_name}] TF_Ret60 already exists, skipping")
            continue

        # Build valid indices
        valid_mask = ~np.isnan(ret60)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            print(f"  [{dataset_name}] Skip TF_Ret60: only {len(valid_indices)} valid samples")
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        train_idx = valid_indices[:split_idx]
        val_idx = valid_indices[split_idx:]

        print(f"  [{dataset_name}] Training TF_Ret60 (train={len(train_idx)}, val={len(val_idx)})...")
        ds_start = time.time()

        val_ic, _ = _train_seq_model_with_oom_recovery(
            features=features,
            labels=ret60,
            train_indices=train_idx,
            val_indices=val_idx,
            dataset_name=dataset_name,
            output_dir=output_dir,
            model_type="transformer",
            target="ret60",
            window_size=WINDOW_RET60,
            batch_size=BATCH_SIZE_W240,
        )
        if val_ic is not None:
            print(f"  [{dataset_name}] TF_Ret60 val_IC={val_ic:.4f} ({time.time()-ds_start:.1f}s)")
        else:
            print(f"  [{dataset_name}] TF_Ret60 FAILED ({time.time()-ds_start:.1f}s)")

        # GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    phase6_elapsed = time.time() - phase6_start
    print(f"\nPhase 6 complete: {phase6_elapsed:.1f}s ({phase6_elapsed/60:.1f} min)")

    # =========================================================================
    # Save ensemble weights (v6 format)
    # =========================================================================
    # Initialize default v6 weights for all datasets
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        if dataset_name not in ensemble_weights:
            ensemble_weights[dataset_name] = {}
        ds_w = ensemble_weights[dataset_name]
        # Set v6 defaults if not already present
        ds_w.setdefault("ret5_w_local", 0.6)
        ds_w.setdefault("ret5_w_global", 0.0)
        ds_w.setdefault("ret5_w_gru_ret5", 0.2)
        ds_w.setdefault("ret5_w_tf_ret5", 0.1)
        ds_w.setdefault("ret5_w_extreme", 0.1)
        ds_w.setdefault("ret60_w_local", 0.5)
        ds_w.setdefault("ret60_w_global", 0.0)
        ds_w.setdefault("ret60_w_gru_ret60", 0.25)
        ds_w.setdefault("ret60_w_tf_ret60", 0.15)
        ds_w.setdefault("ret60_w_extreme", 0.1)

        # If extreme model doesn't exist for this dataset, zero out extreme weight
        ext_r5_path = Path(output_dir) / f"lgb_extreme_ret5_{dataset_name}.txt"
        ext_r60_path = Path(output_dir) / f"lgb_extreme_ret60_{dataset_name}.txt"
        if not ext_r5_path.exists():
            ds_w["ret5_w_extreme"] = 0.0
        if not ext_r60_path.exists():
            ds_w["ret60_w_extreme"] = 0.0

        # If sequence models don't exist, zero out their weights
        if not (Path(output_dir) / f"gru_ret5_{dataset_name}.pt").exists():
            ds_w["ret5_w_gru_ret5"] = 0.0
        if not (Path(output_dir) / f"gru_ret60_{dataset_name}.pt").exists():
            ds_w["ret60_w_gru_ret60"] = 0.0
        if not (Path(output_dir) / f"transformer_ret5_{dataset_name}.pt").exists():
            ds_w["ret5_w_tf_ret5"] = 0.0
        if not (Path(output_dir) / f"transformer_ret60_{dataset_name}.pt").exists():
            ds_w["ret60_w_tf_ret60"] = 0.0

        # Normalize weights to sum to 1.0
        for prefix in ["ret5", "ret60"]:
            keys = [k for k in ds_w if k.startswith(f"{prefix}_w_")]
            total = sum(ds_w[k] for k in keys)
            if total > 0:
                for k in keys:
                    ds_w[k] = round(ds_w[k] / total, 4)
            else:
                # Fallback: pure local
                ds_w[f"{prefix}_w_local"] = 1.0

    with open(weights_path, "w") as f:
        json.dump(ensemble_weights, f, indent=2)
    print(f"\nEnsemble weights saved: {weights_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"v6 Training complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Report total model size
    total_size = 0
    model_count = 0
    size_by_type = {"lgb_local": 0, "lgb_extreme": 0, "gru": 0, "transformer": 0, "other": 0}
    for f_name in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f_name)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath)
            total_size += fsize
            model_count += 1
            if "extreme" in f_name:
                size_by_type["lgb_extreme"] += fsize
            elif f_name.startswith("lgb_"):
                size_by_type["lgb_local"] += fsize
            elif f_name.startswith("gru_"):
                size_by_type["gru"] += fsize
            elif f_name.startswith("transformer_"):
                size_by_type["transformer"] += fsize
            else:
                size_by_type["other"] += fsize

    print(f"\nModel files: {model_count}")
    print(f"  LGB_Local:   {size_by_type['lgb_local']/1024/1024:.2f} MB")
    print(f"  LGB_Extreme: {size_by_type['lgb_extreme']/1024/1024:.2f} MB")
    print(f"  GRU:         {size_by_type['gru']/1024/1024:.2f} MB")
    print(f"  Transformer: {size_by_type['transformer']/1024/1024:.2f} MB")
    print(f"  Other:       {size_by_type['other']/1024/1024:.2f} MB")
    print(f"  TOTAL:       {total_size/1024/1024:.2f} MB")

    if total_size > 150 * 1024 * 1024:
        print("ERROR: Total size exceeds 150 MB submission limit!")
    elif total_size > 144 * 1024 * 1024:
        print("WARNING: Total size exceeds 144 MB safety margin!")
    else:
        print("OK: Total size within 150 MB submission limit.")

    # =========================================================================
    # v6: Auto-run evaluation after training (Task 11.2)
    # =========================================================================
    print(f"\n{'='*60}")
    print("Auto-running local evaluation...")
    print(f"{'='*60}")
    try:
        from evaluate_local import evaluate_all as run_evaluation
        eval_results = run_evaluation(data_dir, output_dir)
        summary = eval_results.get("_summary", {})
        print(f"\nv6 Evaluation complete:")
        print(f"  nR5={summary.get('mean_normal_ret5', float('nan')):.6f}  "
              f"nR60={summary.get('mean_normal_ret60', float('nan')):.6f}  "
              f"eR5={summary.get('mean_extreme_ret5', float('nan')):.6f}  "
              f"eR60={summary.get('mean_extreme_ret60', float('nan')):.6f}")
    except Exception as e:
        print(f"  [WARN] Auto-evaluation failed: {e}")
        print(f"  Run manually: python evaluate_local.py --data-dir {data_dir} --model-dir {output_dir}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train volatility return prediction models")
    parser.add_argument("--data-dir", default="train_dataset",
                        help="Directory containing training .npy files")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save trained model files")
    args = parser.parse_args()

    train_all_models(args.data_dir, args.output_dir)
