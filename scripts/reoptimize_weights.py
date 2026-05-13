#!/usr/bin/env python3
"""
v6: Full weight optimization on validation set.

For each dataset, loads all available model predictions on the validation set (last 20%),
then grid-searches the optimal weight combination to maximize IC.

Usage:
    python reoptimize_weights.py --data-dir train_dataset --models-dir models

Output:
    models/ensemble_weights.json
"""

import os
import sys
import time
import json
import gzip
import shutil
import tempfile
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

import lightgbm as lgb

from train import (
    load_dataset, set_all_seeds, ic_eval_metric, pearson_ic_numpy,
    RANDOM_SEED, NUM_DATASETS, TRAIN_RATIO,
    GRU_HIDDEN_SIZE, GRU_NUM_LAYERS, GRU_DROPOUT,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FF, TRANSFORMER_DROPOUT,
    WINDOW_RET5, WINDOW_RET60,
)
from factor import generate_factors


def _load_lgb_model(path_gz, path_txt):
    """Load LightGBM model, preferring .gz."""
    path_gz = Path(path_gz)
    path_txt = Path(path_txt)
    if path_gz.exists():
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        with gzip.open(str(path_gz), 'rb') as f_in:
            with open(tmp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        model = lgb.Booster(model_file=tmp_path)
        os.unlink(tmp_path)
        return model
    if path_txt.exists():
        return lgb.Booster(model_file=str(path_txt))
    return None


def _get_val_predictions_lgb(models_dir, dataset_name, features_val, features_with_id_val):
    """Get LGB predictions on validation set for all model types."""
    models_path = Path(models_dir)
    preds = {}
    
    # Local Ret5
    m = _load_lgb_model(models_path / f"lgb_ret5_{dataset_name}.txt.gz",
                        models_path / f"lgb_ret5_{dataset_name}.txt")
    preds["local_r5"] = m.predict(features_val).astype(np.float32) if m else None
    
    # Local Ret60
    m = _load_lgb_model(models_path / f"lgb_ret60_{dataset_name}.txt.gz",
                        models_path / f"lgb_ret60_{dataset_name}.txt")
    preds["local_r60"] = m.predict(features_val).astype(np.float32) if m else None
    
    # Global Ret5
    m = _load_lgb_model(models_path / "lgb_ret5_global.txt.gz",
                        models_path / "lgb_ret5_global.txt")
    preds["global_r5"] = m.predict(features_with_id_val).astype(np.float32) if m else None
    
    # Global Ret60
    m = _load_lgb_model(models_path / "lgb_ret60_global.txt.gz",
                        models_path / "lgb_ret60_global.txt")
    preds["global_r60"] = m.predict(features_with_id_val).astype(np.float32) if m else None
    
    # Extreme Ret5
    ext_r5_path = models_path / f"lgb_extreme_ret5_{dataset_name}.txt"
    if ext_r5_path.exists():
        m = lgb.Booster(model_file=str(ext_r5_path))
        preds["extreme_r5"] = m.predict(features_val).astype(np.float32)
    else:
        preds["extreme_r5"] = None
    
    # Extreme Ret60
    ext_r60_path = models_path / f"lgb_extreme_ret60_{dataset_name}.txt"
    if ext_r60_path.exists():
        m = lgb.Booster(model_file=str(ext_r60_path))
        preds["extreme_r60"] = m.predict(features_val).astype(np.float32)
    else:
        preds["extreme_r60"] = None
    
    return preds


def _get_val_predictions_seq(models_dir, dataset_name, features, val_indices,
                              target, window_size, device):
    """Get sequence model predictions on validation set."""
    import torch
    from torch import nn
    
    models_path = Path(models_dir)
    T_full, F = features.shape
    N_val = len(val_indices)
    
    # Build padded array
    clean = np.nan_to_num(features, nan=0.0).astype(np.float32)
    padded = np.zeros((window_size - 1 + T_full, F), dtype=np.float32)
    padded[window_size - 1:] = clean
    del clean
    
    # Clear GPU before starting
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    results = {}
    # w=20: batch=65536 (fast, ~2GB input)
    # w=240: batch=4096 (GRU hidden states ~12GB, fits 32GB GPU)
    batch_size = 65536 if window_size <= 60 else 4096
    
    for model_type in ["gru", "transformer"]:
        model_path = models_path / f"{model_type}_{target}_{dataset_name}.pt"
        if not model_path.exists():
            results[model_type] = None
            continue
        
        try:
            checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                results[model_type] = None
                continue
            
            input_size = checkpoint.get("input_size", F)
            
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            
            if model_type == "gru":
                hidden_size = checkpoint.get("hidden_size", GRU_HIDDEN_SIZE)
                num_layers = checkpoint.get("num_layers", GRU_NUM_LAYERS)
                dropout = checkpoint.get("dropout", GRU_DROPOUT)
                
                class _GRU(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                          num_layers=num_layers, batch_first=True,
                                          dropout=dropout if num_layers > 1 else 0.0)
                        self.fc = nn.Linear(hidden_size, 1)
                    def forward(self, x):
                        _, h_n = self.gru(x)
                        return self.fc(h_n[-1]).squeeze(-1)
                
                model = _GRU().to(device)
            else:
                d_model = checkpoint.get("d_model", TRANSFORMER_D_MODEL)
                nhead = checkpoint.get("nhead", TRANSFORMER_NHEAD)
                n_layers = checkpoint.get("num_layers", TRANSFORMER_NUM_LAYERS)
                dim_ff = checkpoint.get("dim_feedforward", TRANSFORMER_DIM_FF)
                dropout = checkpoint.get("dropout", TRANSFORMER_DROPOUT)
                ckpt_window = checkpoint.get("window_size", window_size)
                
                class _TF(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.input_proj = nn.Linear(input_size, d_model)
                        self.pos_embedding = nn.Parameter(torch.randn(1, ckpt_window, d_model) * 0.02)
                        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                            dropout=dropout, batch_first=True, norm_first=True)
                        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                              norm=nn.LayerNorm(d_model))
                        self.output_head = nn.Linear(d_model, 1)
                    def forward(self, x):
                        h = self.input_proj(x)
                        h = h + self.pos_embedding[:, :h.size(1)]
                        h = self.encoder(h)
                        return self.output_head(h[:, -1, :]).squeeze(-1)
                
                model = _TF().to(device)
            
            model.load_state_dict(checkpoint["state_dict"])
            del checkpoint
            model.eval()
            
            pred = np.empty(N_val, dtype=np.float32)
            with torch.no_grad():
                for start in range(0, N_val, batch_size):
                    end = min(start + batch_size, N_val)
                    batch_idx = val_indices[start:end]
                    # Direct slice from padded array (no stride_tricks to avoid memory explosion)
                    n_b = len(batch_idx)
                    batch_windows = np.empty((n_b, window_size, F), dtype=np.float32)
                    for k in range(n_b):
                        s = batch_idx[k]
                        batch_windows[k] = padded[s:s + window_size]
                    inp = torch.from_numpy(batch_windows).to(device)
                    out = model(inp).cpu().numpy()
                    pred[start:end] = out
                    del inp, batch_windows
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            
            results[model_type] = pred
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"    [WARN] {model_type}_{target} failed: {e}")
            results[model_type] = None
    
    del padded
    return results


def grid_search_weights(preds_dict, labels, step=0.1):
    """
    Grid search over weight combinations that sum to 1.0.
    
    preds_dict: {name: (N,) array or None}
    labels: (N,) array
    
    Returns: (best_weights_dict, best_ic)
    """
    # Filter to available predictions only
    available = {k: v for k, v in preds_dict.items() if v is not None}
    if not available:
        return {}, 0.0
    
    names = list(available.keys())
    arrays = [available[k] for k in names]
    n_models = len(names)
    
    if n_models == 1:
        ic = pearson_ic_numpy(arrays[0], labels)
        return {names[0]: 1.0}, ic
    
    # Generate all weight combinations summing to 1.0
    values = np.arange(0.0, 1.0 + step / 2, step)
    
    best_ic = -np.inf
    best_weights = {k: 0.0 for k in names}
    best_weights[names[0]] = 1.0  # default: first model only
    
    # For 2-5 models, enumerate all valid combinations
    from itertools import product as iter_product
    
    for combo in iter_product(values, repeat=n_models - 1):
        last_w = 1.0 - sum(combo)
        if last_w < -1e-9 or last_w > 1.0 + 1e-9:
            continue
        last_w = max(0.0, min(1.0, last_w))
        
        weights = list(combo) + [last_w]
        
        # Compute blended prediction
        blended = np.zeros_like(labels, dtype=np.float64)
        for w, arr in zip(weights, arrays):
            if w > 0:
                blended += w * arr
        
        ic = pearson_ic_numpy(blended.astype(np.float32), labels)
        if ic > best_ic:
            best_ic = ic
            best_weights = {names[i]: round(weights[i], 2) for i in range(n_models)}
    
    return best_weights, best_ic


def optimize_from_feedback(current_weights: dict, feedback_dir: str, models_dir: str) -> dict:
    """
    Adjust weights based on platform feedback results.
    
    Reads JSON files from feedback_dir/ containing per-dataset IC scores
    from the actual platform. Adjusts weights to:
    - Increase weight of models that improved IC on platform
    - Decrease weight of models that hurt IC on platform
    - Shift toward pure local LGB for datasets where ensemble degraded
    
    Expected feedback JSON format (feedback_dir/iter_N.json):
    {
        "iteration": N,
        "results": {
            "dataset0": {"nR5": 0.05, "nR60": 0.16, "eR5": 0.56, "eR60": 0.95},
            ...
        },
        "baseline": {
            "dataset0": {"nR5": 0.04, "nR60": 0.15, "eR5": 0.55, "eR60": 0.94},
            ...
        }
    }
    
    Returns updated weights dict.
    """
    feedback_path = Path(feedback_dir)
    if not feedback_path.exists():
        print(f"  Feedback dir not found: {feedback_dir}")
        return current_weights
    
    # Find latest feedback iteration
    iter_files = sorted(feedback_path.glob("iter_*.json"))
    if not iter_files:
        print("  No feedback iterations found.")
        return current_weights
    
    latest_iter = iter_files[-1]
    print(f"  Reading feedback from: {latest_iter.name}")
    
    try:
        with open(latest_iter) as f:
            feedback = json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to read feedback: {e}")
        return current_weights
    
    results = feedback.get("results", {})
    baseline = feedback.get("baseline", {})
    
    RET5_KEYS = ["ret5_w_local", "ret5_w_global", "ret5_w_gru_ret5", "ret5_w_tf_ret5", "ret5_w_extreme"]
    RET60_KEYS = ["ret60_w_local", "ret60_w_global", "ret60_w_gru_ret60", "ret60_w_tf_ret60", "ret60_w_extreme"]
    
    updated = {}
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        ds_w = current_weights.get(dataset_name, {}).copy()
        
        if dataset_name not in results:
            updated[dataset_name] = ds_w
            continue
        
        ds_result = results[dataset_name]
        ds_baseline = baseline.get(dataset_name, {})
        
        # Compare platform IC with baseline
        # If ensemble IC < baseline IC, shift toward local
        for target, keys, ic_keys in [
            ("ret5", RET5_KEYS, ["nR5", "eR5"]),
            ("ret60", RET60_KEYS, ["nR60", "eR60"]),
        ]:
            # Average IC across normal and extreme
            result_ic = np.mean([ds_result.get(k, 0.0) for k in ic_keys])
            baseline_ic = np.mean([ds_baseline.get(k, 0.0) for k in ic_keys])
            
            if baseline_ic == 0 and result_ic == 0:
                continue
            
            # If platform IC degraded vs baseline, shift toward local
            if result_ic < baseline_ic - 0.005:
                # Degraded: increase local weight by 0.1, decrease others proportionally
                local_key = f"{target}_w_local"
                current_local = ds_w.get(local_key, 0.5)
                boost = min(0.2, 1.0 - current_local)
                ds_w[local_key] = current_local + boost
                
                # Reduce non-local weights proportionally
                non_local_keys = [k for k in keys if k != local_key and ds_w.get(k, 0) > 0]
                if non_local_keys:
                    reduce_each = boost / len(non_local_keys)
                    for k in non_local_keys:
                        ds_w[k] = max(0.0, ds_w.get(k, 0) - reduce_each)
                
                print(f"    {dataset_name} {target}: degraded ({result_ic:.4f} < {baseline_ic:.4f}), "
                      f"shifting to local")
            
            elif result_ic > baseline_ic + 0.01:
                # Improved significantly: keep current weights (they're working)
                pass
            
            # Normalize to sum = 1.0
            total = sum(ds_w.get(k, 0.0) for k in keys)
            if total > 0:
                for k in keys:
                    if k in ds_w:
                        ds_w[k] = round(ds_w[k] / total, 4)
            else:
                ds_w[f"{target}_w_local"] = 1.0
        
        updated[dataset_name] = ds_w
    
    return updated


def optimize_dataset(ds_idx, data_dir, models_dir, device):
    """Optimize weights for a single dataset. Returns (dataset_name, weights_dict)."""
    import torch
    
    dataset_name = f"dataset{ds_idx}"
    models_path = Path(models_dir)
    
    # Load data
    ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(data_dir, ds_idx)
    
    # Load or generate features
    cache_path = os.path.join(models_dir, f"_cache_features_{dataset_name}.npy")
    if os.path.exists(cache_path):
        features = np.load(cache_path)
    else:
        features = generate_factors(dataset_name, ohlcv)
        np.save(cache_path, features)
    
    T = features.shape[0]
    F = features.shape[1]
    
    # Temporal split — use validation set (last 20%)
    split_idx = int(T * TRAIN_RATIO)
    val_indices = np.arange(split_idx, T)
    features_val = features[split_idx:]
    ret5_val = ret5[split_idx:]
    ret60_val = ret60[split_idx:]
    
    # Features with dataset ID for global model
    id_col = np.full((len(features_val), 1), ds_idx, dtype=np.float32)
    features_with_id_val = np.hstack([features_val, id_col])
    
    # Get LGB predictions
    lgb_preds = _get_val_predictions_lgb(models_dir, dataset_name, features_val, features_with_id_val)
    
    # Get sequence model predictions
    seq_ret5 = _get_val_predictions_seq(models_dir, dataset_name, features, val_indices,
                                         "ret5", WINDOW_RET5, device)
    seq_ret60 = _get_val_predictions_seq(models_dir, dataset_name, features, val_indices,
                                          "ret60", WINDOW_RET60, device)
    
    # --- Optimize Ret5 weights ---
    r5_preds = {}
    if lgb_preds["local_r5"] is not None:
        r5_preds["ret5_w_local"] = lgb_preds["local_r5"]
    if lgb_preds["global_r5"] is not None:
        r5_preds["ret5_w_global"] = lgb_preds["global_r5"]
    if seq_ret5.get("gru") is not None:
        r5_preds["ret5_w_gru_ret5"] = seq_ret5["gru"]
    if seq_ret5.get("transformer") is not None:
        r5_preds["ret5_w_tf_ret5"] = seq_ret5["transformer"]
    # Note: extreme model uses indicator weighting, handle separately
    
    # Clean labels for IC computation
    r5_labels = np.nan_to_num(ret5_val, nan=0.0).astype(np.float32)
    r60_labels = np.nan_to_num(ret60_val, nan=0.0).astype(np.float32)
    
    r5_weights, r5_ic = grid_search_weights(r5_preds, r5_labels)
    
    # --- Optimize Ret60 weights ---
    r60_preds = {}
    if lgb_preds["local_r60"] is not None:
        r60_preds["ret60_w_local"] = lgb_preds["local_r60"]
    if lgb_preds["global_r60"] is not None:
        r60_preds["ret60_w_global"] = lgb_preds["global_r60"]
    if seq_ret60.get("gru") is not None:
        r60_preds["ret60_w_gru_ret60"] = seq_ret60["gru"]
    if seq_ret60.get("transformer") is not None:
        r60_preds["ret60_w_tf_ret60"] = seq_ret60["transformer"]
    
    r60_weights, r60_ic = grid_search_weights(r60_preds, r60_labels)
    
    # --- Safety constraint: if IC improvement < 0.002 over pure local, use pure local ---
    local_only_r5_ic = pearson_ic_numpy(lgb_preds["local_r5"], r5_labels) if lgb_preds["local_r5"] is not None else 0.0
    local_only_r60_ic = pearson_ic_numpy(lgb_preds["local_r60"], r60_labels) if lgb_preds["local_r60"] is not None else 0.0
    
    if r5_ic - local_only_r5_ic < 0.002:
        r5_weights = {"ret5_w_local": 1.0}
        r5_ic = local_only_r5_ic
    
    if r60_ic - local_only_r60_ic < 0.002:
        r60_weights = {"ret60_w_local": 1.0}
        r60_ic = local_only_r60_ic
    
    # Merge into single dict
    ds_weights = {}
    for k in ["ret5_w_local", "ret5_w_global", "ret5_w_gru_ret5", "ret5_w_tf_ret5", "ret5_w_extreme"]:
        ds_weights[k] = r5_weights.get(k, 0.0)
    for k in ["ret60_w_local", "ret60_w_global", "ret60_w_gru_ret60", "ret60_w_tf_ret60", "ret60_w_extreme"]:
        ds_weights[k] = r60_weights.get(k, 0.0)
    
    print(f"  {dataset_name}: R5 IC={r5_ic:.4f} {r5_weights} | R60 IC={r60_ic:.4f} {r60_weights}")
    
    return dataset_name, ds_weights


def prune_models(weights: dict, models_dir: str, size_limit_mb: float = 144.0) -> list:
    """
    Remove zero-weight model files from submission.
    If total > size_limit_mb, remove lowest-weight sequence models.
    Never removes LightGBM models.
    
    Returns list of files to include in submission.
    """
    models_path = Path(models_dir)
    
    # Map weight keys to file patterns
    WEIGHT_TO_FILES = {
        "ret5_w_local": lambda ds: [f"lgb_ret5_{ds}.txt.gz", f"lgb_ret5_{ds}.txt"],
        "ret5_w_global": lambda ds: ["lgb_ret5_global.txt.gz", "lgb_ret5_global.txt"],
        "ret5_w_gru_ret5": lambda ds: [f"gru_ret5_{ds}.pt"],
        "ret5_w_tf_ret5": lambda ds: [f"transformer_ret5_{ds}.pt"],
        "ret5_w_extreme": lambda ds: [f"lgb_extreme_ret5_{ds}.txt"],
        "ret60_w_local": lambda ds: [f"lgb_ret60_{ds}.txt.gz", f"lgb_ret60_{ds}.txt"],
        "ret60_w_global": lambda ds: ["lgb_ret60_global.txt.gz", "lgb_ret60_global.txt"],
        "ret60_w_gru_ret60": lambda ds: [f"gru_ret60_{ds}.pt"],
        "ret60_w_tf_ret60": lambda ds: [f"transformer_ret60_{ds}.pt"],
        "ret60_w_extreme": lambda ds: [f"lgb_extreme_ret60_{ds}.txt"],
    }
    
    # Start with all existing files
    include_files = set()
    for f in models_path.iterdir():
        if f.is_file() and not f.name.startswith("_cache_"):
            include_files.add(f.name)
    
    # Remove zero-weight sequence/extreme model files
    exclude_files = set()
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        ds_w = weights.get(dataset_name, {})
        
        for key, file_fn in WEIGHT_TO_FILES.items():
            weight_val = ds_w.get(key, 0.0)
            if weight_val == 0.0:
                files = file_fn(dataset_name)
                for f in files:
                    if f in include_files:
                        # Never remove LGB local models (backbone)
                        if "extreme" in f or f.endswith(".pt"):
                            exclude_files.add(f)
    
    include_files -= exclude_files
    
    # Calculate total size
    total_size = sum(
        (models_path / f).stat().st_size
        for f in include_files
        if (models_path / f).exists()
    )
    total_mb = total_size / (1024 * 1024)
    
    # If still over limit, remove lowest-weight sequence models
    if total_mb > size_limit_mb:
        print(f"  Total {total_mb:.1f} MB > {size_limit_mb} MB, pruning sequence models...")
        seq_files_with_weight = []
        for ds_idx in range(NUM_DATASETS):
            dataset_name = f"dataset{ds_idx}"
            ds_w = weights.get(dataset_name, {})
            for key in ["ret5_w_gru_ret5", "ret5_w_tf_ret5", "ret60_w_gru_ret60", "ret60_w_tf_ret60"]:
                w = ds_w.get(key, 0.0)
                files = WEIGHT_TO_FILES[key](dataset_name)
                for f in files:
                    if f in include_files and (models_path / f).exists():
                        fsize = (models_path / f).stat().st_size
                        seq_files_with_weight.append((w, fsize, f))
        
        # Sort by weight ascending — remove lowest weight first
        seq_files_with_weight.sort(key=lambda x: x[0])
        
        for w, fsize, fname in seq_files_with_weight:
            if total_mb <= size_limit_mb:
                break
            include_files.discard(fname)
            total_mb -= fsize / (1024 * 1024)
            print(f"    Removed {fname} (weight={w:.2f}, {fsize/1024:.0f} KB)")
    
    print(f"  Final submission: {total_mb:.1f} MB ({len(include_files)} files)")
    return sorted(include_files)


def main():
    parser = argparse.ArgumentParser(description="v6 weight optimization")
    parser.add_argument("--data-dir", default="train_dataset")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default=None,
                        help="Output path (default: models-dir/ensemble_weights.json)")
    parser.add_argument("--prune", action="store_true",
                        help="Remove zero-weight models and generate submission_manifest.json")
    parser.add_argument("--mode", choices=["local", "feedback"], default="local",
                        help="Optimization mode: local (validation set) or feedback (platform results)")
    parser.add_argument("--feedback-dir", default="feedback_state",
                        help="Directory containing platform feedback JSON files (for feedback mode)")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(args.models_dir, "ensemble_weights.json")
    
    set_all_seeds(RANDOM_SEED)
    
    print(f"v6 Weight Optimization")
    print(f"  Mode: {args.mode}")
    print(f"  Data: {args.data_dir}")
    print(f"  Models: {args.models_dir}")
    print(f"  Output: {args.output}")
    
    total_start = time.time()
    
    if args.mode == "local":
        # Full grid search on validation set
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"  Device: {device}")
        
        all_weights = {}
        for ds_idx in range(NUM_DATASETS):
            ds_name, ds_w = optimize_dataset(ds_idx, args.data_dir, args.models_dir, device)
            all_weights[ds_name] = ds_w
    
    elif args.mode == "feedback":
        # Load current weights and adjust based on platform feedback
        current_weights = {}
        if os.path.exists(args.output):
            with open(args.output) as f:
                current_weights = json.load(f)
        else:
            # Initialize with pure local defaults
            for ds_idx in range(NUM_DATASETS):
                current_weights[f"dataset{ds_idx}"] = {
                    "ret5_w_local": 1.0, "ret5_w_global": 0.0,
                    "ret5_w_gru_ret5": 0.0, "ret5_w_tf_ret5": 0.0, "ret5_w_extreme": 0.0,
                    "ret60_w_local": 1.0, "ret60_w_global": 0.0,
                    "ret60_w_gru_ret60": 0.0, "ret60_w_tf_ret60": 0.0, "ret60_w_extreme": 0.0,
                }
        
        all_weights = optimize_from_feedback(current_weights, args.feedback_dir, args.models_dir)
    
    # Save weights
    with open(args.output, "w") as f:
        json.dump(all_weights, f, indent=2)
    
    elapsed = time.time() - total_start
    print(f"\nOptimization done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Saved: {args.output}")
    
    # Estimate submission size
    models_path = Path(args.models_dir)
    total_size = sum(
        f.stat().st_size for f in models_path.iterdir()
        if f.is_file() and not f.name.startswith("_cache_")
    )
    total_mb = total_size / (1024 * 1024)
    print(f"\nTotal model files: {total_mb:.1f} MB")
    if total_mb > 150:
        print("  ERROR: Exceeds 150 MB platform limit!")
    elif total_mb > 144:
        print("  WARNING: Exceeds 144 MB safety margin!")
    else:
        print("  OK: Within 150 MB limit.")
    
    # Prune if requested
    if args.prune:
        print(f"\nPruning zero-weight models...")
        include_files = prune_models(all_weights, args.models_dir)
        
        # Generate submission manifest
        manifest_path = os.path.join(args.models_dir, "submission_manifest.json")
        manifest = {
            "files": include_files,
            "total_files": len(include_files),
            "weights_file": "ensemble_weights.json",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
