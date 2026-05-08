#!/usr/bin/env python3
"""
Local evaluation script for the Volatility Return Prediction system.

Runs the full factor → predict pipeline on training data and computes
Pearson IC scores matching the competition's exact evaluation logic.

Reports 4 IC values per dataset:
  - normal × Ret5
  - normal × Ret60
  - extreme × Ret5
  - extreme × Ret60

Plus mean IC across all 30 datasets for each category.

Usage:
    python evaluate_local.py [--data-dir train_dataset] [--model-dir models]
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path


# =============================================================================
# Pearson IC — exact match to competition reference implementation
# =============================================================================

def pearson_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Pearson IC between predictions and labels.
    Matches the competition evaluation server logic exactly:
    - Invalid values (NaN, ±Inf) are replaced with 0 BEFORE computing correlation
    - Replaced values still participate in mean and variance calculations
    - Returns NaN if n < 2 or denominator is 0
    """
    if predictions.shape[0] < 2:
        return float("nan")
    # Invalid → 0, then full participation in correlation
    p = np.nan_to_num(predictions.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(labels.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = p - p.mean()
    y = y - y.mean()
    denom = np.sqrt((p ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((p * y).sum() / denom)


# =============================================================================
# Normal / Extreme subset splitting
# =============================================================================

def build_extreme_mask(indices: np.ndarray, extreme_intervals: np.ndarray) -> np.ndarray:
    """
    Build a boolean mask marking rows that fall within any extreme interval.

    Args:
        indices: 1D array of integer row indices (from ohlcv[:, 0])
        extreme_intervals: (N, 2) array of [start, end] closed intervals

    Returns:
        Boolean array of shape (T,) — True for extreme rows
    """
    T = indices.shape[0]
    mask = np.zeros(T, dtype=np.bool_)

    if extreme_intervals.shape[0] == 0:
        return mask

    # Convert indices to int for comparison
    idx = indices.astype(np.int64)

    for k in range(extreme_intervals.shape[0]):
        start = int(extreme_intervals[k, 0])
        end = int(extreme_intervals[k, 1])
        # Mark rows where index falls within [start, end]
        mask |= (idx >= start) & (idx <= end)

    return mask


# =============================================================================
# Main evaluation
# =============================================================================

def check_submission_size(model_dir: str) -> dict:
    """
    Check submission package size and report v6 model file breakdown.

    Returns dict with size info and pass/fail status.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return {"total_mb": 0, "pass": True, "warning": False}

    categories = {
        "lgb_local": 0,
        "lgb_extreme": 0,
        "gru_ret5": 0,
        "gru_ret60": 0,
        "transformer_ret5": 0,
        "transformer_ret60": 0,
        "gru_legacy": 0,
        "transformer_legacy": 0,
        "other": 0,
    }

    total_bytes = 0
    for f in model_path.iterdir():
        if not f.is_file():
            continue
        size = f.stat().st_size
        total_bytes += size
        name = f.name

        if name.startswith("lgb_extreme_"):
            categories["lgb_extreme"] += size
        elif name.startswith("lgb_ret5_") or name.startswith("lgb_ret60_") or name.startswith("lgb_global"):
            categories["lgb_local"] += size
        elif name.startswith("gru_ret5_"):
            categories["gru_ret5"] += size
        elif name.startswith("gru_ret60_"):
            categories["gru_ret60"] += size
        elif name.startswith("transformer_ret5_"):
            categories["transformer_ret5"] += size
        elif name.startswith("transformer_ret60_"):
            categories["transformer_ret60"] += size
        elif name.startswith("gru_") and name.endswith(".pt"):
            categories["gru_legacy"] += size
        elif name.startswith("transformer_") and name.endswith(".pt"):
            categories["transformer_legacy"] += size
        else:
            categories["other"] += size

    total_mb = total_bytes / (1024 * 1024)
    warning = total_mb > 144
    fail = total_mb > 150

    return {
        "total_mb": total_mb,
        "categories": {k: v / (1024 * 1024) for k, v in categories.items()},
        "pass": not fail,
        "warning": warning,
    }


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage info if available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return {"allocated_gb": allocated, "reserved_gb": reserved, "total_gb": total}
    except Exception:
        pass
    return {}


def evaluate_all(data_dir: str, model_dir: str) -> dict:
    """
    v6: Run full evaluation on all 30 datasets.

    Supports all v6 model types:
    - LGB_Local, LGB_Extreme
    - Dual-window GRU (Ret5 w=20, Ret60 w=240)
    - Dual-window Transformer (Ret5 w=20, Ret60 w=240)

    Reports:
    - Per-dataset: nR5, nR60, eR5, eR60, inference time, GPU memory peak
    - Aggregate: mean IC across all datasets
    - Submission size check (> 144 MB warning, > 150 MB error)

    Returns:
        Dictionary with per-dataset and aggregate IC results.
    """
    # Import factor and predict modules
    from factor import generate_factors
    import predict
    predict.MODEL_DIR = Path(model_dir)

    num_datasets = 30
    results = {}

    # Accumulators for mean IC
    all_normal_ret5 = []
    all_normal_ret60 = []
    all_extreme_ret5 = []
    all_extreme_ret60 = []

    total_start = time.time()

    # Check GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("GPU: Not available (CPU inference)")
    except Exception:
        print("GPU: Detection failed (CPU inference)")

    print()
    print(f"{'Dataset':>10s}  {'T':>8s}  {'norm':>8s}  {'ext':>7s}  "
          f"{'nR5':>8s}  {'nR60':>8s}  {'eR5':>8s}  {'eR60':>8s}  "
          f"{'Time':>6s}  {'GPU_MB':>7s}")
    print("-" * 100)

    for ds_idx in range(num_datasets):
        dataset_name = f"dataset{ds_idx}"
        ds_start = time.time()

        # Load data
        ohlcv_path = os.path.join(data_dir, f"{dataset_name}_train_ohlcv.npy")
        ext_path = os.path.join(data_dir, f"{dataset_name}_train_extreme_intervals.npy")

        if not os.path.exists(ohlcv_path):
            print(f"{dataset_name:>10s}  SKIPPED (file not found)")
            continue

        raw = np.load(ohlcv_path)
        extreme_intervals = np.load(ext_path)

        indices = raw[:, 0]                          # row index column
        ohlcv = raw[:, 1:6].astype(np.float32)       # [open, high, low, close, volume]
        ret5_true = raw[:, 6]                         # Ret5 labels
        ret60_true = raw[:, 7]                        # Ret60 labels

        T = ohlcv.shape[0]

        # Reset GPU memory tracking
        gpu_peak_mb = 0
        if gpu_available:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # Generate factors
        factors = generate_factors(dataset_name, ohlcv)

        # Generate signals (uses all v6 model types: dual window, extreme, etc.)
        signals = predict.generate_signals(dataset_name, factors)

        pred_ret5 = signals[:, 0]
        pred_ret60 = signals[:, 1]

        # Get GPU peak memory
        if gpu_available:
            try:
                gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            except Exception:
                pass

        # Build extreme/normal masks
        extreme_mask = build_extreme_mask(indices, extreme_intervals)
        normal_mask = ~extreme_mask

        n_extreme = extreme_mask.sum()
        n_normal = normal_mask.sum()

        # Compute 4 IC values
        ic_normal_ret5 = pearson_ic(pred_ret5[normal_mask], ret5_true[normal_mask])
        ic_normal_ret60 = pearson_ic(pred_ret60[normal_mask], ret60_true[normal_mask])
        ic_extreme_ret5 = pearson_ic(pred_ret5[extreme_mask], ret5_true[extreme_mask])
        ic_extreme_ret60 = pearson_ic(pred_ret60[extreme_mask], ret60_true[extreme_mask])

        ds_elapsed = time.time() - ds_start

        results[dataset_name] = {
            "rows": T,
            "normal": int(n_normal),
            "extreme": int(n_extreme),
            "normal_ret5": ic_normal_ret5,
            "normal_ret60": ic_normal_ret60,
            "extreme_ret5": ic_extreme_ret5,
            "extreme_ret60": ic_extreme_ret60,
            "time": ds_elapsed,
            "gpu_peak_mb": gpu_peak_mb,
        }

        # Accumulate for mean (skip NaN)
        if not np.isnan(ic_normal_ret5):
            all_normal_ret5.append(ic_normal_ret5)
        if not np.isnan(ic_normal_ret60):
            all_normal_ret60.append(ic_normal_ret60)
        if not np.isnan(ic_extreme_ret5):
            all_extreme_ret5.append(ic_extreme_ret5)
        if not np.isnan(ic_extreme_ret60):
            all_extreme_ret60.append(ic_extreme_ret60)

        # Print per-dataset results
        print(f"{dataset_name:>10s}  "
              f"T={T:>8d}  "
              f"norm={n_normal:>8d}  ext={n_extreme:>7d}  "
              f"nR5={ic_normal_ret5:>8.4f}  nR60={ic_normal_ret60:>8.4f}  "
              f"eR5={ic_extreme_ret5:>8.4f}  eR60={ic_extreme_ret60:>8.4f}  "
              f"({ds_elapsed:>5.1f}s  {gpu_peak_mb:>6.0f})")

    total_elapsed = time.time() - total_start

    # Compute means
    mean_normal_ret5 = np.mean(all_normal_ret5) if all_normal_ret5 else float("nan")
    mean_normal_ret60 = np.mean(all_normal_ret60) if all_normal_ret60 else float("nan")
    mean_extreme_ret5 = np.mean(all_extreme_ret5) if all_extreme_ret5 else float("nan")
    mean_extreme_ret60 = np.mean(all_extreme_ret60) if all_extreme_ret60 else float("nan")

    # Print summary
    print()
    print("=" * 100)
    print("v6 EVALUATION SUMMARY")
    print("=" * 100)
    print(f"  Mean Normal  Ret5:  {mean_normal_ret5:>8.6f}  ({'PASS' if mean_normal_ret5 > 0 else 'FAIL'})")
    print(f"  Mean Normal  Ret60: {mean_normal_ret60:>8.6f}  ({'PASS' if mean_normal_ret60 > 0 else 'FAIL'})")
    print(f"  Mean Extreme Ret5:  {mean_extreme_ret5:>8.6f}  ({'PASS' if mean_extreme_ret5 > 0 else 'FAIL'})")
    print(f"  Mean Extreme Ret60: {mean_extreme_ret60:>8.6f}  ({'PASS' if mean_extreme_ret60 > 0 else 'FAIL'})")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Datasets evaluated: {len(results)}")
    print(f"  Valid IC counts: nR5={len(all_normal_ret5)}, nR60={len(all_normal_ret60)}, "
          f"eR5={len(all_extreme_ret5)}, eR60={len(all_extreme_ret60)}")

    # v6: Submission size check
    print()
    print("-" * 100)
    print("SUBMISSION SIZE CHECK")
    print("-" * 100)
    size_info = check_submission_size(model_dir)
    print(f"  Total size: {size_info['total_mb']:.2f} MB")
    if "categories" in size_info:
        for cat, size_mb in size_info["categories"].items():
            if size_mb > 0:
                print(f"    {cat:<25s}: {size_mb:>8.2f} MB")
    if not size_info["pass"]:
        print(f"  ERROR: Exceeds 150 MB platform limit!")
    elif size_info["warning"]:
        print(f"  WARNING: Exceeds 144 MB safety margin!")
    else:
        print(f"  ✓ Size check PASSED (< 144 MB)")
    print("=" * 100)

    results["_summary"] = {
        "mean_normal_ret5": mean_normal_ret5,
        "mean_normal_ret60": mean_normal_ret60,
        "mean_extreme_ret5": mean_extreme_ret5,
        "mean_extreme_ret60": mean_extreme_ret60,
        "total_time": total_elapsed,
        "submission_size_mb": size_info["total_mb"],
        "submission_size_pass": size_info["pass"],
    }

    return results


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6 local evaluation for volatility return prediction")
    parser.add_argument("--data-dir", default="train_dataset",
                        help="Directory containing training .npy files")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing trained model files")
    parser.add_argument("--check-size-only", action="store_true",
                        help="Only check submission size without running inference")
    args = parser.parse_args()

    if args.check_size_only:
        size_info = check_submission_size(args.model_dir)
        print(f"Total size: {size_info['total_mb']:.2f} MB")
        if "categories" in size_info:
            for cat, size_mb in size_info["categories"].items():
                if size_mb > 0:
                    print(f"  {cat:<25s}: {size_mb:>8.2f} MB")
        if not size_info["pass"]:
            print("ERROR: Exceeds 150 MB platform limit!")
            sys.exit(1)
        elif size_info["warning"]:
            print("WARNING: Exceeds 144 MB safety margin!")
        else:
            print("✓ Size check PASSED")
    else:
        results = evaluate_all(args.data_dir, args.model_dir)
