#!/usr/bin/env python3
"""
Standalone script to train global LightGBM models (Ret5 + Ret60).

Merges all 30 datasets with dataset_id as categorical feature,
then calls train_global_lgb for each target.

Usage:
    python train_global.py --data-dir train_dataset --output-dir models
    
    # If features are already cached (from Phase 1 of train.py):
    python train_global.py --data-dir train_dataset --output-dir models --cache-dir models
    
    # Control parallelism (default: all CPU cores):
    python train_global.py --data-dir train_dataset --output-dir models --workers 8
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

from train import (
    train_global_lgb,
    load_dataset,
    set_all_seeds,
    RANDOM_SEED,
    NUM_DATASETS,
    TRAIN_RATIO,
)
from factor import generate_factors


def _generate_features_for_dataset(args):
    """
    Worker function for parallel feature generation.
    
    Args:
        args: tuple of (ds_idx, data_dir, cache_dir, numba_threads)
    
    Returns:
        (ds_idx, features, ret5, ret60) or raises on error
    """
    ds_idx, data_dir, cache_dir, numba_threads = args
    dataset_name = f"dataset{ds_idx}"
    
    # Limit Numba threads per worker to avoid over-subscription
    if numba_threads:
        from numba import config as numba_config
        numba_config.NUMBA_NUM_THREADS = numba_threads
        import numba
        numba.set_num_threads(numba_threads)
    
    # Try loading cached features first
    features = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"_cache_features_{dataset_name}.npy")
        if os.path.exists(cache_path):
            features = np.load(cache_path)
    
    # Load raw data
    ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(data_dir, ds_idx)
    
    # Generate features if not cached
    if features is None:
        t0 = time.time()
        features = generate_factors(dataset_name, ohlcv)
        t1 = time.time()
        # Save cache for future use
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"_cache_features_{dataset_name}.npy")
            np.save(cache_path, features)
        print(f"  {dataset_name}: {features.shape} ({t1-t0:.1f}s)")
    else:
        print(f"  {dataset_name}: {features.shape} (cached)")
    
    return (ds_idx, features, ret5, ret60)


def build_global_dataset(data_dir: str, cache_dir: str = None, n_workers: int = None):
    """
    Load all 30 datasets, generate features in parallel, and build global train/val arrays.
    
    Args:
        data_dir: path to training data
        cache_dir: path to feature cache (None = no cache)
        n_workers: number of parallel workers (None = all CPU cores)
    
    Returns:
        X_train: (N_train, 166) — 165 features + dataset_id column
        y_train_r5: (N_train,)
        y_train_r60: (N_train,)
        X_val: (N_val, 166)
        y_val_r5: (N_val,)
        y_val_r60: (N_val,)
    """
    if n_workers is None:
        # Sweet spot: 8-10 workers on 32 cores
        # Each worker is single-threaded Numba, so we want enough workers
        # to keep cores busy, but not so many that memory bandwidth saturates.
        # 30 datasets with huge size variance → use imap_unordered for load balancing.
        n_workers = min(10, cpu_count(), NUM_DATASETS)
    
    total_cores = cpu_count()
    numba_threads_per_worker = 1  # Numba functions are single-threaded (no prange used)
    
    print(f"  Using {n_workers} workers (total cores: {total_cores})")
    
    # Sort datasets by size descending so large ones start first (better load balancing)
    # Estimate size from data files
    dataset_sizes = []
    for ds_idx in range(NUM_DATASETS):
        ohlcv_path = os.path.join(data_dir, f"dataset{ds_idx}_train_ohlcv.npy")
        try:
            size = os.path.getsize(ohlcv_path)
        except OSError:
            size = 0
        dataset_sizes.append((size, ds_idx))
    dataset_sizes.sort(reverse=True)  # largest first
    
    # Prepare arguments — largest datasets first for better scheduling
    worker_args = [(ds_idx, data_dir, cache_dir, numba_threads_per_worker) 
                   for _, ds_idx in dataset_sizes]
    
    # Parallel feature generation with load balancing
    t0 = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            # imap_unordered: as soon as a worker finishes, it picks up the next task
            # Combined with largest-first ordering, this minimizes idle time
            results = list(pool.imap_unordered(_generate_features_for_dataset, worker_args))
    else:
        results = [_generate_features_for_dataset(a) for a in worker_args]
    t1 = time.time()
    print(f"  All features generated in {t1-t0:.1f}s ({n_workers} workers)")
    
    # Sort by ds_idx to ensure consistent ordering
    results.sort(key=lambda x: x[0])
    
    # Build global arrays
    all_X_train = []
    all_y_train_r5 = []
    all_y_train_r60 = []
    all_X_val = []
    all_y_val_r5 = []
    all_y_val_r60 = []

    for ds_idx, features, ret5, ret60 in results:
        T = features.shape[0]
        
        # Temporal 80/20 split
        split_idx = int(T * TRAIN_RATIO)
        
        # Append dataset_id column (float32, used as categorical by LGB)
        id_col = np.full((T, 1), ds_idx, dtype=np.float32)
        X_with_id = np.hstack([features, id_col])  # (T, 166)
        
        # Split
        all_X_train.append(X_with_id[:split_idx])
        all_y_train_r5.append(ret5[:split_idx])
        all_y_train_r60.append(ret60[:split_idx])
        all_X_val.append(X_with_id[split_idx:])
        all_y_val_r5.append(ret5[split_idx:])
        all_y_val_r60.append(ret60[split_idx:])

    # Concatenate all datasets
    X_train = np.concatenate(all_X_train, axis=0)
    y_train_r5 = np.concatenate(all_y_train_r5, axis=0)
    y_train_r60 = np.concatenate(all_y_train_r60, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_val_r5 = np.concatenate(all_y_val_r5, axis=0)
    y_val_r60 = np.concatenate(all_y_val_r60, axis=0)

    print(f"\n  Global dataset built:")
    print(f"    Train: {X_train.shape} ({X_train.nbytes/1024**3:.1f} GB)")
    print(f"    Val:   {X_val.shape} ({X_val.nbytes/1024**3:.1f} GB)")
    
    return X_train, y_train_r5, y_train_r60, X_val, y_val_r5, y_val_r60


def _train_global_worker(args):
    """Worker function for parallel global LGB training."""
    X_train, y_train, X_val, y_val, target, output_dir, num_threads = args
    
    # Override num_threads in params to split cores between workers
    from train import (
        LGB_PARAMS_RET5, LGB_PARAMS_RET60, NUM_BOOST_ROUND_RET5, NUM_BOOST_ROUND_RET60,
        train_lgb_two_phase, ic_eval_metric, save_lgb_model_gz, MIN_BOOST_ROUND,
    )
    import lightgbm as lgb
    
    params = LGB_PARAMS_RET5.copy() if target == "ret5" else LGB_PARAMS_RET60.copy()
    params["num_threads"] = num_threads
    num_boost_round = NUM_BOOST_ROUND_RET5 if target == "ret5" else NUM_BOOST_ROUND_RET60

    # Clean NaN labels
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_v = X_val[val_valid]
    y_v = y_val[val_valid]

    print(f"  Global {target}: train={len(y_tr)}, val={len(y_v)}, threads={num_threads}")

    train_data = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=[165],
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
    print(f"  Global {target}: Trees={model.num_trees()}, Val IC={val_ic:.6f}")

    # Save
    path_gz = Path(output_dir) / f"lgb_{target}_global.txt.gz"
    save_lgb_model_gz(model, path_gz)
    print(f"  Saved: {path_gz}")
    return target, val_ic


def main():
    parser = argparse.ArgumentParser(description="Train global LightGBM models")
    parser.add_argument("--data-dir", default="train_dataset",
                        help="Directory containing training .npy files")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save global model files")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory with cached feature .npy files (default: same as output-dir)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for feature generation (default: auto)")
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = args.output_dir

    set_all_seeds(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if global models already exist
    path_r5 = Path(args.output_dir) / "lgb_ret5_global.txt.gz"
    path_r60 = Path(args.output_dir) / "lgb_ret60_global.txt.gz"
    
    if path_r5.exists() and path_r60.exists():
        print("Global models already exist, skipping.")
        print(f"  {path_r5}")
        print(f"  {path_r60}")
        return

    total_start = time.time()

    # Build global dataset (parallel feature generation)
    print("Loading data and generating features...")
    X_train, y_train_r5, y_train_r60, X_val, y_val_r5, y_val_r60 = \
        build_global_dataset(args.data_dir, args.cache_dir, n_workers=args.workers)

    # Train Ret5 and Ret60 in parallel — each gets half the cores
    total_cores = cpu_count()
    need_r5 = not path_r5.exists()
    need_r60 = not path_r60.exists()
    
    if need_r5 and need_r60:
        # Both needed: run in parallel, split cores 50/50
        threads_per_model = max(1, total_cores // 2)
        print(f"\nTraining Global LGB Ret5 + Ret60 in parallel ({threads_per_model} threads each)...")
        
        from multiprocessing import Process, Queue
        import multiprocessing as mp
        
        # Use fork-based processes to share the numpy arrays (copy-on-write)
        ctx = mp.get_context("fork")
        
        def _worker(target, y_train, y_val):
            _train_global_worker((X_train, y_train, X_val, y_val, 
                                  target, args.output_dir, threads_per_model))
        
        p1 = ctx.Process(target=_worker, args=("ret5", y_train_r5, y_val_r5))
        p2 = ctx.Process(target=_worker, args=("ret60", y_train_r60, y_val_r60))
        
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        if p1.exitcode != 0:
            print("ERROR: Global Ret5 training failed!")
        if p2.exitcode != 0:
            print("ERROR: Global Ret60 training failed!")
    else:
        # Only one needed: use all cores
        if need_r5:
            print(f"\nTraining Global LGB Ret5 ({total_cores} threads)...")
            _train_global_worker((X_train, y_train_r5, X_val, y_val_r5,
                                  "ret5", args.output_dir, total_cores))
        if need_r60:
            print(f"\nTraining Global LGB Ret60 ({total_cores} threads)...")
            _train_global_worker((X_train, y_train_r60, X_val, y_val_r60,
                                  "ret60", args.output_dir, total_cores))

    total_elapsed = time.time() - total_start
    print(f"\nGlobal training complete: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  {path_r5}")
    print(f"  {path_r60}")


if __name__ == "__main__":
    main()
