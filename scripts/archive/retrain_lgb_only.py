#!/usr/bin/env python3
"""
Quick LightGBM-only retraining script for Phase 1 checkpoint.
Retrains only the 60 LightGBM models with v3 two-phase training,
preserving existing GRU models and ensemble weights.
"""

import os
import sys
import time
import json
import numpy as np

from train import (
    load_dataset, set_all_seeds, RANDOM_SEED, NUM_DATASETS,
    LGB_PARAMS_RET5, LGB_PARAMS_RET60,
    NUM_BOOST_ROUND_RET5, NUM_BOOST_ROUND_RET60,
    train_single_model,
)
from factor import generate_factors


def retrain_lgb_only(data_dir: str, output_dir: str) -> None:
    """Retrain only LightGBM models with v3 two-phase training."""
    set_all_seeds(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    total_start = time.time()
    ret5_trees = []
    ret60_trees = []

    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        ds_start = time.time()
        print(f"\n{'='*60}")
        print(f"Training {dataset_name} (LightGBM only)")
        print(f"{'='*60}")

        # Load data
        ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(data_dir, ds_idx)
        print(f"  Loaded: {ohlcv.shape[0]} rows")

        # Generate features
        t0 = time.time()
        features = generate_factors(dataset_name, ohlcv)
        t1 = time.time()
        print(f"  Features: {features.shape} ({t1-t0:.1f}s)")

        # Train Ret5
        print(f"  Training LightGBM Ret5...")
        model_ret5 = train_single_model(
            features, ret5,
            LGB_PARAMS_RET5, NUM_BOOST_ROUND_RET5,
            dataset_name, "ret5",
        )
        path_ret5 = os.path.join(output_dir, f"lgb_ret5_{dataset_name}.txt")
        model_ret5.save_model(path_ret5)
        ret5_trees.append(model_ret5.num_trees())

        # Train Ret60
        print(f"  Training LightGBM Ret60...")
        model_ret60 = train_single_model(
            features, ret60,
            LGB_PARAMS_RET60, NUM_BOOST_ROUND_RET60,
            dataset_name, "ret60",
        )
        path_ret60 = os.path.join(output_dir, f"lgb_ret60_{dataset_name}.txt")
        model_ret60.save_model(path_ret60)
        ret60_trees.append(model_ret60.num_trees())

        ds_elapsed = time.time() - ds_start
        print(f"  Dataset time: {ds_elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"LightGBM Retraining Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\nRet5 tree counts: min={min(ret5_trees)}, max={max(ret5_trees)}, mean={np.mean(ret5_trees):.1f}")
    print(f"Ret60 tree counts: min={min(ret60_trees)}, max={max(ret60_trees)}, mean={np.mean(ret60_trees):.1f}")
    
    all_ret5_pass = all(t >= 30 for t in ret5_trees)
    print(f"\nAll Ret5 models >= 30 trees: {all_ret5_pass}")
    if not all_ret5_pass:
        for i, t in enumerate(ret5_trees):
            if t < 30:
                print(f"  FAIL: dataset{i} has {t} trees")


if __name__ == "__main__":
    retrain_lgb_only("train_dataset", "models_phase3")
