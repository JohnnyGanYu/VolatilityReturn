#!/usr/bin/env python3
"""
Generate multiple submission variants for A/B testing on the platform.

Creates different ensemble_weights.json files representing different strategies:
- Plan A: Pure local LGB (safest baseline, like v5)
- Plan B: Local + Global LGB only (no sequence models, small package)
- Plan C: Full v6 optimized (from reoptimize_weights.py results)
- Plan D: Aggressive sequence models (higher seq weights for datasets where they help)

Usage:
    python generate_submissions.py --data-dir train_dataset --models-dir models
    
Output:
    submissions/plan_a/  (pure local)
    submissions/plan_b/  (local + global)
    submissions/plan_c/  (full optimized)
    submissions/plan_d/  (aggressive seq)
"""

import os
import sys
import json
import shutil
import gzip
import tempfile
import numpy as np
from pathlib import Path

# Add current dir to path
sys.path.insert(0, '.')
from train import load_dataset, set_all_seeds, RANDOM_SEED, NUM_DATASETS, TRAIN_RATIO
from factor import generate_factors

import lightgbm as lgb


def _load_lgb_model(path_gz, path_txt):
    path_gz, path_txt = Path(path_gz), Path(path_txt)
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


def pearson_ic(pred, labels):
    if len(pred) < 2:
        return 0.0
    p = np.nan_to_num(pred.astype(np.float64), nan=0.0)
    y = np.nan_to_num(labels.astype(np.float64), nan=0.0)
    p = p - p.mean()
    y = y - y.mean()
    denom = np.sqrt((p**2).sum() * (y**2).sum())
    return float((p*y).sum() / denom) if denom > 0 else 0.0


def evaluate_lgb_variants(data_dir, models_dir):
    """Evaluate local vs global LGB on validation set for each dataset."""
    models_path = Path(models_dir)
    results = {}
    
    for ds_idx in range(NUM_DATASETS):
        dataset_name = f"dataset{ds_idx}"
        
        # Load data
        ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(data_dir, ds_idx)
        
        # Load or generate features
        cache_path = models_path / f"_cache_features_{dataset_name}.npy"
        if cache_path.exists():
            features = np.load(str(cache_path))
        else:
            features = generate_factors(dataset_name, ohlcv)
            np.save(str(cache_path), features)
        
        T = features.shape[0]
        split_idx = int(T * TRAIN_RATIO)
        features_val = features[split_idx:]
        ret5_val = np.nan_to_num(ret5[split_idx:], nan=0.0).astype(np.float32)
        ret60_val = np.nan_to_num(ret60[split_idx:], nan=0.0).astype(np.float32)
        
        # Features with ID for global model
        id_col = np.full((len(features_val), 1), ds_idx, dtype=np.float32)
        features_with_id = np.hstack([features_val, id_col])
        
        # Local predictions
        m = _load_lgb_model(models_path / f"lgb_ret5_{dataset_name}.txt.gz",
                           models_path / f"lgb_ret5_{dataset_name}.txt")
        local_r5 = m.predict(features_val).astype(np.float32) if m else np.zeros(len(features_val))
        
        m = _load_lgb_model(models_path / f"lgb_ret60_{dataset_name}.txt.gz",
                           models_path / f"lgb_ret60_{dataset_name}.txt")
        local_r60 = m.predict(features_val).astype(np.float32) if m else np.zeros(len(features_val))
        
        # Global predictions
        m = _load_lgb_model(models_path / "lgb_ret5_global.txt.gz",
                           models_path / "lgb_ret5_global.txt")
        global_r5 = m.predict(features_with_id).astype(np.float32) if m else local_r5.copy()
        
        m = _load_lgb_model(models_path / "lgb_ret60_global.txt.gz",
                           models_path / "lgb_ret60_global.txt")
        global_r60 = m.predict(features_with_id).astype(np.float32) if m else local_r60.copy()
        
        # Compute ICs
        ic_local_r5 = pearson_ic(local_r5, ret5_val)
        ic_local_r60 = pearson_ic(local_r60, ret60_val)
        ic_global_r5 = pearson_ic(global_r5, ret5_val)
        ic_global_r60 = pearson_ic(global_r60, ret60_val)
        
        # Grid search local+global blend
        best_w_r5, best_ic_r5 = 1.0, ic_local_r5
        best_w_r60, best_ic_r60 = 1.0, ic_local_r60
        
        for w in np.arange(0.0, 1.05, 0.1):
            blend_r5 = w * local_r5 + (1-w) * global_r5
            ic = pearson_ic(blend_r5, ret5_val)
            if ic > best_ic_r5:
                best_ic_r5 = ic
                best_w_r5 = round(w, 1)
            
            blend_r60 = w * local_r60 + (1-w) * global_r60
            ic = pearson_ic(blend_r60, ret60_val)
            if ic > best_ic_r60:
                best_ic_r60 = ic
                best_w_r60 = round(w, 1)
        
        results[dataset_name] = {
            "ic_local_r5": ic_local_r5,
            "ic_local_r60": ic_local_r60,
            "ic_global_r5": ic_global_r5,
            "ic_global_r60": ic_global_r60,
            "best_blend_w_r5": best_w_r5,  # weight for local
            "best_blend_ic_r5": best_ic_r5,
            "best_blend_w_r60": best_w_r60,
            "best_blend_ic_r60": best_ic_r60,
        }
        
        print(f"  {dataset_name}: "
              f"R5 local={ic_local_r5:.4f} global={ic_global_r5:.4f} best={best_ic_r5:.4f}(L{best_w_r5}) | "
              f"R60 local={ic_local_r60:.4f} global={ic_global_r60:.4f} best={best_ic_r60:.4f}(L{best_w_r60})")
    
    return results


def generate_plan_a(results):
    """Plan A: Pure local LGB (safest, like v5 baseline)"""
    weights = {}
    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        weights[ds] = {
            "ret5_w_local": 1.0, "ret5_w_global": 0.0,
            "ret5_w_gru_ret5": 0.0, "ret5_w_tf_ret5": 0.0, "ret5_w_extreme": 0.0,
            "ret60_w_local": 1.0, "ret60_w_global": 0.0,
            "ret60_w_gru_ret60": 0.0, "ret60_w_tf_ret60": 0.0, "ret60_w_extreme": 0.0,
        }
    return weights


def generate_plan_b(results):
    """Plan B: Local + Global LGB blend (no seq models, small package)"""
    weights = {}
    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        r = results[ds]
        
        w_local_r5 = r["best_blend_w_r5"]
        w_local_r60 = r["best_blend_w_r60"]
        
        # Safety: if blend doesn't improve over local by 0.001, use pure local
        if r["best_blend_ic_r5"] - r["ic_local_r5"] < 0.001:
            w_local_r5 = 1.0
        if r["best_blend_ic_r60"] - r["ic_local_r60"] < 0.001:
            w_local_r60 = 1.0
        
        weights[ds] = {
            "ret5_w_local": w_local_r5, "ret5_w_global": round(1.0 - w_local_r5, 1),
            "ret5_w_gru_ret5": 0.0, "ret5_w_tf_ret5": 0.0, "ret5_w_extreme": 0.0,
            "ret60_w_local": w_local_r60, "ret60_w_global": round(1.0 - w_local_r60, 1),
            "ret60_w_gru_ret60": 0.0, "ret60_w_tf_ret60": 0.0, "ret60_w_extreme": 0.0,
        }
    return weights


def generate_plan_c(results):
    """Plan C: Local + Global with conservative seq model weights (0.1 each where available)"""
    weights = {}
    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        r = results[ds]
        
        # Start with best LGB blend
        w_local_r5 = r["best_blend_w_r5"]
        w_global_r5 = round(1.0 - w_local_r5, 1)
        w_local_r60 = r["best_blend_w_r60"]
        w_global_r60 = round(1.0 - w_local_r60, 1)
        
        # Add small seq model weights (take from the larger LGB weight)
        w_gru_r5 = 0.1
        w_tf_r5 = 0.1
        # Reduce LGB weights proportionally
        lgb_total_r5 = w_local_r5 + w_global_r5
        remaining_r5 = 1.0 - w_gru_r5 - w_tf_r5
        if lgb_total_r5 > 0:
            w_local_r5 = round(w_local_r5 / lgb_total_r5 * remaining_r5, 2)
            w_global_r5 = round(remaining_r5 - w_local_r5, 2)
        
        w_gru_r60 = 0.1
        w_tf_r60 = 0.1
        lgb_total_r60 = w_local_r60 + w_global_r60
        remaining_r60 = 1.0 - w_gru_r60 - w_tf_r60
        if lgb_total_r60 > 0:
            w_local_r60 = round(w_local_r60 / lgb_total_r60 * remaining_r60, 2)
            w_global_r60 = round(remaining_r60 - w_local_r60, 2)
        
        weights[ds] = {
            "ret5_w_local": max(0, w_local_r5), "ret5_w_global": max(0, w_global_r5),
            "ret5_w_gru_ret5": w_gru_r5, "ret5_w_tf_ret5": w_tf_r5, "ret5_w_extreme": 0.0,
            "ret60_w_local": max(0, w_local_r60), "ret60_w_global": max(0, w_global_r60),
            "ret60_w_gru_ret60": w_gru_r60, "ret60_w_tf_ret60": w_tf_r60, "ret60_w_extreme": 0.0,
        }
    return weights


def generate_plan_d(results):
    """Plan D: Aggressive - higher seq model weights (0.2 each)"""
    weights = {}
    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        r = results[ds]
        
        w_gru_r5 = 0.2
        w_tf_r5 = 0.2
        w_gru_r60 = 0.2
        w_tf_r60 = 0.2
        
        # Remaining 0.6 split between local and global based on best blend
        w_local_r5 = r["best_blend_w_r5"]
        lgb_total_r5 = 0.6
        w_local_r5_final = round(w_local_r5 * lgb_total_r5, 2)
        w_global_r5_final = round(lgb_total_r5 - w_local_r5_final, 2)
        
        w_local_r60 = r["best_blend_w_r60"]
        lgb_total_r60 = 0.6
        w_local_r60_final = round(w_local_r60 * lgb_total_r60, 2)
        w_global_r60_final = round(lgb_total_r60 - w_local_r60_final, 2)
        
        weights[ds] = {
            "ret5_w_local": max(0, w_local_r5_final), "ret5_w_global": max(0, w_global_r5_final),
            "ret5_w_gru_ret5": w_gru_r5, "ret5_w_tf_ret5": w_tf_r5, "ret5_w_extreme": 0.0,
            "ret60_w_local": max(0, w_local_r60_final), "ret60_w_global": max(0, w_global_r60_final),
            "ret60_w_gru_ret60": w_gru_r60, "ret60_w_tf_ret60": w_tf_r60, "ret60_w_extreme": 0.0,
        }
    return weights


def estimate_size(models_dir, weights, plan_name):
    """Estimate submission size for a given weight config."""
    models_path = Path(models_dir)
    total = 0
    
    # Always include LGB local (backbone)
    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        for target in ["ret5", "ret60"]:
            for ext in [".txt.gz", ".txt"]:
                p = models_path / f"lgb_{target}_{ds}{ext}"
                if p.exists():
                    total += p.stat().st_size
                    break
    
    # Include global if any dataset uses it
    any_global = any(
        weights[f"dataset{i}"].get("ret5_w_global", 0) > 0 or 
        weights[f"dataset{i}"].get("ret60_w_global", 0) > 0
        for i in range(NUM_DATASETS)
    )
    if any_global:
        for f in ["lgb_ret5_global.txt.gz", "lgb_ret60_global.txt.gz"]:
            p = models_path / f
            if p.exists():
                total += p.stat().st_size
    
    # Include seq models if weights > 0
    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        w = weights[ds]
        if w.get("ret5_w_gru_ret5", 0) > 0:
            p = models_path / f"gru_ret5_{ds}.pt"
            if p.exists(): total += p.stat().st_size
        if w.get("ret5_w_tf_ret5", 0) > 0:
            p = models_path / f"transformer_ret5_{ds}.pt"
            if p.exists(): total += p.stat().st_size
        if w.get("ret60_w_gru_ret60", 0) > 0:
            p = models_path / f"gru_ret60_{ds}.pt"
            if p.exists(): total += p.stat().st_size
        if w.get("ret60_w_tf_ret60", 0) > 0:
            p = models_path / f"transformer_ret60_{ds}.pt"
            if p.exists(): total += p.stat().st_size
    
    return total / (1024 * 1024)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="train_dataset")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-dir", default="submissions")
    args = parser.parse_args()
    
    set_all_seeds(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Evaluating LGB variants on validation set...")
    print("=" * 100)
    results = evaluate_lgb_variants(args.data_dir, args.models_dir)
    
    # Generate plans
    plans = {
        "plan_a": ("Pure Local LGB (safest)", generate_plan_a(results)),
        "plan_b": ("Local + Global LGB blend", generate_plan_b(results)),
        "plan_c": ("LGB blend + conservative seq (0.1)", generate_plan_c(results)),
        "plan_d": ("LGB blend + aggressive seq (0.2)", generate_plan_d(results)),
    }
    
    print("\n" + "=" * 100)
    print("SUBMISSION PLANS")
    print("=" * 100)
    
    for plan_name, (desc, weights) in plans.items():
        size_mb = estimate_size(args.models_dir, weights, plan_name)
        
        # Save weights
        plan_dir = os.path.join(args.output_dir, plan_name)
        os.makedirs(plan_dir, exist_ok=True)
        weights_path = os.path.join(plan_dir, "ensemble_weights.json")
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        
        print(f"\n  {plan_name}: {desc}")
        print(f"    Size: ~{size_mb:.1f} MB")
        print(f"    Weights: {weights_path}")
        
        # Show a few example weights
        for ds in ["dataset0", "dataset6", "dataset21"]:
            w = weights[ds]
            r5_parts = []
            if w["ret5_w_local"] > 0: r5_parts.append(f"L{w['ret5_w_local']:.1f}")
            if w["ret5_w_global"] > 0: r5_parts.append(f"G{w['ret5_w_global']:.1f}")
            if w.get("ret5_w_gru_ret5", 0) > 0: r5_parts.append(f"GRU{w['ret5_w_gru_ret5']:.1f}")
            if w.get("ret5_w_tf_ret5", 0) > 0: r5_parts.append(f"TF{w['ret5_w_tf_ret5']:.1f}")
            print(f"      {ds} R5: {'+'.join(r5_parts)}")
    
    print(f"\n{'='*100}")
    print("TO USE A PLAN:")
    print("  cp submissions/plan_X/ensemble_weights.json models/")
    print("  python evaluate_local.py --data-dir train_dataset --model-dir models")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
