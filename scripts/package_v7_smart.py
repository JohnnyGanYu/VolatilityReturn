#!/usr/bin/env python3
"""
Smart v7 packaging: only includes models actually referenced by weights.

Analyzes the weights JSON to determine which model files are needed,
then packages only those files. Supports gzip compression for .txt files.

Usage:
    python scripts/package_v7_smart.py --weights weights_v7_iter1.json \
        --output submissions/submission_v7_iter1.tar.gz
"""

import os
import sys
import gzip
import json
import shutil
import tarfile
import tempfile
import argparse
from pathlib import Path

NUM_DATASETS = 30
MAX_SIZE_MB = 145


def compress_to_gz(src_path: Path, dst_path: Path, level: int = 9):
    """Compress .txt to .txt.gz with specified compression level."""
    with open(src_path, 'rb') as f_in:
        with gzip.open(str(dst_path), 'wb', compresslevel=level) as f_out:
            shutil.copyfileobj(f_in, f_out)


def analyze_weights(weights: dict) -> dict:
    """Determine which models are needed per dataset."""
    needed = {
        'v5_lgb_local': set(),    # dataset indices
        'v5_lgb_global': False,   # single model
        'v5_gru': set(),
        'v5_tf': set(),
        'v6_lgb_local': set(),
        'v6_lgb_global': False,
        'v6_lgb_extreme': set(),
        'v6_gru': set(),
        'v6_tf': set(),
    }

    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        if ds not in weights:
            continue
        w = weights[ds]

        # v5 LGB local
        if w.get('ret5_w_local', 0) > 0 or w.get('ret60_w_local', 0) > 0:
            needed['v5_lgb_local'].add(i)

        # v5 LGB global
        if w.get('ret5_w_global', 0) > 0 or w.get('ret60_w_global', 0) > 0:
            needed['v5_lgb_global'] = True

        # v5 GRU (dual)
        if w.get('ret5_w_gru', 0) > 0 or w.get('ret60_w_gru', 0) > 0:
            needed['v5_gru'].add(i)

        # v5 TF (dual)
        if w.get('ret5_w_tf', 0) > 0 or w.get('ret60_w_tf', 0) > 0:
            needed['v5_tf'].add(i)

        # v6 LGB local
        if w.get('ret5_w_v6_local', 0) > 0 or w.get('ret60_w_v6_local', 0) > 0:
            needed['v6_lgb_local'].add(i)

        # v6 LGB global
        if w.get('ret5_w_v6_global', 0) > 0 or w.get('ret60_w_v6_global', 0) > 0:
            needed['v6_lgb_global'] = True

        # v6 LGB extreme
        if w.get('ret5_w_extreme', 0) > 0 or w.get('ret60_w_extreme', 0) > 0:
            needed['v6_lgb_extreme'].add(i)

        # v6 GRU single
        if w.get('ret5_w_gru_ret5', 0) > 0 or w.get('ret60_w_gru_ret60', 0) > 0:
            needed['v6_gru'].add(i)

        # v6 TF single
        if w.get('ret5_w_tf_ret5', 0) > 0 or w.get('ret60_w_tf_ret60', 0) > 0:
            needed['v6_tf'].add(i)

    return needed


def package(weights_file: str, output_path: str):
    v5_dir = Path("models_v5")
    v6_dir = Path("models_v6")

    # Load and clean weights
    with open(weights_file) as f:
        weights = json.load(f)
    clean_weights = {k: v for k, v in weights.items()
                     if not k.startswith("_") and isinstance(v, dict)}

    # Analyze what's needed
    needed = analyze_weights(clean_weights)

    print("=" * 60)
    print("v7 Smart Packaging")
    print("=" * 60)
    print(f"Weights: {weights_file}")
    print(f"Output:  {output_path}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        file_count = 0

        # 1. Scripts
        shutil.copy("factor.py", tmp / "factor.py")
        shutil.copy("predict_v7.py", tmp / "predict.py")
        file_count += 2

        # 2. Weights (clean)
        with open(tmp / "ensemble_weights.json", "w") as f:
            json.dump(clean_weights, f, indent=2)
        file_count += 1

        # 3. v5 LGB local
        ds_set = needed['v5_lgb_local']
        if ds_set:
            print(f"[v5_lgb_local] {len(ds_set)} datasets")
            for i in ds_set:
                for target in ["ret5", "ret60"]:
                    src = v5_dir / f"lgb_{target}_dataset{i}.txt.gz"
                    if src.exists():
                        shutil.copy(src, tmp / src.name)
                        file_count += 1

        # 4. v5 LGB global
        if needed['v5_lgb_global']:
            print("[v5_lgb_global] yes")
            for target in ["ret5", "ret60"]:
                src = v5_dir / f"lgb_{target}_global.txt.gz"
                if src.exists():
                    shutil.copy(src, tmp / src.name)
                    file_count += 1

        # 5. v5 GRU (dual)
        ds_set = needed['v5_gru']
        if ds_set:
            print(f"[v5_gru] {len(ds_set)} datasets: {sorted(ds_set)}")
            for i in ds_set:
                src = v5_dir / f"gru_dataset{i}.pt"
                if src.exists():
                    shutil.copy(src, tmp / src.name)
                    file_count += 1

        # 6. v5 TF (dual)
        ds_set = needed['v5_tf']
        if ds_set:
            print(f"[v5_tf] {len(ds_set)} datasets: {sorted(ds_set)}")
            for i in ds_set:
                src = v5_dir / f"transformer_dataset{i}.pt"
                if src.exists():
                    shutil.copy(src, tmp / src.name)
                    file_count += 1

        # 7. v6 LGB local (rename with v6_ prefix)
        ds_set = needed['v6_lgb_local']
        if ds_set:
            print(f"[v6_lgb_local] {len(ds_set)} datasets")
            for i in ds_set:
                for target in ["ret5", "ret60"]:
                    src = v6_dir / f"lgb_{target}_dataset{i}.txt.gz"
                    dst_name = f"v6_lgb_{target}_dataset{i}.txt.gz"
                    if src.exists():
                        shutil.copy(src, tmp / dst_name)
                        file_count += 1

        # 8. v6 LGB global (rename with v6_ prefix)
        if needed['v6_lgb_global']:
            print("[v6_lgb_global] yes")
            for target in ["ret5", "ret60"]:
                src = v6_dir / f"lgb_{target}_global.txt.gz"
                dst_name = f"v6_lgb_{target}_global.txt.gz"
                if src.exists():
                    shutil.copy(src, tmp / dst_name)
                    file_count += 1

        # 9. v6 LGB extreme (compress .txt → .txt.gz)
        ds_set = needed['v6_lgb_extreme']
        if ds_set:
            print(f"[v6_lgb_extreme] {len(ds_set)} datasets (compressing...)")
            for i in ds_set:
                for target in ["ret5", "ret60"]:
                    src = v6_dir / f"lgb_extreme_{target}_dataset{i}.txt"
                    dst = tmp / f"lgb_extreme_{target}_dataset{i}.txt.gz"
                    if src.exists():
                        compress_to_gz(src, dst, level=9)
                        file_count += 1

        # 10. v6 GRU single
        ds_set = needed['v6_gru']
        if ds_set:
            print(f"[v6_gru] {len(ds_set)} datasets: {sorted(ds_set)}")
            for i in ds_set:
                for name in [f"gru_ret5_dataset{i}.pt", f"gru_ret60_dataset{i}.pt"]:
                    src = v6_dir / name
                    if src.exists():
                        shutil.copy(src, tmp / name)
                        file_count += 1

        # 11. v6 TF single
        ds_set = needed['v6_tf']
        if ds_set:
            print(f"[v6_tf] {len(ds_set)} datasets: {sorted(ds_set)}")
            for i in ds_set:
                for name in [f"transformer_ret5_dataset{i}.pt",
                             f"transformer_ret60_dataset{i}.pt"]:
                    src = v6_dir / name
                    if src.exists():
                        shutil.copy(src, tmp / name)
                        file_count += 1

        # Calculate size
        total_size = sum(f.stat().st_size for f in tmp.iterdir())
        total_mb = total_size / (1024 * 1024)
        print(f"\nStaging: {file_count} files, {total_mb:.1f} MB")

        if total_mb > MAX_SIZE_MB:
            print(f"ERROR: {total_mb:.1f} MB exceeds {MAX_SIZE_MB} MB!")
            sys.exit(1)

        # Create archive
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with tarfile.open(output_path, "w:gz") as tar:
            for f in sorted(tmp.iterdir()):
                tar.add(str(f), arcname=f.name)

        archive_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"DONE: {output_path}")
        print(f"  Archive: {archive_mb:.1f} MB, {file_count} files")
        print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Weights JSON file")
    parser.add_argument("--output", default="submissions/submission_v7_iter1.tar.gz")
    args = parser.parse_args()
    package(args.weights, args.output)
