#!/usr/bin/env python3
"""
v7 Submission packaging: v5+v6 multi-model ensemble.

Models included:
  - v5 LGB local:   lgb_ret5_dataset{0..29}.txt.gz (from models_v5)
  - v5 LGB global:  lgb_ret5_global.txt.gz (from models_v5)
  - v6 LGB local:   v6_lgb_ret5_dataset{0..29}.txt.gz (from models_v6, renamed)
  - v6 LGB global:  v6_lgb_ret5_global.txt.gz (from models_v6, renamed)
  - v6 LGB extreme: lgb_extreme_ret5_dataset{0..29}.txt.gz (from models_v6, compressed)

Plus:
  - factor.py
  - predict.py (predict_v7.py renamed)
  - ensemble_weights.json (weights_robust.json)

Usage:
    python scripts/package_v7.py --output submissions/submission_v7.tar.gz
"""

import os
import sys
import gzip
import shutil
import tarfile
import tempfile
import argparse
import json
from pathlib import Path

NUM_DATASETS = 30
MAX_SIZE_MB = 150


def compress_to_gz(src_path: Path, dst_path: Path):
    """Compress a .txt file to .txt.gz."""
    with open(src_path, 'rb') as f_in:
        with gzip.open(str(dst_path), 'wb', compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)


def package_v7(output_path: str, weights_file: str = "weights_robust.json"):
    """Create v7 submission package."""
    v5_dir = Path("models_v5")
    v6_dir = Path("models_v6")

    if not v5_dir.exists():
        print(f"ERROR: {v5_dir} not found")
        sys.exit(1)
    if not v6_dir.exists():
        print(f"ERROR: {v6_dir} not found")
        sys.exit(1)
    if not os.path.exists(weights_file):
        print(f"ERROR: {weights_file} not found")
        sys.exit(1)
    if not os.path.exists("predict_v7.py"):
        print(f"ERROR: predict_v7.py not found")
        sys.exit(1)
    if not os.path.exists("factor.py"):
        print(f"ERROR: factor.py not found")
        sys.exit(1)

    # Create temp dir for staging
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        file_count = 0
        total_size = 0

        print("=" * 60)
        print("v7 Submission Packaging")
        print("=" * 60)

        # 1. Scripts
        print("\n[1] Adding scripts...")
        shutil.copy("factor.py", tmp / "factor.py")
        shutil.copy("predict_v7.py", tmp / "predict.py")  # rename to predict.py
        file_count += 2

        # 2. Weights (strip _val_ic and _meta fields for submission)
        print("[2] Adding weights...")
        with open(weights_file) as f:
            weights = json.load(f)
        # Remove metadata
        clean_weights = {k: v for k, v in weights.items() if not k.startswith("_")}
        for ds in clean_weights:
            if "_val_ic" in clean_weights[ds]:
                del clean_weights[ds]["_val_ic"]
        with open(tmp / "ensemble_weights.json", "w") as f:
            json.dump(clean_weights, f, indent=2)
        file_count += 1

        # 3. v5 LGB local (already .txt.gz)
        print("[3] Adding v5 LGB local (60 files)...")
        for i in range(NUM_DATASETS):
            for target in ["ret5", "ret60"]:
                src = v5_dir / f"lgb_{target}_dataset{i}.txt.gz"
                if src.exists():
                    shutil.copy(src, tmp / src.name)
                    file_count += 1
                else:
                    print(f"  WARN: missing {src}")

        # 4. v5 LGB global (already .txt.gz)
        print("[4] Adding v5 LGB global (2 files)...")
        for target in ["ret5", "ret60"]:
            src = v5_dir / f"lgb_{target}_global.txt.gz"
            if src.exists():
                shutil.copy(src, tmp / src.name)
                file_count += 1

        # 5. v6 LGB local (already .txt.gz, rename with v6_ prefix)
        print("[5] Adding v6 LGB local (60 files, renamed with v6_ prefix)...")
        for i in range(NUM_DATASETS):
            for target in ["ret5", "ret60"]:
                src = v6_dir / f"lgb_{target}_dataset{i}.txt.gz"
                dst_name = f"v6_lgb_{target}_dataset{i}.txt.gz"
                if src.exists():
                    shutil.copy(src, tmp / dst_name)
                    file_count += 1
                else:
                    print(f"  WARN: missing {src}")

        # 6. v6 LGB global (already .txt.gz, rename with v6_ prefix)
        print("[6] Adding v6 LGB global (2 files, renamed with v6_ prefix)...")
        for target in ["ret5", "ret60"]:
            src = v6_dir / f"lgb_{target}_global.txt.gz"
            dst_name = f"v6_lgb_{target}_global.txt.gz"
            if src.exists():
                shutil.copy(src, tmp / dst_name)
                file_count += 1

        # 7. v6 LGB extreme (compress .txt to .txt.gz)
        print("[7] Adding v6 LGB extreme (60 files, compressing...)...")
        for i in range(NUM_DATASETS):
            for target in ["ret5", "ret60"]:
                src = v6_dir / f"lgb_extreme_{target}_dataset{i}.txt"
                dst = tmp / f"lgb_extreme_{target}_dataset{i}.txt.gz"
                if src.exists():
                    compress_to_gz(src, dst)
                    file_count += 1
                else:
                    print(f"  WARN: missing {src}")

        # Calculate total size
        for f in tmp.iterdir():
            total_size += f.stat().st_size
        total_mb = total_size / (1024 * 1024)

        print(f"\n[8] Staging complete: {file_count} files, {total_mb:.1f} MB")

        if total_mb > MAX_SIZE_MB:
            print(f"ERROR: {total_mb:.1f} MB exceeds {MAX_SIZE_MB} MB limit!")
            sys.exit(1)

        # 8. Create tar.gz
        print(f"[9] Creating archive: {output_path}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with tarfile.open(output_path, "w:gz") as tar:
            for f in sorted(tmp.iterdir()):
                tar.add(str(f), arcname=f.name)

        archive_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n{'=' * 60}")
        print(f"DONE!")
        print(f"  Output: {output_path}")
        print(f"  Archive size: {archive_size:.1f} MB")
        print(f"  Files: {file_count}")
        print(f"  Status: {'OK' if archive_size < MAX_SIZE_MB else 'OVER LIMIT!'}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package v7 submission")
    parser.add_argument("--output", default="submissions/submission_v7.tar.gz")
    parser.add_argument("--weights", default="weights_robust.json",
                        help="Weights JSON file to include")
    args = parser.parse_args()

    package_v7(args.output, args.weights)
