#!/usr/bin/env python3
"""
v6: Check submission package size and file structure.

Validates:
- Total size <= 150 MB (hard limit)
- Warning at > 144 MB (safety margin)
- Categorized size breakdown for all v6 model types
- No uncompressed .txt LightGBM model files

Usage:
    python check_submission_size.py [--model-dir models] [--limit-mb 150.0]
"""
import os
import sys
import argparse
from pathlib import Path


# v6 model file categories
MODEL_CATEGORIES = {
    "lgb_local_ret5": {"pattern": "lgb_ret5_dataset", "description": "LGB Local Ret5"},
    "lgb_local_ret60": {"pattern": "lgb_ret60_dataset", "description": "LGB Local Ret60"},
    "lgb_global": {"pattern": "lgb_global", "description": "LGB Global"},
    "lgb_extreme_ret5": {"pattern": "lgb_extreme_ret5_", "description": "LGB Extreme Ret5"},
    "lgb_extreme_ret60": {"pattern": "lgb_extreme_ret60_", "description": "LGB Extreme Ret60"},
    "gru_ret5": {"pattern": "gru_ret5_", "description": "GRU Ret5 (w=20)"},
    "gru_ret60": {"pattern": "gru_ret60_", "description": "GRU Ret60 (w=240)"},
    "transformer_ret5": {"pattern": "transformer_ret5_", "description": "Transformer Ret5 (w=20)"},
    "transformer_ret60": {"pattern": "transformer_ret60_", "description": "Transformer Ret60 (w=240)"},
    "gru_legacy": {"pattern": "gru_dataset", "description": "GRU Legacy (v5)"},
    "transformer_legacy": {"pattern": "transformer_dataset", "description": "Transformer Legacy (v5)"},
    "ensemble_weights": {"pattern": "ensemble_weights", "description": "Ensemble Weights"},
    "submission_manifest": {"pattern": "submission_manifest", "description": "Submission Manifest"},
}


def categorize_file(filename: str) -> str:
    """Categorize a model file by its name pattern."""
    for cat_key, cat_info in MODEL_CATEGORIES.items():
        if filename.startswith(cat_info["pattern"]) or cat_info["pattern"] in filename:
            return cat_key
    return "other"


def check_size(model_dir: str = "models", limit_mb: float = 150.0,
               warning_mb: float = 144.0) -> bool:
    """
    v6: Check submission package size and verify file structure.

    Args:
        model_dir: Directory containing model files.
        limit_mb: Maximum allowed total size in MB (hard limit).
        warning_mb: Warning threshold in MB (safety margin).

    Returns:
        True if all checks pass, False otherwise.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"ERROR: Model directory '{model_dir}' does not exist.")
        return False

    # Collect all files and categorize
    all_files = []
    category_sizes = {}
    category_counts = {}
    total_bytes = 0

    for f in sorted(model_path.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            all_files.append((f, size))
            total_bytes += size

            cat = categorize_file(f.name)
            category_sizes[cat] = category_sizes.get(cat, 0) + size
            category_counts[cat] = category_counts.get(cat, 0) + 1

    total_mb = total_bytes / (1024 * 1024)

    print(f"Model directory: {model_path.resolve()}")
    print(f"Total files: {len(all_files)}")
    print(f"Total size: {total_mb:.2f} MB (limit: {limit_mb:.1f} MB, warning: {warning_mb:.1f} MB)")
    print()

    # --- v6 categorized breakdown ---
    print("File Category Breakdown:")
    print(f"  {'Category':<30s}  {'Count':>5s}  {'Size (MB)':>10s}  {'% Total':>8s}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*10}  {'-'*8}")

    for cat_key in sorted(category_sizes.keys()):
        size_mb = category_sizes[cat_key] / (1024 * 1024)
        count = category_counts[cat_key]
        pct = (category_sizes[cat_key] / total_bytes * 100) if total_bytes > 0 else 0
        desc = MODEL_CATEGORIES.get(cat_key, {}).get("description", cat_key)
        print(f"  {desc:<30s}  {count:>5d}  {size_mb:>10.2f}  {pct:>7.1f}%")

    print(f"  {'─'*30}  {'─'*5}  {'─'*10}  {'─'*8}")
    print(f"  {'TOTAL':<30s}  {len(all_files):>5d}  {total_mb:>10.2f}  {'100.0':>7s}%")
    print()

    # --- Size check ---
    if total_mb > limit_mb:
        print(f"✗ Size check FAILED: {total_mb:.2f} MB > {limit_mb:.1f} MB (HARD LIMIT)")
        size_ok = False
    elif total_mb > warning_mb:
        print(f"⚠ Size check WARNING: {total_mb:.2f} MB > {warning_mb:.1f} MB (safety margin)")
        print(f"  Remaining before hard limit: {limit_mb - total_mb:.2f} MB")
        size_ok = True
    else:
        print(f"✓ Size check PASSED: {total_mb:.2f} MB <= {warning_mb:.1f} MB")
        print(f"  Remaining margin: {warning_mb - total_mb:.2f} MB")
        size_ok = True

    print()

    # --- Compression check: no uncompressed .txt LightGBM models ---
    uncompressed_lgb = []
    for f, size in all_files:
        name = f.name
        if name.endswith(".txt") and not name.endswith(".txt.gz"):
            is_lgb_local = (
                name.startswith("lgb_ret5_dataset") or
                name.startswith("lgb_ret60_dataset") or
                name.startswith("lgb_global")
            )
            # Note: lgb_extreme files are stored as .txt (smaller, no compression needed)
            if is_lgb_local:
                uncompressed_lgb.append(f)

    if not uncompressed_lgb:
        print("✓ Compression check PASSED: No uncompressed .txt LGB local model files found.")
    else:
        print(f"⚠ Compression check: Found {len(uncompressed_lgb)} uncompressed .txt LGB local model(s):")
        for f in uncompressed_lgb[:5]:
            print(f"    {f.name}")
        if len(uncompressed_lgb) > 5:
            print(f"    ... and {len(uncompressed_lgb) - 5} more")
        print("  Consider compressing to .txt.gz for smaller submission size.")

    print()

    # --- v6 model completeness check ---
    print("v6 Model Completeness:")
    expected_per_dataset = {
        "lgb_local_ret5": 30,
        "lgb_local_ret60": 30,
        "gru_ret5": 30,
        "gru_ret60": 30,
        "transformer_ret5": 30,
        "transformer_ret60": 30,
    }

    for cat_key, expected_count in expected_per_dataset.items():
        actual = category_counts.get(cat_key, 0)
        desc = MODEL_CATEGORIES.get(cat_key, {}).get("description", cat_key)
        status = "✓" if actual >= expected_count else f"({actual}/{expected_count})"
        print(f"  {status} {desc}: {actual} files")

    # Extreme models are optional (depends on data)
    ext_r5 = category_counts.get("lgb_extreme_ret5", 0)
    ext_r60 = category_counts.get("lgb_extreme_ret60", 0)
    print(f"  ○ LGB Extreme Ret5: {ext_r5} files (optional, depends on data)")
    print(f"  ○ LGB Extreme Ret60: {ext_r60} files (optional, depends on data)")

    print()

    all_ok = size_ok
    if all_ok:
        print("All checks PASSED.")
    else:
        print("Some checks FAILED.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="v6: Check submission package size and file structure."
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory containing model files (default: models)"
    )
    parser.add_argument(
        "--limit-mb", type=float, default=150.0,
        help="Maximum allowed total size in MB (default: 150.0)"
    )
    parser.add_argument(
        "--warning-mb", type=float, default=144.0,
        help="Warning threshold in MB (default: 144.0)"
    )
    args = parser.parse_args()

    ok = check_size(model_dir=args.model_dir, limit_mb=args.limit_mb,
                    warning_mb=args.warning_mb)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
