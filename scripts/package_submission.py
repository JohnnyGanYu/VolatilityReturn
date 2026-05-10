#!/usr/bin/env python3
"""
Submission packaging script for the Volatility Return Prediction system (v3).

Creates a .tar.gz archive containing:
  Required:
  - factor.py       (feature generation)
  - predict.py      (signal generation)
  - lgb_ret5_dataset{0..29}.txt       (30 Ret5 LightGBM models)
  - lgb_ret60_dataset{0..29}.txt      (30 Ret60 LightGBM models)
  - gru_dataset{0..29}.pt             (30 GRU TorchScript models)
  - transformer_dataset{0..29}.pt     (30 Transformer TorchScript models)
  - ensemble_weights.json             (three-model ensemble weights: alpha+beta+gamma)

  Optional (included if present, not required):
  - feature_selection.json            (top-100 feature indices per target)
  - regime_dataset{0..29}.txt         (regime classifier models)

Performs validation checks:
  1. All required files exist
  2. Total archive size < 200 MB
  3. Only approved library imports
  4. Smoke test: import + synthetic data run (exercises three-model ensemble path)

Usage:
    python package_submission.py [--model-dir models] [--output submission.tar.gz]
"""

import os
import sys
import ast
import tarfile
import tempfile
import argparse
import shutil
import time


# =============================================================================
# Configuration
# =============================================================================

MAX_SUBMISSION_SIZE_MB = 200
NUM_DATASETS = 30

# Approved libraries on the evaluation platform
APPROVED_LIBRARIES = {
    "torch", "numpy", "pandas", "sklearn", "scipy", "bottleneck",
    "numba", "statsmodels", "talib", "lightgbm", "transformers", "polars",
}

# Standard library modules (always available)
STDLIB_MODULES = {
    "os", "sys", "random", "pathlib", "math", "collections", "functools",
    "itertools", "typing", "time", "json", "struct", "io", "abc",
    "hashlib", "copy", "warnings", "contextlib", "operator", "string",
    "re", "pickle", "gzip", "zipfile", "tempfile", "shutil", "glob",
    "logging", "traceback", "inspect", "importlib", "builtins",
    "multiprocessing", "threading", "concurrent", "queue", "ctypes",
    "array", "bisect", "heapq", "decimal", "fractions", "statistics",
    "enum", "dataclasses", "textwrap", "unicodedata", "codecs",
}

# Library versions on the evaluation platform (for documentation)
PLATFORM_VERSIONS = {
    "Python": "3.12",
    "torch": "2.4.0+cu124",
    "numpy": "2.4.4",
    "pandas": "3.0.2",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.1",
    "bottleneck": "1.6.0",
    "numba": "0.65.0",
    "statsmodels": "0.14.6",
    "TA-Lib": "0.6.8",
    "lightgbm": "4.6.0",
    "transformers": "5.5.4",
    "polars": "1.39.3",
}


# =============================================================================
# Validation checks
# =============================================================================

def check_imports(filepath: str) -> list:
    """Check that a Python file only imports from approved or stdlib modules."""
    with open(filepath) as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    violations = []
    for mod in imports:
        if mod not in APPROVED_LIBRARIES and mod not in STDLIB_MODULES:
            violations.append(mod)
    return violations


def check_required_files(model_dir: str) -> list:
    """Check that all required files exist."""
    missing = []

    # Entry scripts
    for script in ["factor.py", "predict.py"]:
        if not os.path.isfile(script):
            missing.append(script)

    # LightGBM model files
    for i in range(NUM_DATASETS):
        for target in ["ret5", "ret60"]:
            fname = f"lgb_{target}_dataset{i}.txt"
            fpath = os.path.join(model_dir, fname)
            if not os.path.isfile(fpath):
                missing.append(fpath)

    # GRU TorchScript model files
    for i in range(NUM_DATASETS):
        fname = f"gru_dataset{i}.pt"
        fpath = os.path.join(model_dir, fname)
        if not os.path.isfile(fpath):
            missing.append(fpath)

    # Transformer TorchScript model files
    for i in range(NUM_DATASETS):
        fname = f"transformer_dataset{i}.pt"
        fpath = os.path.join(model_dir, fname)
        if not os.path.isfile(fpath):
            missing.append(fpath)

    # Ensemble weights
    weights_path = os.path.join(model_dir, "ensemble_weights.json")
    if not os.path.isfile(weights_path):
        missing.append(weights_path)

    return missing


def smoke_test(model_dir: str) -> bool:
    """Run a basic smoke test: import modules and run on synthetic data."""
    import json
    import numpy as np

    print("  Smoke test: importing factor.py...")
    from factor import generate_factors

    print("  Smoke test: importing predict.py...")
    import predict
    from pathlib import Path
    predict.MODEL_DIR = Path(model_dir)

    # Validate ensemble_weights.json has three-weight schema
    weights_path = os.path.join(model_dir, "ensemble_weights.json")
    if os.path.isfile(weights_path):
        print("  Smoke test: validating ensemble_weights.json three-weight schema...")
        with open(weights_path) as wf:
            all_weights = json.load(wf)
        for ds_name, ds_weights in all_weights.items():
            for target in ["ret5", "ret60"]:
                alpha_key = f"{target}_alpha"
                beta_key = f"{target}_beta"
                gamma_key = f"{target}_gamma"
                assert alpha_key in ds_weights, f"{ds_name} missing {alpha_key}"
                assert beta_key in ds_weights, f"{ds_name} missing {beta_key}"
                assert gamma_key in ds_weights, f"{ds_name} missing {gamma_key}"
                a = float(ds_weights[alpha_key])
                b = float(ds_weights[beta_key])
                g = float(ds_weights[gamma_key])
                weight_sum = a + b + g
                assert abs(weight_sum - 1.0) < 0.05, (
                    f"{ds_name} {target} weights sum to {weight_sum:.3f}, expected ~1.0"
                )
        print(f"    Validated {len(all_weights)} datasets with three-weight schema")

    print("  Smoke test: running on synthetic data (T=500)...")
    np.random.seed(42)
    mock_data = np.random.rand(500, 5).astype(np.float32)
    mock_data[:, :4] = mock_data[:, :4] * 0.5 + 0.5  # price-like range
    mock_data[:, 4] = mock_data[:, 4] * 10000          # volume-like range

    factors = generate_factors("dataset0", mock_data)
    assert factors.shape[0] == 500, f"Factor rows mismatch: {factors.shape[0]} != 500"
    assert factors.shape[1] <= 512, f"Too many features: {factors.shape[1]}"
    assert factors.dtype == np.float32, f"Wrong dtype: {factors.dtype}"
    print(f"    Factors: shape={factors.shape}, dtype={factors.dtype}")

    signals = predict.generate_signals("dataset0", factors)
    assert signals.shape == (500, 2), f"Signal shape mismatch: {signals.shape}"
    assert signals.dtype == np.float32, f"Wrong dtype: {signals.dtype}"
    assert np.all(np.isfinite(signals)), "Signals contain NaN or Inf"
    print(f"    Signals: shape={signals.shape}, dtype={signals.dtype}, all_finite=True")

    print("  Smoke test: PASSED")
    return True


# =============================================================================
# Packaging
# =============================================================================

def create_package(model_dir: str, output_path: str) -> None:
    """Create the submission .tar.gz archive."""
    print("=" * 70)
    print("Submission Packaging")
    print("=" * 70)

    # --- Step 1: Check required files ---
    print("\n[1/5] Checking required files...")
    missing = check_required_files(model_dir)
    if missing:
        print(f"  FAIL: Missing files: {missing}")
        sys.exit(1)
    expected_count = 2 + NUM_DATASETS * 2 + NUM_DATASETS + NUM_DATASETS + 1  # scripts + lgb + gru + transformer + json
    print(f"  OK: All {expected_count} required files found")

    # --- Step 2: Check imports ---
    print("\n[2/5] Checking library imports...")
    for script in ["factor.py", "predict.py"]:
        violations = check_imports(script)
        if violations:
            print(f"  FAIL: {script} imports unapproved modules: {violations}")
            sys.exit(1)
        print(f"  OK: {script} — all imports approved")

    # --- Step 3: Smoke test ---
    print("\n[3/5] Running smoke test...")
    try:
        smoke_test(model_dir)
    except Exception as e:
        print(f"  FAIL: Smoke test error: {e}")
        sys.exit(1)

    # --- Step 4: Create archive ---
    print(f"\n[4/5] Creating archive: {output_path}")
    with tarfile.open(output_path, "w:gz") as tar:
        # Add entry scripts
        tar.add("factor.py", arcname="factor.py")
        tar.add("predict.py", arcname="predict.py")
        print(f"  Added: factor.py, predict.py")

        # Add LightGBM model files
        model_count = 0
        for i in range(NUM_DATASETS):
            for target in ["ret5", "ret60"]:
                fname = f"lgb_{target}_dataset{i}.txt"
                fpath = os.path.join(model_dir, fname)
                tar.add(fpath, arcname=fname)
                model_count += 1
        print(f"  Added: {model_count} LightGBM model files")

        # Add GRU TorchScript model files
        gru_count = 0
        for i in range(NUM_DATASETS):
            fname = f"gru_dataset{i}.pt"
            fpath = os.path.join(model_dir, fname)
            tar.add(fpath, arcname=fname)
            gru_count += 1
        print(f"  Added: {gru_count} GRU model files")

        # Add Transformer TorchScript model files
        tf_count = 0
        for i in range(NUM_DATASETS):
            fname = f"transformer_dataset{i}.pt"
            fpath = os.path.join(model_dir, fname)
            tar.add(fpath, arcname=fname)
            tf_count += 1
        print(f"  Added: {tf_count} Transformer model files")

        # Add ensemble weights
        weights_fname = "ensemble_weights.json"
        tar.add(os.path.join(model_dir, weights_fname), arcname=weights_fname)
        print(f"  Added: {weights_fname}")

        # Add optional files (if present)
        optional_count = 0
        # Feature selection
        fs_path = os.path.join(model_dir, "feature_selection.json")
        if os.path.isfile(fs_path):
            tar.add(fs_path, arcname="feature_selection.json")
            optional_count += 1
            print(f"  Added: feature_selection.json")

        # Regime classifiers
        for i in range(NUM_DATASETS):
            fname = f"regime_dataset{i}.txt"
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath):
                tar.add(fpath, arcname=fname)
                optional_count += 1
        if optional_count > 1:
            print(f"  Added: {optional_count - 1} regime classifier files")
        elif optional_count == 1:
            # Only feature_selection.json was added, no regime files
            pass

    # --- Step 5: Verify size ---
    print("\n[5/5] Verifying archive...")
    archive_size = os.path.getsize(output_path)
    archive_size_mb = archive_size / (1024 * 1024)
    print(f"  Archive size: {archive_size_mb:.2f} MB")

    if archive_size_mb > MAX_SUBMISSION_SIZE_MB:
        print(f"  FAIL: Exceeds {MAX_SUBMISSION_SIZE_MB} MB limit!")
        sys.exit(1)
    print(f"  OK: Within {MAX_SUBMISSION_SIZE_MB} MB limit")

    # Verify archive contents
    with tarfile.open(output_path, "r:gz") as tar:
        members = tar.getnames()
    print(f"  Archive contains {len(members)} files")
    assert "factor.py" in members, "factor.py missing from archive"
    assert "predict.py" in members, "predict.py missing from archive"

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PACKAGING COMPLETE")
    print("=" * 70)
    print(f"  Output: {output_path}")
    print(f"  Size: {archive_size_mb:.2f} MB")
    print(f"  Files: {len(members)} (2 scripts + {model_count} LightGBM + {gru_count} GRU + {tf_count} Transformer + 1 JSON + {optional_count} optional)")
    print()
    print("  Platform library versions:")
    for lib, ver in PLATFORM_VERSIONS.items():
        print(f"    {lib:>15s}: {ver}")
    print()
    print("  Ready for submission!")
    print("=" * 70)


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package submission for competition")
    parser.add_argument("--model-dir", default="models",
                        help="Directory containing trained model files")
    parser.add_argument("--output", default="submission.tar.gz",
                        help="Output archive path")
    args = parser.parse_args()

    create_package(args.model_dir, args.output)
