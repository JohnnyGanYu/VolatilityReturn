#!/usr/bin/env python3
"""
Compare global LGB IC vs local LGB IC per dataset.
Sets use_global_model_ret5/ret60 flags in ensemble_weights.json.
"""
import os, json, gzip, shutil, tempfile, time
import numpy as np
import lightgbm as lgb
from pathlib import Path
from factor import generate_factors
from train import load_dataset, ic_eval_metric, TRAIN_RATIO, NUM_DATASETS, MIN_VALID_SAMPLES

DATA_DIR = "train_dataset"
MODEL_DIR = Path("models_v4")


def load_lgb_gz(path_gz):
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    with gzip.open(str(path_gz), 'rb') as f_in:
        with open(tmp_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    model = lgb.Booster(model_file=tmp_path)
    os.unlink(tmp_path)
    return model


def main():
    # Load ensemble weights
    weights_path = MODEL_DIR / "ensemble_weights.json"
    with open(weights_path) as f:
        weights = json.load(f)

    # Load global models
    print("Loading global models...")
    global_r5 = load_lgb_gz(MODEL_DIR / "lgb_ret5_global.txt.gz")
    global_r60 = load_lgb_gz(MODEL_DIR / "lgb_ret60_global.txt.gz")

    print(f"{'Dataset':<15} {'Local R5':>10} {'Global R5':>10} {'Use R5':>8} "
          f"{'Local R60':>10} {'Global R60':>10} {'Use R60':>8}")

    for ds_idx in range(NUM_DATASETS):
        ds_name = f"dataset{ds_idx}"
        ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(DATA_DIR, ds_idx)
        features = generate_factors(ds_name, ohlcv)

        # Valid mask: both ret5 and ret60 non-NaN
        valid_mask = ~np.isnan(ret5) & ~np.isnan(ret60)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            weights[ds_name]["use_global_model_ret5"] = False
            weights[ds_name]["use_global_model_ret60"] = False
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        val_idx = valid_indices[split_idx:]

        # Local LGB predictions
        local_r5_model = load_lgb_gz(MODEL_DIR / f"lgb_ret5_{ds_name}.txt.gz")
        local_r60_model = load_lgb_gz(MODEL_DIR / f"lgb_ret60_{ds_name}.txt.gz")
        local_pred_r5 = local_r5_model.predict(features[val_idx])
        local_pred_r60 = local_r60_model.predict(features[val_idx])

        local_ds_r5 = lgb.Dataset(features[val_idx], label=ret5[val_idx])
        _, local_ic_r5, _ = ic_eval_metric(local_pred_r5, local_ds_r5)
        local_ds_r60 = lgb.Dataset(features[val_idx], label=ret60[val_idx])
        _, local_ic_r60, _ = ic_eval_metric(local_pred_r60, local_ds_r60)

        # Global LGB predictions (need dataset ID column)
        dataset_id = ds_idx
        id_col = np.full((len(val_idx), 1), dataset_id, dtype=np.int32)
        features_with_id = np.hstack([features[val_idx], id_col])

        global_pred_r5 = global_r5.predict(features_with_id)
        global_pred_r60 = global_r60.predict(features_with_id)

        global_ds_r5 = lgb.Dataset(features_with_id, label=ret5[val_idx])
        _, global_ic_r5, _ = ic_eval_metric(global_pred_r5, global_ds_r5)
        global_ds_r60 = lgb.Dataset(features_with_id, label=ret60[val_idx])
        _, global_ic_r60, _ = ic_eval_metric(global_pred_r60, global_ds_r60)

        use_r5 = bool(global_ic_r5 > local_ic_r5)
        use_r60 = bool(global_ic_r60 > local_ic_r60)

        if ds_name not in weights:
            weights[ds_name] = {}
        weights[ds_name]["use_global_model_ret5"] = use_r5
        weights[ds_name]["use_global_model_ret60"] = use_r60

        print(f"{ds_name:<15} {local_ic_r5:>10.4f} {global_ic_r5:>10.4f} {str(use_r5):>8} "
              f"{local_ic_r60:>10.4f} {global_ic_r60:>10.4f} {str(use_r60):>8}")

    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nSaved: {weights_path}")


if __name__ == "__main__":
    main()
