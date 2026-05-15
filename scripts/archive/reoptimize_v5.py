#!/usr/bin/env python3
"""
Joint optimization of global/local LGB + GRU + Transformer ensemble weights.
Instead of first choosing global vs local, then optimizing 3-model weights,
this script jointly searches over all 4 models at once.

For each dataset and target:
  pred = w_local * local_lgb + w_global * global_lgb + w_gru * gru + w_tf * transformer
  where w_local + w_global + w_gru + w_tf = 1.0

Uses validation set (last 20%) for weight optimization.
"""
import os, json, gzip, shutil, tempfile, time
import numpy as np
import torch
import lightgbm as lgb
from pathlib import Path
from factor import generate_factors
from train import (
    load_dataset, build_sliding_windows_for_indices, pearson_ic_numpy,
    _batch_predict, GRU_WINDOW_SIZE, GRU_BATCH_SIZE,
    TRANSFORMER_BATCH_SIZE, TRAIN_RATIO, NUM_DATASETS, MIN_VALID_SAMPLES,
)

DATA_DIR = "train_dataset"
MODEL_DIR = Path("models_v5")


def load_lgb_gz(path_gz):
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name
    with gzip.open(str(path_gz), 'rb') as f_in:
        with open(tmp_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    model = lgb.Booster(model_file=tmp_path)
    os.unlink(tmp_path)
    return model


def load_sequence_model(model_path, device, features_dim):
    """Load GRU or Transformer model (state_dict format)."""
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        return None

    from torch import nn
    model_type = checkpoint.get("model_type", "transformer")
    input_size = checkpoint.get("input_size", features_dim)

    if model_type == "gru":
        class _GRU(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0)
                self.fc = nn.Linear(hidden_size, 2)
            def forward(self, x):
                _, h_n = self.gru(x)
                return self.fc(h_n[-1])
        model = _GRU(input_size, checkpoint.get("hidden_size", 64),
                      checkpoint.get("num_layers", 2), checkpoint.get("dropout", 0.1)).to(device)
    else:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        class _TF(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, dim_ff, dropout, max_seq_len):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                    dropout=dropout, batch_first=True, norm_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                      norm=nn.LayerNorm(d_model))
                self.output_head = nn.Linear(d_model, 2)
            def forward(self, x):
                h = self.input_proj(x)
                h = h + self.pos_embedding[:, :h.size(1)]
                h = self.encoder(h)
                return self.output_head(h[:, -1, :])
        model = _TF(input_size, checkpoint.get("d_model", 64), checkpoint.get("nhead", 4),
                     checkpoint.get("num_layers", 4), checkpoint.get("dim_feedforward", 256),
                     checkpoint.get("dropout", 0.1), checkpoint.get("window_size", 60)).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def grid_search_4model(local_pred, global_pred, gru_pred, tf_pred, labels,
                        local_ok=True, global_ok=True, gru_ok=True, tf_ok=True):
    """
    Grid search over 4-model weights: w_local + w_global + w_gru + w_tf = 1.0
    Step size 0.1. Returns best (w_local, w_global, w_gru, w_tf).
    """
    step = 0.1
    values = np.arange(0.0, 1.0 + step/2, step)
    values = [round(v, 1) for v in values]

    best_ic = -np.inf
    best_w = (1.0, 0.0, 0.0, 0.0)  # default: pure local LGB

    for wl in values:
        if not local_ok and wl > 0:
            continue
        for wg in values:
            if not global_ok and wg > 0:
                continue
            for wb in values:
                if not gru_ok and wb > 0:
                    continue
                wt = round(1.0 - wl - wg - wb, 1)
                if wt < -0.01 or wt > 1.01:
                    continue
                if not tf_ok and wt > 0:
                    continue
                wt = max(0.0, min(1.0, wt))

                blended = wl * local_pred + wg * global_pred + wb * gru_pred + wt * tf_pred
                ic = pearson_ic_numpy(blended, labels)
                if ic > best_ic:
                    best_ic = ic
                    best_w = (round(wl, 1), round(wg, 1), round(wb, 1), round(wt, 1))

    # Safety: if best IC barely better than pure local, use pure local
    local_ic = pearson_ic_numpy(local_pred, labels)
    if best_w != (1.0, 0.0, 0.0, 0.0):
        if best_ic - local_ic < 0.002:
            best_w = (1.0, 0.0, 0.0, 0.0)
            best_ic = local_ic
        # Safety: LGB total (local + global) must be >= 0.3
        elif best_w[0] + best_w[1] < 0.3:
            # Sequence models dominating — cap them, redistribute to LGB
            lgb_total = 0.3
            seq_total = 0.7
            wl, wg, wb, wt = best_w
            old_seq = wb + wt
            if old_seq > 0:
                wb_new = round(seq_total * wb / old_seq, 1)
                wt_new = round(seq_total * wt / old_seq, 1)
            else:
                wb_new, wt_new = 0.0, 0.0
            # Split lgb_total proportionally between local and global
            old_lgb = wl + wg
            if old_lgb > 0:
                wl_new = round(lgb_total * wl / old_lgb, 1)
                wg_new = round(lgb_total * wg / old_lgb, 1)
            else:
                wl_new, wg_new = lgb_total, 0.0
            best_w = (wl_new, wg_new, wb_new, wt_new)

    return best_w, best_ic


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Load global models
    print("Loading global models...")
    global_r5 = load_lgb_gz(MODEL_DIR / "lgb_ret5_global.txt.gz")
    global_r60 = load_lgb_gz(MODEL_DIR / "lgb_ret60_global.txt.gz")

    weights = {}
    total_start = time.time()

    print(f"\n{'Dataset':<12} {'nR5 w':<20} {'nR60 w':<20} {'nR5 IC':>8} {'nR60 IC':>8}")

    for ds_idx in range(NUM_DATASETS):
        ds_name = f"dataset{ds_idx}"
        t0 = time.time()

        ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(DATA_DIR, ds_idx)
        features = generate_factors(ds_name, ohlcv)

        # Valid indices
        valid_mask = ~np.isnan(ret5) & ~np.isnan(ret60)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            weights[ds_name] = {"ret5_w": [1,0,0,0], "ret60_w": [1,0,0,0]}
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        val_idx = valid_indices[split_idx:]
        # Use full validation set (no subsampling)

        val_r5 = np.nan_to_num(ret5[val_idx], nan=0.0)
        val_r60 = np.nan_to_num(ret60[val_idx], nan=0.0)

        # Local LGB predictions
        local_r5_model = load_lgb_gz(MODEL_DIR / f"lgb_ret5_{ds_name}.txt.gz")
        local_r60_model = load_lgb_gz(MODEL_DIR / f"lgb_ret60_{ds_name}.txt.gz")
        local_pred_r5 = local_r5_model.predict(features[val_idx])
        local_pred_r60 = local_r60_model.predict(features[val_idx])

        # Global LGB predictions
        id_col = np.full((len(val_idx), 1), ds_idx, dtype=np.int32)
        features_with_id = np.hstack([features[val_idx], id_col])
        global_pred_r5 = global_r5.predict(features_with_id)
        global_pred_r60 = global_r60.predict(features_with_id)

        # GRU predictions
        gru_r5 = np.zeros(len(val_idx))
        gru_r60 = np.zeros(len(val_idx))
        gru_ok = False
        gru_path = MODEL_DIR / f"gru_{ds_name}.pt"
        if gru_path.exists():
            try:
                gru_model = load_sequence_model(gru_path, device, features.shape[1])
                if gru_model is not None:
                    windows = build_sliding_windows_for_indices(features, val_idx)
                    preds = _batch_predict(gru_model, windows, device, GRU_BATCH_SIZE)
                    gru_r5 = preds[:, 0]
                    gru_r60 = preds[:, 1]
                    gru_ic = (pearson_ic_numpy(gru_r5, val_r5) + pearson_ic_numpy(gru_r60, val_r60)) / 2
                    gru_ok = gru_ic >= 0.01
                    del gru_model, windows
            except Exception:
                pass

        # Transformer predictions
        tf_r5 = np.zeros(len(val_idx))
        tf_r60 = np.zeros(len(val_idx))
        tf_ok = False
        tf_path = MODEL_DIR / f"transformer_{ds_name}.pt"
        if tf_path.exists():
            try:
                tf_model = load_sequence_model(tf_path, device, features.shape[1])
                if tf_model is not None:
                    windows = build_sliding_windows_for_indices(features, val_idx)
                    preds = _batch_predict(tf_model, windows, device, TRANSFORMER_BATCH_SIZE)
                    tf_r5 = preds[:, 0]
                    tf_r60 = preds[:, 1]
                    tf_ic = (pearson_ic_numpy(tf_r5, val_r5) + pearson_ic_numpy(tf_r60, val_r60)) / 2
                    tf_ok = tf_ic >= 0.01
                    del tf_model, windows
            except Exception:
                pass

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Joint 4-model grid search
        w_r5, ic_r5 = grid_search_4model(
            local_pred_r5, global_pred_r5, gru_r5, tf_r5, val_r5,
            local_ok=True, global_ok=True, gru_ok=gru_ok, tf_ok=tf_ok)
        w_r60, ic_r60 = grid_search_4model(
            local_pred_r60, global_pred_r60, gru_r60, tf_r60, val_r60,
            local_ok=True, global_ok=True, gru_ok=gru_ok, tf_ok=tf_ok)

        weights[ds_name] = {
            "ret5_w_local": w_r5[0], "ret5_w_global": w_r5[1],
            "ret5_w_gru": w_r5[2], "ret5_w_tf": w_r5[3],
            "ret60_w_local": w_r60[0], "ret60_w_global": w_r60[1],
            "ret60_w_gru": w_r60[2], "ret60_w_tf": w_r60[3],
        }

        r5_str = f"L{w_r5[0]} G{w_r5[1]} B{w_r5[2]} T{w_r5[3]}"
        r60_str = f"L{w_r60[0]} G{w_r60[1]} B{w_r60[2]} T{w_r60[3]}"
        print(f"{ds_name:<12} {r5_str:<20} {r60_str:<20} {ic_r5:>8.4f} {ic_r60:>8.4f}  ({time.time()-t0:.1f}s)")

    # Convert to predict.py compatible format
    final_weights = {}
    for ds_name, w in weights.items():
        fw = {}
        # For predict.py: use_global decides LGB source, alpha/beta/gamma for 3-model blend
        # New approach: store all 4 weights directly
        fw["ret5_w_local"] = w.get("ret5_w_local", 1.0)
        fw["ret5_w_global"] = w.get("ret5_w_global", 0.0)
        fw["ret5_w_gru"] = w.get("ret5_w_gru", 0.0)
        fw["ret5_w_tf"] = w.get("ret5_w_tf", 0.0)
        fw["ret60_w_local"] = w.get("ret60_w_local", 1.0)
        fw["ret60_w_global"] = w.get("ret60_w_global", 0.0)
        fw["ret60_w_gru"] = w.get("ret60_w_gru", 0.0)
        fw["ret60_w_tf"] = w.get("ret60_w_tf", 0.0)
        final_weights[ds_name] = fw

    with open(MODEL_DIR / "ensemble_weights.json", "w") as f:
        json.dump(final_weights, f, indent=2)
    print(f"\nDone! Total: {time.time()-total_start:.1f}s")
    print(f"Saved: {MODEL_DIR / 'ensemble_weights.json'}")


if __name__ == "__main__":
    main()
