#!/usr/bin/env python3
"""
Re-optimize ensemble weights with Transformer included.
Only runs inference + grid search, no training.
"""
import os, json, time, gzip, shutil, tempfile
import numpy as np
import lightgbm as lgb
import torch
from pathlib import Path
from factor import generate_factors
from train import (
    load_dataset, build_sliding_windows_for_indices, pearson_ic_numpy,
    optimize_three_model_ensemble, _batch_predict,
    GRU_WINDOW_SIZE, GRU_BATCH_SIZE, GRU_MIN_IC_THRESHOLD,
    TRANSFORMER_MIN_IC_THRESHOLD, TRANSFORMER_BATCH_SIZE,
    TRAIN_RATIO, NUM_DATASETS, MIN_VALID_SAMPLES,
)

DATA_DIR = "train_dataset"
MODEL_DIR = Path("models_v4")
MAX_VAL_SAMPLES = 40000


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
    weights = {}
    # Load existing weights to preserve use_global_model flags
    existing_path = MODEL_DIR / "ensemble_weights.json"
    if existing_path.exists():
        with open(existing_path) as f:
            weights = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Disable Flash/MemEfficient SDP for Transformer compatibility
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    total_start = time.time()

    for ds_idx in range(NUM_DATASETS):
        ds_name = f"dataset{ds_idx}"
        t0 = time.time()
        print(f"\n{'='*40} {ds_name} {'='*40}")

        ohlcv, ret5, ret60, indices, extreme_intervals = load_dataset(DATA_DIR, ds_idx)
        features = generate_factors(ds_name, ohlcv)
        print(f"  Features: {features.shape}")

        # Valid indices (both ret5 and ret60 non-NaN)
        valid_mask = ~np.isnan(ret5) & ~np.isnan(ret60)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < MIN_VALID_SAMPLES:
            print(f"  Skipping: only {len(valid_indices)} valid samples")
            continue

        split_idx = int(len(valid_indices) * TRAIN_RATIO)
        val_idx = valid_indices[split_idx:]

        # Subsample val for speed
        if len(val_idx) > MAX_VAL_SAMPLES:
            step = len(val_idx) // MAX_VAL_SAMPLES
            val_idx = val_idx[::step][:MAX_VAL_SAMPLES]

        # LGB predictions
        lgb_r5_model = load_lgb_gz(MODEL_DIR / f"lgb_ret5_{ds_name}.txt.gz")
        lgb_r60_model = load_lgb_gz(MODEL_DIR / f"lgb_ret60_{ds_name}.txt.gz")
        lgb_pred_r5 = lgb_r5_model.predict(features[val_idx])
        lgb_pred_r60 = lgb_r60_model.predict(features[val_idx])

        # GRU predictions
        gru_path = MODEL_DIR / f"gru_{ds_name}.pt"
        gru_r5 = np.zeros(len(val_idx))
        gru_r60 = np.zeros(len(val_idx))
        gru_ic = -np.inf
        if gru_path.exists():
            try:
                # Try state_dict format first (v4), fall back to TorchScript (v3)
                checkpoint = torch.load(str(gru_path), map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    from torch import nn
                    hidden_size = checkpoint.get("hidden_size", 64)
                    num_layers = checkpoint.get("num_layers", 2)
                    dropout = checkpoint.get("dropout", 0.1)
                    input_size = checkpoint.get("input_size", features.shape[1])

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

                    gru_model = _GRU(input_size, hidden_size, num_layers, dropout).to(device)
                    gru_model.load_state_dict(checkpoint["state_dict"])
                    del checkpoint
                else:
                    gru_model = torch.jit.load(str(gru_path), map_location=device)
                gru_model.eval()
                windows = build_sliding_windows_for_indices(features, val_idx)
                preds = _batch_predict(gru_model, windows, device, GRU_BATCH_SIZE)
                gru_r5 = preds[:, 0]
                gru_r60 = preds[:, 1]
                gru_ic = (pearson_ic_numpy(gru_r5, np.nan_to_num(ret5[val_idx], nan=0.0)) +
                          pearson_ic_numpy(gru_r60, np.nan_to_num(ret60[val_idx], nan=0.0))) / 2
                del gru_model, windows
                print(f"  GRU val IC: {gru_ic:.4f}")
            except Exception as e:
                print(f"  GRU failed: {e}")

        # Transformer predictions
        tf_path = MODEL_DIR / f"transformer_{ds_name}.pt"
        tf_r5 = np.zeros(len(val_idx))
        tf_r60 = np.zeros(len(val_idx))
        tf_ic = -np.inf
        if tf_path.exists():
            try:
                checkpoint = torch.load(str(tf_path), map_location=device)
                from torch import nn

                class _TF(nn.Module):
                    def __init__(self, input_size, d_model=64, nhead=4, num_layers=4,
                                 dim_feedforward=256, dropout=0.1, max_seq_len=60):
                        super().__init__()
                        self.input_proj = nn.Linear(input_size, d_model)
                        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
                        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                            dropout=dropout, batch_first=True, norm_first=True)
                        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                              norm=nn.LayerNorm(d_model))
                        self.output_head = nn.Linear(d_model, 2)
                    def forward(self, x):
                        h = self.input_proj(x)
                        h = h + self.pos_embedding[:, :h.size(1)]
                        h = self.encoder(h)
                        return self.output_head(h[:, -1, :])

                input_size = checkpoint.get("input_size", features.shape[1])
                tf_model = _TF(input_size=input_size).to(device)
                tf_model.load_state_dict(checkpoint["state_dict"])
                tf_model.eval()
                windows = build_sliding_windows_for_indices(features, val_idx)
                preds = _batch_predict(tf_model, windows, device, TRANSFORMER_BATCH_SIZE)
                tf_r5 = preds[:, 0]
                tf_r60 = preds[:, 1]
                tf_ic = (pearson_ic_numpy(tf_r5, np.nan_to_num(ret5[val_idx], nan=0.0)) +
                         pearson_ic_numpy(tf_r60, np.nan_to_num(ret60[val_idx], nan=0.0))) / 2
                del tf_model, windows, checkpoint
                print(f"  Transformer val IC: {tf_ic:.4f}")
            except Exception as e:
                print(f"  Transformer failed: {e}")

        # Optimize ensemble
        gru_enabled = gru_ic >= GRU_MIN_IC_THRESHOLD
        tf_enabled = tf_ic >= TRANSFORMER_MIN_IC_THRESHOLD
        print(f"  GRU enabled={gru_enabled}, TF enabled={tf_enabled}")

        w = optimize_three_model_ensemble(
            lgb_pred_r5, lgb_pred_r60,
            gru_r5, gru_r60,
            tf_r5, tf_r60,
            np.nan_to_num(ret5[val_idx], nan=0.0),
            np.nan_to_num(ret60[val_idx], nan=0.0),
            gru_enabled=gru_enabled,
            tf_enabled=tf_enabled,
        )

        # Preserve existing use_global_model flags
        if ds_name in weights:
            w["use_global_model_ret5"] = weights[ds_name].get("use_global_model_ret5", False)
            w["use_global_model_ret60"] = weights[ds_name].get("use_global_model_ret60", False)

        weights[ds_name] = w
        print(f"  Ensemble: r5=({w['ret5_alpha']},{w['ret5_beta']},{w['ret5_gamma']}), "
              f"r60=({w['ret60_alpha']},{w['ret60_beta']},{w['ret60_gamma']})")
        print(f"  Time: {time.time()-t0:.1f}s")

    with open(MODEL_DIR / "ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nDone! Total: {time.time()-total_start:.1f}s")
    print(f"Saved: {MODEL_DIR / 'ensemble_weights.json'}")


if __name__ == "__main__":
    main()
