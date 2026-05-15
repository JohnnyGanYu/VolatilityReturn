#!/usr/bin/env python3
"""
v7 Predict: Multi-version ensemble (v5 + v6 models).

Supports all signal sources:
  1. v5 LGB local      (lgb_ret5_datasetX.txt.gz / .txt)
  2. v5 LGB global     (lgb_ret5_global.txt.gz)
  3. v5 GRU dual       (gru_datasetX.pt, window=60, output_dim=2)
  4. v5 TF dual        (transformer_datasetX.pt, window=60, output_dim=2)
  5. v6 LGB local      (v6_lgb_ret5_datasetX.txt.gz)
  6. v6 LGB global     (v6_lgb_ret5_global.txt.gz)
  7. v6 LGB extreme    (lgb_extreme_ret5_datasetX.txt)
  8. v6 GRU single     (gru_ret5_datasetX.pt, w=20 / gru_ret60_datasetX.pt, w=240)
  9. v6 TF single      (transformer_ret5_datasetX.pt / transformer_ret60_datasetX.pt)

Weight format (ensemble_weights.json):
{
  "dataset0": {
    "ret5_w_local": 0.3,        // v5 lgb local
    "ret5_w_global": 0.2,       // v5 lgb global
    "ret5_w_gru": 0.1,          // v5 gru (dual)
    "ret5_w_tf": 0.05,          // v5 tf (dual)
    "ret5_w_v6_local": 0.2,     // v6 lgb local
    "ret5_w_v6_global": 0.0,    // v6 lgb global
    "ret5_w_extreme": 0.05,     // v6 lgb extreme
    "ret5_w_gru_ret5": 0.05,    // v6 gru single
    "ret5_w_tf_ret5": 0.05,     // v6 tf single
    // same for ret60_w_*
  }
}
"""

import numpy as np
import random
import os
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lightgbm as lgb

# Model directory on the evaluation platform
MODEL_DIR = Path("/workspace/submission")


# =============================================================================
# Utility functions
# =============================================================================

def _set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_lgb_model(path_gz: Path, path_txt: Path):
    """Load LightGBM model. Returns None if not found."""
    import gzip, shutil, tempfile
    if path_gz.exists():
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp_path = tmp.name
            with gzip.open(str(path_gz), 'rb') as f_in:
                with open(tmp_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            model = lgb.Booster(model_file=tmp_path)
            os.unlink(tmp_path)
            return model
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    if path_txt.exists():
        return lgb.Booster(model_file=str(path_txt))
    return None


# =============================================================================
# Sliding window helper
# =============================================================================

def _build_sliding_windows_batched(padded: np.ndarray, start: int, end: int,
                                   window_size: int) -> np.ndarray:
    T_padded, F = padded.shape
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    n_windows = T_padded - window_size + 1
    all_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(n_windows, window_size, F), strides=strides
    )
    return np.ascontiguousarray(all_windows[start:end])


# =============================================================================
# Sequence model runners
# =============================================================================

def _run_dual_target_model(model_path: Path, padded: np.ndarray, T: int, F: int,
                           device, batch_size: int, window_size: int) -> np.ndarray:
    """Run v5 dual-target model. Returns (T, 2)."""
    import torch
    from torch import nn

    if not model_path.exists():
        return np.zeros((T, 2), dtype=np.float32)

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            return np.zeros((T, 2), dtype=np.float32)

        # Infer model type: check checkpoint field first, then filename
        model_type = checkpoint.get("model_type", None)
        if model_type is None:
            fname = model_path.name.lower()
            if "transformer" in fname:
                model_type = "transformer"
            else:
                model_type = "gru"
        input_size = checkpoint.get("input_size", F)

        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        if model_type == "gru":
            hidden_size = checkpoint.get("hidden_size", 64)
            num_layers = checkpoint.get("num_layers", 2)
            dropout = checkpoint.get("dropout", 0.1)

            class _GRU(nn.Module):
                def __init__(self, inp, hid, nl, do):
                    super().__init__()
                    self.gru = nn.GRU(input_size=inp, hidden_size=hid,
                                      num_layers=nl, batch_first=True,
                                      dropout=do if nl > 1 else 0.0)
                    self.fc = nn.Linear(hid, 2)
                def forward(self, x):
                    _, h_n = self.gru(x)
                    return self.fc(h_n[-1])

            model = _GRU(input_size, hidden_size, num_layers, dropout).to(device)
        else:
            ckpt_window = checkpoint.get("window_size", window_size)
            d_model = checkpoint.get("d_model", 64)
            nhead = checkpoint.get("nhead", 4)
            n_layers = checkpoint.get("num_layers", 4)
            dim_ff = checkpoint.get("dim_feedforward", 256)
            dropout = checkpoint.get("dropout", 0.1)

            class _TF(nn.Module):
                def __init__(self, inp, dm, nh, nl, dff, do, msl):
                    super().__init__()
                    self.input_proj = nn.Linear(inp, dm)
                    self.pos_embedding = nn.Parameter(torch.randn(1, msl, dm) * 0.02)
                    el = nn.TransformerEncoderLayer(d_model=dm, nhead=nh,
                        dim_feedforward=dff, dropout=do, batch_first=True, norm_first=True)
                    self.encoder = nn.TransformerEncoder(el, num_layers=nl,
                        norm=nn.LayerNorm(dm))
                    self.output_head = nn.Linear(dm, 2)
                def forward(self, x):
                    h = self.input_proj(x)
                    h = h + self.pos_embedding[:, :h.size(1)]
                    h = self.encoder(h)
                    return self.output_head(h[:, -1, :])

            model = _TF(input_size, d_model, nhead, n_layers, dim_ff,
                        dropout, ckpt_window).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        model.eval()

        pred = np.empty((T, 2), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch = _build_sliding_windows_batched(padded, start, end, window_size)
                inp = torch.from_numpy(batch).to(device)
                out = model(inp).cpu().numpy()
                del inp
                pred[start:end] = out

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return pred

    except Exception as e:
        print(f"    [WARN] dual model failed: {model_path.name}: {e}")
        return np.zeros((T, 2), dtype=np.float32)


def _run_single_target_model(model_path: Path, padded: np.ndarray, T: int, F: int,
                             device, batch_size: int, window_size: int) -> np.ndarray:
    """Run v6 single-target model. Returns (T,)."""
    import torch
    from torch import nn

    if not model_path.exists():
        return np.zeros(T, dtype=np.float32)

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            return np.zeros(T, dtype=np.float32)

        # Infer model type: check checkpoint field first, then filename
        model_type = checkpoint.get("model_type", None)
        if model_type is None:
            fname = model_path.name.lower()
            if "transformer" in fname:
                model_type = "transformer"
            else:
                model_type = "gru"
        input_size = checkpoint.get("input_size", F)
        ckpt_window = checkpoint.get("window_size", window_size)

        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)

        if model_type == "gru":
            hidden_size = checkpoint.get("hidden_size", 64)
            num_layers = checkpoint.get("num_layers", 2)
            dropout = checkpoint.get("dropout", 0.1)

            class _GRU(nn.Module):
                def __init__(self, inp, hid, nl, do):
                    super().__init__()
                    self.gru = nn.GRU(input_size=inp, hidden_size=hid,
                                      num_layers=nl, batch_first=True,
                                      dropout=do if nl > 1 else 0.0)
                    self.fc = nn.Linear(hid, 1)
                def forward(self, x):
                    _, h_n = self.gru(x)
                    return self.fc(h_n[-1]).squeeze(-1)

            model = _GRU(input_size, hidden_size, num_layers, dropout).to(device)
        else:
            d_model = checkpoint.get("d_model", 64)
            nhead = checkpoint.get("nhead", 4)
            n_layers = checkpoint.get("num_layers", 4)
            dim_ff = checkpoint.get("dim_feedforward", 256)
            dropout = checkpoint.get("dropout", 0.1)

            class _TF(nn.Module):
                def __init__(self, inp, dm, nh, nl, dff, do, msl):
                    super().__init__()
                    self.input_proj = nn.Linear(inp, dm)
                    self.pos_embedding = nn.Parameter(torch.randn(1, msl, dm) * 0.02)
                    el = nn.TransformerEncoderLayer(d_model=dm, nhead=nh,
                        dim_feedforward=dff, dropout=do, batch_first=True, norm_first=True)
                    self.encoder = nn.TransformerEncoder(el, num_layers=nl,
                        norm=nn.LayerNorm(dm))
                    self.output_head = nn.Linear(dm, 1)
                def forward(self, x):
                    h = self.input_proj(x)
                    h = h + self.pos_embedding[:, :h.size(1)]
                    h = self.encoder(h)
                    return self.output_head(h[:, -1, :]).squeeze(-1)

            model = _TF(input_size, d_model, nhead, n_layers, dim_ff,
                        dropout, ckpt_window).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        model.eval()

        pred = np.empty(T, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch = _build_sliding_windows_batched(padded, start, end, window_size)
                inp = torch.from_numpy(batch).to(device)
                out = model(inp).cpu().numpy()
                del inp
                pred[start:end] = out

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return pred

    except Exception as e:
        print(f"    [WARN] single model failed: {model_path.name}: {e}")
        return np.zeros(T, dtype=np.float32)


# =============================================================================
# Main signal generation
# =============================================================================

def generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray:
    """
    v7: Generate prediction signals using all v5+v6 models.

    Args:
        dataset_name: e.g. "dataset0"
        factors: (T, F) feature matrix

    Returns:
        (T, 2) array [Ret5, Ret60], all finite.
    """
    _set_seeds(42)

    T, F = factors.shape
    dataset_id = int(dataset_name.replace("dataset", ""))

    # =========================================================================
    # Load weights
    # =========================================================================
    # Default: pure v5 LGB local (safe fallback)
    weights = {
        "ret5_w_local": 1.0, "ret5_w_global": 0.0,
        "ret5_w_gru": 0.0, "ret5_w_tf": 0.0,
        "ret5_w_v6_local": 0.0, "ret5_w_v6_global": 0.0,
        "ret5_w_extreme": 0.0,
        "ret5_w_gru_ret5": 0.0, "ret5_w_tf_ret5": 0.0,
        "ret60_w_local": 1.0, "ret60_w_global": 0.0,
        "ret60_w_gru": 0.0, "ret60_w_tf": 0.0,
        "ret60_w_v6_local": 0.0, "ret60_w_v6_global": 0.0,
        "ret60_w_extreme": 0.0,
        "ret60_w_gru_ret60": 0.0, "ret60_w_tf_ret60": 0.0,
    }

    weights_path = MODEL_DIR / "ensemble_weights.json"
    if weights_path.exists():
        try:
            with open(weights_path) as wf:
                all_weights = json.load(wf)
            ds_w = all_weights.get(dataset_name, {})
            for k in weights:
                if k in ds_w:
                    weights[k] = float(ds_w[k])
        except Exception:
            pass

    # Extract individual weights
    w = weights
    w_r5 = [w["ret5_w_local"], w["ret5_w_global"], w["ret5_w_gru"], w["ret5_w_tf"],
             w["ret5_w_v6_local"], w["ret5_w_v6_global"], w["ret5_w_extreme"],
             w["ret5_w_gru_ret5"], w["ret5_w_tf_ret5"]]
    w_r60 = [w["ret60_w_local"], w["ret60_w_global"], w["ret60_w_gru"], w["ret60_w_tf"],
              w["ret60_w_v6_local"], w["ret60_w_v6_global"], w["ret60_w_extreme"],
              w["ret60_w_gru_ret60"], w["ret60_w_tf_ret60"]]

    # =========================================================================
    # Prepare inputs
    # =========================================================================
    id_col = np.full((T, 1), dataset_id, dtype=np.int32)
    factors_with_id = np.hstack([factors, id_col])

    # =========================================================================
    # 1. v5 LGB local
    # =========================================================================
    pred_v5_local = np.zeros((T, 2), dtype=np.float32)
    if w_r5[0] > 0 or w_r60[0] > 0:
        m = _load_lgb_model(MODEL_DIR / f"lgb_ret5_{dataset_name}.txt.gz",
                            MODEL_DIR / f"lgb_ret5_{dataset_name}.txt")
        if m: pred_v5_local[:, 0] = m.predict(factors).astype(np.float32)

        m = _load_lgb_model(MODEL_DIR / f"lgb_ret60_{dataset_name}.txt.gz",
                            MODEL_DIR / f"lgb_ret60_{dataset_name}.txt")
        if m: pred_v5_local[:, 1] = m.predict(factors).astype(np.float32)

    # =========================================================================
    # 2. v5 LGB global
    # =========================================================================
    pred_v5_global = np.zeros((T, 2), dtype=np.float32)
    if w_r5[1] > 0 or w_r60[1] > 0:
        m = _load_lgb_model(MODEL_DIR / "lgb_ret5_global.txt.gz",
                            MODEL_DIR / "lgb_ret5_global.txt")
        if m: pred_v5_global[:, 0] = m.predict(factors_with_id).astype(np.float32)

        m = _load_lgb_model(MODEL_DIR / "lgb_ret60_global.txt.gz",
                            MODEL_DIR / "lgb_ret60_global.txt")
        if m: pred_v5_global[:, 1] = m.predict(factors_with_id).astype(np.float32)

    # =========================================================================
    # 3-4. v5 GRU + TF (dual-target, window=60)
    # =========================================================================
    pred_v5_gru = np.zeros((T, 2), dtype=np.float32)
    pred_v5_tf = np.zeros((T, 2), dtype=np.float32)

    need_v5_seq = (w_r5[2] > 0 or w_r60[2] > 0 or w_r5[3] > 0 or w_r60[3] > 0)
    if need_v5_seq:
        import torch
        device = _get_device()
        clean = np.nan_to_num(factors, nan=0.0).astype(np.float32)
        v5_window = 60
        batch_size = 32768

        padded_v5 = np.zeros((v5_window - 1 + T, F), dtype=np.float32)
        padded_v5[v5_window - 1:] = clean

        if w_r5[2] > 0 or w_r60[2] > 0:
            pred_v5_gru = _run_dual_target_model(
                MODEL_DIR / f"gru_{dataset_name}.pt",
                padded_v5, T, F, device, batch_size, v5_window)

        if w_r5[3] > 0 or w_r60[3] > 0:
            pred_v5_tf = _run_dual_target_model(
                MODEL_DIR / f"transformer_{dataset_name}.pt",
                padded_v5, T, F, device, batch_size, v5_window)

        del padded_v5

    # =========================================================================
    # 5. v6 LGB local
    # =========================================================================
    pred_v6_local = np.zeros((T, 2), dtype=np.float32)
    if w_r5[4] > 0 or w_r60[4] > 0:
        m = _load_lgb_model(MODEL_DIR / f"v6_lgb_ret5_{dataset_name}.txt.gz",
                            MODEL_DIR / f"v6_lgb_ret5_{dataset_name}.txt")
        if m: pred_v6_local[:, 0] = m.predict(factors).astype(np.float32)

        m = _load_lgb_model(MODEL_DIR / f"v6_lgb_ret60_{dataset_name}.txt.gz",
                            MODEL_DIR / f"v6_lgb_ret60_{dataset_name}.txt")
        if m: pred_v6_local[:, 1] = m.predict(factors).astype(np.float32)

    # =========================================================================
    # 6. v6 LGB global
    # =========================================================================
    pred_v6_global = np.zeros((T, 2), dtype=np.float32)
    if w_r5[5] > 0 or w_r60[5] > 0:
        m = _load_lgb_model(MODEL_DIR / "v6_lgb_ret5_global.txt.gz",
                            MODEL_DIR / "v6_lgb_ret5_global.txt")
        if m: pred_v6_global[:, 0] = m.predict(factors_with_id).astype(np.float32)

        m = _load_lgb_model(MODEL_DIR / "v6_lgb_ret60_global.txt.gz",
                            MODEL_DIR / "v6_lgb_ret60_global.txt")
        if m: pred_v6_global[:, 1] = m.predict(factors_with_id).astype(np.float32)

    # =========================================================================
    # 7. v6 LGB extreme
    # =========================================================================
    pred_v6_extreme = np.zeros((T, 2), dtype=np.float32)
    if w_r5[6] > 0 or w_r60[6] > 0:
        ext_r5_gz = MODEL_DIR / f"lgb_extreme_ret5_{dataset_name}.txt.gz"
        ext_r5_txt = MODEL_DIR / f"lgb_extreme_ret5_{dataset_name}.txt"
        ext_r60_gz = MODEL_DIR / f"lgb_extreme_ret60_{dataset_name}.txt.gz"
        ext_r60_txt = MODEL_DIR / f"lgb_extreme_ret60_{dataset_name}.txt"

        m = _load_lgb_model(ext_r5_gz, ext_r5_txt)
        if m: pred_v6_extreme[:, 0] = m.predict(factors).astype(np.float32)

        m = _load_lgb_model(ext_r60_gz, ext_r60_txt)
        if m: pred_v6_extreme[:, 1] = m.predict(factors).astype(np.float32)

    # =========================================================================
    # 8-9. v6 GRU + TF (single-target)
    # =========================================================================
    pred_v6_gru = np.zeros((T, 2), dtype=np.float32)
    pred_v6_tf = np.zeros((T, 2), dtype=np.float32)

    need_v6_seq = (w_r5[7] > 0 or w_r60[7] > 0 or w_r5[8] > 0 or w_r60[8] > 0)
    if need_v6_seq:
        import torch
        device = _get_device()
        clean = np.nan_to_num(factors, nan=0.0).astype(np.float32)

        # Ret5 models (window=20)
        if w_r5[7] > 0 or w_r5[8] > 0:
            w_r5_win = 20
            batch_r5 = 32768
            padded_r5 = np.zeros((w_r5_win - 1 + T, F), dtype=np.float32)
            padded_r5[w_r5_win - 1:] = clean

            if w_r5[7] > 0:
                pred_v6_gru[:, 0] = _run_single_target_model(
                    MODEL_DIR / f"gru_ret5_{dataset_name}.pt",
                    padded_r5, T, F, device, batch_r5, w_r5_win)

            if w_r5[8] > 0:
                pred_v6_tf[:, 0] = _run_single_target_model(
                    MODEL_DIR / f"transformer_ret5_{dataset_name}.pt",
                    padded_r5, T, F, device, batch_r5, w_r5_win)

            del padded_r5

        # Ret60 models (window=240)
        if w_r60[7] > 0 or w_r60[8] > 0:
            w_r60_win = 240
            batch_r60 = 8192
            padded_r60 = np.zeros((w_r60_win - 1 + T, F), dtype=np.float32)
            padded_r60[w_r60_win - 1:] = clean

            if w_r60[7] > 0:
                pred_v6_gru[:, 1] = _run_single_target_model(
                    MODEL_DIR / f"gru_ret60_{dataset_name}.pt",
                    padded_r60, T, F, device, batch_r60, w_r60_win)

            if w_r60[8] > 0:
                pred_v6_tf[:, 1] = _run_single_target_model(
                    MODEL_DIR / f"transformer_ret60_{dataset_name}.pt",
                    padded_r60, T, F, device, batch_r60, w_r60_win)

            del padded_r60

        del clean

    # =========================================================================
    # Ensemble: weighted sum
    # =========================================================================
    all_preds = [pred_v5_local, pred_v5_global, pred_v5_gru, pred_v5_tf,
                 pred_v6_local, pred_v6_global, pred_v6_extreme,
                 pred_v6_gru, pred_v6_tf]

    pred_ret5 = sum(w_r5[i] * all_preds[i][:, 0] for i in range(9))
    pred_ret60 = sum(w_r60[i] * all_preds[i][:, 1] for i in range(9))

    signals = np.column_stack([pred_ret5, pred_ret60]).astype(np.float32)
    return np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
