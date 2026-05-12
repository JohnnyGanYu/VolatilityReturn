import numpy as np
import random
import os
import json
from pathlib import Path

import lightgbm as lgb


# Model directory on the evaluation platform
MODEL_DIR = Path("/workspace/submission")


def _load_lgb_model(path_gz: Path, path_txt: Path) -> lgb.Booster:
    """Load LightGBM model, preferring .gz, falling back to .txt.

    Args:
        path_gz: Path to the gzip-compressed model file (.txt.gz)
        path_txt: Path to the plain text model file (.txt)

    Returns:
        Loaded lgb.Booster instance.

    Raises:
        FileNotFoundError: If neither file exists.
    """
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
    raise FileNotFoundError(f"Model file not found: {path_gz} or {path_txt}")

# Sliding window size (must match training)
GRU_WINDOW_SIZE = 60
GRU_BATCH_SIZE = 32768  # Reduced from 65536 to fit RTX 4090 24GB


def _set_seeds(seed: int = 42) -> None:
    """Set all random seeds for strict reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _get_device():
    """Return best available PyTorch device: CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_sliding_windows_batched(factors: np.ndarray, start: int, end: int,
                                     padded: np.ndarray, window_size: int) -> np.ndarray:
    """Build sliding windows for a batch of indices from pre-padded array.

    Uses numpy stride_tricks for zero-copy vectorized window construction.
    """
    T_padded, F = padded.shape
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    n_windows = T_padded - window_size + 1
    all_windows = np.lib.stride_tricks.as_strided(
        padded, shape=(n_windows, window_size, F), strides=strides
    )
    return np.ascontiguousarray(all_windows[start:end])


def _run_sequence_model(model_path, padded, T, F, device):
    """Run batched inference for a TorchScript sequence model (GRU or Transformer).

    Returns (T, 2) numpy array. Falls back to zeros on any failure.
    """
    import torch
    if not model_path.exists():
        return np.zeros((T, 2), dtype=np.float32)
    try:
        # Load model: try state_dict format first (v4), fall back to TorchScript (v3)
        checkpoint = None
        try:
            checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        except Exception:
            pass

        if checkpoint is not None and isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # state_dict format (GRU or Transformer)
            from torch import nn

            # Disable Flash/MemEfficient SDP for Transformer compatibility
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(False)

            model_type = checkpoint.get("model_type", "transformer")
            input_size = checkpoint.get("input_size", F)

            if model_type == "gru":
                hidden_size = checkpoint.get("hidden_size", 64)
                num_layers = checkpoint.get("num_layers", 2)
                dropout = checkpoint.get("dropout", 0.1)

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

                model = _GRU(input_size, hidden_size, num_layers, dropout).to(device)
            else:
                # Transformer
                window_size = checkpoint.get("window_size", GRU_WINDOW_SIZE)
                d_model = checkpoint.get("d_model", 64)
                nhead = checkpoint.get("nhead", 4)
                n_layers = checkpoint.get("num_layers", 4)
                dim_ff = checkpoint.get("dim_feedforward", 256)
                dropout = checkpoint.get("dropout", 0.1)

                class _TF(nn.Module):
                    def __init__(self, input_size, d_model, nhead, num_layers,
                                 dim_feedforward, dropout, max_seq_len):
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

                model = _TF(input_size, d_model, nhead, n_layers, dim_ff, dropout, window_size).to(device)

            model.load_state_dict(checkpoint["state_dict"])
            del checkpoint
        else:
            # TorchScript format (v3 legacy)
            model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        pred = np.empty((T, 2), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, T, GRU_BATCH_SIZE):
                end = min(start + GRU_BATCH_SIZE, T)
                batch = _build_sliding_windows_batched(None, start, end, padded, GRU_WINDOW_SIZE)
                inp = torch.from_numpy(batch).to(device)
                out = model(inp).cpu().numpy()
                del inp  # free GPU tensor immediately
                pred[start:end] = out
        # Explicitly free model and GPU cache
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return pred
    except Exception as e:
        print(f"    [WARN] Sequence model failed for {model_path.name}: {e}")
        return np.zeros((T, 2), dtype=np.float32)


def generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray:
    """
    Generate prediction signals from a feature matrix.

    Args:
        dataset_name: e.g. "dataset0" through "dataset29"
        factors: np.ndarray of shape (T, F), dtype float32

    Returns:
        np.ndarray of shape (T, 2), dtype float32
        Column 0: Ret5 prediction signal
        Column 1: Ret60 prediction signal
        All values are finite (no NaN, no Inf).
    """
    _set_seeds(42)

    T, F = factors.shape

    # Parse dataset ID for global model inference
    dataset_id = int(dataset_name.replace("dataset", ""))

    # --- Load ensemble weights ---
    # Support two formats:
    # v5 format: ret5_w_local, ret5_w_global, ret5_w_gru, ret5_w_tf (4-model joint)
    # v4 format: use_global_model_ret5, ret5_alpha, ret5_beta, ret5_gamma (backward compat)
    w_local_r5 = 1.0; w_global_r5 = 0.0; w_gru_r5 = 0.0; w_tf_r5 = 0.0
    w_local_r60 = 1.0; w_global_r60 = 0.0; w_gru_r60 = 0.0; w_tf_r60 = 0.0

    weights_path = MODEL_DIR / "ensemble_weights.json"
    if weights_path.exists():
        try:
            with open(weights_path) as wf:
                all_weights = json.load(wf)
            ds_w = all_weights.get(dataset_name, {})

            if "ret5_w_local" in ds_w:
                # v5 format: 4-model joint weights
                w_local_r5 = float(ds_w.get("ret5_w_local", 1.0))
                w_global_r5 = float(ds_w.get("ret5_w_global", 0.0))
                w_gru_r5 = float(ds_w.get("ret5_w_gru", 0.0))
                w_tf_r5 = float(ds_w.get("ret5_w_tf", 0.0))
                w_local_r60 = float(ds_w.get("ret60_w_local", 1.0))
                w_global_r60 = float(ds_w.get("ret60_w_global", 0.0))
                w_gru_r60 = float(ds_w.get("ret60_w_gru", 0.0))
                w_tf_r60 = float(ds_w.get("ret60_w_tf", 0.0))
            else:
                # v4 backward compat format
                use_global_r5 = bool(ds_w.get("use_global_model_ret5", False))
                use_global_r60 = bool(ds_w.get("use_global_model_ret60", False))
                alpha_r5 = float(ds_w.get("ret5_alpha", 1.0))
                beta_r5 = float(ds_w.get("ret5_beta", 0.0))
                gamma_r5 = float(ds_w.get("ret5_gamma", 0.0))
                alpha_r60 = float(ds_w.get("ret60_alpha", 1.0))
                beta_r60 = float(ds_w.get("ret60_beta", 0.0))
                gamma_r60 = float(ds_w.get("ret60_gamma", 0.0))
                # Convert to 4-model format
                if use_global_r5:
                    w_local_r5 = 0.0; w_global_r5 = alpha_r5
                else:
                    w_local_r5 = alpha_r5; w_global_r5 = 0.0
                w_gru_r5 = beta_r5; w_tf_r5 = gamma_r5
                if use_global_r60:
                    w_local_r60 = 0.0; w_global_r60 = alpha_r60
                else:
                    w_local_r60 = alpha_r60; w_global_r60 = 0.0
                w_gru_r60 = beta_r60; w_tf_r60 = gamma_r60
        except Exception:
            pass  # keep defaults

    # --- Optional: Regime feature augmentation ---
    regime_path = MODEL_DIR / f"regime_{dataset_name}.txt"
    if regime_path.exists():
        try:
            regime_model = lgb.Booster(model_file=str(regime_path))
            regime_features = factors[:, 82:94]  # Regime feature subset
            regime_prob = regime_model.predict(regime_features).astype(np.float32)
            factors_augmented = np.column_stack([factors, regime_prob])
        except Exception:
            factors_augmented = factors
    else:
        factors_augmented = factors

    # --- Optional: Feature selection ---
    fs_path = MODEL_DIR / "feature_selection.json"
    if fs_path.exists():
        try:
            with open(fs_path) as f:
                fs = json.load(f)
            ret5_indices = fs["ret5_features"]
            ret60_indices = fs["ret60_features"]
            factors_ret5 = factors_augmented[:, ret5_indices]
            factors_ret60 = factors_augmented[:, ret60_indices]
        except Exception:
            factors_ret5 = factors_augmented
            factors_ret60 = factors_augmented
    else:
        factors_ret5 = factors_augmented
        factors_ret60 = factors_augmented

    # --- LightGBM inference (CPU) ---
    id_col = np.full((T, 1), dataset_id, dtype=np.int32)
    factors_with_id = np.hstack([factors_augmented, id_col])

    # Local LGB
    need_local = (w_local_r5 > 0 or w_local_r60 > 0)
    local_pred_r5 = np.zeros(T)
    local_pred_r60 = np.zeros(T)
    if need_local:
        try:
            m = _load_lgb_model(MODEL_DIR / f"lgb_ret5_{dataset_name}.txt.gz",
                                MODEL_DIR / f"lgb_ret5_{dataset_name}.txt")
            local_pred_r5 = m.predict(factors_ret5)
        except FileNotFoundError:
            pass
        try:
            m = _load_lgb_model(MODEL_DIR / f"lgb_ret60_{dataset_name}.txt.gz",
                                MODEL_DIR / f"lgb_ret60_{dataset_name}.txt")
            local_pred_r60 = m.predict(factors_ret60)
        except FileNotFoundError:
            pass

    # Global LGB
    need_global = (w_global_r5 > 0 or w_global_r60 > 0)
    global_pred_r5 = np.zeros(T)
    global_pred_r60 = np.zeros(T)
    if need_global:
        try:
            m = _load_lgb_model(MODEL_DIR / "lgb_ret5_global.txt.gz",
                                MODEL_DIR / "lgb_ret5_global.txt")
            global_pred_r5 = m.predict(factors_with_id)
        except FileNotFoundError:
            global_pred_r5 = local_pred_r5  # fallback
        try:
            m = _load_lgb_model(MODEL_DIR / "lgb_ret60_global.txt.gz",
                                MODEL_DIR / "lgb_ret60_global.txt")
            global_pred_r60 = m.predict(factors_with_id)
        except FileNotFoundError:
            global_pred_r60 = local_pred_r60  # fallback

    # Check if sequence models needed
    need_gru = (w_gru_r5 > 0 or w_gru_r60 > 0)
    need_transformer = (w_tf_r5 > 0 or w_tf_r60 > 0)

    # Time safety: skip sequence models for very large datasets
    MAX_ROWS_FOR_SEQ = 3_000_000
    if T > MAX_ROWS_FOR_SEQ and (need_gru or need_transformer):
        max_seq_weight = max(w_gru_r5, w_gru_r60, w_tf_r5, w_tf_r60)
        if max_seq_weight <= 0.2:
            need_gru = False
            need_transformer = False

    gru_pred = np.zeros((T, 2), dtype=np.float32)
    tf_pred = np.zeros((T, 2), dtype=np.float32)

    if need_gru or need_transformer:
        import torch
        clean = np.nan_to_num(factors, nan=0.0).astype(np.float32)
        padded = np.zeros((GRU_WINDOW_SIZE - 1 + T, F), dtype=np.float32)
        padded[GRU_WINDOW_SIZE - 1:] = clean
        device = _get_device()

        if need_gru:
            gru_pred = _run_sequence_model(
                MODEL_DIR / f"gru_{dataset_name}.pt", padded, T, F, device)
        if need_transformer:
            tf_pred = _run_sequence_model(
                MODEL_DIR / f"transformer_{dataset_name}.pt", padded, T, F, device)
        del padded, clean

    # --- 4-model ensemble ---
    pred_ret5 = (w_local_r5 * local_pred_r5 + w_global_r5 * global_pred_r5 +
                 w_gru_r5 * gru_pred[:, 0] + w_tf_r5 * tf_pred[:, 0])
    pred_ret60 = (w_local_r60 * local_pred_r60 + w_global_r60 * global_pred_r60 +
                  w_gru_r60 * gru_pred[:, 1] + w_tf_r60 * tf_pred[:, 1])

    signals = np.column_stack([pred_ret5, pred_ret60]).astype(np.float32)
    return np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
