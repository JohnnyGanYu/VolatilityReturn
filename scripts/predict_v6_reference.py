import numpy as np
import random
import os
import json
from pathlib import Path

# Avoid CUDA memory fragmentation on RTX 4090 (24GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def _get_adaptive_batch_size(window_size: int, device) -> int:
    """
    v6: Adaptive batch size based on window size and available GPU memory.
    - window=20 -> batch=32768
    - window=240 -> batch=4096 (GRU hidden states need ~12GB at larger batch)
    - If available VRAM < 8 GB, halve the batch size.
    """
    import torch
    if window_size <= 60:
        base_batch = 32768
    else:
        base_batch = 8192

    if device.type == "cuda":
        try:
            avail_vram, _ = torch.cuda.mem_get_info()
            if avail_vram < 8 * 1024**3:  # < 8 GB
                base_batch = base_batch // 2
                print(f"    [WARN] Low VRAM ({avail_vram/1024**3:.1f} GB), reducing batch to {base_batch}")
        except Exception:
            pass

    return base_batch


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


def detect_extreme_regime(close_prices: np.ndarray, window: int = 60,
                          threshold_mult: float = 2.0) -> np.ndarray:
    """
    Causal extreme regime detection for inference. Only uses data at or before time t.

    Algorithm:
    1. Compute close-to-close log returns
    2. Compute rolling_std(return, window) using only historical data
    3. Mark |current_return| > threshold_mult * rolling_std as extreme

    Returns:
        (T,) bool array, True = extreme regime at that time step
    """
    T = len(close_prices)
    close = np.asarray(close_prices, dtype=np.float64)

    # Compute log returns (causal)
    log_returns = np.zeros(T, dtype=np.float64)
    for i in range(1, T):
        if close[i - 1] > 0 and close[i] > 0:
            log_returns[i] = np.log(close[i] / close[i - 1])

    # Compute rolling std and extreme mask (causal)
    extreme_mask = np.zeros(T, dtype=np.bool_)

    for t in range(1, T):
        start = max(1, t - window + 1)
        window_returns = log_returns[start:t + 1]

        if len(window_returns) < 2:
            continue

        rolling_std = np.std(window_returns, ddof=0)

        if rolling_std > 0:
            threshold = threshold_mult * rolling_std
            if abs(log_returns[t]) > threshold:
                extreme_mask[t] = True

    return extreme_mask


def _run_single_target_model(model_path: Path, padded: np.ndarray, T: int, F: int,
                              device, batch_size: int, window_size: int) -> np.ndarray:
    """
    v6: Run inference for a single-target sequence model (output_dim=1).

    Returns (T,) numpy array. Falls back to zeros on any failure.
    """
    import torch
    from torch import nn

    if not model_path.exists():
        return np.zeros(T, dtype=np.float32)

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            print(f"    [WARN] Invalid checkpoint format: {model_path.name}")
            return np.zeros(T, dtype=np.float32)

        model_type = checkpoint.get("model_type", "gru")
        input_size = checkpoint.get("input_size", F)
        ckpt_window = checkpoint.get("window_size", window_size)

        # Disable Flash/MemEfficient SDP for Transformer compatibility
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(False)

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
                    self.fc = nn.Linear(hidden_size, 1)
                def forward(self, x):
                    _, h_n = self.gru(x)
                    return self.fc(h_n[-1]).squeeze(-1)

            model = _GRU(input_size, hidden_size, num_layers, dropout).to(device)
        else:
            # Transformer
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
                    self.output_head = nn.Linear(d_model, 1)
                def forward(self, x):
                    h = self.input_proj(x)
                    h = h + self.pos_embedding[:, :h.size(1)]
                    h = self.encoder(h)
                    return self.output_head(h[:, -1, :]).squeeze(-1)

            model = _TF(input_size, d_model, nhead, n_layers, dim_ff, dropout, ckpt_window).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        model.eval()

        pred = np.empty(T, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch = _build_sliding_windows_batched(None, start, end, padded, window_size)
                inp = torch.from_numpy(batch).to(device)
                out = model(inp).cpu().numpy()
                del inp
                pred[start:end] = out

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return pred

    except Exception as e:
        print(f"    [WARN] Single-target model failed for {model_path.name}: {e}")
        return np.zeros(T, dtype=np.float32)


def _run_v5_sequence_model(model_path: Path, padded: np.ndarray, T: int, F: int,
                            device, batch_size: int, window_size: int) -> np.ndarray:
    """
    v5 backward compat: Run inference for a dual-target model (output_dim=2).

    Returns (T, 2) numpy array. Falls back to zeros on any failure.
    """
    import torch
    from torch import nn

    if not model_path.exists():
        return np.zeros((T, 2), dtype=np.float32)

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_type = checkpoint.get("model_type", "gru")
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
                ckpt_window = checkpoint.get("window_size", window_size)
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

                model = _TF(input_size, d_model, nhead, n_layers, dim_ff, dropout, ckpt_window).to(device)

            model.load_state_dict(checkpoint["state_dict"])
            del checkpoint
        else:
            # TorchScript format (v3 legacy)
            model = torch.jit.load(str(model_path), map_location=device)

        model.eval()
        pred = np.empty((T, 2), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch = _build_sliding_windows_batched(None, start, end, padded, window_size)
                inp = torch.from_numpy(batch).to(device)
                out = model(inp).cpu().numpy()
                del inp
                pred[start:end] = out

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return pred

    except Exception as e:
        print(f"    [WARN] v5 sequence model failed for {model_path.name}: {e}")
        return np.zeros((T, 2), dtype=np.float32)


def generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray:
    """
    v6: Generate prediction signals from a feature matrix.

    Supports:
    - Dual window inference (Ret5 w=20, Ret60 w=240)
    - Extreme regime detection and fusion
    - v5 backward compatibility (fallback to gru_*.pt if v6 files missing)
    - Adaptive batch size
    - Extended v6 weight format

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
    dataset_id = int(dataset_name.replace("dataset", ""))

    # =========================================================================
    # Load ensemble weights (v6 extended format with v5 backward compat)
    # =========================================================================
    # v6 defaults
    w_local_r5 = 1.0
    w_global_r5 = 0.0
    w_gru_ret5 = 0.0
    w_tf_ret5 = 0.0
    w_extreme_r5 = 0.0
    w_local_r60 = 1.0
    w_global_r60 = 0.0
    w_gru_ret60 = 0.0
    w_tf_ret60 = 0.0
    w_extreme_r60 = 0.0

    weights_path = MODEL_DIR / "ensemble_weights.json"
    if weights_path.exists():
        try:
            with open(weights_path) as wf:
                all_weights = json.load(wf)
            ds_w = all_weights.get(dataset_name, {})

            if "ret5_w_gru_ret5" in ds_w or "ret5_w_extreme" in ds_w:
                # v6 format
                w_local_r5 = float(ds_w.get("ret5_w_local", 1.0))
                w_global_r5 = float(ds_w.get("ret5_w_global", 0.0))
                w_gru_ret5 = float(ds_w.get("ret5_w_gru_ret5", 0.0))
                w_tf_ret5 = float(ds_w.get("ret5_w_tf_ret5", 0.0))
                w_extreme_r5 = float(ds_w.get("ret5_w_extreme", 0.0))
                w_local_r60 = float(ds_w.get("ret60_w_local", 1.0))
                w_global_r60 = float(ds_w.get("ret60_w_global", 0.0))
                w_gru_ret60 = float(ds_w.get("ret60_w_gru_ret60", 0.0))
                w_tf_ret60 = float(ds_w.get("ret60_w_tf_ret60", 0.0))
                w_extreme_r60 = float(ds_w.get("ret60_w_extreme", 0.0))
            elif "ret5_w_local" in ds_w:
                # v5 format: ret5_w_local, ret5_w_global, ret5_w_gru, ret5_w_tf
                w_local_r5 = float(ds_w.get("ret5_w_local", 1.0))
                w_global_r5 = float(ds_w.get("ret5_w_global", 0.0))
                w_gru_ret5 = float(ds_w.get("ret5_w_gru", 0.0))
                w_tf_ret5 = float(ds_w.get("ret5_w_tf", 0.0))
                w_extreme_r5 = 0.0
                w_local_r60 = float(ds_w.get("ret60_w_local", 1.0))
                w_global_r60 = float(ds_w.get("ret60_w_global", 0.0))
                w_gru_ret60 = float(ds_w.get("ret60_w_gru", 0.0))
                w_tf_ret60 = float(ds_w.get("ret60_w_tf", 0.0))
                w_extreme_r60 = 0.0
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
                if use_global_r5:
                    w_local_r5 = 0.0; w_global_r5 = alpha_r5
                else:
                    w_local_r5 = alpha_r5; w_global_r5 = 0.0
                w_gru_ret5 = beta_r5; w_tf_ret5 = gamma_r5
                if use_global_r60:
                    w_local_r60 = 0.0; w_global_r60 = alpha_r60
                else:
                    w_local_r60 = alpha_r60; w_global_r60 = 0.0
                w_gru_ret60 = beta_r60; w_tf_ret60 = gamma_r60
        except Exception:
            pass  # keep defaults

    # =========================================================================
    # Feature preparation
    # =========================================================================
    # Optional: Feature selection
    fs_path = MODEL_DIR / "feature_selection.json"
    if fs_path.exists():
        try:
            with open(fs_path) as f:
                fs = json.load(f)
            ret5_indices = fs["ret5_features"]
            ret60_indices = fs["ret60_features"]
            factors_ret5 = factors[:, ret5_indices]
            factors_ret60 = factors[:, ret60_indices]
        except Exception:
            factors_ret5 = factors
            factors_ret60 = factors
    else:
        factors_ret5 = factors
        factors_ret60 = factors

    # =========================================================================
    # LightGBM Local inference (CPU)
    # =========================================================================
    id_col = np.full((T, 1), dataset_id, dtype=np.int32)
    factors_with_id = np.hstack([factors, id_col])

    local_pred_r5 = np.zeros(T, dtype=np.float32)
    local_pred_r60 = np.zeros(T, dtype=np.float32)
    if w_local_r5 > 0 or w_local_r60 > 0:
        try:
            m = _load_lgb_model(MODEL_DIR / f"lgb_ret5_{dataset_name}.txt.gz",
                                MODEL_DIR / f"lgb_ret5_{dataset_name}.txt")
            local_pred_r5 = m.predict(factors_ret5).astype(np.float32)
        except FileNotFoundError:
            pass
        try:
            m = _load_lgb_model(MODEL_DIR / f"lgb_ret60_{dataset_name}.txt.gz",
                                MODEL_DIR / f"lgb_ret60_{dataset_name}.txt")
            local_pred_r60 = m.predict(factors_ret60).astype(np.float32)
        except FileNotFoundError:
            pass

    # =========================================================================
    # LightGBM Global inference (CPU)
    # =========================================================================
    global_pred_r5 = np.zeros(T, dtype=np.float32)
    global_pred_r60 = np.zeros(T, dtype=np.float32)
    if w_global_r5 > 0 or w_global_r60 > 0:
        try:
            m = _load_lgb_model(MODEL_DIR / "lgb_ret5_global.txt.gz",
                                MODEL_DIR / "lgb_ret5_global.txt")
            global_pred_r5 = m.predict(factors_with_id).astype(np.float32)
        except FileNotFoundError:
            global_pred_r5 = local_pred_r5.copy()
        try:
            m = _load_lgb_model(MODEL_DIR / "lgb_ret60_global.txt.gz",
                                MODEL_DIR / "lgb_ret60_global.txt")
            global_pred_r60 = m.predict(factors_with_id).astype(np.float32)
        except FileNotFoundError:
            global_pred_r60 = local_pred_r60.copy()

    # =========================================================================
    # Extreme regime detection and LGB_Extreme inference
    # =========================================================================
    extreme_pred_r5 = np.zeros(T, dtype=np.float32)
    extreme_pred_r60 = np.zeros(T, dtype=np.float32)
    extreme_indicator = np.zeros(T, dtype=np.float32)

    # Always try to load extreme models (auto-blend in extreme regime)
    ext_r5_path = MODEL_DIR / f"lgb_extreme_ret5_{dataset_name}.txt"
    ext_r60_path = MODEL_DIR / f"lgb_extreme_ret60_{dataset_name}.txt"
    has_any_extreme = ext_r5_path.exists() or ext_r60_path.exists()

    if has_any_extreme:
        # Detect extreme regime (causal, based on rolling volatility)
        close_proxy = factors[:, 3] if F > 3 else factors[:, 0]
        extreme_indicator = detect_extreme_regime(close_proxy, window=60, threshold_mult=2.0).astype(np.float32)

        # Load extreme LGB models
        if ext_r5_path.exists():
            try:
                m = lgb.Booster(model_file=str(ext_r5_path))
                extreme_pred_r5 = m.predict(factors_ret5).astype(np.float32)
            except Exception:
                pass

        if ext_r60_path.exists():
            try:
                m = lgb.Booster(model_file=str(ext_r60_path))
                extreme_pred_r60 = m.predict(factors_ret60).astype(np.float32)
            except Exception:
                pass

    # =========================================================================
    # Sequence model inference (GPU)
    # =========================================================================
    # Check if sequence models are needed
    need_seq_ret5 = (w_gru_ret5 > 0 or w_tf_ret5 > 0)
    need_seq_ret60 = (w_gru_ret60 > 0 or w_tf_ret60 > 0)

    # Time safety: skip sequence models for very large datasets with low weights
    MAX_ROWS_FOR_SEQ = 3_000_000
    if T > MAX_ROWS_FOR_SEQ:
        max_seq_weight = max(w_gru_ret5, w_tf_ret5, w_gru_ret60, w_tf_ret60)
        if max_seq_weight <= 0.2:
            need_seq_ret5 = False
            need_seq_ret60 = False

    gru_ret5_pred = np.zeros(T, dtype=np.float32)
    tf_ret5_pred = np.zeros(T, dtype=np.float32)
    gru_ret60_pred = np.zeros(T, dtype=np.float32)
    tf_ret60_pred = np.zeros(T, dtype=np.float32)

    if need_seq_ret5 or need_seq_ret60:
        import torch
        clean = np.nan_to_num(factors, nan=0.0).astype(np.float32)
        device = _get_device()

        # --- Ret5 sequence models (window=20) ---
        if need_seq_ret5:
            window_r5 = 20
            batch_r5 = _get_adaptive_batch_size(window_r5, device)

            padded_r5 = np.zeros((window_r5 - 1 + T, F), dtype=np.float32)
            padded_r5[window_r5 - 1:] = clean

            # Try v6 models first, fall back to v5
            gru_ret5_v6_path = MODEL_DIR / f"gru_ret5_{dataset_name}.pt"
            gru_v5_path = MODEL_DIR / f"gru_{dataset_name}.pt"

            if w_gru_ret5 > 0:
                if gru_ret5_v6_path.exists():
                    gru_ret5_pred = _run_single_target_model(
                        gru_ret5_v6_path, padded_r5, T, F, device, batch_r5, window_r5)
                elif gru_v5_path.exists():
                    # v5 fallback: dual-target model with window=60
                    v5_window = 60
                    padded_v5 = np.zeros((v5_window - 1 + T, F), dtype=np.float32)
                    padded_v5[v5_window - 1:] = clean
                    v5_pred = _run_v5_sequence_model(
                        gru_v5_path, padded_v5, T, F, device, batch_r5, v5_window)
                    gru_ret5_pred = v5_pred[:, 0]
                    del padded_v5

            tf_ret5_v6_path = MODEL_DIR / f"transformer_ret5_{dataset_name}.pt"
            tf_v5_path = MODEL_DIR / f"transformer_{dataset_name}.pt"

            if w_tf_ret5 > 0:
                if tf_ret5_v6_path.exists():
                    tf_ret5_pred = _run_single_target_model(
                        tf_ret5_v6_path, padded_r5, T, F, device, batch_r5, window_r5)
                elif tf_v5_path.exists():
                    v5_window = 60
                    padded_v5 = np.zeros((v5_window - 1 + T, F), dtype=np.float32)
                    padded_v5[v5_window - 1:] = clean
                    v5_pred = _run_v5_sequence_model(
                        tf_v5_path, padded_v5, T, F, device, batch_r5, v5_window)
                    tf_ret5_pred = v5_pred[:, 0]
                    del padded_v5

            del padded_r5

        # --- Ret60 sequence models (window=240) ---
        if need_seq_ret60:
            window_r60 = 240

            # Short dataset safety: if T < 240, output zeros for Ret60 seq models
            if T < window_r60:
                gru_ret60_pred = np.zeros(T, dtype=np.float32)
                tf_ret60_pred = np.zeros(T, dtype=np.float32)
            else:
                batch_r60 = _get_adaptive_batch_size(window_r60, device)

                padded_r60 = np.zeros((window_r60 - 1 + T, F), dtype=np.float32)
                padded_r60[window_r60 - 1:] = clean

                gru_ret60_v6_path = MODEL_DIR / f"gru_ret60_{dataset_name}.pt"
                gru_v5_path = MODEL_DIR / f"gru_{dataset_name}.pt"

                if w_gru_ret60 > 0:
                    if gru_ret60_v6_path.exists():
                        gru_ret60_pred = _run_single_target_model(
                            gru_ret60_v6_path, padded_r60, T, F, device, batch_r60, window_r60)
                    elif gru_v5_path.exists():
                        # v5 fallback
                        v5_window = 60
                        padded_v5 = np.zeros((v5_window - 1 + T, F), dtype=np.float32)
                        padded_v5[v5_window - 1:] = clean
                        v5_pred = _run_v5_sequence_model(
                            gru_v5_path, padded_v5, T, F, device, batch_r60, v5_window)
                        gru_ret60_pred = v5_pred[:, 1]
                        del padded_v5

                tf_ret60_v6_path = MODEL_DIR / f"transformer_ret60_{dataset_name}.pt"
                tf_v5_path = MODEL_DIR / f"transformer_{dataset_name}.pt"

                if w_tf_ret60 > 0:
                    if tf_ret60_v6_path.exists():
                        tf_ret60_pred = _run_single_target_model(
                            tf_ret60_v6_path, padded_r60, T, F, device, batch_r60, window_r60)
                    elif tf_v5_path.exists():
                        v5_window = 60
                        padded_v5 = np.zeros((v5_window - 1 + T, F), dtype=np.float32)
                        padded_v5[v5_window - 1:] = clean
                        v5_pred = _run_v5_sequence_model(
                            tf_v5_path, padded_v5, T, F, device, batch_r60, v5_window)
                        tf_ret60_pred = v5_pred[:, 1]
                        del padded_v5

                del padded_r60

        del clean

    # =========================================================================
    # Ensemble: weighted combination
    # =========================================================================
    # Normal regime prediction (all models except extreme)
    pred_ret5 = (w_local_r5 * local_pred_r5 +
                 w_global_r5 * global_pred_r5 +
                 w_gru_ret5 * gru_ret5_pred +
                 w_tf_ret5 * tf_ret5_pred)

    pred_ret60 = (w_local_r60 * local_pred_r60 +
                  w_global_r60 * global_pred_r60 +
                  w_gru_ret60 * gru_ret60_pred +
                  w_tf_ret60 * tf_ret60_pred)

    # Extreme regime: auto-blend when detected (50% normal + 50% extreme)
    # Only activates if extreme model files exist and were loaded successfully
    EXTREME_BLEND = 0.0  # DISABLED for Round 2 A/B test — set to 0.5 to re-enable
    has_extreme_r5 = np.any(extreme_pred_r5 != 0) and EXTREME_BLEND > 0
    has_extreme_r60 = np.any(extreme_pred_r60 != 0) and EXTREME_BLEND > 0

    if has_extreme_r5:
        extreme_mask = extreme_indicator.astype(np.bool_)
        pred_ret5[extreme_mask] = (
            (1 - EXTREME_BLEND) * pred_ret5[extreme_mask] +
            EXTREME_BLEND * extreme_pred_r5[extreme_mask]
        )

    if has_extreme_r60:
        extreme_mask = extreme_indicator.astype(np.bool_)
        pred_ret60[extreme_mask] = (
            (1 - EXTREME_BLEND) * pred_ret60[extreme_mask] +
            EXTREME_BLEND * extreme_pred_r60[extreme_mask]
        )

    signals = np.column_stack([pred_ret5, pred_ret60]).astype(np.float32)
    return np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
