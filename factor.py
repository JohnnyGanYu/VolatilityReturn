# Version: v7 — 165-dim features (final)
import numpy as np
import random
from numba import njit, prange


# =============================================================================
# Seed initialization
# =============================================================================

def _set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# Numba-accelerated helper: rolling window computations
# =============================================================================

@njit(cache=True)
def _rolling_mean(arr, window):
    """Causal rolling mean. First window-1 values are NaN."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window - 1] = np.nan
    s = 0.0
    cnt = 0
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            # NaN poisons the window; we still advance but mark output NaN
            # Use a simpler approach: recompute from scratch when NaN present
            pass
        if i >= window:
            pass
        # Simple approach: compute from scratch for robustness
        if i < window - 1:
            continue
        total = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            vj = arr[j]
            if not np.isnan(vj):
                total += vj
                count += 1
        if count > 0:
            out[i] = total / count
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def _rolling_sum(arr, window):
    """Causal rolling sum. First window-1 values are NaN."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        total = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            vj = arr[j]
            if not np.isnan(vj):
                total += vj
                count += 1
        if count > 0:
            out[i] = total
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def _rolling_std(arr, window):
    """Causal rolling standard deviation. First window-1 values are NaN."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        total = 0.0
        total_sq = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            vj = arr[j]
            if not np.isnan(vj):
                total += vj
                total_sq += vj * vj
                count += 1
        if count >= 2:
            mean = total / count
            var = total_sq / count - mean * mean
            if var < 0.0:
                var = 0.0
            out[i] = np.sqrt(var)
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def _rolling_max(arr, window):
    """Causal rolling max. First window-1 values are NaN."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        mx = -np.inf
        found = False
        for j in range(i - window + 1, i + 1):
            vj = arr[j]
            if not np.isnan(vj):
                if vj > mx:
                    mx = vj
                found = True
        if found:
            out[i] = mx
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def _rolling_min(arr, window):
    """Causal rolling min. First window-1 values are NaN."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window - 1] = np.nan
    for i in range(window - 1, n):
        mn = np.inf
        found = False
        for j in range(i - window + 1, i + 1):
            vj = arr[j]
            if not np.isnan(vj):
                if vj < mn:
                    mn = vj
                found = True
        if found:
            out[i] = mn
        else:
            out[i] = np.nan
    return out


# =============================================================================
# Momentum features (Task 1.2)
# =============================================================================

@njit(cache=True)
def _compute_log_returns(close, window):
    """Log return over `window` bars: log(close[i] / close[i-window])."""
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window] = np.nan
    for i in range(window, n):
        c_now = close[i]
        c_prev = close[i - window]
        if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0.0:
            out[i] = np.nan
        else:
            out[i] = np.log(c_now / c_prev)
    return out


@njit(cache=True)
def _compute_momentum_features(open_, high, low, close, volume):
    """Compute momentum features. Returns list of 1D arrays."""
    n = close.shape[0]
    lookbacks = np.array([1, 3, 5, 10, 20, 60, 120])
    num_lb = lookbacks.shape[0]
    # log returns: 7 features
    # rate of change (close/close_prev - 1): 7 features
    # Total: 14 features
    num_features = num_lb * 2
    result = np.empty((n, num_features), dtype=np.float64)

    for k in range(num_lb):
        w = lookbacks[k]
        for i in range(n):
            if i < w:
                result[i, k] = np.nan
                result[i, k + num_lb] = np.nan
            else:
                c_now = close[i]
                c_prev = close[i - w]
                if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0.0:
                    result[i, k] = np.nan
                    result[i, k + num_lb] = np.nan
                else:
                    result[i, k] = np.log(c_now / c_prev)          # log return
                    result[i, k + num_lb] = c_now / c_prev - 1.0   # rate of change
    return result


# =============================================================================
# Volatility features (Task 1.3)
# =============================================================================

@njit(cache=True)
def _compute_volatility_features(open_, high, low, close, volume):
    """
    Compute volatility features:
    - Rolling std of 1-bar log returns over windows [5, 10, 20, 60, 120]  (5)
    - Parkinson volatility over same windows                               (5)
    - Garman-Klass volatility over same windows                            (5)
    - ATR over same windows                                                (5)
    Total: 20 features
    """
    n = close.shape[0]
    windows = np.array([5, 10, 20, 60, 120])
    nw = windows.shape[0]
    num_features = nw * 4
    result = np.empty((n, num_features), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(close[i] / close[i - 1])

    # Pre-compute true range
    true_range = np.empty(n, dtype=np.float64)
    true_range[0] = np.nan
    for i in range(1, n):
        h = high[i]
        l = low[i]
        cp = close[i - 1]
        if np.isnan(h) or np.isnan(l) or np.isnan(cp):
            true_range[i] = np.nan
        else:
            tr1 = h - l
            tr2 = abs(h - cp)
            tr3 = abs(l - cp)
            true_range[i] = max(tr1, max(tr2, tr3))

    for wi in range(nw):
        w = windows[wi]
        col_std = wi
        col_park = wi + nw
        col_gk = wi + nw * 2
        col_atr = wi + nw * 3

        for i in range(w, n):
            # Rolling std of log returns
            total = 0.0
            total_sq = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = log_ret[j]
                if not np.isnan(v):
                    total += v
                    total_sq += v * v
                    cnt += 1
            if cnt >= 2:
                mean = total / cnt
                var = total_sq / cnt - mean * mean
                if var < 0.0:
                    var = 0.0
                result[i, col_std] = np.sqrt(var)
            else:
                result[i, col_std] = np.nan

            # Parkinson volatility: sqrt(1/(4*n*ln2) * sum(ln(H/L))^2)
            park_sum = 0.0
            park_cnt = 0
            for j in range(i - w + 1, i + 1):
                h = high[j]
                l = low[j]
                if not np.isnan(h) and not np.isnan(l) and l > 0.0:
                    hl = np.log(h / l)
                    park_sum += hl * hl
                    park_cnt += 1
            if park_cnt >= 1:
                result[i, col_park] = np.sqrt(park_sum / (4.0 * park_cnt * 0.6931471805599453))
            else:
                result[i, col_park] = np.nan

            # Garman-Klass volatility
            gk_sum = 0.0
            gk_cnt = 0
            for j in range(i - w + 1, i + 1):
                h = high[j]
                l = low[j]
                o = open_[j]
                c = close[j]
                if not np.isnan(h) and not np.isnan(l) and not np.isnan(o) and not np.isnan(c) and l > 0.0 and o > 0.0:
                    hl = np.log(h / l)
                    co = np.log(c / o)
                    gk_sum += 0.5 * hl * hl - (2.0 * 0.6931471805599453 - 1.0) * co * co
                    gk_cnt += 1
            if gk_cnt >= 1:
                result[i, col_gk] = np.sqrt(abs(gk_sum / gk_cnt))
            else:
                result[i, col_gk] = np.nan

            # ATR (Average True Range)
            atr_sum = 0.0
            atr_cnt = 0
            for j in range(i - w + 1, i + 1):
                tr = true_range[j]
                if not np.isnan(tr):
                    atr_sum += tr
                    atr_cnt += 1
            if atr_cnt >= 1:
                result[i, col_atr] = atr_sum / atr_cnt
            else:
                result[i, col_atr] = np.nan

    return result


# =============================================================================
# Volume and microstructure features (Task 1.4)
# =============================================================================

@njit(cache=True)
def _compute_volume_features(open_, high, low, close, volume):
    """
    Volume features:
    - Volume MA ratios (volume / rolling_mean_volume) for windows [5, 10, 20, 60]  (4)
    - Log volume                                                                     (1)
    - Volume change (volume[i] / volume[i-1])                                        (1)
    - VWAP deviation: (close - vwap) / close for windows [5, 10, 20, 60]            (4)
    - OBV (On-Balance Volume) change over windows [5, 10, 20, 60]                   (4)
    Total: 14 features
    """
    n = close.shape[0]
    windows = np.array([5, 10, 20, 60])
    nw = windows.shape[0]
    num_features = nw + 1 + 1 + nw + nw  # 14
    result = np.empty((n, num_features), dtype=np.float64)
    result[:] = np.nan

    # Log volume (feature 0 after vol_ma_ratios)
    col_logvol = nw
    for i in range(n):
        v = volume[i]
        if np.isnan(v) or v < 0.0:
            result[i, col_logvol] = np.nan
        else:
            result[i, col_logvol] = np.log(v + 1.0)

    # Volume change
    col_volchg = nw + 1
    result[0, col_volchg] = np.nan
    for i in range(1, n):
        v_now = volume[i]
        v_prev = volume[i - 1]
        if np.isnan(v_now) or np.isnan(v_prev) or v_prev <= 0.0:
            result[i, col_volchg] = np.nan
        else:
            result[i, col_volchg] = v_now / v_prev - 1.0

    # Pre-compute typical price for VWAP
    typical = np.empty(n, dtype=np.float64)
    for i in range(n):
        h = high[i]
        l = low[i]
        c = close[i]
        if np.isnan(h) or np.isnan(l) or np.isnan(c):
            typical[i] = np.nan
        else:
            typical[i] = (h + l + c) / 3.0

    # Pre-compute OBV
    obv = np.empty(n, dtype=np.float64)
    obv[0] = 0.0
    for i in range(1, n):
        c_now = close[i]
        c_prev = close[i - 1]
        v = volume[i]
        if np.isnan(c_now) or np.isnan(c_prev) or np.isnan(v):
            obv[i] = obv[i - 1]
        elif c_now > c_prev:
            obv[i] = obv[i - 1] + v
        elif c_now < c_prev:
            obv[i] = obv[i - 1] - v
        else:
            obv[i] = obv[i - 1]

    for wi in range(nw):
        w = windows[wi]
        col_vol_ratio = wi
        col_vwap_dev = nw + 2 + wi
        col_obv_chg = nw + 2 + nw + wi

        for i in range(w - 1, n):
            # Volume MA ratio
            vol_sum = 0.0
            vol_cnt = 0
            for j in range(i - w + 1, i + 1):
                vj = volume[j]
                if not np.isnan(vj):
                    vol_sum += vj
                    vol_cnt += 1
            if vol_cnt > 0 and vol_sum > 0.0:
                vol_ma = vol_sum / vol_cnt
                v_now = volume[i]
                if not np.isnan(v_now):
                    result[i, col_vol_ratio] = v_now / vol_ma
                else:
                    result[i, col_vol_ratio] = np.nan
            else:
                result[i, col_vol_ratio] = np.nan

            # VWAP deviation
            tp_vol_sum = 0.0
            vol_sum2 = 0.0
            for j in range(i - w + 1, i + 1):
                tp = typical[j]
                vj = volume[j]
                if not np.isnan(tp) and not np.isnan(vj):
                    tp_vol_sum += tp * vj
                    vol_sum2 += vj
            c = close[i]
            if vol_sum2 > 0.0 and not np.isnan(c) and c > 0.0:
                vwap = tp_vol_sum / vol_sum2
                result[i, col_vwap_dev] = (c - vwap) / c
            else:
                result[i, col_vwap_dev] = np.nan

            # OBV change over window
            if i >= w:
                result[i, col_obv_chg] = obv[i] - obv[i - w]
            else:
                result[i, col_obv_chg] = np.nan

    return result


@njit(cache=True)
def _compute_microstructure_features(open_, high, low, close, volume):
    """
    Microstructure features:
    - Spread proxy: (high - low) / close                                    (1)
    - Upper shadow ratio: (high - max(open, close)) / (high - low + 1e-10) (1)
    - Lower shadow ratio: (min(open, close) - low) / (high - low + 1e-10)  (1)
    - Bar return: (close - open) / open                                     (1)
    - Bar return / range: (close - open) / (high - low + 1e-10)            (1)
    - Rolling spread proxy mean for windows [5, 10, 20]                     (3)
    - Rolling bar return skewness for windows [20, 60]                      (2)
    - Amihud illiquidity: |return| / volume for windows [5, 10, 20]        (3)
    - High-close / close-low ratio                                          (1)
    Total: 14 features
    """
    n = close.shape[0]
    windows_spread = np.array([5, 10, 20])
    windows_skew = np.array([20, 60])
    windows_amihud = np.array([5, 10, 20])
    num_features = 5 + 3 + 2 + 3 + 1  # 14
    result = np.empty((n, num_features), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute per-bar features
    spread_proxy = np.empty(n, dtype=np.float64)
    bar_return = np.empty(n, dtype=np.float64)
    abs_ret_over_vol = np.empty(n, dtype=np.float64)

    for i in range(n):
        h = high[i]
        l = low[i]
        c = close[i]
        o = open_[i]
        v = volume[i]

        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(o):
            spread_proxy[i] = np.nan
            bar_return[i] = np.nan
            result[i, 0] = np.nan  # spread proxy
            result[i, 1] = np.nan  # upper shadow
            result[i, 2] = np.nan  # lower shadow
            result[i, 3] = np.nan  # bar return
            result[i, 4] = np.nan  # bar return / range
            result[i, 13] = np.nan  # high-close / close-low
        else:
            hl = h - l
            sp = hl / c if c > 0.0 else np.nan
            spread_proxy[i] = sp
            result[i, 0] = sp

            # Upper shadow
            body_top = max(o, c)
            body_bot = min(o, c)
            if hl > 1e-10:
                result[i, 1] = (h - body_top) / hl
                result[i, 2] = (body_bot - l) / hl
                result[i, 4] = (c - o) / hl
            else:
                result[i, 1] = 0.0
                result[i, 2] = 0.0
                result[i, 4] = 0.0

            # Bar return
            if o > 0.0:
                br = (c - o) / o
                bar_return[i] = br
                result[i, 3] = br
            else:
                bar_return[i] = np.nan
                result[i, 3] = np.nan

            # High-close / close-low ratio
            hc = h - c
            cl = c - l
            if cl > 1e-10:
                result[i, 13] = hc / cl
            else:
                result[i, 13] = np.nan

        # Amihud: |1-bar return| / volume
        if i == 0:
            abs_ret_over_vol[i] = np.nan
        else:
            c_now = close[i]
            c_prev = close[i - 1]
            if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0.0 or np.isnan(v) or v <= 0.0:
                abs_ret_over_vol[i] = np.nan
            else:
                abs_ret_over_vol[i] = abs(np.log(c_now / c_prev)) / v

    # Rolling spread proxy mean
    for wi in range(3):
        w = windows_spread[wi]
        col = 5 + wi
        for i in range(w - 1, n):
            total = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                vj = spread_proxy[j]
                if not np.isnan(vj):
                    total += vj
                    cnt += 1
            if cnt > 0:
                result[i, col] = total / cnt
            else:
                result[i, col] = np.nan

    # Rolling bar return skewness
    for wi in range(2):
        w = windows_skew[wi]
        col = 8 + wi
        for i in range(w - 1, n):
            total = 0.0
            total_sq = 0.0
            total_cb = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                vj = bar_return[j]
                if not np.isnan(vj):
                    total += vj
                    total_sq += vj * vj
                    total_cb += vj * vj * vj
                    cnt += 1
            if cnt >= 3:
                mean = total / cnt
                var = total_sq / cnt - mean * mean
                if var > 1e-20:
                    std = np.sqrt(var)
                    m3 = total_cb / cnt - 3.0 * mean * total_sq / cnt + 2.0 * mean * mean * mean
                    result[i, col] = m3 / (std * std * std)
                else:
                    result[i, col] = 0.0
            else:
                result[i, col] = np.nan

    # Rolling Amihud illiquidity
    for wi in range(3):
        w = windows_amihud[wi]
        col = 10 + wi
        for i in range(w - 1, n):
            total = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                vj = abs_ret_over_vol[j]
                if not np.isnan(vj):
                    total += vj
                    cnt += 1
            if cnt > 0:
                result[i, col] = total / cnt
            else:
                result[i, col] = np.nan

    return result


# =============================================================================
# Technical indicator features (Task 1.5)
# =============================================================================

@njit(cache=True)
def _ema(arr, span):
    """Exponential moving average (causal). Uses span to compute alpha."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        v = arr[i]
        if np.isnan(v):
            out[i] = out[i - 1]
        elif np.isnan(out[i - 1]):
            out[i] = v
        else:
            out[i] = alpha * v + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def _compute_rsi(close, window):
    """RSI (Relative Strength Index). Causal, NaN-safe."""
    n = close.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:window] = np.nan

    # Compute 1-bar changes
    delta = np.empty(n, dtype=np.float64)
    delta[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]):
            delta[i] = np.nan
        else:
            delta[i] = close[i] - close[i - 1]

    # First RSI value: simple average of gains/losses
    gain_sum = 0.0
    loss_sum = 0.0
    cnt = 0
    for j in range(1, window + 1):
        d = delta[j]
        if not np.isnan(d):
            if d > 0:
                gain_sum += d
            else:
                loss_sum += (-d)
            cnt += 1
    if cnt > 0:
        avg_gain = gain_sum / cnt
        avg_loss = loss_sum / cnt
    else:
        avg_gain = 0.0
        avg_loss = 0.0

    if avg_loss == 0.0:
        out[window] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        out[window] = 100.0 - 100.0 / (1.0 + rs)

    # Subsequent values: exponential smoothing
    alpha = 1.0 / window
    for i in range(window + 1, n):
        d = delta[i]
        if np.isnan(d):
            out[i] = out[i - 1]
        else:
            g = d if d > 0 else 0.0
            l = -d if d < 0 else 0.0
            avg_gain = alpha * g + (1.0 - alpha) * avg_gain
            avg_loss = alpha * l + (1.0 - alpha) * avg_loss
            if avg_loss == 0.0:
                out[i] = 100.0 if avg_gain > 0.0 else 50.0
            else:
                rs = avg_gain / avg_loss
                out[i] = 100.0 - 100.0 / (1.0 + rs)

    return out


@njit(cache=True)
def _compute_technical_features(open_, high, low, close, volume):
    """
    Technical indicator features:
    - RSI(14), RSI(28)                                                      (2)
    - MACD line (EMA12 - EMA26)                                             (1)
    - MACD signal (EMA9 of MACD line)                                       (1)
    - MACD histogram                                                        (1)
    - Bollinger Band width for windows [20, 60]                             (2)
    - Bollinger %B for windows [20, 60]                                     (2)
    - Stochastic %K for windows [14, 28]                                    (2)
    - Stochastic %D (3-period SMA of %K) for windows [14, 28]              (2)
    - CCI for windows [20, 60]                                              (2)
    - Williams %R for windows [14, 28]                                      (2)
    - Price position in range (close - low) / (high - low) rolling [20, 60] (2)
    - Momentum acceleration (return[i] - return[i-5])                       (1)
    Total: 20 features
    """
    n = close.shape[0]
    num_features = 20
    result = np.empty((n, num_features), dtype=np.float64)
    result[:] = np.nan

    # RSI(14), RSI(28)
    rsi14 = _compute_rsi(close, 14)
    rsi28 = _compute_rsi(close, 28)
    result[:, 0] = rsi14
    result[:, 1] = rsi28

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(ema12[i]) or np.isnan(ema26[i]):
            macd_line[i] = np.nan
        else:
            macd_line[i] = ema12[i] - ema26[i]
    macd_signal = _ema(macd_line, 9)
    for i in range(n):
        result[i, 2] = macd_line[i]
        result[i, 3] = macd_signal[i]
        if np.isnan(macd_line[i]) or np.isnan(macd_signal[i]):
            result[i, 4] = np.nan
        else:
            result[i, 4] = macd_line[i] - macd_signal[i]

    # Bollinger Bands
    bb_windows = np.array([20, 60])
    for wi in range(2):
        w = bb_windows[wi]
        col_width = 5 + wi
        col_pctb = 7 + wi
        for i in range(w - 1, n):
            total = 0.0
            total_sq = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = close[j]
                if not np.isnan(v):
                    total += v
                    total_sq += v * v
                    cnt += 1
            if cnt >= 2:
                mean = total / cnt
                var = total_sq / cnt - mean * mean
                if var < 0.0:
                    var = 0.0
                std = np.sqrt(var)
                upper = mean + 2.0 * std
                lower = mean - 2.0 * std
                if mean > 0.0:
                    result[i, col_width] = 4.0 * std / mean  # BB width
                else:
                    result[i, col_width] = np.nan
                band_range = upper - lower
                if band_range > 1e-10:
                    result[i, col_pctb] = (close[i] - lower) / band_range  # %B
                else:
                    result[i, col_pctb] = 0.5

    # Stochastic %K and %D
    stoch_windows = np.array([14, 28])
    for wi in range(2):
        w = stoch_windows[wi]
        col_k = 9 + wi
        col_d = 11 + wi
        pct_k = np.empty(n, dtype=np.float64)
        pct_k[:] = np.nan
        for i in range(w - 1, n):
            hh = -np.inf
            ll = np.inf
            found = False
            for j in range(i - w + 1, i + 1):
                h = high[j]
                l = low[j]
                if not np.isnan(h) and not np.isnan(l):
                    if h > hh:
                        hh = h
                    if l < ll:
                        ll = l
                    found = True
            c = close[i]
            if found and not np.isnan(c) and (hh - ll) > 1e-10:
                pct_k[i] = (c - ll) / (hh - ll) * 100.0
            elif found:
                pct_k[i] = 50.0
        result[:, col_k] = pct_k

        # %D = 3-period SMA of %K
        for i in range(2, n):
            total = 0.0
            cnt = 0
            for j in range(max(0, i - 2), i + 1):
                v = pct_k[j]
                if not np.isnan(v):
                    total += v
                    cnt += 1
            if cnt > 0:
                result[i, col_d] = total / cnt

    # CCI
    cci_windows = np.array([20, 60])
    for wi in range(2):
        w = cci_windows[wi]
        col = 13 + wi
        for i in range(w - 1, n):
            h = high[i]
            l = low[i]
            c = close[i]
            if np.isnan(h) or np.isnan(l) or np.isnan(c):
                continue
            tp = (h + l + c) / 3.0
            # Mean of typical prices
            total = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                hj = high[j]
                lj = low[j]
                cj = close[j]
                if not np.isnan(hj) and not np.isnan(lj) and not np.isnan(cj):
                    total += (hj + lj + cj) / 3.0
                    cnt += 1
            if cnt >= 2:
                mean_tp = total / cnt
                # Mean absolute deviation
                mad = 0.0
                for j in range(i - w + 1, i + 1):
                    hj = high[j]
                    lj = low[j]
                    cj = close[j]
                    if not np.isnan(hj) and not np.isnan(lj) and not np.isnan(cj):
                        mad += abs((hj + lj + cj) / 3.0 - mean_tp)
                mad /= cnt
                if mad > 1e-10:
                    result[i, col] = (tp - mean_tp) / (0.015 * mad)
                else:
                    result[i, col] = 0.0

    # Williams %R
    wr_windows = np.array([14, 28])
    for wi in range(2):
        w = wr_windows[wi]
        col = 15 + wi
        for i in range(w - 1, n):
            hh = -np.inf
            ll = np.inf
            found = False
            for j in range(i - w + 1, i + 1):
                h = high[j]
                l = low[j]
                if not np.isnan(h) and not np.isnan(l):
                    if h > hh:
                        hh = h
                    if l < ll:
                        ll = l
                    found = True
            c = close[i]
            if found and not np.isnan(c) and (hh - ll) > 1e-10:
                result[i, col] = (hh - c) / (hh - ll) * -100.0
            elif found:
                result[i, col] = -50.0

    # Price position in range
    pp_windows = np.array([20, 60])
    for wi in range(2):
        w = pp_windows[wi]
        col = 17 + wi
        for i in range(w - 1, n):
            hh = -np.inf
            ll = np.inf
            found = False
            for j in range(i - w + 1, i + 1):
                h = high[j]
                l = low[j]
                if not np.isnan(h) and not np.isnan(l):
                    if h > hh:
                        hh = h
                    if l < ll:
                        ll = l
                    found = True
            c = close[i]
            if found and not np.isnan(c) and (hh - ll) > 1e-10:
                result[i, col] = (c - ll) / (hh - ll)
            elif found:
                result[i, col] = 0.5

    # Momentum acceleration: return[i] - return[i-5]
    col_acc = 19
    for i in range(6, n):
        c_now = close[i]
        c_1 = close[i - 1]
        c_5 = close[i - 5]
        c_6 = close[i - 6]
        if np.isnan(c_now) or np.isnan(c_1) or np.isnan(c_5) or np.isnan(c_6) or c_1 <= 0.0 or c_6 <= 0.0:
            result[i, col_acc] = np.nan
        else:
            ret_now = np.log(c_now / c_1)
            ret_5 = np.log(c_5 / c_6)
            result[i, col_acc] = ret_now - ret_5

    return result


# =============================================================================
# Regime detection and cross-interaction features (Task 1.6)
# =============================================================================

@njit(cache=True)
def _compute_regime_features(open_, high, low, close, volume):
    """
    Regime features:
    - Vol-of-vol (rolling std of rolling volatility) for windows [20, 60, 120]  (3)
    - Max drawdown over windows [20, 60, 120]                                   (3)
    - Drawdown from rolling max for windows [20, 60, 120]                       (3)
    - Extreme move flag: |1-bar return| > 2 * rolling_std(20)                   (1)
    - Up/down streak length                                                      (1)
    - Volume spike: volume / rolling_mean(20) > 3                               (1)
    Total: 12 features
    """
    n = close.shape[0]
    windows = np.array([20, 60, 120])
    nw = windows.shape[0]
    num_features = nw + nw + nw + 1 + 1 + 1  # 12
    result = np.empty((n, num_features), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(close[i] / close[i - 1])

    # Pre-compute rolling volatility (std of returns, window=20)
    roll_vol_20 = np.empty(n, dtype=np.float64)
    roll_vol_20[:] = np.nan
    for i in range(19, n):
        total = 0.0
        total_sq = 0.0
        cnt = 0
        for j in range(i - 19, i + 1):
            v = log_ret[j]
            if not np.isnan(v):
                total += v
                total_sq += v * v
                cnt += 1
        if cnt >= 2:
            mean = total / cnt
            var = total_sq / cnt - mean * mean
            if var < 0.0:
                var = 0.0
            roll_vol_20[i] = np.sqrt(var)

    # Vol-of-vol: rolling std of roll_vol_20
    for wi in range(nw):
        w = windows[wi]
        col = wi
        for i in range(w - 1, n):
            total = 0.0
            total_sq = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = roll_vol_20[j]
                if not np.isnan(v):
                    total += v
                    total_sq += v * v
                    cnt += 1
            if cnt >= 2:
                mean = total / cnt
                var = total_sq / cnt - mean * mean
                if var < 0.0:
                    var = 0.0
                result[i, col] = np.sqrt(var)

    # Max drawdown over windows
    for wi in range(nw):
        w = windows[wi]
        col_mdd = nw + wi
        col_dd = nw * 2 + wi
        for i in range(w - 1, n):
            # Find rolling max of close in window
            roll_max = -np.inf
            found = False
            for j in range(i - w + 1, i + 1):
                c = close[j]
                if not np.isnan(c) and c > roll_max:
                    roll_max = c
                    found = True
            if not found or np.isnan(close[i]):
                continue

            # Max drawdown: max peak-to-trough in window
            peak = close[i - w + 1] if not np.isnan(close[i - w + 1]) else -np.inf
            max_dd = 0.0
            for j in range(i - w + 1, i + 1):
                c = close[j]
                if np.isnan(c):
                    continue
                if c > peak:
                    peak = c
                if peak > 0.0:
                    dd = (peak - c) / peak
                    if dd > max_dd:
                        max_dd = dd
            result[i, col_mdd] = max_dd

            # Current drawdown from rolling max
            if roll_max > 0.0:
                result[i, col_dd] = (roll_max - close[i]) / roll_max

    # Extreme move flag
    col_extreme = nw * 3
    for i in range(1, n):
        r = log_ret[i]
        rv = roll_vol_20[i]
        if not np.isnan(r) and not np.isnan(rv) and rv > 0.0:
            result[i, col_extreme] = 1.0 if abs(r) > 2.0 * rv else 0.0

    # Up/down streak
    col_streak = nw * 3 + 1
    streak = 0.0
    for i in range(1, n):
        r = log_ret[i]
        if np.isnan(r):
            result[i, col_streak] = 0.0
        elif r > 0:
            if streak > 0:
                streak += 1.0
            else:
                streak = 1.0
            result[i, col_streak] = streak
        elif r < 0:
            if streak < 0:
                streak -= 1.0
            else:
                streak = -1.0
            result[i, col_streak] = streak
        else:
            streak = 0.0
            result[i, col_streak] = 0.0

    # Volume spike flag
    col_vspike = nw * 3 + 2
    for i in range(19, n):
        vol_sum = 0.0
        vol_cnt = 0
        for j in range(i - 19, i + 1):
            v = volume[j]
            if not np.isnan(v):
                vol_sum += v
                vol_cnt += 1
        if vol_cnt > 0:
            vol_ma = vol_sum / vol_cnt
            v_now = volume[i]
            if not np.isnan(v_now) and vol_ma > 0.0:
                result[i, col_vspike] = v_now / vol_ma
            else:
                result[i, col_vspike] = np.nan

    return result


@njit(cache=True)
def _compute_cross_features(open_, high, low, close, volume):
    """
    Cross-interaction features:
    - Momentum × Volatility: ret_w * vol_w for w in [5, 10, 20, 60]        (4)
    - Volume × Return: volume_ratio_w * ret_w for w in [5, 10, 20, 60]     (4)
    - Return / Volatility (Sharpe-like): ret_w / vol_w for w in [5, 10, 20, 60] (4)
    - Price acceleration: ret_5 - ret_10                                     (1)
    - Volume trend: vol_ma_5 / vol_ma_20                                     (1)
    - Volatility ratio: vol_5 / vol_60                                       (1)
    Total: 15 features
    """
    n = close.shape[0]
    windows = np.array([5, 10, 20, 60])
    nw = windows.shape[0]
    num_features = nw * 3 + 3  # 15
    result = np.empty((n, num_features), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute log returns for each window
    rets = np.empty((n, nw), dtype=np.float64)
    rets[:] = np.nan
    for wi in range(nw):
        w = windows[wi]
        for i in range(w, n):
            if np.isnan(close[i]) or np.isnan(close[i - w]) or close[i - w] <= 0.0:
                rets[i, wi] = np.nan
            else:
                rets[i, wi] = np.log(close[i] / close[i - w])

    # Pre-compute rolling volatility for each window
    log_ret1 = np.empty(n, dtype=np.float64)
    log_ret1[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret1[i] = np.nan
        else:
            log_ret1[i] = np.log(close[i] / close[i - 1])

    vols = np.empty((n, nw), dtype=np.float64)
    vols[:] = np.nan
    for wi in range(nw):
        w = windows[wi]
        for i in range(w - 1, n):
            total = 0.0
            total_sq = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = log_ret1[j]
                if not np.isnan(v):
                    total += v
                    total_sq += v * v
                    cnt += 1
            if cnt >= 2:
                mean = total / cnt
                var = total_sq / cnt - mean * mean
                if var < 0.0:
                    var = 0.0
                vols[i, wi] = np.sqrt(var)

    # Pre-compute volume MA ratios
    vol_ratios = np.empty((n, nw), dtype=np.float64)
    vol_ratios[:] = np.nan
    for wi in range(nw):
        w = windows[wi]
        for i in range(w - 1, n):
            vol_sum = 0.0
            vol_cnt = 0
            for j in range(i - w + 1, i + 1):
                v = volume[j]
                if not np.isnan(v):
                    vol_sum += v
                    vol_cnt += 1
            if vol_cnt > 0 and vol_sum > 0.0:
                vol_ma = vol_sum / vol_cnt
                v_now = volume[i]
                if not np.isnan(v_now):
                    vol_ratios[i, wi] = v_now / vol_ma

    # Momentum × Volatility
    for wi in range(nw):
        col = wi
        for i in range(n):
            r = rets[i, wi]
            v = vols[i, wi]
            if not np.isnan(r) and not np.isnan(v):
                result[i, col] = r * v

    # Volume × Return
    for wi in range(nw):
        col = nw + wi
        for i in range(n):
            vr = vol_ratios[i, wi]
            r = rets[i, wi]
            if not np.isnan(vr) and not np.isnan(r):
                result[i, col] = vr * r

    # Return / Volatility (Sharpe-like)
    for wi in range(nw):
        col = nw * 2 + wi
        for i in range(n):
            r = rets[i, wi]
            v = vols[i, wi]
            if not np.isnan(r) and not np.isnan(v) and v > 1e-10:
                result[i, col] = r / v

    # Price acceleration: ret_5 - ret_10
    col_acc = nw * 3
    for i in range(n):
        r5 = rets[i, 0]   # window=5
        r10 = rets[i, 1]  # window=10
        if not np.isnan(r5) and not np.isnan(r10):
            result[i, col_acc] = r5 - r10

    # Volume trend: vol_ma_5 / vol_ma_20
    col_vtrend = nw * 3 + 1
    for i in range(n):
        vr5 = vol_ratios[i, 0]   # window=5 ratio
        vr20 = vol_ratios[i, 2]  # window=20 ratio
        # We need actual vol_ma values, not ratios. Approximate via ratio of ratios.
        # Actually, let's compute vol_ma_5 / vol_ma_20 directly
        if i >= 19:
            vol_sum5 = 0.0
            cnt5 = 0
            for j in range(i - 4, i + 1):
                v = volume[j]
                if not np.isnan(v):
                    vol_sum5 += v
                    cnt5 += 1
            vol_sum20 = 0.0
            cnt20 = 0
            for j in range(i - 19, i + 1):
                v = volume[j]
                if not np.isnan(v):
                    vol_sum20 += v
                    cnt20 += 1
            if cnt5 > 0 and cnt20 > 0:
                ma5 = vol_sum5 / cnt5
                ma20 = vol_sum20 / cnt20
                if ma20 > 0.0:
                    result[i, col_vtrend] = ma5 / ma20

    # Volatility ratio: vol_5 / vol_60
    col_vratio = nw * 3 + 2
    for i in range(n):
        v5 = vols[i, 0]   # window=5
        v60 = vols[i, 3]  # window=60
        if not np.isnan(v5) and not np.isnan(v60) and v60 > 1e-10:
            result[i, col_vratio] = v5 / v60

    return result


# =============================================================================
# New features: EMA ratios (Task 3.1)
# =============================================================================

@njit(cache=True)
def _compute_ema_ratios(close):
    """
    EMA ratio features: close[i] / EMA(close, span)[i] for spans [5, 10, 20, 60, 120].
    Measures price deviation from smoothed trend at different time scales.
    Total: 5 features
    """
    n = close.shape[0]
    spans = np.array([5, 10, 20, 60, 120])
    result = np.empty((n, 5), dtype=np.float64)
    result[:] = np.nan

    for si in range(5):
        span = spans[si]
        alpha = 2.0 / (span + 1.0)
        ema_val = np.nan
        for i in range(n):
            c = close[i]
            if np.isnan(c):
                result[i, si] = np.nan
            elif np.isnan(ema_val):
                ema_val = c
                result[i, si] = 0.0  # close/ema - 1 = 0 at initialization
            else:
                ema_val = alpha * c + (1.0 - alpha) * ema_val
                if ema_val > 0.0:
                    result[i, si] = c / ema_val - 1.0
                else:
                    result[i, si] = np.nan
    return result


# =============================================================================
# New features: Rolling skewness and kurtosis (Task 3.3)
# =============================================================================

@njit(cache=True)
def _compute_rolling_skew_kurt(close):
    """
    Rolling skewness and kurtosis of 1-bar log returns.
    Skewness over windows [20, 60, 120] (3) + Kurtosis over [20, 60, 120] (3).
    Total: 6 features
    """
    n = close.shape[0]
    windows = np.array([20, 60, 120])
    nw = 3
    result = np.empty((n, 6), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(close[i] / close[i - 1])

    for wi in range(nw):
        w = windows[wi]
        col_skew = wi
        col_kurt = nw + wi
        for i in range(w - 1, n):
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = log_ret[j]
                if not np.isnan(v):
                    s1 += v
                    s2 += v * v
                    s3 += v * v * v
                    s4 += v * v * v * v
                    cnt += 1
            if cnt >= 3:
                mean = s1 / cnt
                var = s2 / cnt - mean * mean
                if var > 1e-20:
                    std = np.sqrt(var)
                    # Skewness: E[(x-mu)^3] / std^3
                    m3 = s3 / cnt - 3.0 * mean * s2 / cnt + 2.0 * mean * mean * mean
                    result[i, col_skew] = m3 / (std * std * std)
                    # Kurtosis: E[(x-mu)^4] / std^4 - 3 (excess)
                    m4 = (s4 / cnt - 4.0 * mean * s3 / cnt
                           + 6.0 * mean * mean * s2 / cnt
                           - 3.0 * mean * mean * mean * mean)
                    result[i, col_kurt] = m4 / (var * var) - 3.0
                else:
                    result[i, col_skew] = 0.0
                    result[i, col_kurt] = 0.0
    return result


# =============================================================================
# New features: Close-to-open gaps (Task 3.4)
# =============================================================================

@njit(cache=True)
def _compute_close_to_open_gaps(open_, close):
    """
    Close-to-open gap features:
    - Raw gap: log(open[i] / close[i-1])                    (1)
    - Rolling mean of gap over windows [5, 10, 20]           (3)
    - Rolling std of gap over windows [5, 10, 20]            (3)
    Total: 7 features
    """
    n = close.shape[0]
    windows = np.array([5, 10, 20])
    result = np.empty((n, 7), dtype=np.float64)
    result[:] = np.nan

    # Raw gap
    gap = np.empty(n, dtype=np.float64)
    gap[0] = np.nan
    for i in range(1, n):
        o = open_[i]
        cp = close[i - 1]
        if np.isnan(o) or np.isnan(cp) or cp <= 0.0 or o <= 0.0:
            gap[i] = np.nan
        else:
            gap[i] = np.log(o / cp)
    result[:, 0] = gap

    # Rolling mean and std
    for wi in range(3):
        w = windows[wi]
        col_mean = 1 + wi
        col_std = 4 + wi
        for i in range(w - 1, n):
            s1 = 0.0
            s2 = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = gap[j]
                if not np.isnan(v):
                    s1 += v
                    s2 += v * v
                    cnt += 1
            if cnt >= 1:
                mean = s1 / cnt
                result[i, col_mean] = mean
                if cnt >= 2:
                    var = s2 / cnt - mean * mean
                    if var < 0.0:
                        var = 0.0
                    result[i, col_std] = np.sqrt(var)
    return result


# =============================================================================
# New features: Volume-weighted returns (Task 3.5)
# =============================================================================

@njit(cache=True)
def _compute_volume_weighted_returns(close, volume):
    """
    Volume-weighted return features:
    - Per-bar VWR: log_ret * (volume / vol_ma_w) for w in [5, 10, 20, 60]  (4)
    - Rolling sum of VWR over same windows                                   (4)
    Total: 8 features
    """
    n = close.shape[0]
    windows = np.array([5, 10, 20, 60])
    nw = 4
    result = np.empty((n, 8), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(close[i] / close[i - 1])

    for wi in range(nw):
        w = windows[wi]
        col_vwr = wi
        col_sum = nw + wi

        # Compute per-bar VWR and rolling sum
        vwr = np.empty(n, dtype=np.float64)
        vwr[:] = np.nan

        for i in range(w - 1, n):
            # Volume MA over window
            vol_sum = 0.0
            vol_cnt = 0
            for j in range(i - w + 1, i + 1):
                v = volume[j]
                if not np.isnan(v):
                    vol_sum += v
                    vol_cnt += 1
            if vol_cnt > 0 and vol_sum > 0.0:
                vol_ma = vol_sum / vol_cnt
                r = log_ret[i]
                v_now = volume[i]
                if not np.isnan(r) and not np.isnan(v_now) and vol_ma > 0.0:
                    vwr[i] = r * (v_now / vol_ma)
                    result[i, col_vwr] = vwr[i]

        # Rolling sum of VWR
        for i in range(w - 1, n):
            total = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                v = vwr[j]
                if not np.isnan(v):
                    total += v
                    cnt += 1
            if cnt > 0:
                result[i, col_sum] = total
    return result


# =============================================================================
# New features: Return autocorrelation (Task 3.6)
# =============================================================================

@njit(cache=True)
def _compute_return_autocorrelation(close):
    """
    Return autocorrelation features:
    - Lag-1 autocorrelation over windows [20, 60]   (2)
    - Lag-5 autocorrelation over windows [20, 60]   (2)
    Total: 4 features
    """
    n = close.shape[0]
    windows = np.array([20, 60])
    lags = np.array([1, 5])
    result = np.empty((n, 4), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(close[i] / close[i - 1])

    for li in range(2):
        lag = lags[li]
        for wi in range(2):
            w = windows[wi]
            col = li * 2 + wi
            for i in range(w + lag - 1, n):
                sx = 0.0
                sy = 0.0
                sxy = 0.0
                sx2 = 0.0
                sy2 = 0.0
                cnt = 0
                for j in range(i - w + 1 + lag, i + 1):
                    x = log_ret[j]
                    y = log_ret[j - lag]
                    if not np.isnan(x) and not np.isnan(y):
                        sx += x
                        sy += y
                        sxy += x * y
                        sx2 += x * x
                        sy2 += y * y
                        cnt += 1
                if cnt >= lag + 3:
                    mx = sx / cnt
                    my = sy / cnt
                    cov = sxy / cnt - mx * my
                    vx = sx2 / cnt - mx * mx
                    vy = sy2 / cnt - my * my
                    if vx > 1e-20 and vy > 1e-20:
                        result[i, col] = cov / np.sqrt(vx * vy)
                    else:
                        result[i, col] = 0.0
    return result


# =============================================================================
# New features: Realized variance (Task 3.7)
# =============================================================================

@njit(cache=True)
def _compute_realized_variance(close):
    """
    Realized variance proxy features:
    - RV: sum of squared 1-bar log returns over windows [5, 10, 20, 60]   (4)
    - log-RV: log(RV) for each window                                      (4)
    Total: 8 features
    """
    n = close.shape[0]
    windows = np.array([5, 10, 20, 60])
    nw = 4
    result = np.empty((n, 8), dtype=np.float64)
    result[:] = np.nan

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i - 1]) or close[i - 1] <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(close[i] / close[i - 1])

    for wi in range(nw):
        w = windows[wi]
        col_rv = wi
        col_logrv = nw + wi
        for i in range(w - 1, n):
            total = 0.0
            cnt = 0
            for j in range(i - w + 1, i + 1):
                r = log_ret[j]
                if not np.isnan(r):
                    total += r * r
                    cnt += 1
            if cnt > 0:
                rv = total
                result[i, col_rv] = rv
                if rv > 0.0:
                    result[i, col_logrv] = np.log(rv)
                else:
                    result[i, col_logrv] = np.nan
    return result


# =============================================================================
# Long-period features (v4 Task 1.1)
# =============================================================================

@njit(cache=True)
def _compute_long_period_features(open_, high, low, close, volume):
    """
    Long-period features for windows [240, 480]. Returns (n, 18) float64 array.

    Columns:
      0: log(close[i]/close[i-240])          - long momentum log return w=240
      1: close[i]/close[i-240] - 1           - long momentum rate of change w=240
      2: log(close[i]/close[i-480])          - long momentum log return w=480
      3: close[i]/close[i-480] - 1           - long momentum rate of change w=480
      4: rolling std of 1-bar log returns, w=240
      5: rolling std of 1-bar log returns, w=480
      6: Parkinson volatility, w=240
      7: Parkinson volatility, w=480
      8: ATR, w=240
      9: ATR, w=480
     10: close[i]/EMA(close, 240)[i] - 1
     11: close[i]/EMA(close, 480)[i] - 1
     12: RV(240) = sum of squared 1-bar log returns over window 240
     13: RV(480)
     14: log(RV(240))
     15: log(RV(480))
     16: (close[i] - min(close,240)[i]) / (max(close,240)[i] - min(close,240)[i])
     17: (close[i] - min(close,480)[i]) / (max(close,480)[i] - min(close,480)[i])
    """
    n = close.shape[0]
    result = np.empty((n, 18), dtype=np.float64)
    result[:] = np.nan

    LN2 = 0.6931471805599453

    # Pre-compute 1-bar log returns
    log_ret = np.empty(n, dtype=np.float64)
    log_ret[0] = np.nan
    for i in range(1, n):
        c_now = close[i]
        c_prev = close[i - 1]
        if np.isnan(c_now) or np.isnan(c_prev) or c_prev <= 0.0:
            log_ret[i] = np.nan
        else:
            log_ret[i] = np.log(c_now / c_prev)

    # Pre-compute true range
    true_range = np.empty(n, dtype=np.float64)
    true_range[0] = np.nan
    for i in range(1, n):
        h = high[i]
        l = low[i]
        cp = close[i - 1]
        if np.isnan(h) or np.isnan(l) or np.isnan(cp):
            true_range[i] = np.nan
        else:
            tr1 = h - l
            tr2 = abs(h - cp)
            tr3 = abs(l - cp)
            true_range[i] = max(tr1, max(tr2, tr3))

    # Pre-compute EMA(close, 240) and EMA(close, 480)
    ema240 = _ema(close, 240)
    ema480 = _ema(close, 480)

    # Columns 0-3: Long-period momentum
    for i in range(n):
        # w=240
        if i >= 240:
            c_now = close[i]
            c_prev = close[i - 240]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0.0:
                result[i, 0] = np.log(c_now / c_prev)
                result[i, 1] = c_now / c_prev - 1.0
            # else stays NaN
        # w=480
        if i >= 480:
            c_now = close[i]
            c_prev = close[i - 480]
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 0.0:
                result[i, 2] = np.log(c_now / c_prev)
                result[i, 3] = c_now / c_prev - 1.0

    # Columns 4-9: Long-period volatility (rolling std, Parkinson, ATR)
    # w=240
    for i in range(240 - 1, n):
        # Rolling std of log returns (col 4)
        total = 0.0
        total_sq = 0.0
        cnt = 0
        for j in range(i - 240 + 1, i + 1):
            v = log_ret[j]
            if not np.isnan(v):
                total += v
                total_sq += v * v
                cnt += 1
        if cnt >= 2:
            mean = total / cnt
            var = total_sq / cnt - mean * mean
            if var < 0.0:
                var = 0.0
            result[i, 4] = np.sqrt(var)

        # Parkinson volatility (col 6)
        park_sum = 0.0
        park_cnt = 0
        for j in range(i - 240 + 1, i + 1):
            h = high[j]
            l = low[j]
            if not np.isnan(h) and not np.isnan(l) and l > 0.0:
                hl = np.log(h / l)
                park_sum += hl * hl
                park_cnt += 1
        if park_cnt >= 1:
            result[i, 6] = np.sqrt(park_sum / (4.0 * park_cnt * LN2))

        # ATR (col 8)
        atr_sum = 0.0
        atr_cnt = 0
        for j in range(i - 240 + 1, i + 1):
            tr = true_range[j]
            if not np.isnan(tr):
                atr_sum += tr
                atr_cnt += 1
        if atr_cnt >= 1:
            result[i, 8] = atr_sum / atr_cnt

    # w=480
    for i in range(480 - 1, n):
        # Rolling std of log returns (col 5)
        total = 0.0
        total_sq = 0.0
        cnt = 0
        for j in range(i - 480 + 1, i + 1):
            v = log_ret[j]
            if not np.isnan(v):
                total += v
                total_sq += v * v
                cnt += 1
        if cnt >= 2:
            mean = total / cnt
            var = total_sq / cnt - mean * mean
            if var < 0.0:
                var = 0.0
            result[i, 5] = np.sqrt(var)

        # Parkinson volatility (col 7)
        park_sum = 0.0
        park_cnt = 0
        for j in range(i - 480 + 1, i + 1):
            h = high[j]
            l = low[j]
            if not np.isnan(h) and not np.isnan(l) and l > 0.0:
                hl = np.log(h / l)
                park_sum += hl * hl
                park_cnt += 1
        if park_cnt >= 1:
            result[i, 7] = np.sqrt(park_sum / (4.0 * park_cnt * LN2))

        # ATR (col 9)
        atr_sum = 0.0
        atr_cnt = 0
        for j in range(i - 480 + 1, i + 1):
            tr = true_range[j]
            if not np.isnan(tr):
                atr_sum += tr
                atr_cnt += 1
        if atr_cnt >= 1:
            result[i, 9] = atr_sum / atr_cnt

    # Columns 10-11: EMA deviation
    for i in range(n):
        c = close[i]
        e240 = ema240[i]
        if not np.isnan(c) and not np.isnan(e240) and e240 > 0.0:
            result[i, 10] = c / e240 - 1.0
        e480 = ema480[i]
        if not np.isnan(c) and not np.isnan(e480) and e480 > 0.0:
            result[i, 11] = c / e480 - 1.0

    # Columns 12-15: Realized variance (RV and log-RV)
    # w=240
    for i in range(240 - 1, n):
        total = 0.0
        cnt = 0
        for j in range(i - 240 + 1, i + 1):
            r = log_ret[j]
            if not np.isnan(r):
                total += r * r
                cnt += 1
        if cnt > 0:
            rv = total
            result[i, 12] = rv
            if rv > 0.0:
                result[i, 14] = np.log(rv)

    # w=480
    for i in range(480 - 1, n):
        total = 0.0
        cnt = 0
        for j in range(i - 480 + 1, i + 1):
            r = log_ret[j]
            if not np.isnan(r):
                total += r * r
                cnt += 1
        if cnt > 0:
            rv = total
            result[i, 13] = rv
            if rv > 0.0:
                result[i, 15] = np.log(rv)

    # Columns 16-17: Price range position
    # w=240
    for i in range(240 - 1, n):
        c = close[i]
        if np.isnan(c):
            continue
        mn = np.inf
        mx = -np.inf
        for j in range(i - 240 + 1, i + 1):
            v = close[j]
            if not np.isnan(v):
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        rng = mx - mn
        if rng > 0.0:
            result[i, 16] = (c - mn) / rng
        else:
            result[i, 16] = 0.5

    # w=480
    for i in range(480 - 1, n):
        c = close[i]
        if np.isnan(c):
            continue
        mn = np.inf
        mx = -np.inf
        for j in range(i - 480 + 1, i + 1):
            v = close[j]
            if not np.isnan(v):
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        rng = mx - mn
        if rng > 0.0:
            result[i, 17] = (c - mn) / rng
        else:
            result[i, 17] = 0.5

    return result


# =============================================================================
# Main entry point (Task 1.1)
# =============================================================================

def generate_factors(dataset_name: str, data: np.ndarray) -> np.ndarray:
    """
    Generate features from OHLCV data.

    Args:
        dataset_name: e.g. "dataset0" through "dataset29"
        data: np.ndarray of shape (T, 5), dtype float32
              columns: [open, high, low, close, volume]

    Returns:
        np.ndarray of shape (T, 165), dtype float32
        # v4: 147 + 18 = 165 dims
    """
    _set_seeds(42)

    T = data.shape[0]

    # Unpack OHLCV columns as float64 for computation precision
    open_ = data[:, 0].astype(np.float64)
    high = data[:, 1].astype(np.float64)
    low = data[:, 2].astype(np.float64)
    close = data[:, 3].astype(np.float64)
    volume = data[:, 4].astype(np.float64)

    # Compute baseline feature groups (109 features, unchanged)
    momentum = _compute_momentum_features(open_, high, low, close, volume)       # 14
    volatility = _compute_volatility_features(open_, high, low, close, volume)   # 20
    vol_feats = _compute_volume_features(open_, high, low, close, volume)        # 14
    micro = _compute_microstructure_features(open_, high, low, close, volume)    # 14
    tech = _compute_technical_features(open_, high, low, close, volume)          # 20
    regime = _compute_regime_features(open_, high, low, close, volume)           # 12
    cross = _compute_cross_features(open_, high, low, close, volume)             # 15

    # New feature groups (38 features, appended after baseline)
    ema_ratios = _compute_ema_ratios(close)                                      # 5
    skew_kurt = _compute_rolling_skew_kurt(close)                                # 6
    gaps = _compute_close_to_open_gaps(open_, close)                             # 7
    vwr = _compute_volume_weighted_returns(close, volume)                        # 8
    autocorr = _compute_return_autocorrelation(close)                            # 4
    rv = _compute_realized_variance(close)                                       # 8

    # Long-period features (v4: 18 features, cols 147-164)
    long_period = _compute_long_period_features(open_, high, low, close, volume) # 18

    # Stack all features: 109 baseline + 38 new + 18 long-period = 165 total
    # v4: 147 + 18 = 165 dims
    features = np.hstack([
        np.column_stack([
            momentum,
            volatility,
            vol_feats,
            micro,
            tech,
            regime,
            cross,
            ema_ratios,
            skew_kurt,
            gaps,
            vwr,
            autocorr,
            rv,
        ]),
        long_period,
    ])

    assert features.shape[0] == T
    assert features.shape[1] == 165
    assert features.shape[1] <= 512

    return features.astype(np.float32)
