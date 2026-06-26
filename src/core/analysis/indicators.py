"""
China-compatible technical indicators — 通达信 (TDX) / 同花顺 algorithm equivalents.

Standard libraries like TA-Lib are INCOMPATIBLE with Chinese broker conventions:
  - TA-Lib STOCH() returns only K, D — missing J (= 3K - 2D).
  - TA-Lib RSI can exceed 100 (Chinese brokers cap at 0-100).
  - TA-Lib MACD uses EMA; Chinese brokers use a TDX-specific SMA variant.

This module provides NumPy implementations that match 通达信 output exactly.

All functions accept 1-D numpy arrays. NaN-padding is applied at the beginning
for indicators that need warm-up periods.
"""

from __future__ import annotations

import numpy as np


# ─── TDX SMA (the core smoothing algorithm) ──────────────────────────────────


def sma_tdx(
    series: np.ndarray, n_weight: int, m_weight: int
) -> np.ndarray:
    """
    通达信 SMA 算法 — the foundational smoothing function for KDJ and MACD.

    Formula: SMA_i = (M × close_i + (N - M) × SMA_{i-1}) / N

    This is NOT the same as a simple moving average. It's a recursive
    weighted smoothing where the weight of the new close is M/N and the
    weight of the previous smoothed value is (N-M)/N.

    Args:
        series: Price series (1-D).
        n_weight: Denominator weight N.
        m_weight: Numerator weight M (must be ≤ N).

    Returns:
        Smoothed series, same length as input.
    """
    if m_weight > n_weight:
        raise ValueError(f"m_weight ({m_weight}) must be ≤ n_weight ({n_weight})")

    result = np.empty_like(series, dtype=np.float64)
    result[0] = float(series[0])

    for i in range(1, len(series)):
        result[i] = (
            m_weight * float(series[i]) + (n_weight - m_weight) * result[i - 1]
        ) / n_weight

    return result


# ─── KDJ ────────────────────────────────────────────────────────────────────


def kdj(
    high_or_close: np.ndarray,
    low: np.ndarray | None = None,
    close: np.ndarray | None = None,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KDJ indicator — 通达信 compatible.

    Standard call: kdj(high, low, close, n=9, m1=3, m2=3)
    Convenience call: kdj(close_array, n=9) — uses close as high/low/close.

    Algorithm:
      1. RSV = (close - low_n) / (high_n - low_n) × 100
      2. K = SMA(RSV, m1, 1)
      3. D = SMA(K, m2, 1)
      4. J = 3K - 2D

    Returns:
        (K, D, J) — each a numpy array of same length as input.
        Leading n-1 values are NaN (warm-up period).
    """
    # Handle overloaded signatures
    if low is None and close is None:
        arr = high_or_close
        high_arr = arr
        low_arr = arr
        close_arr = arr
    elif low is not None and close is not None:
        high_arr = high_or_close
        low_arr = low
        close_arr = close
    else:
        raise ValueError("Provide either (close_array,) or (high, low, close)")

    length = len(close_arr)
    rsv = np.full(length, np.nan, dtype=np.float64)

    for i in range(n - 1, length):
        window = close_arr[i - n + 1 : i + 1]
        high_n = np.max(high_arr[i - n + 1 : i + 1])
        low_n = np.min(low_arr[i - n + 1 : i + 1])
        if high_n != low_n:
            rsv[i] = (float(close_arr[i]) - low_n) / (high_n - low_n) * 100.0
        else:
            rsv[i] = 50.0  # all same → neutral

    k_vals = _sma_nan(rsv, n_weight=m1, m_weight=1)
    d_vals = _sma_nan(k_vals, n_weight=m2, m_weight=1)
    j_vals = 3.0 * k_vals - 2.0 * d_vals

    return k_vals, d_vals, j_vals


# ─── MACD (TDX algorithm) ───────────────────────────────────────────────────


def macd_tdx(
    close: np.ndarray,
    short: int = 12,
    long: int = 26,
    mid: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD using 通达信 EMA → SMA pipeline.

    Unlike standard MACD (EMA-based), the 通达信 version:
      1. Computes short and long EMAs.
      2. DIF = short_ema - long_ema
      3. DEA = SMA(DIF, mid, 1)  ← TDX-specific
      4. MACD histogram = 2 × (DIF - DEA)

    Returns:
        (DIF, DEA, MACD_histogram) — each a numpy array.
    """
    ema_short = _ema(close, short)
    ema_long = _ema(close, long)
    diff = ema_short - ema_long

    # 通达信 uses SMA for DEA, not EMA
    dea = _sma_nan(diff, n_weight=mid, m_weight=1)

    macd_hist = 2.0 * (diff - dea)
    return diff, dea, macd_hist


# ─── RSI ─────────────────────────────────────────────────────────────────────


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    RSI capped at [0, 100] — 通达信 compatible.

    Unlike TA-Lib which can return RSI > 100, this implementation clamps
    the output to [0, 100] matching Chinese broker behavior.

    Algorithm: Wilder's smoothed average of gains/losses.
    """
    length = len(close)
    if length < period + 1:
        return np.full(length, np.nan, dtype=np.float64)

    result = np.full(length, np.nan, dtype=np.float64)
    diffs = np.diff(close)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)

    # Initial average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder's smoothing
    for i in range(period + 1, length):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)

    # Clamp to [0, 100] — 通达信 convention
    return np.clip(result, 0.0, 100.0)


# ─── Bollinger Bands ────────────────────────────────────────────────────────


def bollinger_bands(
    close: np.ndarray, period: int = 20, std_mult: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands — 20-period MA ± std_mult × standard deviation.

    Returns:
        (upper, middle, lower) — each a numpy array.
        Leading period-1 values are NaN.
    """
    length = len(close)
    upper = np.full(length, np.nan, dtype=np.float64)
    middle = np.full(length, np.nan, dtype=np.float64)
    lower = np.full(length, np.nan, dtype=np.float64)

    for i in range(period - 1, length):
        window = close[i - period + 1 : i + 1]
        ma = np.mean(window)
        std = np.std(window, ddof=0)
        middle[i] = ma
        upper[i] = ma + std_mult * std
        lower[i] = ma - std_mult * std

    return upper, middle, lower


# ─── Internal Helpers ───────────────────────────────────────────────────────


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.full_like(series, np.nan, dtype=np.float64)
    if len(series) < period:
        return result
    result[period - 1] = np.mean(series[:period])
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(series)):
        result[i] = (series[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def _sma_nan(series: np.ndarray, n_weight: int, m_weight: int) -> np.ndarray:
    """SMA that preserves NaN at the input positions — used for KDJ and DEA."""
    result = np.full_like(series, np.nan, dtype=np.float64)
    # Find first non-NaN index
    valid = np.where(~np.isnan(series))[0]
    if len(valid) == 0:
        return result
    first_valid = valid[0]
    result[first_valid] = float(series[first_valid])
    for i in range(first_valid + 1, len(series)):
        if np.isnan(series[i]):
            result[i] = np.nan
        elif np.isnan(result[i - 1]):
            result[i] = float(series[i])
        else:
            result[i] = (
                m_weight * float(series[i]) + (n_weight - m_weight) * result[i - 1]
            ) / n_weight
    return result
