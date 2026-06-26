"""
Tests for src/core/analysis/indicators.py — China-compatible technical indicators.

Uses MyTT-style algorithms that match 通达信/TDX conventions:
  - KDJ includes J-line (TA-Lib only returns K,D)
  - MACD uses SMA variant (not EMA)
  - RSI capped at 0-100
  - BOLL uses 20-period MA ± 2×std

All functions accept numpy arrays or polars Series and return numpy arrays.
"""

import numpy as np
import polars as pl
import pytest

from src.core.analysis.indicators import (
    bollinger_bands,
    kdj,
    macd_tdx,
    rsi,
    sma_tdx,
)


# ─── Test Data ──────────────────────────────────────────────────────────────

@pytest.fixture
def prices() -> np.ndarray:
    """Simple price series: 10 data points."""
    return np.array([10.0, 10.2, 10.5, 10.3, 10.8, 11.0, 10.9, 11.2, 11.5, 11.3], dtype=np.float64)


@pytest.fixture
def prices_flat() -> np.ndarray:
    """Flat price: all same value."""
    return np.ones(20, dtype=np.float64) * 10.0


# ─── SMA (TDX algorithm) ────────────────────────────────────────────────────

class TestSMATDX:
    """通达信SMA — the core smoothing algorithm for KDJ and MACD."""

    def test_sma_basic(self) -> None:
        """SMA with weight N=5, M=1 should match expected recursive values."""
        series = np.array([10.0, 12.0, 11.0, 13.0, 14.0], dtype=np.float64)
        result = sma_tdx(series, n_weight=5, m_weight=1)
        assert len(result) == 5
        # First element = input
        assert result[0] == pytest.approx(10.0)
        # SMA(2) = (11.0 + 4*10.0) / 5 = 10.2
        # Actually: sma_tdx formula = (M*close + (N-M)*prev_sma) / N
        # result[1] = (1*12.0 + 4*10.0)/5 = 10.4
        assert result[1] == pytest.approx(10.4)

    def test_sma_returns_float_array(self, prices: np.ndarray) -> None:
        result = sma_tdx(prices, n_weight=9, m_weight=3)
        assert result.dtype == np.float64
        assert len(result) == len(prices)


# ─── KDJ ────────────────────────────────────────────────────────────────────

class TestKDJ:
    """KDJ must return K, D, J values (TA-Lib only returns K, D)."""

    def test_kdj_returns_three_arrays(self, prices: np.ndarray) -> None:
        k, d, j = kdj(prices, n=9, m1=3, m2=3)
        for arr in (k, d, j):
            assert isinstance(arr, np.ndarray)
            assert len(arr) == len(prices)

    def test_kdj_j_is_3k_minus_2d(self) -> None:
        """J = 3K - 2D relationship must hold exactly for valid (non-NaN) points."""
        prices_long = np.sin(np.linspace(0, 4 * np.pi, 50)) * 5 + 20
        k, d, j = kdj(prices_long, n=9, m1=3, m2=3)
        valid = ~np.isnan(k)
        for i in np.where(valid)[0]:
            assert j[i] == pytest.approx(3 * k[i] - 2 * d[i], abs=1e-6), f"Mismatch at i={i}"

    def test_kdj_values_in_reasonable_range(self) -> None:
        """KDJ values should be approximately in [-20, 120] range for valid points."""
        prices_long = np.sin(np.linspace(0, 6 * np.pi, 80)) * 5 + 20
        k, d, _j = kdj(prices_long, n=14, m1=3, m2=3)
        valid = ~np.isnan(k)
        k_valid = k[valid]
        d_valid = d[valid]
        for label, arr in [("K", k_valid), ("D", d_valid)]:
            assert arr.min() >= -20, f"{label} min too low: {arr.min()}"
            assert arr.max() <= 120, f"{label} max too high: {arr.max()}"


# ─── MACD (TDX-compatible) ──────────────────────────────────────────────────

class TestMACDTDX:
    """MACD using TDX SMA algorithm."""

    def test_macd_returns_diff_dea_macd(self, prices: np.ndarray) -> None:
        diff, dea, macd = macd_tdx(prices, short=12, long=26, mid=9)
        for arr in (diff, dea, macd):
            assert len(arr) == len(prices)
            assert arr.dtype == np.float64

    def test_macd_diff_positive_for_uptrend(self) -> None:
        """In a strong uptrend, DIF should go positive."""
        uptrend = np.linspace(10, 20, 100)
        diff, _dea, _macd = macd_tdx(uptrend)
        assert diff[-1] > 0


# ─── RSI ────────────────────────────────────────────────────────────────────

class TestRSI:
    """RSI must be capped at 0-100 (unlike TA-Lib which can exceed 100)."""

    def test_rsi_range_zero_to_hundred(self, prices: np.ndarray) -> None:
        result = rsi(prices, period=6)
        valid = result[~np.isnan(result)]
        for val in valid:
            assert 0 <= val <= 100, f"RSI out of range: {val}"

    def test_rsi_flat_prices(self, prices_flat: np.ndarray) -> None:
        """Flat prices → all gains, no losses → RSI should be close to 100 after init."""
        result = rsi(prices_flat, period=6)
        valid = result[~np.isnan(result)]
        for val in valid:
            assert val >= 90  # near 100 for all-up


# ─── Bollinger Bands ────────────────────────────────────────────────────────

class TestBollingerBands:
    """Bollinger Bands: middle=MA20, upper/lower=MA±2σ."""

    def test_returns_three_arrays(self, prices: np.ndarray) -> None:
        upper, middle, lower = bollinger_bands(prices, period=5, std_mult=2.0)
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_upper_above_lower(self, prices: np.ndarray) -> None:
        upper, middle, lower = bollinger_bands(prices, period=5, std_mult=2.0)
        valid_start = 5  # first 5 points may be NaN
        for i in range(valid_start, len(prices)):
            assert upper[i] >= middle[i] >= lower[i], f"Band violation at i={i}"

    def test_flat_prices_zero_width(self, prices_flat: np.ndarray) -> None:
        """Flat prices → std=0 → upper = middle = lower."""
        upper, middle, lower = bollinger_bands(prices_flat, period=5, std_mult=2.0)
        valid = ~np.isnan(middle)
        assert np.allclose(upper[valid], middle[valid])
        assert np.allclose(lower[valid], middle[valid])
