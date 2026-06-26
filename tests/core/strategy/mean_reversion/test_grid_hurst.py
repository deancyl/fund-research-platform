"""
TDD tests for GridHurstStrategy (S10) — 网格交易 + Hurst 一票否决.

Spec:
  - Target: CSI 300 ETF (510300)
  - Grid params: 5% spacing, 5 layers, equal position per layer
  - Hurst >= 0.6 → VETO (one-vote rejection, block all grid signals)
  - Hurst < 0.4 → full grid BUY signals (strong mean-reversion)
  - Eligible regimes: SIDEWAYS only
  - Hurst estimator: R/S analysis on close prices
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl

from src.core.strategy.base import MarketRegime, SignalDirection
from src.core.strategy.mean_reversion.grid_hurst import GridHurstStrategy

# ─── Test Fixtures ───────────────────────────────────────────────────────────


def _make_df(close_arr: np.ndarray) -> pl.DataFrame:
    """Create a polars DataFrame with a close column from a numpy array."""
    return pl.DataFrame({"close": close_arr.tolist()})


def _make_market_data(
    close_arr: np.ndarray, fund_code: str = "510300",
) -> dict[str, pl.DataFrame]:
    """Create market_data dict from a numpy close-price array."""
    return {fund_code: _make_df(close_arr)}


def _price_trending(n: int = 256) -> np.ndarray:
    """Strong uptrend — prices rise monotonically (Hurst >> 0.6)."""
    # Clean linear trend from 3.0 to 6.0 over n points → Hurst ≈ 1.0
    return np.linspace(3.0, 6.0, n, dtype=np.float64)


def _price_mean_reverting(n: int = 256) -> np.ndarray:
    """Strong mean-reverting process (Ornstein-Uhlenbeck) — Hurst < 0.5."""
    rng = np.random.default_rng(99)
    theta = 0.9
    mu = 3.0
    sigma = 0.003
    ou = np.zeros(n, dtype=np.float64)
    ou[0] = mu
    for t in range(1, n):
        ou[t] = ou[t - 1] + theta * (mu - ou[t - 1]) + sigma * rng.normal()
    return ou


def _price_constant(n: int = 100) -> np.ndarray:
    """Constant price series — edge case for Hurst estimator."""
    return np.full(n, 5.0, dtype=np.float64)


# ─── Strategy Configuration ──────────────────────────────────────────────────


class TestStrategyConfig:
    """GridHurstStrategy default parameters, eligibility, and data requirements."""

    def test_default_params(self) -> None:
        """S10 defaults: 5% spacing, 5 layers, veto at 0.6."""
        strat = GridHurstStrategy()
        assert strat.grid_spacing_pct == 0.05
        assert strat.grid_layers == 5
        assert strat.hurst_veto_threshold == 0.6
        assert strat.name == "grid_hurst"

    def test_eligible_regimes(self) -> None:
        """Strategy operates in SIDEWAYS regime only."""
        strat = GridHurstStrategy()
        assert strat.is_eligible(MarketRegime.SIDEWAYS)
        for regime in MarketRegime:
            if regime != MarketRegime.SIDEWAYS:
                assert not strat.is_eligible(regime), f"Should not be eligible in {regime}"

    def test_required_data(self) -> None:
        """Strategy needs close prices for Hurst and grid computation."""
        strat = GridHurstStrategy()
        required = strat.required_data()
        assert "close" in required

    def test_validate_default_is_clean(self) -> None:
        """Default parameters must pass validation."""
        strat = GridHurstStrategy()
        errors = strat.validate()
        assert errors == []

    def test_validate_rejects_zero_spacing(self) -> None:
        """grid_spacing_pct must be > 0."""
        strat = GridHurstStrategy(grid_spacing_pct=0.0)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_validate_rejects_zero_layers(self) -> None:
        """grid_layers must be >= 1."""
        strat = GridHurstStrategy(grid_layers=0)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_custom_params(self) -> None:
        """Strategy accepts custom frozen parameters."""
        strat = GridHurstStrategy(
            grid_spacing_pct=0.08,
            grid_layers=3,
            hurst_veto_threshold=0.55,
        )
        assert strat.grid_spacing_pct == 0.08
        assert strat.grid_layers == 3
        assert strat.hurst_veto_threshold == 0.55


# ─── Hurst Veto — Trending Market ────────────────────────────────────────────


class TestHurstVeto:
    """Hurst >= veto threshold must produce HOLD — no BUY/SELL signals."""

    def test_trending_price_vetos_buy(self) -> None:
        """Strong trend → Hurst >= 0.6 → BUY signals blocked."""
        close = _price_trending()
        market_data = _make_market_data(close)
        strat = GridHurstStrategy()
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        # Verify no BUY or ACCUMULATE signals
        buy_dirs = {SignalDirection.BUY, SignalDirection.ACCUMULATE}
        for s in signals:
            assert s["direction"] not in buy_dirs, (
                f"VETO should block BUY, got {s['direction']}"
            )

    def test_veto_signal_contains_hurst_reason(self) -> None:
        """Veto signal's reason field must mention Hurst."""
        close = _price_trending()
        market_data = _make_market_data(close)
        strat = GridHurstStrategy()
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        assert len(signals) >= 1
        for s in signals:
            assert "hurst" in s["reason"].lower(), f"Reason missing Hurst: {s['reason']}"


# ─── Full Grid — Mean-Reverting Market ───────────────────────────────────────


class TestFullGrid:
    """Hurst < 0.4 → full grid BUY signals generated."""

    def test_mean_reverting_generates_buy(self) -> None:
        """Strong mean reversion → Hurst < 0.4 → BUY signals at grid layers."""
        close = _price_mean_reverting()
        market_data = _make_market_data(close)
        strat = GridHurstStrategy(grid_layers=3)
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) > 0, "Expected BUY signals for mean-reverting market"

    def test_buy_signals_have_correct_shape(self) -> None:
        """Each BUY signal must have all required keys with valid values."""
        close = _price_mean_reverting()
        market_data = _make_market_data(close)
        strat = GridHurstStrategy(grid_layers=3)
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        for s in buy_signals:
            assert s["fund_code"] == "510300"
            assert 0.0 < s["confidence"] <= 1.0
            assert s["target_weight"] > 0.0
            assert "grid" in s["reason"].lower()

    def test_buy_layer_count_matches_layers(self) -> None:
        """Number of BUY signals equals grid_layers when Hurst < 0.4."""
        close = _price_mean_reverting()
        market_data = _make_market_data(close)
        strat = GridHurstStrategy(grid_layers=3)
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 3, f"Expected 3 BUY signals, got {len(buy_signals)}"


# ─── Hurst Estimator ─────────────────────────────────────────────────────────


class TestHurstEstimator:
    """R/S Hurst exponent estimator correctness."""

    def test_monotonic_ordering(self) -> None:
        """Trending series H > random walk H > mean-reverting H."""
        rng = np.random.default_rng(7)
        n = 256

        # Perfect trend → H ≈ 1.0
        trend = np.arange(n, dtype=np.float64)
        # Random walk → H ≈ 0.5
        rw = np.cumsum(rng.normal(0.0, 1.0, n).astype(np.float64))
        # Mean-reverting → H < 0.5
        theta = 0.8
        mu = 0.0
        sigma = 0.05
        mr = np.zeros(n, dtype=np.float64)
        for t in range(1, n):
            mr[t] = mr[t - 1] + theta * (mu - mr[t - 1]) + sigma * rng.normal()

        h_trend = GridHurstStrategy._hurst_rs(trend)
        h_rw = GridHurstStrategy._hurst_rs(rw)
        h_mr = GridHurstStrategy._hurst_rs(mr)

        assert h_mr < h_rw < h_trend, (
            f"Expected h_mr < h_rw < h_trend, got {h_mr=:.3f} < {h_rw=:.3f} < {h_trend=:.3f}"
        )

    def test_constant_series(self) -> None:
        """Constant series should not crash or return NaN."""
        close = _price_constant()
        h = GridHurstStrategy._hurst_rs(close)
        assert not math.isnan(h)
        assert isinstance(h, float)

    def test_result_in_range(self) -> None:
        """Hurst exponent must be clamped to [0, 1]."""
        rng = np.random.default_rng(123)
        prices = np.cumsum(rng.normal(0.0, 0.01, 200).astype(np.float64)) + 3.0
        h = GridHurstStrategy._hurst_rs(prices)
        assert 0.0 <= h <= 1.0, f"Hurst {h} out of [0, 1]"


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Empty data, insufficient data, and multiple funds."""

    def test_empty_market_data(self) -> None:
        """Empty market_data → empty signal list."""
        strat = GridHurstStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 1), market_data={})
        assert signals == []

    def test_insufficient_data(self) -> None:
        """Too few data points → no crash, no BUY signals."""
        close = np.array([3.0, 3.01, 3.02], dtype=np.float64)
        market_data = _make_market_data(close)
        strat = GridHurstStrategy()
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        buy_dirs = {SignalDirection.BUY, SignalDirection.ACCUMULATE}
        for s in signals:
            assert s["direction"] not in buy_dirs

    def test_multiple_funds(self) -> None:
        """Each fund in market_data receives its own signals."""
        close_trend = _price_trending()
        close_mr = _price_mean_reverting()
        market_data = {
            "510300": _make_df(close_trend),
            "159915": _make_df(close_mr),
        }
        strat = GridHurstStrategy()
        signals = strat.generate_signals(dt=date(2026, 12, 31), market_data=market_data)
        fund_codes = {s["fund_code"] for s in signals}
        assert "510300" in fund_codes
        assert "159915" in fund_codes
