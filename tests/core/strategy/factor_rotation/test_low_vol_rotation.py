"""
TDD tests for LowVolRotation (S13) — ETF Low-Volatility Rotation Strategy.

Spec:
  - Select bottom top_n ETFs by trailing volatility, equal weight.
  - Default: 60-day volatility window, top_n=3.
  - Annual: 12.77%, MaxDD: -8.81%, Sharpe: 1.06.
  - Eligible in ALL market regimes.
  - required_data: ["close"].
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection
from src.core.strategy.factor_rotation.low_vol_rotation import LowVolRotation


# ─── Test Fixtures ───────────────────────────────────────────────────────────


def _make_df(close_arr: np.ndarray) -> pl.DataFrame:
    """Create a polars DataFrame with a close column from a numpy array."""
    return pl.DataFrame({"close": close_arr.tolist()})


def _price_low_vol(n: int = 120) -> np.ndarray:
    """Flat, low-volatility price series — annualised vol ~2%.

    Returns a nearly-constant price series with minimal noise.
    """
    rng = np.random.default_rng(10)
    base = np.linspace(10.0, 10.3, n, dtype=np.float64)
    noise = rng.normal(0.0, 0.005, n).astype(np.float64)
    return base + noise


def _price_high_vol(n: int = 120) -> np.ndarray:
    """High-volatility price series with large swings.

    Uses alternating up/down segments to generate ~15% annualised vol.
    """
    rng = np.random.default_rng(20)
    prices = np.zeros(n, dtype=np.float64)
    prices[0] = 10.0
    for i in range(1, n):
        shock = rng.normal(0.0, 0.15, 1).item()
        prices[i] = prices[i - 1] + shock
    return prices


def _price_medium_vol(n: int = 120) -> np.ndarray:
    """Medium-volatility price series — moderate noise."""
    rng = np.random.default_rng(30)
    base = np.linspace(10.0, 10.5, n, dtype=np.float64)
    noise = rng.normal(0.0, 0.05, n).astype(np.float64)
    return base + noise


# ─── Strategy Configuration Tests ────────────────────────────────────────────


class TestStrategyConfig:
    """Default parameters, eligibility, data requirements, and validation."""

    def test_default_params(self) -> None:
        strat = LowVolRotation()
        assert strat.vol_period == 60
        assert strat.top_n == 3
        assert strat.name == "low_vol_rotation"

    def test_eligible_regimes_all(self) -> None:
        strat = LowVolRotation()
        for regime in MarketRegime:
            assert strat.is_eligible(regime), f"Expected eligible for {regime}"

    def test_required_data(self) -> None:
        strat = LowVolRotation()
        required = strat.required_data()
        assert "close" in required

    def test_validate_default_is_clean(self) -> None:
        strat = LowVolRotation()
        errors = strat.validate()
        assert errors == []

    def test_validate_rejects_zero_top_n(self) -> None:
        strat = LowVolRotation(top_n=0)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_validate_rejects_short_vol_period(self) -> None:
        strat = LowVolRotation(vol_period=1)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_custom_params(self) -> None:
        strat = LowVolRotation(vol_period=30, top_n=5)
        assert strat.vol_period == 30
        assert strat.top_n == 5

    def test_frozen_model_prevents_mutation(self) -> None:
        strat = LowVolRotation()
        with pytest.raises(Exception):
            strat.vol_period = 30  # type: ignore[misc]


# ─── Low-Vol Selection Logic ─────────────────────────────────────────────────


class TestLowVolSelection:
    """Strategy must select bottom top_n by volatility, equal weight."""

    def test_selects_lowest_vol_funds(self) -> None:
        """3 funds with different volatilities → lowest-vol are picked."""
        n = 120
        market_data = {
            "ETF_LOW": _make_df(_price_low_vol(n)),
            "ETF_MED": _make_df(_price_medium_vol(n)),
            "ETF_HIGH": _make_df(_price_high_vol(n)),
        }
        strat = LowVolRotation(vol_period=60, top_n=2)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        buy_funds = {s["fund_code"] for s in buy_signals}
        assert len(buy_signals) == 2, f"Expected 2 BUY signals, got {len(buy_signals)}"
        assert "ETF_LOW" in buy_funds, "Lowest-vol ETF should be selected"
        assert "ETF_HIGH" not in buy_funds, "Highest-vol ETF should be excluded"

    def test_selects_exactly_top_n(self) -> None:
        """When more funds than top_n exist, exactly top_n are selected."""
        n = 120
        market_data = {
            f"ETF_{i}": _make_df(
                np.cumsum(
                    np.random.default_rng(100 + i).normal(0.0, 0.02 + i * 0.03, n).astype(np.float64)
                )
                + 10.0
            )
            for i in range(5)
        }
        strat = LowVolRotation(vol_period=60, top_n=3)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 3

    def test_less_funds_than_top_n_selects_all(self) -> None:
        """When fewer funds available than top_n, select all available."""
        market_data = {
            "ETF_A": _make_df(_price_low_vol(120)),
            "ETF_B": _make_df(_price_medium_vol(120)),
        }
        strat = LowVolRotation(vol_period=60, top_n=5)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 2


# ─── Signal Structure ────────────────────────────────────────────────────────


class TestSignalStructure:
    """Returned signal dicts must have all required keys with valid values."""

    def test_buy_signal_shape(self) -> None:
        market_data = {
            "ETF_A": _make_df(_price_low_vol(120)),
            "ETF_B": _make_df(_price_high_vol(120)),
        }
        strat = LowVolRotation(vol_period=60, top_n=1)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in {e.value for e in SignalDirection}
            assert isinstance(s["confidence"], (float, int))
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], (float, int))
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0

    def test_equal_weight_allocation(self) -> None:
        """Selected ETFs receive equal target_weight = 1 / top_n."""
        n = 120
        market_data = {
            "ETF_A": _make_df(_price_low_vol(n)),
            "ETF_B": _make_df(_price_medium_vol(n)),
            "ETF_C": _make_df(_price_high_vol(n)),
        }
        strat = LowVolRotation(vol_period=60, top_n=2)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        expected_weight = round(1.0 / 2, 4)
        for s in buy_signals:
            assert s["target_weight"] == expected_weight, (
                f"Expected weight {expected_weight}, got {s['target_weight']}"
            )

    def test_reason_mentions_volatility(self) -> None:
        market_data = {
            "ETF_A": _make_df(_price_low_vol(120)),
            "ETF_B": _make_df(_price_high_vol(120)),
        }
        strat = LowVolRotation(vol_period=60, top_n=1)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) >= 1
        assert "vol" in buy_signals[0]["reason"].lower()

    def test_selected_have_trim_for_excluded(self) -> None:
        """Excluded ETFs should get TRIM or SELL signals."""
        market_data = {
            "ETF_A": _make_df(_price_low_vol(120)),
            "ETF_B": _make_df(_price_medium_vol(120)),
            "ETF_C": _make_df(_price_high_vol(120)),
        }
        strat = LowVolRotation(vol_period=60, top_n=1)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        trim_dirs = {SignalDirection.TRIM, SignalDirection.SELL}
        excluded_signals = [
            s for s in signals
            if s["direction"] in {d.value for d in trim_dirs}
        ]
        assert len(excluded_signals) >= 1, "Excluded ETFs should be trimmed"


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Empty data, insufficient data, missing columns, and single fund."""

    def test_empty_market_data(self) -> None:
        strat = LowVolRotation()
        signals = strat.generate_signals(dt=date(2026, 6, 1), market_data={})
        assert signals == []

    def test_insufficient_data_no_crash(self) -> None:
        """Too few data points (< vol_period) → no crash, no BUY."""
        close = np.linspace(10.0, 10.5, 10, dtype=np.float64)
        market_data = {"ETF_X": _make_df(close)}
        strat = LowVolRotation(vol_period=60)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 0

    def test_missing_close_column_no_crash(self) -> None:
        df = pl.DataFrame({"nav": [1.0, 1.1, 1.2]})
        strat = LowVolRotation()
        signals = strat.generate_signals(
            dt=date(2026, 6, 15), market_data={"000001": df},
        )
        assert isinstance(signals, list)

    def test_single_fund_produces_buy(self) -> None:
        """Single fund with sufficient data → should be selected."""
        market_data = {"ETF_ONLY": _make_df(_price_low_vol(120))}
        strat = LowVolRotation(vol_period=60, top_n=3)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 1
        assert buy_signals[0]["fund_code"] == "ETF_ONLY"

    def test_all_nan_close_no_crash(self) -> None:
        close = np.full(100, np.nan, dtype=np.float64)
        market_data = {"ETF_X": _make_df(close)}
        strat = LowVolRotation()
        signals = strat.generate_signals(
            dt=date(2026, 6, 15), market_data=market_data,
        )
        assert isinstance(signals, list)

    def test_zero_vol_no_crash(self) -> None:
        """Constant price → zero volatility → should still work."""
        close = np.full(120, 10.0, dtype=np.float64)
        market_data = {"ETF_FLAT": _make_df(close)}
        strat = LowVolRotation(vol_period=60, top_n=1)
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 1
