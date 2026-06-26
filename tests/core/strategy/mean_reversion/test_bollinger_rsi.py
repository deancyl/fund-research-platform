"""
TDD tests for BollingerRsiStrategy (S9) — Bollinger Band + RSI Composite.

Spec:
  - Bollinger(20, 2.0) + RSI(14) composite signal.
  - BUY when RSI < 30 (oversold) AND price below lower Bollinger band.
  - Win rate: 68-70% in range-bound (SIDEWAYS) markets.
  - Eligible regimes: SIDEWAYS only.
  - Default params: boll_period=20, boll_std=2.0, rsi_period=14,
    rsi_oversold=30, rsi_overbought=70.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection
from src.core.strategy.mean_reversion.bollinger_rsi import BollingerRsiStrategy


# ─── Test Fixtures ───────────────────────────────────────────────────────────


def _make_df(close_arr: np.ndarray) -> pl.DataFrame:
    """Create a polars DataFrame with a close column from a numpy array."""
    return pl.DataFrame({"close": close_arr.tolist()})


def _price_oversold_below_band(n: int = 100) -> np.ndarray:
    """Price declining below Bollinger lower band + RSI < 30.

    Uses tight consolidation (std ~0.2) for most of the series so Bollinger
    bands stay narrow, then a 3-point crash breaches the lower band while
    the consecutive losses drive RSI into oversold territory.
    """
    rng = np.random.default_rng(42)
    # 80 points: very tight range [49.6, 50.4] → Bollinger bands are narrow
    tight = 50.0 + rng.normal(0.0, 0.2, 80).astype(np.float64)
    # 17 more tight points so the 20-period window still has narrow std
    tight2 = 50.0 + rng.normal(0.0, 0.2, 17).astype(np.float64)
    # 3-point crash drives price below lower band and RSI oversold
    crash = np.array([30.0, 25.0, 20.0], dtype=np.float64)
    return np.concatenate([tight, tight2, crash])


def _price_overbought_above_band(n: int = 100) -> np.ndarray:
    """Price rising above Bollinger upper band + RSI > 70.

    Tight consolidation then 3-point surge that breaches the upper band
    while consecutive gains drive RSI above 70.
    """
    rng = np.random.default_rng(43)
    tight = 50.0 + rng.normal(0.0, 0.2, 80).astype(np.float64)
    tight2 = 50.0 + rng.normal(0.0, 0.2, 17).astype(np.float64)
    surge = np.array([70.0, 75.0, 80.0], dtype=np.float64)
    return np.concatenate([tight, tight2, surge])


def _price_sideways(n: int = 100) -> np.ndarray:
    """Price oscillating inside Bollinger bands with neutral RSI."""
    rng = np.random.default_rng(44)
    return np.full(n, 50.0, dtype=np.float64) + rng.normal(0.0, 1.5, n).astype(np.float64)


def _price_strong_rsi_oversold_inside_band(n: int = 100) -> np.ndarray:
    """RSI oversold but price stays inside Bollinger band — no BUY.

    A slow, steady grind down that stays inside the band envelope.
    """
    return np.linspace(55.0, 45.0, n, dtype=np.float64)


# ─── Strategy Configuration Tests ────────────────────────────────────────────


class TestStrategyConfig:
    """Default parameters, eligibility, data requirements, and validation."""

    def test_default_params(self) -> None:
        strat = BollingerRsiStrategy()
        assert strat.boll_period == 20
        assert strat.boll_std == 2.0
        assert strat.rsi_period == 14
        assert strat.rsi_oversold == 30.0
        assert strat.rsi_overbought == 70.0
        assert strat.name == "bollinger_rsi"

    def test_eligible_regimes(self) -> None:
        strat = BollingerRsiStrategy()
        assert strat.is_eligible(MarketRegime.SIDEWAYS)
        for regime in MarketRegime:
            if regime != MarketRegime.SIDEWAYS:
                assert not strat.is_eligible(regime), (
                    f"Should not be eligible in {regime}"
                )

    def test_required_data(self) -> None:
        strat = BollingerRsiStrategy()
        required = strat.required_data()
        assert "close" in required

    def test_validate_default_is_clean(self) -> None:
        strat = BollingerRsiStrategy()
        errors = strat.validate()
        assert errors == []

    def test_validate_rejects_oversold_above_overbought(self) -> None:
        strat = BollingerRsiStrategy(rsi_oversold=80.0, rsi_overbought=30.0)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_validate_rejects_zero_period(self) -> None:
        strat = BollingerRsiStrategy(boll_period=0)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_validate_rejects_negative_std(self) -> None:
        strat = BollingerRsiStrategy(boll_std=-1.0)
        errors = strat.validate()
        assert len(errors) >= 1

    def test_custom_params(self) -> None:
        strat = BollingerRsiStrategy(
            boll_period=10,
            boll_std=2.5,
            rsi_period=7,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
        )
        assert strat.boll_period == 10
        assert strat.boll_std == 2.5
        assert strat.rsi_period == 7
        assert strat.rsi_oversold == 25.0
        assert strat.rsi_overbought == 75.0

    def test_frozen_model_prevents_mutation(self) -> None:
        strat = BollingerRsiStrategy()
        with pytest.raises(Exception):
            strat.boll_period = 10  # type: ignore[misc]


# ─── Composite Signal Logic ──────────────────────────────────────────────────


class TestCompositeSignal:
    """Both RSI < oversold AND price < lower band must hold for BUY."""

    def test_oversold_below_band_produces_buy(self) -> None:
        """RSI < 30 AND price below lower band → BUY signal."""
        close = _price_oversold_below_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) >= 1, f"Expected BUY signal, got {signals}"

    def test_overbought_above_band_produces_sell(self) -> None:
        """RSI > 70 AND price above upper band → SELL signal."""
        close = _price_overbought_above_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        sell_signals = [s for s in signals if s["direction"] == SignalDirection.SELL]
        assert len(sell_signals) >= 1, f"Expected SELL signal, got {signals}"

    def test_sideways_produces_hold(self) -> None:
        """Neutral market → HOLD (no extreme RSI or band breach)."""
        close = _price_sideways()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        hold_signals = [s for s in signals if s["direction"] == SignalDirection.HOLD]
        assert len(hold_signals) >= 1

    def test_oversold_inside_band_no_buy(self) -> None:
        """RSI oversold but price inside band → no BUY (AND gate)."""
        close = _price_strong_rsi_oversold_inside_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 0, (
            f"Should not BUY when price is inside band, got {buy_signals}"
        )


# ─── Signal Structure ────────────────────────────────────────────────────────


class TestSignalStructure:
    """Returned signal dicts must have all required keys with valid values."""

    def test_buy_signal_shape(self) -> None:
        close = _price_oversold_below_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
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

    def test_buy_reason_mentions_bollinger_and_rsi(self) -> None:
        close = _price_oversold_below_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) >= 1
        reason_lower = buy_signals[0]["reason"].lower()
        assert "boll" in reason_lower or "rsi" in reason_lower

    def test_sell_target_weight_is_zero(self) -> None:
        close = _price_overbought_above_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        sell_signals = [s for s in signals if s["direction"] == SignalDirection.SELL]
        for s in sell_signals:
            assert s["target_weight"] == 0.0

    def test_buy_target_weight_positive(self) -> None:
        close = _price_oversold_below_band()
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        for s in buy_signals:
            assert s["target_weight"] > 0.0
            assert s["target_weight"] <= strat.config.max_position_pct


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Empty data, insufficient data, missing columns, and multiple funds."""

    def test_empty_market_data(self) -> None:
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 1), market_data={})
        assert signals == []

    def test_insufficient_data_no_crash(self) -> None:
        """Too few data points (< boll_period) → no crash, no BUY."""
        close = np.array([10.0, 10.5, 10.2], dtype=np.float64)
        market_data = {"510300": _make_df(close)}
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        assert len(buy_signals) == 0

    def test_missing_close_column_no_crash(self) -> None:
        df = pl.DataFrame({"nav": [1.0, 1.1, 1.2]})
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(
            dt=date(2026, 6, 15), market_data={"000001": df},
        )
        assert isinstance(signals, list)

    def test_all_nan_close_no_crash(self) -> None:
        close = np.full(50, np.nan, dtype=np.float64)
        df = _make_df(close)
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(
            dt=date(2026, 6, 15), market_data={"510300": df},
        )
        assert isinstance(signals, list)

    def test_multiple_funds(self) -> None:
        close_buy = _price_oversold_below_band()
        close_sideways = _price_sideways()
        market_data = {
            "510300": _make_df(close_buy),
            "159915": _make_df(close_sideways),
        }
        strat = BollingerRsiStrategy()
        signals = strat.generate_signals(dt=date(2026, 6, 15), market_data=market_data)
        fund_codes = {s["fund_code"] for s in signals}
        assert "510300" in fund_codes
        assert "159915" in fund_codes
        # 510300 should have a BUY
        buy_510300 = [
            s for s in signals
            if s["fund_code"] == "510300" and s["direction"] == SignalDirection.BUY
        ]
        assert len(buy_510300) >= 1
