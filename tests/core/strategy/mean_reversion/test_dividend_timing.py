"""
Tests for DividendTimingStrategy (S8) — 红利低波 + 股息率择时.

Scoring formula: RSI(20) * 30% + MA deviation * 35% + Bollinger deviation * 35%
  - Low score (<=25) -> BUY
  - High score (>=75) -> SELL
  - Mid score (25-75) -> HOLD
  - Empty data -> empty signals
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection
from src.core.strategy.mean_reversion.dividend_timing import DividendTimingStrategy

# ─── Test Fixtures ───────────────────────────────────────────────────────────


def _make_df(close_arr: np.ndarray) -> pl.DataFrame:
    """Create a polars DataFrame with a close column from a numpy array."""
    return pl.DataFrame({"close": close_arr.tolist()})


def _price_downtrend(n: int = 60) -> np.ndarray:
    """Strong downtrend: prices decline from 100 to 50."""
    return np.linspace(100.0, 50.0, n, dtype=np.float64)


def _price_uptrend(n: int = 60) -> np.ndarray:
    """Strong uptrend: prices rise from 50 to 100."""
    return np.linspace(50.0, 100.0, n, dtype=np.float64)


def _price_sideways(n: int = 60) -> np.ndarray:
    """Sideways chop around 50 with small noise."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.5, n).astype(np.float64)
    base = np.full(n, 50.0, dtype=np.float64)
    return base + noise


@pytest.fixture
def strategy_default() -> DividendTimingStrategy:
    """Strategy with default parameters from the S8 spec."""
    return DividendTimingStrategy(
        rsi_period=20,
        ma_period=20,
        boll_period=20,
        score_buy_threshold=25.0,
        score_sell_threshold=75.0,
    )


# ─── Score Computation ───────────────────────────────────────────────────────


class TestScoreComputation:
    """Internal _compute_score method must produce values in [0, 100]."""

    def test_score_in_range_uptrend(self, strategy_default: DividendTimingStrategy) -> None:
        """Score for any valid close array must be in [0, 100]."""
        close = _price_uptrend()
        score = strategy_default._compute_score(close)
        assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] for uptrend"

    def test_score_in_range_downtrend(self, strategy_default: DividendTimingStrategy) -> None:
        close = _price_downtrend()
        score = strategy_default._compute_score(close)
        assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] for downtrend"

    def test_score_in_range_sideways(self, strategy_default: DividendTimingStrategy) -> None:
        close = _price_sideways()
        score = strategy_default._compute_score(close)
        assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] for sideways"


# ─── Signal Generation ───────────────────────────────────────────────────────


class TestSignalDirection:
    """generate_signals must produce correct SignalDirection values."""

    def test_downtrend_produces_buy(self, strategy_default: DividendTimingStrategy) -> None:
        """Strong downtrend → oversold → low score → BUY signal."""
        close = _price_downtrend()
        market_data = {"000922": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        assert len(signals) >= 1
        directions = {s["direction"] for s in signals}
        assert "BUY" in directions, f"Expected BUY in downtrend, got {directions}"

    def test_uptrend_produces_sell(self, strategy_default: DividendTimingStrategy) -> None:
        """Strong uptrend → overbought → high score → SELL signal."""
        close = _price_uptrend()
        market_data = {"000922": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        assert len(signals) >= 1
        directions = {s["direction"] for s in signals}
        assert "SELL" in directions, f"Expected SELL in uptrend, got {directions}"

    def test_sideways_produces_hold(self, strategy_default: DividendTimingStrategy) -> None:
        """Sideways chop → mid score → HOLD signal."""
        close = _price_sideways()
        market_data = {"000922": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        assert len(signals) >= 1
        directions = {s["direction"] for s in signals}
        assert "HOLD" in directions, f"Expected HOLD for sideways, got {directions}"


class TestEmptyOrInsufficientData:
    """Edge cases: no data or too little data."""

    def test_empty_market_data_returns_empty_list(
        self, strategy_default: DividendTimingStrategy,
    ) -> None:
        signals = strategy_default.generate_signals(dt=date(2025, 1, 1), market_data={})
        assert signals == []

    def test_insufficient_data_returns_hold_or_empty(
        self, strategy_default: DividendTimingStrategy,
    ) -> None:
        """Only 3 price points → not enough for 20-period indicators."""
        close = np.array([10.0, 10.5, 10.2], dtype=np.float64)
        market_data = {"000922": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        # Either no signals or HOLD (no crash, no wrong signal)
        for s in signals:
            assert s["direction"] in ("HOLD", SignalDirection.HOLD)


class TestSignalShape:
    """Returned signal dicts must have all required keys."""

    def test_signal_dict_keys(self, strategy_default: DividendTimingStrategy) -> None:
        close = _price_uptrend()
        market_data = {"000922": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        for s in signals:
            assert "fund_code" in s
            assert "direction" in s
            assert "confidence" in s
            assert "target_weight" in s
            assert "reason" in s
            assert 0.0 <= s["confidence"] <= 1.0


# ─── Strategy Configuration ──────────────────────────────────────────────────


class TestStrategyConfig:
    """Strategy parameters and eligibility."""

    def test_default_params(self, strategy_default: DividendTimingStrategy) -> None:
        assert strategy_default.rsi_period == 20
        assert strategy_default.ma_period == 20
        assert strategy_default.boll_period == 20
        assert strategy_default.score_buy_threshold == 25.0
        assert strategy_default.score_sell_threshold == 75.0

    def test_eligible_regimes(self, strategy_default: DividendTimingStrategy) -> None:
        assert strategy_default.is_eligible(MarketRegime.TRENDING_DOWN)
        assert strategy_default.is_eligible(MarketRegime.SIDEWAYS)
        assert strategy_default.is_eligible(MarketRegime.HIGH_VOL)
        assert not strategy_default.is_eligible(MarketRegime.TRENDING_UP)
        assert not strategy_default.is_eligible(MarketRegime.CRISIS)

    def test_required_data(self, strategy_default: DividendTimingStrategy) -> None:
        required = strategy_default.required_data()
        assert "close" in required

    def test_validate_returns_empty_on_default(
        self, strategy_default: DividendTimingStrategy,
    ) -> None:
        errors = strategy_default.validate()
        assert errors == []

    def test_validate_rejects_buy_above_sell(self) -> None:
        strategy = DividendTimingStrategy(
            score_buy_threshold=60.0,
            score_sell_threshold=40.0,
        )
        errors = strategy.validate()
        assert len(errors) >= 1


# ─── Custom Parameters ───────────────────────────────────────────────────────


class TestCustomParameters:
    """Strategy must accept custom parameters through its frozen model."""

    def test_custom_thresholds(self) -> None:
        strategy = DividendTimingStrategy(
            rsi_period=14,
            ma_period=10,
            boll_period=10,
            score_buy_threshold=20.0,
            score_sell_threshold=80.0,
        )
        assert strategy.rsi_period == 14
        assert strategy.score_buy_threshold == 20.0
        assert strategy.score_sell_threshold == 80.0

    def test_multiple_funds(self, strategy_default: DividendTimingStrategy) -> None:
        """Multiple funds in market_data should all be processed."""
        close_down = _price_downtrend()
        close_up = _price_uptrend()
        market_data = {
            "000922": _make_df(close_down),
            "510880": _make_df(close_up),
        }
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        fund_codes = {s["fund_code"] for s in signals}
        assert "000922" in fund_codes
        assert "510880" in fund_codes
