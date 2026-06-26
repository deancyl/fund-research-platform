"""
Tests for RsiFixedIncomeStrategy (S7) — RSI均值回归 + 固收.

Logic: RSI(14) < 35 → BUY index ETF; RSI > 70 → SELL; else BUY bond fund.
  35 <= RSI <= 70   → neutral zone → rotate into bond ETF (defensive).
  RSI < 35          → oversold   → buy the dip on equity.
  RSI > 70          → overbought → take profit, sell equity.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection
from src.core.strategy.mean_reversion.rsi_fixed_income import RsiFixedIncomeStrategy

# ─── Test Fixtures ───────────────────────────────────────────────────────────


def _make_df(close_arr: np.ndarray) -> pl.DataFrame:
    """Create a polars DataFrame with a close column from a numpy array."""
    return pl.DataFrame({"close": close_arr.tolist()})


def _price_downtrend(n: int = 60) -> np.ndarray:
    """Steady decline from 100 to 50 → RSI well below 35 (oversold)."""
    return np.linspace(100.0, 50.0, n, dtype=np.float64)


def _price_uptrend(n: int = 60) -> np.ndarray:
    """Steady rise from 50 to 100 → RSI well above 70 (overbought)."""
    return np.linspace(50.0, 100.0, n, dtype=np.float64)


def _price_sideways(n: int = 60) -> np.ndarray:
    """Sideways chop around 50 → RSI ≈ 50 (neutral zone 35-70)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.5, n).astype(np.float64)
    base = np.full(n, 50.0, dtype=np.float64)
    return base + noise


@pytest.fixture
def strategy_default() -> RsiFixedIncomeStrategy:
    """Strategy with default parameters from the S7 spec."""
    return RsiFixedIncomeStrategy()


@pytest.fixture
def strategy_custom() -> RsiFixedIncomeStrategy:
    """Strategy with custom band thresholds."""
    return RsiFixedIncomeStrategy(
        rsi_period=14,
        rsi_oversold=30.0,
        rsi_overbought=80.0,
        bond_fund_code="511880",
    )


# ─── Signal Direction ────────────────────────────────────────────────────────


class TestSignalDirection:
    """generate_signals must produce correct SignalDirection values."""

    def test_downtrend_produces_buy_equity(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Steady decline → RSI << 35 → BUY equity ETF (oversold)."""
        close = _price_downtrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        assert len(signals) >= 1
        directions = {s["direction"] for s in signals}
        assert "BUY" in directions, f"Expected BUY in downtrend, got {directions}"

    def test_uptrend_produces_sell_equity(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Steady rise → RSI >> 70 → SELL equity ETF (overbought)."""
        close = _price_uptrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        assert len(signals) >= 1
        directions = {s["direction"] for s in signals}
        assert "SELL" in directions, f"Expected SELL in uptrend, got {directions}"

    def test_sideways_produces_buy_bond(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Sideways → RSI ≈ 50 (neutral) → BUY bond fund (rotate to fixed income)."""
        close = _price_sideways()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        assert len(signals) >= 1
        # In neutral zone, signal should point to the bond fund
        bond_signals = [s for s in signals if s["fund_code"] == "511260"]
        assert len(bond_signals) >= 1, f"Expected BUY bond signal, got {signals}"
        assert bond_signals[0]["direction"] == SignalDirection.BUY

    def test_downtrend_signal_fund_code_is_equity(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Oversold → BUY should target the equity ETF, not the bond fund."""
        close = _price_downtrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        equity_signals = [s for s in signals if s["fund_code"] == "510050"]
        assert len(equity_signals) >= 1
        assert equity_signals[0]["direction"] == SignalDirection.BUY

    def test_uptrend_signal_fund_code_is_equity(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Overbought → SELL should target the equity ETF."""
        close = _price_uptrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        equity_signals = [s for s in signals if s["fund_code"] == "510050"]
        assert len(equity_signals) >= 1
        assert equity_signals[0]["direction"] == SignalDirection.SELL


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestEmptyOrInsufficientData:
    """Edge cases: no data or too little data."""

    def test_empty_market_data_returns_empty_list(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Empty market_data → empty signals."""
        signals = strategy_default.generate_signals(dt=date(2025, 1, 1), market_data={})
        assert signals == []

    def test_insufficient_data_returns_hold_or_empty(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Only 3 price points → not enough for 14-period RSI → HOLD only."""
        close = np.array([10.0, 10.5, 10.2], dtype=np.float64)
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        for s in signals:
            assert s["direction"] in (SignalDirection.HOLD, "HOLD")

    def test_missing_close_column_no_crash(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """DataFrame without 'close' column → no signal, no crash."""
        df = pl.DataFrame({"nav": [1.0, 1.1, 1.2]})
        signals = strategy_default.generate_signals(
            dt=date(2025, 6, 30), market_data={"000001": df},
        )
        # Should not crash; may return empty or skip the fund
        assert isinstance(signals, list)

    def test_empty_close_array_no_crash(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """Empty close array → no signal, no crash."""
        df = pl.DataFrame({"close": []})
        signals = strategy_default.generate_signals(
            dt=date(2025, 6, 30), market_data={"510050": df},
        )
        assert isinstance(signals, list)

    def test_all_nan_close_no_crash(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """All-NaN close array → must not crash."""
        close = np.full(30, np.nan, dtype=np.float64)
        df = _make_df(close)
        signals = strategy_default.generate_signals(
            dt=date(2025, 6, 30), market_data={"510050": df},
        )
        assert isinstance(signals, list)


# ─── Signal Shape ────────────────────────────────────────────────────────────


class TestSignalShape:
    """Returned signal dicts must have all required keys."""

    def test_signal_dict_keys_downtrend(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        close = _price_downtrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        for s in signals:
            assert "fund_code" in s
            assert "direction" in s
            assert "confidence" in s
            assert "target_weight" in s
            assert "reason" in s
            assert 0.0 <= s["confidence"] <= 1.0

    def test_signal_dict_keys_uptrend(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        close = _price_uptrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        for s in signals:
            assert "fund_code" in s
            assert "direction" in s
            assert "confidence" in s
            assert "target_weight" in s
            assert "reason" in s

    def test_signal_dict_keys_sideways(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        close = _price_sideways()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        for s in signals:
            assert "fund_code" in s
            assert "direction" in s
            assert "confidence" in s
            assert "target_weight" in s
            assert "reason" in s


# ─── Strategy Configuration ──────────────────────────────────────────────────


class TestStrategyConfig:
    """Strategy parameters, eligibility, and validation."""

    def test_default_params(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        assert strategy_default.rsi_period == 14
        assert strategy_default.rsi_oversold == 35.0
        assert strategy_default.rsi_overbought == 70.0
        assert strategy_default.bond_fund_code == "511260"

    def test_eligible_regimes_all(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        """S7 is eligible for ALL market regimes."""
        for regime in MarketRegime:
            assert strategy_default.is_eligible(regime), (
                f"Expected eligible for {regime}"
            )

    def test_required_data(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        required = strategy_default.required_data()
        assert "close" in required
        assert len(required) == 1

    def test_validate_returns_empty_on_default(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        errors = strategy_default.validate()
        assert errors == []

    def test_validate_rejects_invalid_thresholds(self) -> None:
        """rsi_oversold must be less than rsi_overbought."""
        strategy = RsiFixedIncomeStrategy(
            rsi_oversold=60.0,
            rsi_overbought=40.0,
        )
        errors = strategy.validate()
        assert len(errors) >= 1

    def test_name_matches_spec(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        assert strategy_default.name == "rsi_fixed_income"

    def test_version_default(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        assert strategy_default.config.version == "1.0.0"


# ─── Custom Parameters ───────────────────────────────────────────────────────


class TestCustomParameters:
    """Strategy must accept custom parameters through its frozen model."""

    def test_custom_thresholds(self, strategy_custom: RsiFixedIncomeStrategy) -> None:
        assert strategy_custom.rsi_period == 14
        assert strategy_custom.rsi_oversold == 30.0
        assert strategy_custom.rsi_overbought == 80.0
        assert strategy_custom.bond_fund_code == "511880"

    def test_custom_bond_thresholds_produce_expected_signals(
        self, strategy_custom: RsiFixedIncomeStrategy,
    ) -> None:
        """With wider bands (30/80), sideways data should still be neutral."""
        close = _price_sideways()
        market_data = {"510050": _make_df(close)}
        signals = strategy_custom.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        # With wider bands, sideways RSI ≈ 50 should still trigger bond buy
        bond_signals = [s for s in signals if s["fund_code"] == "511880"]
        assert len(bond_signals) >= 1

    def test_multiple_funds(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        """Multiple equity funds in market_data should all be processed."""
        close_down = _price_downtrend()
        close_up = _price_uptrend()
        market_data = {
            "510050": _make_df(close_down),
            "510300": _make_df(close_up),
        }
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        fund_codes = {s["fund_code"] for s in signals}
        # Both equity funds should appear in results
        assert "510050" in fund_codes or len(signals) >= 2
        assert "510300" in fund_codes or len(signals) >= 2

    def test_frozen_model_prevents_mutation(self) -> None:
        """Strategy is frozen — field mutation raises an error."""
        strategy = RsiFixedIncomeStrategy()
        with pytest.raises(Exception):  # may be ValidationError, FrozenInstanceError, etc.
            strategy.rsi_period = 10  # type: ignore[misc]


# ─── Confidence and Target Weight ────────────────────────────────────────────


class TestConfidenceAndWeight:
    """Confidence must be in [0, 1] and weight must be reasonable."""

    def test_buy_confidence_positive(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        """Oversold BUY should have confidence > 0."""
        close = _price_downtrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY
                       and s["fund_code"] == "510050"]
        assert len(buy_signals) >= 1
        assert buy_signals[0]["confidence"] > 0.0

    def test_sell_confidence_positive(self, strategy_default: RsiFixedIncomeStrategy) -> None:
        """Overbought SELL should have confidence > 0."""
        close = _price_uptrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        sell_signals = [s for s in signals if s["direction"] == SignalDirection.SELL]
        assert len(sell_signals) >= 1
        assert sell_signals[0]["confidence"] > 0.0

    def test_buy_target_weight_within_bounds(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """BUY target_weight should be in (0, max_position_pct]."""
        close = _price_downtrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY]
        for s in buy_signals:
            assert 0.0 < s["target_weight"] <= strategy_default.config.max_position_pct

    def test_sell_target_weight_is_zero(
        self, strategy_default: RsiFixedIncomeStrategy,
    ) -> None:
        """SELL should have target_weight == 0."""
        close = _price_uptrend()
        market_data = {"510050": _make_df(close)}
        signals = strategy_default.generate_signals(dt=date(2025, 6, 30), market_data=market_data)
        sell_signals = [s for s in signals if s["direction"] == SignalDirection.SELL]
        for s in sell_signals:
            assert s["target_weight"] == 0.0
