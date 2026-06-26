"""
Tests for North-bound Flow Following strategy (S16).

S16 — North-bound Flow Following: 北向资金连续3日净流入→BUY, 连续3日净流出→SELL.
Win rate: N/A (flow-following), eligible: TRENDING_UP + SIDEWAYS.
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection
from src.core.strategy.china_specific.north_bound_flow import (
    NorthBoundFlow,
    NorthBoundFlowConfig,
)


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _make_flow_df(*flows: float) -> pl.DataFrame:
    """Build a polars DataFrame with north_bound_flow column.
    Each float becomes one row. Also adds a close column for data completeness.
    """
    return pl.DataFrame({
        "north_bound_flow": list(flows),
        "close": [1.0] * len(flows),
    })


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> NorthBoundFlow:
    config = NorthBoundFlowConfig(
        name="north_bound_flow",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        consecutive_days=3,
        flow_threshold=0.0,
    )
    return NorthBoundFlow(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2026, 6, 15)


@pytest.fixture
def inflow_data() -> dict[str, pl.DataFrame]:
    """3 consecutive days of positive north-bound flow."""
    return {"FUND_A": _make_flow_df(50.0, 30.0, 20.0)}


@pytest.fixture
def outflow_data() -> dict[str, pl.DataFrame]:
    """3 consecutive days of negative north-bound flow."""
    return {"FUND_A": _make_flow_df(-50.0, -30.0, -20.0)}


@pytest.fixture
def mixed_flow_data() -> dict[str, pl.DataFrame]:
    """Mixed direction — not 3 consecutive same-sign."""
    return {"FUND_A": _make_flow_df(50.0, -10.0, 20.0)}


@pytest.fixture
def multi_fund_data() -> dict[str, pl.DataFrame]:
    """Multiple funds with different flow patterns."""
    return {
        "FUND_IN": _make_flow_df(50.0, 30.0, 20.0),
        "FUND_OUT": _make_flow_df(-40.0, -25.0, -15.0),
        "FUND_MIXED": _make_flow_df(10.0, -5.0, 8.0),
    }


# ─── Configuration Tests ────────────────────────────────────────────────────────


class TestNorthBoundFlowConfig:
    def test_default_config(self) -> None:
        config = NorthBoundFlowConfig(name="north_bound_flow")
        assert config.consecutive_days == 3
        assert config.flow_threshold == 0.0
        assert config.max_position_pct == 0.20

    def test_custom_consecutive_days(self) -> None:
        config = NorthBoundFlowConfig(name="north_bound_flow", consecutive_days=5)
        assert config.consecutive_days == 5

    def test_custom_flow_threshold(self) -> None:
        config = NorthBoundFlowConfig(name="north_bound_flow", flow_threshold=10.0)
        assert config.flow_threshold == 10.0

    def test_rejects_invalid_consecutive_days(self) -> None:
        with pytest.raises(ValueError):
            NorthBoundFlowConfig(name="north_bound_flow", consecutive_days=0)

    def test_rejects_negative_consecutive_days(self) -> None:
        with pytest.raises(ValueError):
            NorthBoundFlowConfig(name="north_bound_flow", consecutive_days=-1)

    def test_inherits_strategy_config(self) -> None:
        config = NorthBoundFlowConfig(name="north_bound_flow")
        from src.core.strategy.base import StrategyConfig
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ─────────────────────────────────────────────────────────


class TestRegimeEligibility:
    def test_eligible_in_trending_up_and_sideways(self, strategy: NorthBoundFlow) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is True

    def test_not_eligible_in_other_regimes(self, strategy: NorthBoundFlow) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ──────────────────────────────────────────────────────────────


class TestRequiredData:
    def test_returns_required_fields(self, strategy: NorthBoundFlow) -> None:
        fields = set(strategy.required_data())
        assert "close" in fields
        assert "north_bound_flow" in fields

    def test_required_data_is_list_of_str(self, strategy: NorthBoundFlow) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ─────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    def test_consecutive_inflow_generates_buy(
        self, strategy: NorthBoundFlow, trade_date: date, inflow_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, inflow_data)
        assert len(signals) == 1
        assert signals[0]["fund_code"] == "FUND_A"
        assert signals[0]["direction"] == SignalDirection.BUY.value

    def test_consecutive_outflow_generates_sell(
        self, strategy: NorthBoundFlow, trade_date: date, outflow_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, outflow_data)
        assert len(signals) == 1
        assert signals[0]["fund_code"] == "FUND_A"
        assert signals[0]["direction"] == SignalDirection.SELL.value

    def test_mixed_flow_generates_hold(
        self, strategy: NorthBoundFlow, trade_date: date, mixed_flow_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, mixed_flow_data)
        assert len(signals) == 1
        assert signals[0]["direction"] == SignalDirection.HOLD.value

    def test_multi_fund_signals(
        self, strategy: NorthBoundFlow, trade_date: date, multi_fund_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, multi_fund_data)
        assert len(signals) == 3

        buy = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        sell = [s for s in signals if s["direction"] == SignalDirection.SELL.value]
        hold = [s for s in signals if s["direction"] == SignalDirection.HOLD.value]

        assert len(buy) == 1 and buy[0]["fund_code"] == "FUND_IN"
        assert len(sell) == 1 and sell[0]["fund_code"] == "FUND_OUT"
        assert len(hold) == 1 and hold[0]["fund_code"] == "FUND_MIXED"

    def test_signal_structure(
        self, strategy: NorthBoundFlow, trade_date: date, inflow_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, inflow_data)
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in {e.value for e in SignalDirection}
            assert isinstance(s["confidence"], float)
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], float)
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0

    def test_five_consecutive_days(
        self, trade_date: date
    ) -> None:
        config = NorthBoundFlowConfig(name="north_bound_flow", consecutive_days=5)
        strategy_5 = NorthBoundFlow(config=config)
        # 5 consecutive inflow days → BUY
        data = {"FUND_A": _make_flow_df(10.0, 20.0, 30.0, 15.0, 25.0)}
        signals = strategy_5.generate_signals(trade_date, data)
        assert signals[0]["direction"] == SignalDirection.BUY.value

    def test_five_consecutive_not_met(
        self, trade_date: date
    ) -> None:
        config = NorthBoundFlowConfig(name="north_bound_flow", consecutive_days=5)
        strategy_5 = NorthBoundFlow(config=config)
        # 5 rows but only last 4 are inflow, first is negative → HOLD
        data = {"FUND_A": _make_flow_df(-5.0, 10.0, 20.0, 30.0, 15.0)}
        signals = strategy_5.generate_signals(trade_date, data)
        assert signals[0]["direction"] == SignalDirection.HOLD.value

    def test_custom_flow_threshold_filters(
        self, trade_date: date
    ) -> None:
        # flow_threshold=10: flows between -10 and 10 treated as "flat"
        config = NorthBoundFlowConfig(name="north_bound_flow", flow_threshold=10.0)
        strategy_thresh = NorthBoundFlow(config=config)
        data = {"FUND_A": _make_flow_df(5.0, 5.0, 5.0)}
        signals = strategy_thresh.generate_signals(trade_date, data)
        # All flows below threshold → not "inflow" → HOLD
        assert signals[0]["direction"] == SignalDirection.HOLD.value

    def test_buy_confidence_exceeds_hold(
        self, strategy: NorthBoundFlow, trade_date: date
    ) -> None:
        inflow = {"FUND_A": _make_flow_df(50.0, 30.0, 20.0)}
        mixed = {"FUND_B": _make_flow_df(10.0, -5.0, 8.0)}
        buy_sig = strategy.generate_signals(trade_date, inflow)[0]
        hold_sig = strategy.generate_signals(trade_date, mixed)[0]
        assert buy_sig["confidence"] > hold_sig["confidence"]


# ─── Signal Generation — Edge Cases ─────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    def test_empty_market_data(self, strategy: NorthBoundFlow, trade_date: date) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_missing_north_bound_flow_column(
        self, strategy: NorthBoundFlow, trade_date: date
    ) -> None:
        df = pl.DataFrame({"close": [1.0, 1.0, 1.0]})
        signals = strategy.generate_signals(trade_date, {"FUND_A": df})
        assert signals == []

    def test_insufficient_rows(
        self, strategy: NorthBoundFlow, trade_date: date
    ) -> None:
        # Fewer rows than consecutive_days
        df = _make_flow_df(10.0, 20.0)  # only 2 rows, need 3
        signals = strategy.generate_signals(trade_date, {"FUND_A": df})
        assert signals == []  # insufficient data → no signal

    def test_exactly_consecutive_days_rows(
        self, strategy: NorthBoundFlow, trade_date: date, inflow_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, inflow_data)
        assert len(signals) > 0

    def test_more_rows_than_consecutive_days(
        self, strategy: NorthBoundFlow, trade_date: date
    ) -> None:
        df = _make_flow_df(-5.0, 10.0, 20.0, 30.0)  # last 3 are positive
        signals = strategy.generate_signals(trade_date, {"FUND_A": df})
        assert signals[0]["direction"] == SignalDirection.BUY.value

    def test_zero_flow_treated_as_neutral(
        self, strategy: NorthBoundFlow, trade_date: date
    ) -> None:
        # flow_threshold=0, zero is not > 0 and not < 0 → HOLD
        df = _make_flow_df(0.0, 0.0, 0.0)
        signals = strategy.generate_signals(trade_date, {"FUND_A": df})
        assert signals[0]["direction"] == SignalDirection.HOLD.value

    def test_mixed_valid_invalid_funds(
        self, strategy: NorthBoundFlow, trade_date: date, inflow_data: dict[str, pl.DataFrame]
    ) -> None:
        mixed = dict(inflow_data)
        mixed["FUND_BAD"] = pl.DataFrame({"close": [1.0] * 5})
        signals = strategy.generate_signals(trade_date, mixed)
        bad_signals = [s for s in signals if s["fund_code"] == "FUND_BAD"]
        assert len(bad_signals) == 0


# ─── Strategy Identity ──────────────────────────────────────────────────────────


class TestStrategyIdentity:
    def test_name_matches_config(self, strategy: NorthBoundFlow) -> None:
        assert strategy.name == "north_bound_flow"

    def test_validate_passes(self, strategy: NorthBoundFlow) -> None:
        assert strategy.validate() == []

    def test_strategy_is_frozen(self, strategy: NorthBoundFlow) -> None:
        with pytest.raises(Exception):
            strategy.config.consecutive_days = 99  # type: ignore[misc]
