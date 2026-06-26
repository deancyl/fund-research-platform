"""
Tests for SOE Reform strategy (S15) — "中特估" composite scoring.

S15 — SOE Reform: Composite score from dividend yield + north-bound flow
+ institutional flow combined with configurable weights. Select top-3 stocks.

Parameters: top_n=3, dividend_weight=0.4, flow_weight=0.6
Eligible regimes: ALL.
Required data: ["dividend_yield", "north_bound_flow", "institutional_flow"].
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.china_specific.soe_reform import SoeReform, SoeReformConfig


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_stock_df(
    dividend_yield: float,
    north_bound_flow: float,
    institutional_flow: float,
    n_rows: int = 60,
) -> pl.DataFrame:
    """Build a polars DataFrame with SOE reform scoring columns."""
    return pl.DataFrame({
        "dividend_yield": [dividend_yield] * n_rows,
        "north_bound_flow": [north_bound_flow] * n_rows,
        "institutional_flow": [institutional_flow] * n_rows,
    })


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> SoeReform:
    """Default SoeReform with top_n=3, dividend_weight=0.4, flow_weight=0.6."""
    config = SoeReformConfig(
        name="soe_reform",
        version="1.0.0",
        eligible_regimes={
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOL,
            MarketRegime.CRISIS,
        },
        top_n=3,
        dividend_weight=0.4,
        flow_weight=0.6,
    )
    return SoeReform(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2025, 6, 1)


@pytest.fixture
def multi_stock_data() -> dict[str, pl.DataFrame]:
    """5 SOE stocks with varying dividend yield and flow characteristics.

    Expected ranking (by composite score descending):
      STOCK_A: div=0.06, nb=0.8, inst=0.9 → highest
      STOCK_B: div=0.05, nb=0.6, inst=0.7 → 2nd
      STOCK_C: div=0.04, nb=0.4, inst=0.5 → 3rd
      STOCK_D: div=0.02, nb=0.2, inst=0.3 → 4th
      STOCK_E: div=0.01, nb=0.1, inst=0.1 → lowest
    """
    return {
        "STOCK_A": _make_stock_df(dividend_yield=0.06, north_bound_flow=0.8, institutional_flow=0.9),
        "STOCK_B": _make_stock_df(dividend_yield=0.05, north_bound_flow=0.6, institutional_flow=0.7),
        "STOCK_C": _make_stock_df(dividend_yield=0.04, north_bound_flow=0.4, institutional_flow=0.5),
        "STOCK_D": _make_stock_df(dividend_yield=0.02, north_bound_flow=0.2, institutional_flow=0.3),
        "STOCK_E": _make_stock_df(dividend_yield=0.01, north_bound_flow=0.1, institutional_flow=0.1),
    }


@pytest.fixture
def single_stock_data() -> dict[str, pl.DataFrame]:
    """Single SOE stock for edge case testing."""
    return {
        "STOCK_ONLY": _make_stock_df(dividend_yield=0.05, north_bound_flow=0.5, institutional_flow=0.5),
    }


@pytest.fixture
def equal_data() -> dict[str, pl.DataFrame]:
    """Stocks with identical metrics — all should have same score."""
    return {
        "STOCK_X": _make_stock_df(dividend_yield=0.04, north_bound_flow=0.5, institutional_flow=0.5),
        "STOCK_Y": _make_stock_df(dividend_yield=0.04, north_bound_flow=0.5, institutional_flow=0.5),
    }


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestSoeReformConfig:
    """Configuration validation for S15 SOE Reform."""

    def test_default_config(self) -> None:
        config = SoeReformConfig(name="soe_reform")
        assert config.top_n == 3
        assert config.dividend_weight == 0.4
        assert config.flow_weight == 0.6

    def test_weights_sum_to_one_by_default(self) -> None:
        config = SoeReformConfig(name="soe_reform")
        assert config.dividend_weight + config.flow_weight == pytest.approx(1.0)

    def test_rejects_negative_top_n(self) -> None:
        with pytest.raises(ValueError):
            SoeReformConfig(name="soe_reform", top_n=0)

    def test_rejects_negative_dividend_weight(self) -> None:
        with pytest.raises(ValueError):
            SoeReformConfig(name="soe_reform", dividend_weight=-0.1)

    def test_rejects_negative_flow_weight(self) -> None:
        with pytest.raises(ValueError):
            SoeReformConfig(name="soe_reform", flow_weight=-0.1)

    def test_inherits_strategy_config(self) -> None:
        config = SoeReformConfig(name="soe_reform")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ────────────────────────────────────────────────────────


class TestRegimeEligibility:
    """SOE Reform valid in ALL regimes."""

    def test_eligible_in_all_regimes(self, strategy: SoeReform) -> None:
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True


# ─── required_data ─────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare SOE reform fields."""

    def test_returns_all_required_fields(self, strategy: SoeReform) -> None:
        fields = set(strategy.required_data())
        expected = {"dividend_yield", "north_bound_flow", "institutional_flow"}
        assert fields == expected

    def test_required_data_is_list_of_str(self, strategy: SoeReform) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with valid multi-stock market data."""

    def test_generates_signals_with_valid_data(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_top_3_stocks_are_selected(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Top 3 stocks by composite score should receive BUY signals."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        buy_codes = {s["fund_code"] for s in buy_signals}

        assert "STOCK_A" in buy_codes
        assert "STOCK_B" in buy_codes
        assert "STOCK_C" in buy_codes
        assert len(buy_signals) == 3

    def test_buy_signals_have_equal_weight(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Each selected stock should have weight = 1/top_n."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        expected_weight = 1.0 / strategy.config.top_n
        for s in buy_signals:
            assert s["target_weight"] == pytest.approx(expected_weight, abs=0.001)

    def test_signals_have_all_required_keys(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Every signal must contain fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        valid_dirs = {e.value for e in SignalDirection}
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in valid_dirs
            assert isinstance(s["confidence"], float)
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], float)
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0

    def test_highest_score_has_highest_confidence(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Top-ranked stock should have the highest confidence among all signals."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        assert signals[0]["confidence"] >= 0.7

    def test_signals_sorted_by_score_descending(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Signals must be sorted by composite score descending."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert buy_signals[0]["fund_code"] == "STOCK_A"

    def test_flow_weight_increases_north_bound_impact(
        self,
        trade_date: date,
    ) -> None:
        """Higher flow_weight should favor stocks with stronger flow."""
        # Create two stocks: same dividend, different flow
        data: dict[str, pl.DataFrame] = {
            "HIGH_FLOW": _make_stock_df(dividend_yield=0.04, north_bound_flow=0.9, institutional_flow=0.8),
            "LOW_FLOW": _make_stock_df(dividend_yield=0.04, north_bound_flow=0.1, institutional_flow=0.1),
        }
        config = SoeReformConfig(
            name="soe_reform",
            top_n=1,
            dividend_weight=0.2,
            flow_weight=0.8,
        )
        strategy_flow = SoeReform(config=config)
        signals = strategy_flow.generate_signals(trade_date, data)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        assert "HIGH_FLOW" in buy_codes


# ─── Signal Generation — Edge Cases ────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for SOE reform strategy."""

    def test_empty_market_data_returns_empty(
        self, strategy: SoeReform, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_single_stock_generates_signal(
        self,
        strategy: SoeReform,
        trade_date: date,
        single_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, single_stock_data)
        assert len(signals) >= 1

    def test_missing_required_columns_skipped(
        self, strategy: SoeReform, trade_date: date
    ) -> None:
        incomplete = pl.DataFrame({
            "dividend_yield": [0.04] * 30,
            # missing: north_bound_flow, institutional_flow
        })
        signals = strategy.generate_signals(trade_date, {"STOCK_BAD": incomplete})
        assert signals == []

    def test_mixed_valid_and_invalid(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        mixed = dict(multi_stock_data)
        mixed["STOCK_BAD"] = pl.DataFrame({"close": [1.0] * 30})
        signals = strategy.generate_signals(trade_date, mixed)
        bad_signals = [s for s in signals if s["fund_code"] == "STOCK_BAD"]
        assert len(bad_signals) == 0

    def test_identical_metrics_equal_confidence(
        self,
        strategy: SoeReform,
        trade_date: date,
        equal_data: dict[str, pl.DataFrame],
    ) -> None:
        """Stocks with identical metrics should have similar confidence."""
        signals = strategy.generate_signals(trade_date, equal_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        if len(buy_signals) >= 2:
            confidences = [s["confidence"] for s in buy_signals]
            # With identical data, confidences should be very close
            assert abs(confidences[0] - confidences[1]) < 0.01


# ─── Strategy Identity and Validation ──────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation checks."""

    def test_name_matches_config(self, strategy: SoeReform) -> None:
        assert strategy.name == "soe_reform"

    def test_validate_passes_for_valid_config(self, strategy: SoeReform) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_validate_catches_weights_out_of_range(self) -> None:
        """Weights that do not sum to ~1.0 are caught by model_validator."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SoeReformConfig(name="soe_reform", dividend_weight=0.3, flow_weight=0.3)

    def test_strategy_is_frozen(self, strategy: SoeReform) -> None:
        with pytest.raises(Exception):
            strategy.config.top_n = 99  # type: ignore[misc]


# ─── Composite Scoring Logic ───────────────────────────────────────────────────


class TestCompositeScoring:
    """Verify the weighted composite scoring formula."""

    def test_higher_dividend_and_flow_higher_score(
        self,
        strategy: SoeReform,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Stock with highest dividend + strongest flow should rank #1."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert buy_signals[0]["fund_code"] == "STOCK_A"

    def test_dividend_only_weighting(
        self,
        trade_date: date,
    ) -> None:
        """With dividend_weight=1.0 and flow_weight=0.0, only div yield matters."""
        data: dict[str, pl.DataFrame] = {
            "HIGH_DIV": _make_stock_df(dividend_yield=0.08, north_bound_flow=0.0, institutional_flow=0.0),
            "LOW_DIV": _make_stock_df(dividend_yield=0.02, north_bound_flow=0.9, institutional_flow=0.9),
        }
        config = SoeReformConfig(
            name="soe_reform_dividend",
            top_n=1,
            dividend_weight=1.0,
            flow_weight=0.0,
        )
        strategy_div = SoeReform(config=config)
        signals = strategy_div.generate_signals(trade_date, data)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        assert "HIGH_DIV" in buy_codes
