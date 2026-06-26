"""
Tests for src/core/strategy/mean_reversion/pe_pb_band.py — PE/PB Band strategy (S6).

The strategy uses historical PE/PB percentile bands for Chinese index funds
(CSI 300, CSI 500, ChiNext). When PE falls below the buy threshold AND PB
falls below its buy threshold, an ACCUMULATE signal fires. When PE exceeds
the sell threshold OR PB exceeds its sell threshold, a SELL signal fires.
Otherwise, HOLD.
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.mean_reversion.pe_pb_band import (
    DEFAULT_PB_BUY,
    DEFAULT_PB_SELL,
    DEFAULT_PE_BUY,
    DEFAULT_PE_SELL,
    PEPBBandStrategy,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def default_strategy() -> PEPBBandStrategy:
    """PE/PB Band strategy with CSI 300-optimised defaults."""
    return PEPBBandStrategy(
        config=StrategyConfig(name="pe_pb_band_csi300"),
        pe_buy_threshold=DEFAULT_PE_BUY,
        pe_sell_threshold=DEFAULT_PE_SELL,
        pb_buy_threshold=DEFAULT_PB_BUY,
        pb_sell_threshold=DEFAULT_PB_SELL,
    )


@pytest.fixture
def test_date() -> date:
    return date(2026, 6, 25)


def _make_market_data(fund_code: str, pe: float, pb: float) -> dict[str, pl.DataFrame]:
    """Helper: build a single-fund market_data dict with PE/PB columns."""
    return {
        fund_code: pl.DataFrame(
            {"pe": [pe], "pb": [pb]},
            schema={"pe": pl.Float64, "pb": pl.Float64},
        )
    }


# ─── Signal Generation ──────────────────────────────────────────────────────


class TestBuySignal:
    """ACCUMULATE signal when PE and PB are both below their buy thresholds."""

    def test_both_pe_and_pb_below_buy_thresholds(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=9.0 (< 10.5), PB=1.0 (< 1.2) → ACCUMULATE."""
        market = _make_market_data("510300", pe=9.0, pb=1.0)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        sig = signals[0]
        assert sig["fund_code"] == "510300"
        assert sig["direction"] == SignalDirection.ACCUMULATE
        assert 0.0 <= sig["confidence"] <= 1.0
        assert sig["target_weight"] > 0
        assert "PE" in sig["reason"]
        assert "PB" in sig["reason"]

    def test_pe_below_but_pb_above_buy_threshold_holds(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=9.0 (< 10.5) but PB=1.5 (≥ 1.2) → HOLD (both must be below)."""
        market = _make_market_data("510300", pe=9.0, pb=1.5)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        assert signals[0]["direction"] == SignalDirection.HOLD


class TestSellSignal:
    """SELL signal when PE or PB exceeds its sell threshold."""

    def test_pe_above_sell_threshold(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=15.0 (> 14.5) → SELL."""
        market = _make_market_data("510300", pe=15.0, pb=1.5)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        assert signals[0]["direction"] == SignalDirection.SELL

    def test_pb_above_sell_threshold(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PB=2.0 (> 1.7) → SELL, even though PE is mid-range."""
        market = _make_market_data("510300", pe=12.0, pb=2.0)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        assert signals[0]["direction"] == SignalDirection.SELL

    def test_both_pe_and_pb_extreme_sell(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=18.0 (> 14.5), PB=2.5 (> 1.7) → SELL with high confidence."""
        market = _make_market_data("510300", pe=18.0, pb=2.5)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        sig = signals[0]
        assert sig["direction"] == SignalDirection.SELL
        assert sig["confidence"] >= 0.7


class TestHoldSignal:
    """HOLD when PE and PB are in the neutral zone."""

    def test_mid_range_pe_and_pb(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=12.0 (10.5-14.5), PB=1.5 (1.2-1.7) → HOLD."""
        market = _make_market_data("510300", pe=12.0, pb=1.5)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        sig = signals[0]
        assert sig["direction"] == SignalDirection.HOLD
        assert 0.3 <= sig["confidence"] <= 0.7

    def test_pe_at_boundary_buy(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=10.5 (exactly at buy threshold) → HOLD (not below)."""
        market = _make_market_data("510300", pe=10.5, pb=1.0)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        assert signals[0]["direction"] == SignalDirection.HOLD

    def test_pe_at_boundary_sell(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """PE=14.5 (exactly at sell threshold) → HOLD (not above)."""
        market = _make_market_data("510300", pe=14.5, pb=1.5)
        signals = default_strategy.generate_signals(test_date, market)

        assert len(signals) == 1
        assert signals[0]["direction"] == SignalDirection.HOLD


# ─── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Graceful handling of edge conditions."""

    def test_empty_market_data_returns_empty_list(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """No funds in market_data → empty signal list."""
        signals = default_strategy.generate_signals(test_date, {})
        assert signals == []

    def test_missing_pe_column_skips_fund(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """Fund DataFrames without 'pe' column are skipped gracefully."""
        market = {
            "510300": pl.DataFrame({"pb": [1.0]}, schema={"pb": pl.Float64}),
        }
        signals = default_strategy.generate_signals(test_date, market)
        assert signals == []

    def test_missing_pb_column_skips_fund(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """Fund DataFrames without 'pb' column are skipped gracefully."""
        market = {
            "510300": pl.DataFrame({"pe": [12.0]}, schema={"pe": pl.Float64}),
        }
        signals = default_strategy.generate_signals(test_date, market)
        assert signals == []

    def test_multiple_funds_mixed_signals(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """Multiple funds: one BUY, one SELL, one HOLD."""
        market = {
            "510300": pl.DataFrame(
                {"pe": [9.0], "pb": [1.0]},  # BUY
                schema={"pe": pl.Float64, "pb": pl.Float64},
            ),
            "510500": pl.DataFrame(
                {"pe": [15.0], "pb": [1.5]},  # SELL
                schema={"pe": pl.Float64, "pb": pl.Float64},
            ),
            "159915": pl.DataFrame(
                {"pe": [12.0], "pb": [1.5]},  # HOLD
                schema={"pe": pl.Float64, "pb": pl.Float64},
            ),
        }
        signals = default_strategy.generate_signals(test_date, market)
        assert len(signals) == 3

        by_code = {s["fund_code"]: s for s in signals}
        assert by_code["510300"]["direction"] == SignalDirection.ACCUMULATE
        assert by_code["510500"]["direction"] == SignalDirection.SELL
        assert by_code["159915"]["direction"] == SignalDirection.HOLD

    def test_nan_pe_value_skips_fund(
        self, default_strategy: PEPBBandStrategy, test_date: date
    ) -> None:
        """NaN PE → fund skipped, no crash."""
        market = {
            "510300": pl.DataFrame(
                {"pe": [float("nan")], "pb": [1.0]},
                schema={"pe": pl.Float64, "pb": pl.Float64},
            ),
        }
        signals = default_strategy.generate_signals(test_date, market)
        assert signals == []


# ─── Config & Metadata ──────────────────────────────────────────────────────


class TestConfig:
    """Strategy metadata, validation, and eligibility."""

    def test_name_comes_from_config(self) -> None:
        strategy = PEPBBandStrategy(
            config=StrategyConfig(name="my_pe_pb"),
        )
        assert strategy.name == "my_pe_pb"

    def test_required_data_includes_pe_and_pb(
        self, default_strategy: PEPBBandStrategy
    ) -> None:
        fields = default_strategy.required_data()
        assert "pe" in fields
        assert "pb" in fields

    def test_eligible_in_all_regimes(
        self, default_strategy: PEPBBandStrategy
    ) -> None:
        for regime in MarketRegime:
            assert default_strategy.is_eligible(regime), f"Not eligible for {regime}"

    def test_validate_passes_with_valid_config(
        self, default_strategy: PEPBBandStrategy
    ) -> None:
        errors = default_strategy.validate()
        assert errors == []

    def test_validate_fails_when_buy_above_sell(self) -> None:
        """pe_buy_threshold >= pe_sell_threshold is invalid."""
        strategy = PEPBBandStrategy(
            config=StrategyConfig(name="bad"),
            pe_buy_threshold=15.0,
            pe_sell_threshold=14.0,
        )
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("PE buy" in e for e in errors)

    def test_validate_fails_when_pb_buy_above_sell(self) -> None:
        """pb_buy_threshold >= pb_sell_threshold is invalid."""
        strategy = PEPBBandStrategy(
            config=StrategyConfig(name="bad"),
            pb_buy_threshold=2.0,
            pb_sell_threshold=1.5,
        )
        errors = strategy.validate()
        assert len(errors) > 0
        assert any("PB buy" in e for e in errors)

    def test_custom_thresholds_override_defaults(self, test_date: date) -> None:
        """Strategy respects custom thresholds, not just defaults."""
        strategy = PEPBBandStrategy(
            config=StrategyConfig(name="tight"),
            pe_buy_threshold=8.0,
            pe_sell_threshold=12.0,
            pb_buy_threshold=0.8,
            pb_sell_threshold=1.3,
        )
        # PE=9.0 would be BUY with default (10.5) but HOLD with tight (8.0)
        # PE=9.0 > 8.0 → not below buy → only PB condition met → HOLD
        market = _make_market_data("510300", pe=9.0, pb=0.7)
        signals = strategy.generate_signals(test_date, market)
        assert signals[0]["direction"] == SignalDirection.HOLD

        # PE=11.0, PB=1.4 would be HOLD with default but SELL with tight
        market2 = _make_market_data("510300", pe=11.0, pb=1.4)
        signals2 = strategy.generate_signals(test_date, market2)
        assert signals2[0]["direction"] == SignalDirection.SELL


# ─── Immutability ───────────────────────────────────────────────────────────


class TestFrozenModel:
    """PEPBBandStrategy must be a frozen Pydantic model."""

    def test_cannot_mutate_thresholds(
        self, default_strategy: PEPBBandStrategy
    ) -> None:
        with pytest.raises(Exception):
            default_strategy.pe_buy_threshold = 5.0  # type: ignore[misc]

    def test_cannot_mutate_config(
        self, default_strategy: PEPBBandStrategy
    ) -> None:
        with pytest.raises(Exception):
            default_strategy.config.name = "hacked"  # type: ignore[misc]
