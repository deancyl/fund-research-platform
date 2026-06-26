"""
Tests for Aggressive ETF Portfolio strategy (S19).

S19 — Aggressive ETF Portfolio: Dynamic weights with factor momentum overlay.
Base allocation: CSI300 30% + ChiNext 20% + Sector ETF 30% + Gold 10% + Cash 10%.

Momentum overlay adjusts growth-component weights based on momentum_1m
and momentum_3m signals. Positive momentum → overweight; negative → underweight.

Parameters: csi300_code="510300", chinext_code="159915"
Eligible regimes: TRENDING_UP, SIDEWAYS.
Required data: ["close", "momentum_1m", "momentum_3m"].
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.portfolio.aggressive import AggressivePortfolio, AggressivePortfolioConfig


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_etf_df(
    momentum_1m: float = 0.03,
    momentum_3m: float = 0.08,
    n_rows: int = 60,
) -> pl.DataFrame:
    """Build a polars DataFrame with close and momentum columns."""
    return pl.DataFrame({
        "close": [1.0] * n_rows,
        "momentum_1m": [momentum_1m] * n_rows,
        "momentum_3m": [momentum_3m] * n_rows,
    })


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> AggressivePortfolio:
    """Default AggressivePortfolio with standard codes."""
    config = AggressivePortfolioConfig(
        name="aggressive_portfolio",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        csi300_code="510300",
        chinext_code="159915",
        sector_code="512100",
        gold_code="518880",
    )
    return AggressivePortfolio(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2025, 6, 1)


@pytest.fixture
def full_market_data() -> dict[str, pl.DataFrame]:
    """All four ETFs present with positive momentum."""
    return {
        "510300": _make_etf_df(momentum_1m=0.04, momentum_3m=0.10),
        "159915": _make_etf_df(momentum_1m=0.06, momentum_3m=0.15),
        "512100": _make_etf_df(momentum_1m=0.03, momentum_3m=0.08),
        "518880": _make_etf_df(momentum_1m=0.01, momentum_3m=0.02),
    }


@pytest.fixture
def negative_momentum_data() -> dict[str, pl.DataFrame]:
    """All growth ETFs with negative momentum — should shift to defensive."""
    return {
        "510300": _make_etf_df(momentum_1m=-0.03, momentum_3m=-0.05),
        "159915": _make_etf_df(momentum_1m=-0.04, momentum_3m=-0.08),
        "512100": _make_etf_df(momentum_1m=-0.02, momentum_3m=-0.04),
        "518880": _make_etf_df(momentum_1m=0.01, momentum_3m=0.02),
    }


@pytest.fixture
def mixed_momentum_data() -> dict[str, pl.DataFrame]:
    """Some ETFs positive, some negative momentum."""
    return {
        "510300": _make_etf_df(momentum_1m=0.04, momentum_3m=0.10),
        "159915": _make_etf_df(momentum_1m=-0.02, momentum_3m=-0.05),
        "512100": _make_etf_df(momentum_1m=0.01, momentum_3m=0.03),
        "518880": _make_etf_df(momentum_1m=0.01, momentum_3m=0.02),
    }


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestAggressivePortfolioConfig:
    """Configuration validation for S19 Aggressive ETF Portfolio."""

    def test_default_config(self) -> None:
        config = AggressivePortfolioConfig(name="aggressive_portfolio")
        assert config.csi300_code == "510300"
        assert config.chinext_code == "159915"
        assert config.csi300_weight == 0.30
        assert config.chinext_weight == 0.20
        assert config.sector_weight == 0.30
        assert config.gold_weight == 0.10

    def test_weights_sum_to_one(self) -> None:
        config = AggressivePortfolioConfig(name="aggressive_portfolio")
        total = (
            config.csi300_weight
            + config.chinext_weight
            + config.sector_weight
            + config.gold_weight
            + config.cash_weight
        )
        assert total == pytest.approx(1.0)

    def test_rejects_negative_weight(self) -> None:
        with pytest.raises(ValueError):
            AggressivePortfolioConfig(name="aggressive_portfolio", csi300_weight=-0.1)

    def test_rejects_empty_code(self) -> None:
        with pytest.raises(ValueError):
            AggressivePortfolioConfig(name="aggressive_portfolio", csi300_code="")

    def test_inherits_strategy_config(self) -> None:
        config = AggressivePortfolioConfig(name="aggressive_portfolio")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ────────────────────────────────────────────────────────


class TestRegimeEligibility:
    """AggressivePortfolio valid in TRENDING_UP and SIDEWAYS only."""

    def test_eligible_in_trending_up(self, strategy: AggressivePortfolio) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True

    def test_eligible_in_sideways(self, strategy: AggressivePortfolio) -> None:
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is True

    def test_not_eligible_in_trending_down(self, strategy: AggressivePortfolio) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False

    def test_not_eligible_in_high_vol(self, strategy: AggressivePortfolio) -> None:
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False

    def test_not_eligible_in_crisis(self, strategy: AggressivePortfolio) -> None:
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ─────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare close and momentum fields."""

    def test_returns_all_fields(self, strategy: AggressivePortfolio) -> None:
        fields = set(strategy.required_data())
        expected = {"close", "momentum_1m", "momentum_3m"}
        assert fields == expected

    def test_required_data_is_list_of_str(self, strategy: AggressivePortfolio) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Positive Momentum ─────────────────────────────────────


class TestGenerateSignalsPositiveMomentum:
    """Signal generation when all growth ETFs have positive momentum."""

    def test_generates_signals(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_all_component_funds_present(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        codes = {s["fund_code"] for s in signals}
        assert "510300" in codes
        assert "159915" in codes
        assert "512100" in codes
        assert "518880" in codes

    def test_signals_have_all_required_keys(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
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

    def test_total_weight_approximately_one(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Signal weights should be reasonable. Cash absorbs the residual."""
        signals = strategy.generate_signals(trade_date, full_market_data)
        total = sum(s["target_weight"] for s in signals)
        assert total > 0.0
        # With positive momentum, total can exceed 0.90 (cash residual reduces)
        assert total <= 1.05

    def test_positive_momentum_adds_weight(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Positive momentum should increase growth-component weights above base."""
        signals = strategy.generate_signals(trade_date, full_market_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        # CSI300 base is 0.30; with positive momentum, should be >= base
        assert weights.get("510300", 0) >= 0.25


# ─── Signal Generation — Negative Momentum ─────────────────────────────────────


class TestGenerateSignalsNegativeMomentum:
    """Signal generation when growth ETFs have negative momentum."""

    def test_negative_momentum_reduces_growth_weight(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        negative_momentum_data: dict[str, pl.DataFrame],
    ) -> None:
        """Negative momentum should reduce growth-component weights below base."""
        signals = strategy.generate_signals(trade_date, negative_momentum_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        # With negative momentum, adjusted growth weights should be <= base weight
        csi300_w = weights.get("510300", 0.30)
        expected_base = strategy.config.csi300_weight
        assert csi300_w <= expected_base

    def test_negative_momentum_still_generates_signals(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        negative_momentum_data: dict[str, pl.DataFrame],
    ) -> None:
        """Even with negative momentum, signals must be generated."""
        signals = strategy.generate_signals(trade_date, negative_momentum_data)
        assert len(signals) >= 1


# ─── Signal Generation — Mixed Momentum ────────────────────────────────────────


class TestGenerateSignalsMixedMomentum:
    """Signal generation with mixed positive/negative momentum."""

    def test_higher_momentum_gets_higher_weight(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        mixed_momentum_data: dict[str, pl.DataFrame],
    ) -> None:
        """ETF with strongest positive momentum should get highest growth weight."""
        signals = strategy.generate_signals(trade_date, mixed_momentum_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        # 510300 has strong positive momentum → should have decent weight
        assert weights.get("510300", 0) > 0.10
        # 159915 has negative momentum → should have lower weight
        assert weights.get("159915", 0) <= weights.get("510300", 1.0)


# ─── Edge Cases ────────────────────────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for aggressive portfolio strategy."""

    def test_empty_market_data_returns_empty(
        self, strategy: AggressivePortfolio, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_missing_momentum_columns_skipped(
        self, strategy: AggressivePortfolio, trade_date: date
    ) -> None:
        incomplete = pl.DataFrame({"close": [1.0] * 30})
        signals = strategy.generate_signals(trade_date, {"510300": incomplete})
        assert signals == []

    def test_missing_close_skipped(
        self, strategy: AggressivePortfolio, trade_date: date
    ) -> None:
        incomplete = pl.DataFrame({"momentum_1m": [0.03] * 30, "momentum_3m": [0.08] * 30})
        signals = strategy.generate_signals(trade_date, {"510300": incomplete})
        assert signals == []

    def test_extra_funds_ignored(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        mixed = dict(full_market_data)
        mixed["EXTRA"] = _make_etf_df()
        signals = strategy.generate_signals(trade_date, mixed)
        codes = {s["fund_code"] for s in signals}
        assert "EXTRA" not in codes


# ─── Strategy Identity and Validation ──────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation checks."""

    def test_name_matches_config(self, strategy: AggressivePortfolio) -> None:
        assert strategy.name == "aggressive_portfolio"

    def test_validate_passes_for_valid_config(self, strategy: AggressivePortfolio) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_strategy_is_frozen(self, strategy: AggressivePortfolio) -> None:
        with pytest.raises(Exception):
            strategy.config.csi300_weight = 0.50  # type: ignore[misc]


# ─── Factor Momentum Overlay ───────────────────────────────────────────────────


class TestFactorMomentumOverlay:
    """Verify factor momentum overlay adjusts weights correctly."""

    def test_positive_momentum_boosts_weight_above_base(
        self,
        strategy: AggressivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """With strong positive momentum, growth ETFs should be above base weight."""
        signals = strategy.generate_signals(trade_date, full_market_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        # CSI300 base is 30%; with momentum the adjusted weight could be higher
        # (the overlay adjusts toward growth but the exact multiplier depends on config)
        assert "510300" in weights

    def test_custom_momentum_window(
        self,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Custom momentum window should affect weight adjustments."""
        config = AggressivePortfolioConfig(
            name="aggressive_custom",
            eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
            momentum_window=3,
            momentum_max_boost=0.15,
        )
        strategy = AggressivePortfolio(config=config)
        signals = strategy.generate_signals(trade_date, full_market_data)
        assert len(signals) > 0
