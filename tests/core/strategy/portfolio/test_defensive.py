"""
Tests for 3-Layer Defensive Portfolio strategy (S18).

S18 — 3-Layer Defensive Portfolio: Fixed allocation of dividend ETF 40%,
bond ETF 40%, and gold ETF 20%. Annual rebalance.

Parameters: dividend_code="510880", bond_code="511260", gold_code="518880"
Eligible regimes: ALL.
Required data: ["close"].
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.portfolio.defensive import DefensivePortfolio, DefensivePortfolioConfig


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_price_df(n_rows: int = 60, price: float = 1.0) -> pl.DataFrame:
    """Build a polars DataFrame with close prices."""
    return pl.DataFrame({"close": [price] * n_rows})


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> DefensivePortfolio:
    """Default DefensivePortfolio with standard fund codes."""
    config = DefensivePortfolioConfig(
        name="defensive_portfolio",
        version="1.0.0",
        eligible_regimes={
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOL,
            MarketRegime.CRISIS,
        },
        dividend_code="510880",
        bond_code="511260",
        gold_code="518880",
    )
    return DefensivePortfolio(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2025, 6, 1)


@pytest.fixture
def full_market_data() -> dict[str, pl.DataFrame]:
    """All three ETFs present with price data."""
    return {
        "510880": _make_price_df(),
        "511260": _make_price_df(),
        "518880": _make_price_df(),
    }


@pytest.fixture
def partial_market_data() -> dict[str, pl.DataFrame]:
    """Only dividend and bond ETFs present — gold missing."""
    return {
        "510880": _make_price_df(),
        "511260": _make_price_df(),
    }


@pytest.fixture
def single_fund_data() -> dict[str, pl.DataFrame]:
    """Only one of the three ETFs present."""
    return {
        "510880": _make_price_df(),
    }


@pytest.fixture
def extra_fund_data() -> dict[str, pl.DataFrame]:
    """All three ETFs plus an extra fund."""
    return {
        "510880": _make_price_df(),
        "511260": _make_price_df(),
        "518880": _make_price_df(),
        "EXTRA": _make_price_df(),
    }


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestDefensivePortfolioConfig:
    """Configuration validation for S18 3-Layer Defensive Portfolio."""

    def test_default_config(self) -> None:
        config = DefensivePortfolioConfig(name="defensive_portfolio")
        assert config.dividend_code == "510880"
        assert config.bond_code == "511260"
        assert config.gold_code == "518880"
        assert config.dividend_weight == 0.40
        assert config.bond_weight == 0.40
        assert config.gold_weight == 0.20

    def test_weights_sum_to_one_by_default(self) -> None:
        config = DefensivePortfolioConfig(name="defensive_portfolio")
        total = config.dividend_weight + config.bond_weight + config.gold_weight
        assert total == pytest.approx(1.0)

    def test_rejects_negative_dividend_weight(self) -> None:
        with pytest.raises(ValueError):
            DefensivePortfolioConfig(name="defensive_portfolio", dividend_weight=-0.1)

    def test_rejects_weight_not_sum_to_one(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DefensivePortfolioConfig(name="defensive_portfolio", dividend_weight=0.5, bond_weight=0.5, gold_weight=0.5)

    def test_rejects_empty_dividend_code(self) -> None:
        with pytest.raises(ValueError):
            DefensivePortfolioConfig(name="defensive_portfolio", dividend_code="")

    def test_inherits_strategy_config(self) -> None:
        config = DefensivePortfolioConfig(name="defensive_portfolio")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ────────────────────────────────────────────────────────


class TestRegimeEligibility:
    """DefensivePortfolio valid in ALL regimes."""

    def test_eligible_in_all_regimes(self, strategy: DefensivePortfolio) -> None:
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True


# ─── required_data ─────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare close prices."""

    def test_returns_close(self, strategy: DefensivePortfolio) -> None:
        assert strategy.required_data() == ["close"]

    def test_required_data_is_list_of_str(self, strategy: DefensivePortfolio) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with complete ETF market data."""

    def test_generates_three_signals(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        assert len(signals) == 3

    def test_all_three_are_buy_signals(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        for s in signals:
            assert s["direction"] == SignalDirection.BUY.value

    def test_fixed_weights_are_correct(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Dividend=40%, Bond=40%, Gold=20%."""
        signals = strategy.generate_signals(trade_date, full_market_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        assert weights["510880"] == pytest.approx(0.40, abs=0.001)
        assert weights["511260"] == pytest.approx(0.40, abs=0.001)
        assert weights["518880"] == pytest.approx(0.20, abs=0.001)

    def test_signals_have_all_required_keys(
        self,
        strategy: DefensivePortfolio,
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

    def test_total_weight_equals_one(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        total = sum(s["target_weight"] for s in signals)
        assert total == pytest.approx(1.0)

    def test_signals_contain_all_three_codes(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        codes = {s["fund_code"] for s in signals}
        assert codes == {"510880", "511260", "518880"}

    def test_extra_funds_ignored(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        extra_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        """Extra funds in market_data are not included in signals."""
        signals = strategy.generate_signals(trade_date, extra_fund_data)
        codes = {s["fund_code"] for s in signals}
        assert "EXTRA" not in codes
        assert codes == {"510880", "511260", "518880"}


# ─── Signal Generation — Edge Cases ────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for defensive portfolio strategy."""

    def test_empty_market_data_returns_empty(
        self, strategy: DefensivePortfolio, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_partial_data_some_missing(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        partial_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Missing gold ETF — generates signals for available funds only."""
        signals = strategy.generate_signals(trade_date, partial_market_data)
        assert len(signals) > 0
        codes = {s["fund_code"] for s in signals}
        assert "518880" not in codes

    def test_single_fund_generates_signal(
        self,
        strategy: DefensivePortfolio,
        trade_date: date,
        single_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, single_fund_data)
        assert len(signals) == 1


# ─── Strategy Identity and Validation ──────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation checks."""

    def test_name_matches_config(self, strategy: DefensivePortfolio) -> None:
        assert strategy.name == "defensive_portfolio"

    def test_validate_passes_for_valid_config(self, strategy: DefensivePortfolio) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_validate_catches_bad_weights(self) -> None:
        """Weights that do not sum to 1.0 are caught by model_validator."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DefensivePortfolioConfig(
                name="defensive_portfolio",
                dividend_weight=0.5,
                bond_weight=0.5,
                gold_weight=0.5,
            )

    def test_strategy_is_frozen(self, strategy: DefensivePortfolio) -> None:
        with pytest.raises(Exception):
            strategy.config.dividend_weight = 0.5  # type: ignore[misc]


# ─── Custom Allocation Logic ───────────────────────────────────────────────────


class TestCustomAllocation:
    """Verify custom weight configurations work correctly."""

    def test_custom_weights(
        self,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Custom weights should be reflected in signals."""
        config = DefensivePortfolioConfig(
            name="defensive_custom",
            dividend_weight=0.50,
            bond_weight=0.30,
            gold_weight=0.20,
        )
        strategy = DefensivePortfolio(config=config)
        signals = strategy.generate_signals(trade_date, full_market_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        assert weights["510880"] == pytest.approx(0.50, abs=0.001)
        assert weights["511260"] == pytest.approx(0.30, abs=0.001)
        assert weights["518880"] == pytest.approx(0.20, abs=0.001)

    def test_custom_fund_codes(
        self,
        trade_date: date,
    ) -> None:
        """Custom fund codes should appear in signals."""
        data = {
            "DIV_ETF": _make_price_df(),
            "BOND_ETF": _make_price_df(),
            "GOLD_ETF": _make_price_df(),
        }
        config = DefensivePortfolioConfig(
            name="defensive_custom_codes",
            dividend_code="DIV_ETF",
            bond_code="BOND_ETF",
            gold_code="GOLD_ETF",
        )
        strategy = DefensivePortfolio(config=config)
        signals = strategy.generate_signals(trade_date, data)
        codes = {s["fund_code"] for s in signals}
        assert codes == {"DIV_ETF", "BOND_ETF", "GOLD_ETF"}
