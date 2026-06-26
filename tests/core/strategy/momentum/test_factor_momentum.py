"""
Tests for Factor Momentum strategy (S1) — Ma, Liao & Jiang (2024).

Factor Momentum: 10 common factors ranked monthly by 1-month momentum.
Buy top-ranked factor group. Rebalance monthly.

Covers: signal generation, empty data, regime eligibility, signal structure,
required_data declaration, non-rebalance day behavior.
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.momentum.factor_momentum import FactorMomentum, FactorMomentumConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> FactorMomentum:
    """Default FactorMomentum strategy with top_n=3."""
    config = FactorMomentumConfig(
        name="factor_momentum",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        max_position_pct=0.20,
        min_holding_days=21,
        top_n=3,
        rebalance_day=1,
    )
    return FactorMomentum(config=config)


@pytest.fixture
def rebalance_date() -> date:
    """First day of month — triggers rebalance."""
    return date(2026, 6, 1)


@pytest.fixture
def non_rebalance_date() -> date:
    """Mid-month date — should not trigger rebalance."""
    return date(2026, 6, 15)


@pytest.fixture
def sample_market_data() -> dict[str, pl.DataFrame]:
    """5 funds with deterministic factor data for ranking tests.

    Expected ranking by momentum_1m (descending):
      000003: 0.15 (highest)
      000001: 0.12
      000002: 0.08
      000005: 0.02
      000004: -0.03 (lowest)

    With top_n=3 and min_positive_threshold=0.0, funds 000003, 000001, 000002
    should receive BUY/ACCUMULATE signals.
    """
    base_row: dict[str, object] = {
        "date": date(2026, 6, 1),
        "close": 1.5,
        "volume": 50000,
        "pe_ratio": 15.0,
        "pb_ratio": 2.0,
        "roe": 0.12,
        "momentum_3m": 0.05,
        "momentum_6m": 0.08,
        "volatility_1m": 0.02,
        "dividend_yield": 0.02,
    }

    def make_fund(momentum_1m: float, **overrides: object) -> pl.DataFrame:
        row = {**base_row, "momentum_1m": momentum_1m, **overrides}
        return pl.DataFrame([row])

    return {
        "000001": make_fund(0.12, roe=0.18),
        "000002": make_fund(0.08),
        "000003": make_fund(0.15, pe_ratio=10.0, pb_ratio=1.5),
        "000004": make_fund(-0.03),
        "000005": make_fund(0.02),
    }


# ─── Configuration ────────────────────────────────────────────────────────────


class TestFactorMomentumConfig:
    """Strategy configuration validation."""

    def test_default_config(self) -> None:
        """Default config has sane limits."""
        config = FactorMomentumConfig(
            name="factor_momentum",
        )
        assert config.top_n == 3
        assert config.rebalance_day == 1
        assert config.min_momentum_threshold == 0.0
        assert config.max_position_pct == 0.20

    def test_rejects_invalid_top_n(self) -> None:
        """top_n must be >= 1."""
        with pytest.raises(ValueError):
            FactorMomentumConfig(name="factor_momentum", top_n=0)

    def test_rejects_invalid_rebalance_day(self) -> None:
        """rebalance_day must be 1-28."""
        with pytest.raises(ValueError):
            FactorMomentumConfig(name="factor_momentum", rebalance_day=32)

    def test_inherits_strategy_config(self) -> None:
        """FactorMomentumConfig is a StrategyConfig subclass."""
        config = FactorMomentumConfig(name="factor_momentum")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ───────────────────────────────────────────────────────


class TestRegimeEligibility:
    """Factor momentum is only valid in TRENDING_UP and SIDEWAYS regimes."""

    def test_eligible_in_trending_up(self, strategy: FactorMomentum) -> None:
        """TRENDING_UP is an eligible regime."""
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True

    def test_eligible_in_sideways(self, strategy: FactorMomentum) -> None:
        """SIDEWAYS is an eligible regime."""
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is True

    def test_not_eligible_in_trending_down(self, strategy: FactorMomentum) -> None:
        """TRENDING_DOWN is NOT eligible — momentum breaks down in downtrends."""
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False

    def test_not_eligible_in_high_vol(self, strategy: FactorMomentum) -> None:
        """HIGH_VOL is NOT eligible — noisy signals."""
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False

    def test_not_eligible_in_crisis(self, strategy: FactorMomentum) -> None:
        """CRISIS is NOT eligible — momentum fails in crises."""
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare all factor fields the strategy needs."""

    def test_returns_all_required_fields(self, strategy: FactorMomentum) -> None:
        """All 10 required fields must be declared."""
        fields = set(strategy.required_data())
        expected = {
            "close",
            "volume",
            "pe_ratio",
            "pb_ratio",
            "roe",
            "momentum_1m",
            "momentum_3m",
            "momentum_6m",
            "volatility_1m",
            "dividend_yield",
        }
        assert fields == expected

    def test_required_data_is_list(self, strategy: FactorMomentum) -> None:
        """Return type must be list[str]."""
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ───────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with valid market data on rebalance day."""

    def test_generates_signals_on_rebalance_day(
        self, strategy: FactorMomentum, rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Signals should be generated when data is valid and date is rebalance day."""
        signals = strategy.generate_signals(rebalance_date, sample_market_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_top_n_signals_are_buy(
        self, strategy: FactorMomentum, rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Top-n ranked funds should receive BUY or ACCUMULATE signals."""
        signals = strategy.generate_signals(rebalance_date, sample_market_data)
        directions = {s["fund_code"]: s["direction"] for s in signals}

        # Top 3 by momentum_1m: 000003 (0.15), 000001 (0.12), 000002 (0.08)
        for code in ["000003", "000001", "000002"]:
            assert directions[code] in {SignalDirection.BUY.value, SignalDirection.ACCUMULATE.value}

    def test_lowest_momentum_is_trim_or_sell(
        self, strategy: FactorMomentum, rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Low-momentum funds should get TRIM or SELL signals."""
        signals = strategy.generate_signals(rebalance_date, sample_market_data)
        directions = {s["fund_code"]: s["direction"] for s in signals}

        # 000004 has -3% momentum — should be sold
        assert directions["000004"] in {SignalDirection.TRIM.value, SignalDirection.SELL.value}

    def test_signals_have_required_keys(
        self, strategy: FactorMomentum, rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Every signal dict must contain fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(rebalance_date, sample_market_data)
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in {e.value for e in SignalDirection}
            assert isinstance(s["confidence"], float)
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], float)
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0

    def test_target_weights_sum_to_one_or_less(
        self, strategy: FactorMomentum, rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Sum of target_weights across signals should not exceed 1.0."""
        signals = strategy.generate_signals(rebalance_date, sample_market_data)
        total_weight = sum(s["target_weight"] for s in signals)
        assert total_weight <= 1.0

    def test_positive_momentum_funds_have_higher_confidence(
        self, strategy: FactorMomentum, rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Higher momentum funds should have confidence >= 0.5."""
        signals = strategy.generate_signals(rebalance_date, sample_market_data)
        directions = {s["fund_code"]: s["confidence"] for s in signals}

        # Top 3 funds should have confidence >= 0.5
        for code in ["000003", "000001", "000002"]:
            assert directions[code] >= 0.5, f"Fund {code} confidence too low: {directions[code]}"


# ─── Signal Generation — Edge Cases ───────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge cases for signal generation."""

    def test_empty_data_returns_empty_signals(
        self, strategy: FactorMomentum, rebalance_date: date
    ) -> None:
        """Empty market_data should return empty signal list."""
        signals = strategy.generate_signals(rebalance_date, {})
        assert signals == []

    def test_single_fund_returns_signal(
        self, strategy: FactorMomentum, rebalance_date: date
    ) -> None:
        """Single fund should still generate a signal."""
        df = pl.DataFrame([{
            "date": date(2026, 6, 1),
            "close": 1.5,
            "volume": 50000,
            "pe_ratio": 15.0,
            "pb_ratio": 2.0,
            "roe": 0.12,
            "momentum_1m": 0.10,
            "momentum_3m": 0.05,
            "momentum_6m": 0.08,
            "volatility_1m": 0.02,
            "dividend_yield": 0.02,
        }])
        signals = strategy.generate_signals(rebalance_date, {"000001": df})
        assert len(signals) == 1
        assert signals[0]["fund_code"] == "000001"

    def test_non_rebalance_day_returns_empty(
        self, strategy: FactorMomentum, non_rebalance_date: date, sample_market_data: dict[str, pl.DataFrame]
    ) -> None:
        """Non-rebalance day should return empty signals (monthly rebalance only)."""
        signals = strategy.generate_signals(non_rebalance_date, sample_market_data)
        assert signals == []

    def test_all_negative_momentum_returns_no_buy(
        self, strategy: FactorMomentum, rebalance_date: date
    ) -> None:
        """When all funds have negative momentum, no BUY signals should be generated."""
        base: dict[str, object] = {
            "date": date(2026, 6, 1),
            "close": 1.5,
            "volume": 50000,
            "pe_ratio": 15.0,
            "pb_ratio": 2.0,
            "roe": 0.05,
            "momentum_3m": -0.05,
            "momentum_6m": -0.08,
            "volatility_1m": 0.03,
            "dividend_yield": 0.01,
        }
        data = {
            "000001": pl.DataFrame([{**base, "momentum_1m": -0.02}]),
            "000002": pl.DataFrame([{**base, "momentum_1m": -0.05}]),
            "000003": pl.DataFrame([{**base, "momentum_1m": -0.01}]),
        }
        signals = strategy.generate_signals(rebalance_date, data)
        buy_signals = [s for s in signals if s["direction"] in {SignalDirection.BUY.value, SignalDirection.ACCUMULATE.value}]
        assert len(buy_signals) == 0

    def test_missing_factor_columns_return_empty(
        self, strategy: FactorMomentum, rebalance_date: date
    ) -> None:
        """DataFrames missing required factor columns should return empty signals."""
        df = pl.DataFrame([{
            "date": date(2026, 6, 1),
            "close": 1.5,
            "volume": 50000,
        }])
        signals = strategy.generate_signals(rebalance_date, {"000001": df})
        assert signals == []


# ─── Strategy Name and Identity ───────────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation."""

    def test_name(self, strategy: FactorMomentum) -> None:
        """Strategy name should match config."""
        assert strategy.name == "factor_momentum"

    def test_validate_returns_empty(self, strategy: FactorMomentum) -> None:
        """Default strategy should pass validation with no errors."""
        errors = strategy.validate()
        assert errors == []
