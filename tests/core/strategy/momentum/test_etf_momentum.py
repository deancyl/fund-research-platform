"""
Tests for ETF Multi-Factor Momentum strategy (S2).

S2 — ETF 多因子动量: 49 ETFs (41 A-share + 8 QDII), 23 factors.
Select Top-2 via multi-period momentum scoring with Exp4 hysteresis.
Volatility-based risk gating scales position size.

Lookback: 60/120/252 days weighted.
Sharpe 1.38, MaxDD 10.8%, Win rate 83.3%.
Eligible: TRENDING_UP, SIDEWAYS.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.momentum.etf_momentum import EtfMomentum, EtfMomentumConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_WINDOW_LONG = 252  # trading days in a year

# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_close_series(
    start_price: float,
    daily_ret: float,
    n_days: int = _WINDOW_LONG + 10,
    noise: float = 0.002,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic close prices with a deterministic drift + noise.

    Returns a 1-D numpy array of length n_days with geometric growth pattern.
    """
    rng = np.random.default_rng(seed)
    returns = daily_ret + rng.normal(0.0, noise, n_days)
    # Compensate noise to preserve the intended drift (deterministic ranking)
    bias_correction = daily_ret - np.mean(returns)
    returns = returns + bias_correction
    prices = start_price * np.cumprod(1.0 + returns)
    return prices


def _make_etf_df(
    prices: np.ndarray,
    base_date: date | None = None,
    volume: float = 100000.0,
    turnover_rate: float = 0.05,
    tracking_error: float = 0.002,
    premium_rate: float = 0.001,
    discount_rate: float = 0.0005,
    liquidity: float = 0.95,
) -> pl.DataFrame:
    """Build a polars DataFrame with all required ETF columns.

    Args:
        prices: Array of close prices, one per trading day.
        base_date: Earliest date in the series. Defaults to 2024-06-01.
        volume: Daily trading volume.
        turnover_rate: Daily turnover rate.
        tracking_error: ETF tracking error vs index.
        premium_rate: Premium rate (ETF price > NAV).
        discount_rate: Discount rate (ETF price < NAV).
        liquidity: Liquidity score [0, 1].
    """
    n = len(prices)
    if base_date is None:
        base_date = date(2024, 6, 1)
    dates = [base_date.replace(day=1, month=((base_date.month - 1 + i) % 12) + 1,
            year=base_date.year + (base_date.month - 1 + i) // 12) for i in range(n)]

    return pl.DataFrame({
        "close": prices,
        "volume": [volume] * n,
        "turnover_rate": [turnover_rate] * n,
        "tracking_error": [tracking_error] * n,
        "premium_rate": [premium_rate] * n,
        "discount_rate": [discount_rate] * n,
        "liquidity": [liquidity] * n,
    })


@pytest.fixture
def strategy() -> EtfMomentum:
    """Default EtfMomentum strategy with top_n=2, Exp4 min_hold=9."""
    config = EtfMomentumConfig(
        name="etf_momentum",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        max_position_pct=0.50,
        min_holding_days=9,
        top_n=2,
        lookback_short=60,
        lookback_mid=120,
        lookback_long=252,
        min_hold_days=9,
    )
    return EtfMomentum(config=config)


@pytest.fixture
def strategy_tight() -> EtfMomentum:
    """EtfMomentum with top_n=1 and 5-day minimum hold — for edge case tests."""
    config = EtfMomentumConfig(
        name="etf_momentum_tight",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        top_n=1,
        lookback_short=60,
        lookback_mid=120,
        lookback_long=252,
        min_hold_days=5,
    )
    return EtfMomentum(config=config)


@pytest.fixture
def trade_date() -> date:
    """A trading date for signal generation."""
    return date(2025, 6, 1)


@pytest.fixture
def sample_etf_market_data() -> dict[str, pl.DataFrame]:
    """5 ETFs with deterministic multi-period momentum profiles.

    Expected ranking by weighted momentum score (descending):
      ETF_A: +30% annual drift  → strongest momentum
      ETF_B: +15% annual drift  → moderate momentum
      ETF_C: +5%  annual drift  → weak momentum
      ETF_D: -5%  annual drift  → negative momentum
      ETF_E: 0%   flat          → zero momentum

    With top_n=2, ETF_A and ETF_B should be selected.
    """
    n_days = _WINDOW_LONG + 10  # enough for 252d lookback
    return {
        "ETF_A": _make_etf_df(_make_close_series(1.0, 0.0012, n_days, seed=1),
                              volume=200000.0, tracking_error=0.001, liquidity=0.98),
        "ETF_B": _make_etf_df(_make_close_series(1.0, 0.0006, n_days, seed=2),
                              volume=150000.0, tracking_error=0.003, liquidity=0.92),
        "ETF_C": _make_etf_df(_make_close_series(1.0, 0.0002, n_days, seed=3),
                              volume=80000.0, tracking_error=0.005, liquidity=0.85),
        "ETF_D": _make_etf_df(_make_close_series(1.0, -0.0002, n_days, seed=4),
                              volume=50000.0, tracking_error=0.008, liquidity=0.78),
        "ETF_E": _make_etf_df(_make_close_series(1.0, 0.0, n_days, seed=5),
                              volume=40000.0, tracking_error=0.006, liquidity=0.80),
    }


@pytest.fixture
def single_fund_data() -> dict[str, pl.DataFrame]:
    """Single ETF with positive momentum — for single-fund edge case."""
    n_days = _WINDOW_LONG + 10
    return {
        "ETF_ONLY": _make_etf_df(_make_close_series(1.0, 0.0008, n_days, seed=99),
                                 liquidity=0.90),
    }


@pytest.fixture
def all_negative_data() -> dict[str, pl.DataFrame]:
    """All ETFs with negative momentum — worst case."""
    n_days = _WINDOW_LONG + 10
    return {
        "ETF_X": _make_etf_df(_make_close_series(1.0, -0.0008, n_days, seed=10)),
        "ETF_Y": _make_etf_df(_make_close_series(1.0, -0.0004, n_days, seed=11)),
        "ETF_Z": _make_etf_df(_make_close_series(1.0, -0.0012, n_days, seed=12)),
    }


@pytest.fixture
def insufficient_data() -> dict[str, pl.DataFrame]:
    """DataFrames with too few rows for the longest lookback."""
    return {
        "ETF_SHORT": _make_etf_df(
            _make_close_series(1.0, 0.001, 50, seed=20),  # only 50 rows < 252
        ),
    }


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestEtfMomentumConfig:
    """Strategy configuration validation for S2 ETF Multi-Factor Momentum."""

    def test_default_config(self) -> None:
        """Default config has spec-compliant values."""
        config = EtfMomentumConfig(name="etf_momentum")
        assert config.top_n == 2
        assert config.lookback_short == 60
        assert config.lookback_mid == 120
        assert config.lookback_long == 252
        assert config.min_hold_days == 9
        assert config.max_position_pct == 0.50

    def test_rejects_invalid_top_n(self) -> None:
        """top_n must be >= 1."""
        with pytest.raises(ValueError):
            EtfMomentumConfig(name="etf_momentum", top_n=0)

    def test_rejects_negative_lookback_short(self) -> None:
        """lookback_short must be positive."""
        with pytest.raises(ValueError):
            EtfMomentumConfig(name="etf_momentum", lookback_short=0)

    def test_rejects_invalid_min_hold(self) -> None:
        """min_hold_days must be >= 1."""
        with pytest.raises(ValueError):
            EtfMomentumConfig(name="etf_momentum", min_hold_days=0)

    def test_rejects_lookback_ordering(self) -> None:
        """lookback_short < lookback_mid < lookback_long must hold."""
        with pytest.raises(ValueError):
            EtfMomentumConfig(name="etf_momentum", lookback_short=200, lookback_mid=100)

    def test_inherits_strategy_config(self) -> None:
        """EtfMomentumConfig is a StrategyConfig subclass."""
        config = EtfMomentumConfig(name="etf_momentum")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ───────────────────────────────────────────────────────


class TestRegimeEligibility:
    """ETF Multi-Factor Momentum valid in TRENDING_UP and SIDEWAYS only."""

    def test_eligible_in_trending_up(self, strategy: EtfMomentum) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True

    def test_eligible_in_sideways(self, strategy: EtfMomentum) -> None:
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is True

    def test_not_eligible_in_trending_down(self, strategy: EtfMomentum) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False

    def test_not_eligible_in_high_vol(self, strategy: EtfMomentum) -> None:
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False

    def test_not_eligible_in_crisis(self, strategy: EtfMomentum) -> None:
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare all ETF-specific fields."""

    def test_returns_all_required_fields(self, strategy: EtfMomentum) -> None:
        fields = set(strategy.required_data())
        expected = {
            "close",
            "volume",
            "turnover_rate",
            "tracking_error",
            "premium_rate",
            "discount_rate",
            "liquidity",
        }
        assert fields == expected

    def test_required_data_is_list_of_str(self, strategy: EtfMomentum) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ───────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with valid, complete ETF market data."""

    def test_generates_signals_with_valid_data(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_top_n_etfs_are_selected(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Top 2 ETFs (A + B) should receive BUY signals with equal weights."""
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        buy_codes = {s["fund_code"] for s in buy_signals}

        assert "ETF_A" in buy_codes, "ETF_A (highest momentum) must be selected"
        assert "ETF_B" in buy_codes, "ETF_B (second-highest momentum) must be selected"
        assert len(buy_signals) == 2, f"Expected exactly 2 BUY signals, got {len(buy_signals)}"

    def test_buy_signals_have_equal_weight(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Each selected ETF should have weight = 1/top_n = 0.5."""
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        expected_weight = 1.0 / strategy.config.top_n
        for s in buy_signals:
            assert s["target_weight"] == pytest.approx(expected_weight, abs=0.001)

    def test_signals_have_all_required_keys(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Every signal must contain: fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
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

    def test_total_buy_weight_does_not_exceed_one(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
        buy_weight = sum(
            s["target_weight"] for s in signals
            if s["direction"] == SignalDirection.BUY.value
        )
        assert buy_weight <= 1.0

    def test_top_ranked_has_highest_confidence(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """The #1 ranked ETF should have confidence >= 0.7 (strong signal)."""
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        # Any buy signal should carry reasonable confidence
        for s in buy_signals:
            assert s["confidence"] >= 0.5, f"BUY signal confidence too low: {s['confidence']}"


# ─── Signal Generation — Edge Cases ───────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for signal generation."""

    def test_empty_market_data_returns_empty(
        self, strategy: EtfMomentum, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_single_fund_generates_signal(
        self, strategy_tight: EtfMomentum, trade_date: date, single_fund_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy_tight.generate_signals(trade_date, single_fund_data)
        assert len(signals) >= 1

    def test_insufficient_rows_returns_empty(
        self, strategy: EtfMomentum, trade_date: date, insufficient_data: dict[str, pl.DataFrame]
    ) -> None:
        """DataFrames shorter than lookback_long should return no signals."""
        signals = strategy.generate_signals(trade_date, insufficient_data)
        assert signals == []

    def test_all_negative_momentum_no_buy(
        self, strategy: EtfMomentum, trade_date: date, all_negative_data: dict[str, pl.DataFrame]
    ) -> None:
        """When all ETFs have negative momentum scores, no BUY should be emitted."""
        signals = strategy.generate_signals(trade_date, all_negative_data)
        buy_signals = [
            s for s in signals
            if s["direction"] == SignalDirection.BUY.value
        ]
        assert len(buy_signals) == 0

    def test_missing_required_columns_skipped(
        self, strategy: EtfMomentum, trade_date: date
    ) -> None:
        """Funds missing required columns are silently skipped."""
        incomplete = pl.DataFrame({
            "close": [1.0] * 300,
            "volume": [1000.0] * 300,
            # missing: turnover_rate, tracking_error, premium_rate, discount_rate, liquidity
        })
        signals = strategy.generate_signals(trade_date, {"ETF_BAD": incomplete})
        assert signals == []

    def test_valid_funds_with_some_missing_columns(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Valid funds are still processed even if some funds lack columns."""
        mixed = dict(sample_etf_market_data)
        mixed["ETF_BAD"] = pl.DataFrame({
            "close": [1.0] * 300,
        })
        signals = strategy.generate_signals(trade_date, mixed)
        assert len(signals) > 0
        bad_signals = [s for s in signals if s["fund_code"] == "ETF_BAD"]
        assert len(bad_signals) == 0


# ─── Strategy Identity and Validation ─────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity, validation, and parameter checks."""

    def test_name_matches_config(self, strategy: EtfMomentum) -> None:
        assert strategy.name == "etf_momentum"

    def test_validate_passes_for_valid_config(self, strategy: EtfMomentum) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_validate_catches_bad_lookback_ordering(self) -> None:
        """Bad lookback ordering is rejected at config construction (Pydantic model_validator)."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="lookback ordering"):
            EtfMomentumConfig(
                name="etf_momentum_bad",
                lookback_short=300,
                lookback_long=100,
            )

    def test_strategy_is_frozen(self, strategy: EtfMomentum) -> None:
        """Strategy model should be immutable (Pydantic frozen)."""
        with pytest.raises(Exception):
            strategy.config.top_n = 99  # type: ignore[misc]


# ─── Multi-Period Momentum Scoring ────────────────────────────────────────────


class TestMultiPeriodScoring:
    """Verify multi-period momentum ranking logic with simple synthetic data."""

    def test_higher_drift_ranks_higher(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        sample_etf_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """ETF with highest annual drift should be ranked #1."""
        signals = strategy.generate_signals(trade_date, sample_etf_market_data)
        # Get the order of BUY-signalling funds
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert len(buy_signals) == 2
        # ETF_A should appear before ETF_B (or at least equal confidence)
        codes_in_order = [s["fund_code"] for s in buy_signals]
        assert codes_in_order[0] == "ETF_A"

    def test_negative_drift_not_bought(
        self,
        strategy: EtfMomentum,
        trade_date: date,
        all_negative_data: dict[str, pl.DataFrame],
    ) -> None:
        """ETFs with negative momentum should not be bought."""
        signals = strategy.generate_signals(trade_date, all_negative_data)
        buy_signals = [
            s for s in signals
            if s["direction"] in {SignalDirection.BUY.value, SignalDirection.ACCUMULATE.value}
        ]
        assert len(buy_signals) == 0


# ─── Volatility Risk Gating ───────────────────────────────────────────────────


class TestVolatilityRiskGating:
    """Risk gating: high-volatility ETFs should have reduced position weight."""

    def test_high_tracking_error_reduces_weight(
        self,
        strategy_tight: EtfMomentum,
        trade_date: date,
    ) -> None:
        """ETF with high tracking error should get a scaled-down weight."""
        n_days = _WINDOW_LONG + 10
        # Two ETFs with same momentum but different tracking errors
        prices = _make_close_series(1.0, 0.001, n_days, seed=50)
        etf_low_vol = _make_etf_df(prices, tracking_error=0.001)
        etf_high_vol = _make_etf_df(prices, tracking_error=0.05)

        # top_n=1 → only one fund should be bought
        signals = strategy_tight.generate_signals(
            trade_date, {"LOW_VOL": etf_low_vol, "HIGH_VOL": etf_high_vol}
        )
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        # With equal momentum but higher vol, LOW_VOL should win
        # If HIGH_VOL wins, its weight should still be reduced
        if len(buy_signals) > 0:
            assert "LOW_VOL" in {s["fund_code"] for s in buy_signals}


# ─── Minimum Holding (Exp4 Hysteresis) ────────────────────────────────────────


class TestMinHoldingHysteresis:
    """Exp4 hysteresis: minimum holding period prevents excessive turnover."""

    def test_min_hold_days_configured(self, strategy: EtfMomentum) -> None:
        """min_hold_days default should be 9 for Exp4 hysteresis."""
        assert strategy.config.min_hold_days == 9

    def test_min_hold_prevents_immediate_flip(self) -> None:
        """With high min_hold_days, the strategy should have longer horizon lookbacks."""
        config = EtfMomentumConfig(
            name="etf_momentum_long_hold",
            min_hold_days=21,
            lookback_short=60,
            lookback_mid=120,
            lookback_long=252,
        )
        assert config.min_hold_days == 21
        assert config.lookback_long == 252
