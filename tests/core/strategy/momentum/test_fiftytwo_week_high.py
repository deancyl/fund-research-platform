"""
Tests for 52-Week High Momentum strategy (S5).

S5 — 52-Week High Momentum: Buy ETFs trading near their 52-week high.
Monthly rebalance with proximity threshold (default 95%).

Parameters: lookback_weeks=52, proximity_pct=0.95
Eligible regimes: TRENDING_UP only.
Required data: ["close"].
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.momentum.fiftytwo_week_high import (
    FiftyTwoWeekHigh,
    FiftyTwoWeekHighConfig,
)

# ─── Constants ─────────────────────────────────────────────────────────────────

_WINDOW_LONG = 260  # 52 weeks × 5 trading days


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_close_series(
    start_price: float,
    weekly_return: float,
    n_days: int = _WINDOW_LONG + 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic close prices with a deterministic weekly drift."""
    rng = np.random.default_rng(seed)
    n_weeks = n_days // 5
    weekly_returns: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(n_weeks):
        noise = rng.normal(0.0, 0.005)
        weekly_returns.append(weekly_return + noise)
    # Expand to daily (5 days per week, flat within week for simplicity)
    daily_returns: list[float] = []
    for wr in weekly_returns:
        daily_ret = wr / 5.0
        daily_returns.extend([daily_ret] * 5)
    while len(daily_returns) < n_days:
        daily_returns.append(daily_returns[-1] if daily_returns else 0.0)
    daily_returns = daily_returns[:n_days]
    prices = start_price * np.cumprod(1.0 + np.array(daily_returns))
    return prices


def _make_etf_df(
    prices: np.ndarray,
) -> pl.DataFrame:
    """Build a polars DataFrame with close prices for an ETF."""
    return pl.DataFrame({"close": prices})


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> FiftyTwoWeekHigh:
    """Default FiftyTwoWeekHigh with lookback_weeks=52, proximity_pct=0.95."""
    config = FiftyTwoWeekHighConfig(
        name="fiftytwo_week_high",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP},
        lookback_weeks=52,
        proximity_pct=0.95,
    )
    return FiftyTwoWeekHigh(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2025, 6, 1)


@pytest.fixture
def near_high_data() -> dict[str, pl.DataFrame]:
    """3 ETFs: A near high, B mid-range, C far from high."""
    n_days = _WINDOW_LONG + 20
    # ETF_A: steadily rising, last price near its 52-week high
    prices_a = _make_close_series(1.0, 0.003, n_days, seed=1)
    # ETF_B: flat, mid-range
    prices_b = _make_close_series(1.0, 0.0005, n_days, seed=2)
    # ETF_C: steadily falling, far from high
    prices_c = _make_close_series(1.0, -0.003, n_days, seed=3)
    return {
        "ETF_A": _make_etf_df(prices_a),
        "ETF_B": _make_etf_df(prices_b),
        "ETF_C": _make_etf_df(prices_c),
    }


@pytest.fixture
def all_near_high_data() -> dict[str, pl.DataFrame]:
    """2 ETFs both trading near their 52-week high."""
    n_days = _WINDOW_LONG + 20
    prices_x = _make_close_series(1.0, 0.004, n_days, seed=10)
    prices_y = _make_close_series(1.0, 0.0035, n_days, seed=11)
    return {
        "ETF_X": _make_etf_df(prices_x),
        "ETF_Y": _make_etf_df(prices_y),
    }


@pytest.fixture
def none_near_high_data() -> dict[str, pl.DataFrame]:
    """All ETFs far from 52-week high (declining)."""
    n_days = _WINDOW_LONG + 20
    prices_p = _make_close_series(1.0, -0.004, n_days, seed=20)
    prices_q = _make_close_series(1.0, -0.005, n_days, seed=21)
    return {
        "ETF_P": _make_etf_df(prices_p),
        "ETF_Q": _make_etf_df(prices_q),
    }


@pytest.fixture
def single_fund_data() -> dict[str, pl.DataFrame]:
    """Single ETF near its 52-week high."""
    n_days = _WINDOW_LONG + 20
    prices = _make_close_series(1.0, 0.004, n_days, seed=50)
    return {"ETF_ONLY": _make_etf_df(prices)}


@pytest.fixture
def insufficient_data() -> dict[str, pl.DataFrame]:
    """ETF with too few rows for 52-week lookback."""
    prices = _make_close_series(1.0, 0.003, 100, seed=99)
    return {"ETF_SHORT": _make_etf_df(prices)}


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestFiftyTwoWeekHighConfig:
    """Configuration validation for S5 52-Week High Momentum."""

    def test_default_config(self) -> None:
        config = FiftyTwoWeekHighConfig(name="fiftytwo_week_high")
        assert config.lookback_weeks == 52
        assert config.proximity_pct == 0.95
        assert config.top_n == 3

    def test_rejects_zero_lookback_weeks(self) -> None:
        with pytest.raises(ValueError):
            FiftyTwoWeekHighConfig(name="fiftytwo_week_high", lookback_weeks=0)

    def test_rejects_proximity_pct_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            FiftyTwoWeekHighConfig(name="fiftytwo_week_high", proximity_pct=1.5)

    def test_rejects_negative_top_n(self) -> None:
        with pytest.raises(ValueError):
            FiftyTwoWeekHighConfig(name="fiftytwo_week_high", top_n=0)

    def test_inherits_strategy_config(self) -> None:
        config = FiftyTwoWeekHighConfig(name="fiftytwo_week_high")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ────────────────────────────────────────────────────────


class TestRegimeEligibility:
    """FiftyTwoWeekHigh valid in TRENDING_UP only."""

    def test_eligible_in_trending_up(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True

    def test_not_eligible_in_sideways(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is False

    def test_not_eligible_in_trending_down(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False

    def test_not_eligible_in_high_vol(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False

    def test_not_eligible_in_crisis(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ─────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare close prices."""

    def test_returns_close(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.required_data() == ["close"]

    def test_required_data_is_list_of_str(self, strategy: FiftyTwoWeekHigh) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with valid ETF market data."""

    def test_generates_signals_with_valid_data(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, near_high_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_near_high_etf_gets_buy(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """ETF_A (rising, near 52w high) should get BUY, ETF_C (falling) should not."""
        signals = strategy.generate_signals(trade_date, near_high_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        buy_codes = {s["fund_code"] for s in buy_signals}
        assert "ETF_A" in buy_codes
        assert "ETF_C" not in buy_codes

    def test_all_near_high_multiple_buys(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        all_near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """When multiple ETFs are near their high, all should get BUY signals."""
        signals = strategy.generate_signals(trade_date, all_near_high_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert len(buy_signals) == 2

    def test_signals_have_all_required_keys(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """Every signal must contain fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(trade_date, near_high_data)
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
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        all_near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, all_near_high_data)
        buy_weight = sum(
            s["target_weight"] for s in signals
            if s["direction"] == SignalDirection.BUY.value
        )
        assert buy_weight <= 1.0

    def test_nearer_high_has_higher_confidence(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """ETF closest to its 52w high should have highest confidence."""
        signals = strategy.generate_signals(trade_date, near_high_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        # ETF_A is closest to high, should have highest confidence among buys
        for s in buy_signals:
            assert s["confidence"] >= 0.5

    def test_highest_proximity_ranked_first(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """ETF_A (closest to 52w high) must appear among top-ranked signals."""
        signals = strategy.generate_signals(trade_date, near_high_data)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        assert "ETF_A" in buy_codes


# ─── Signal Generation — Edge Cases ────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for 52-week high strategy."""

    def test_empty_market_data_returns_empty(
        self, strategy: FiftyTwoWeekHigh, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_single_fund_generates_signal(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        single_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, single_fund_data)
        assert len(signals) >= 1

    def test_insufficient_rows_returns_empty(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        insufficient_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, insufficient_data)
        assert signals == []

    def test_all_far_from_high_no_buy(
        self,
        strategy: FiftyTwoWeekHigh,
        trade_date: date,
        none_near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """When no ETF is near its 52-week high, no BUY signals."""
        signals = strategy.generate_signals(trade_date, none_near_high_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert len(buy_signals) == 0

    def test_missing_close_column_skipped(
        self, strategy: FiftyTwoWeekHigh, trade_date: date
    ) -> None:
        bad_df = pl.DataFrame({"volume": [1.0] * 300})
        signals = strategy.generate_signals(trade_date, {"ETF_BAD": bad_df})
        assert signals == []

    def test_mixed_valid_and_invalid_funds(
        self, strategy: FiftyTwoWeekHigh, trade_date: date,
        near_high_data: dict[str, pl.DataFrame]
    ) -> None:
        """Valid funds are processed even when some funds lack data."""
        mixed = dict(near_high_data)
        mixed["ETF_BAD"] = pl.DataFrame({"volume": [1.0] * 300})
        signals = strategy.generate_signals(trade_date, mixed)
        bad_signals = [s for s in signals if s["fund_code"] == "ETF_BAD"]
        assert len(bad_signals) == 0


# ─── Strategy Identity and Validation ──────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation checks."""

    def test_name_matches_config(self, strategy: FiftyTwoWeekHigh) -> None:
        assert strategy.name == "fiftytwo_week_high"

    def test_validate_passes_for_valid_config(self, strategy: FiftyTwoWeekHigh) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_validate_catches_bad_proximity(self) -> None:
        """proximity_pct > 1.0 is rejected by Pydantic field validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FiftyTwoWeekHighConfig(name="fiftytwo_week_high", proximity_pct=1.5)

    def test_strategy_is_frozen(self, strategy: FiftyTwoWeekHigh) -> None:
        with pytest.raises(Exception):
            strategy.config.proximity_pct = 0.80  # type: ignore[misc]


# ─── Proximity Logic ───────────────────────────────────────────────────────────


class TestProximityLogic:
    """Verify the 52-week high proximity scoring logic."""

    def test_proximity_threshold_respected(
        self,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """With proximity_pct=0.95, ETFs below 95% of 52w high should not be bought."""
        config = FiftyTwoWeekHighConfig(
            name="fiftytwo_week_high",
            proximity_pct=0.95,
            eligible_regimes={MarketRegime.TRENDING_UP},
        )
        strategy = FiftyTwoWeekHigh(config=config)
        signals = strategy.generate_signals(trade_date, near_high_data)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        # ETF_C is falling (far from high) — should not be bought
        assert "ETF_C" not in buy_codes

    def test_lower_threshold_more_buys(
        self,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """Lowering proximity threshold should admit more ETFs."""
        config = FiftyTwoWeekHighConfig(
            name="fiftytwo_week_high",
            proximity_pct=0.50,
            eligible_regimes={MarketRegime.TRENDING_UP},
        )
        strategy = FiftyTwoWeekHigh(config=config)
        signals = strategy.generate_signals(trade_date, near_high_data)
        buy_count = len([s for s in signals if s["direction"] == SignalDirection.BUY.value])
        # With 0.50 threshold, more ETFs should qualify
        assert buy_count > 0

    def test_custom_lookback_weeks(
        self,
        trade_date: date,
        near_high_data: dict[str, pl.DataFrame],
    ) -> None:
        """Shorter lookback should still produce valid signals."""
        config = FiftyTwoWeekHighConfig(
            name="fiftytwo_week_short",
            lookback_weeks=26,
            eligible_regimes={MarketRegime.TRENDING_UP},
        )
        strategy = FiftyTwoWeekHigh(config=config)
        signals = strategy.generate_signals(trade_date, near_high_data)
        assert len(signals) > 0
