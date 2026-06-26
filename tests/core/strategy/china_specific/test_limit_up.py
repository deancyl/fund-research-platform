"""
Tests for Limit-up Probability strategy (S17) — 量化打板概率模型.

S17 — Limit-up Probability: Monitor order-by-order data to estimate probability
of hitting limit-up. Marked as HIGH_RISK per 2026 regulations. Not recommended
for retail investors.

Parameters: limit_up_pct=0.10 (main board), volume_spike_threshold=3.0.
Eligible regimes: TRENDING_UP only (informational display, strong warning).
Required data: ["close", "volume", "turnover", "bid_ask_spread"].
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.china_specific.limit_up import LimitUp, LimitUpConfig


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _make_stock_df(
    close_prices: list[float] | None = None,
    volumes: list[float] | None = None,
    turnover: float = 0.05,
    bid_ask_spread: float = 0.001,
    n_rows: int = 60,
) -> pl.DataFrame:
    """Build a polars DataFrame with columns needed for limit-up analysis."""
    if close_prices is None:
        close_prices = [10.0] * n_rows
    if volumes is None:
        volumes = [1_000_000.0] * n_rows
    return pl.DataFrame({
        "close": close_prices,
        "volume": volumes,
        "turnover": [turnover] * n_rows,
        "bid_ask_spread": [bid_ask_spread] * n_rows,
    })


def _approaching_limit_up_df(
    limit_up_pct: float = 0.10,
    n_rows: int = 60,
) -> pl.DataFrame:
    """Build a DataFrame simulating price approaching limit-up.

    Last close is within 2% of limit-up; volume spikes in last 3 rows.
    """
    prev_close = 10.0
    limit_price = prev_close * (1.0 + limit_up_pct)  # 11.0 for 10% limit
    close_prices = [prev_close] * (n_rows - 3) + [
        prev_close * 1.05,  # +5%
        prev_close * 1.07,  # +7%
        prev_close * 1.09,  # +9% — close to limit
    ]
    normal_vol = 1_000_000.0
    volumes = [normal_vol] * (n_rows - 3) + [
        normal_vol * 2.0,
        normal_vol * 3.5,
        normal_vol * 5.0,  # spike
    ]
    return pl.DataFrame({
        "close": close_prices,
        "volume": volumes,
        "turnover": [0.12] * n_rows,  # high turnover
        "bid_ask_spread": [0.0003] * n_rows,  # tight spread
    })


def _normal_trading_df(n_rows: int = 60) -> pl.DataFrame:
    """Build a DataFrame simulating normal trading conditions."""
    return pl.DataFrame({
        "close": [10.0 + i * 0.01 for i in range(n_rows)],
        "volume": [1_000_000.0] * n_rows,
        "turnover": [0.03] * n_rows,
        "bid_ask_spread": [0.005] * n_rows,
    })


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> LimitUp:
    """Default LimitUp with main-board 10% limit and 3x volume spike threshold."""
    config = LimitUpConfig(
        name="limit_up",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP},
        limit_up_pct=0.10,
        volume_spike_threshold=3.0,
    )
    return LimitUp(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2025, 6, 1)


@pytest.fixture
def approaching_data() -> dict[str, pl.DataFrame]:
    """Stock approaching limit-up with volume spike."""
    return {"STOCK_APPR": _approaching_limit_up_df()}


@pytest.fixture
def normal_data() -> dict[str, pl.DataFrame]:
    """Stock trading under normal conditions."""
    return {"STOCK_NORM": _normal_trading_df()}


@pytest.fixture
def multi_stock_data() -> dict[str, pl.DataFrame]:
    """Multiple stocks: one approaching limit-up, one normal, one missing data."""
    return {
        "STOCK_HOT": _approaching_limit_up_df(),
        "STOCK_COLD": _normal_trading_df(),
        "STOCK_BAD": pl.DataFrame({"unrelated": [1.0] * 10}),
    }


# ─── Configuration Tests ────────────────────────────────────────────────────────


class TestLimitUpConfig:
    """Configuration validation for S17 Limit-up Probability."""

    def test_default_config(self) -> None:
        config = LimitUpConfig(name="limit_up")
        assert config.limit_up_pct == 0.10
        assert config.volume_spike_threshold == 3.0

    def test_custom_limit_up_pct(self) -> None:
        """ChiNext/STAR boards have 20% limit."""
        config = LimitUpConfig(name="limit_up", limit_up_pct=0.20)
        assert config.limit_up_pct == 0.20

    def test_custom_volume_threshold(self) -> None:
        config = LimitUpConfig(name="limit_up", volume_spike_threshold=5.0)
        assert config.volume_spike_threshold == 5.0

    def test_rejects_non_positive_limit_up_pct(self) -> None:
        with pytest.raises(ValueError):
            LimitUpConfig(name="limit_up", limit_up_pct=0.0)

    def test_rejects_negative_limit_up_pct(self) -> None:
        with pytest.raises(ValueError):
            LimitUpConfig(name="limit_up", limit_up_pct=-0.05)

    def test_rejects_non_positive_volume_threshold(self) -> None:
        with pytest.raises(ValueError):
            LimitUpConfig(name="limit_up", volume_spike_threshold=0.0)

    def test_inherits_strategy_config(self) -> None:
        config = LimitUpConfig(name="limit_up")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ─────────────────────────────────────────────────────────


class TestRegimeEligibility:
    """Limit-up strategy only valid in TRENDING_UP per 2026 regulations."""

    def test_eligible_in_trending_up(self, strategy: LimitUp) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True

    def test_not_eligible_in_sideways(self, strategy: LimitUp) -> None:
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is False

    def test_not_eligible_in_trending_down(self, strategy: LimitUp) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False

    def test_not_eligible_in_high_vol(self, strategy: LimitUp) -> None:
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False

    def test_not_eligible_in_crisis(self, strategy: LimitUp) -> None:
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ──────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare all limit-up analysis fields."""

    def test_returns_all_required_fields(self, strategy: LimitUp) -> None:
        fields = set(strategy.required_data())
        expected = {"close", "volume", "turnover", "bid_ask_spread"}
        assert fields == expected

    def test_required_data_is_list_of_str(self, strategy: LimitUp) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ─────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with valid market data."""

    def test_generates_signals_with_approaching_stock(
        self,
        strategy: LimitUp,
        trade_date: date,
        approaching_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, approaching_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_approaching_limit_up_generates_higher_probability(
        self,
        strategy: LimitUp,
        trade_date: date,
        approaching_data: dict[str, pl.DataFrame],
        normal_data: dict[str, pl.DataFrame],
    ) -> None:
        """Stock approaching limit-up should have higher probability than normal."""
        sig_approaching = strategy.generate_signals(trade_date, approaching_data)
        sig_normal = strategy.generate_signals(trade_date, normal_data)
        assert sig_approaching[0]["confidence"] >= sig_normal[0]["confidence"]

    def test_signals_have_all_required_keys(
        self,
        strategy: LimitUp,
        trade_date: date,
        approaching_data: dict[str, pl.DataFrame],
    ) -> None:
        """Every signal must contain fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(trade_date, approaching_data)
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

    def test_risk_warning_in_reason(
        self,
        strategy: LimitUp,
        trade_date: date,
        approaching_data: dict[str, pl.DataFrame],
    ) -> None:
        """Every signal reason MUST include a risk_warning per 2026 regulations."""
        signals = strategy.generate_signals(trade_date, approaching_data)
        for s in signals:
            assert "risk_warning" in s
            assert isinstance(s["risk_warning"], str)
            assert len(s["risk_warning"]) > 0
            # The reason should also mention the warning
            reason_lower = s["reason"].lower()
            assert any(
                keyword in reason_lower
                for keyword in ["risk", "warning", "high_risk", "regulation"]
            ), f"Reason missing risk warning: {s['reason']}"

    def test_buy_signal_when_high_probability(
        self,
        strategy: LimitUp,
        trade_date: date,
        approaching_data: dict[str, pl.DataFrame],
    ) -> None:
        """High limit-up probability should trigger ACCUMULATE or BUY signal."""
        signals = strategy.generate_signals(trade_date, approaching_data)
        for s in signals:
            direction = s["direction"]
            assert direction in {
                SignalDirection.ACCUMULATE.value,
                SignalDirection.BUY.value,
            }

    def test_multi_stock_different_probabilities(
        self,
        strategy: LimitUp,
        trade_date: date,
        multi_stock_data: dict[str, pl.DataFrame],
    ) -> None:
        """Different stocks should receive different limit-up probabilities."""
        signals = strategy.generate_signals(trade_date, multi_stock_data)
        assert len(signals) >= 1
        # Hot stock should be present
        codes = {s["fund_code"] for s in signals}
        assert "STOCK_HOT" in codes

    def test_higher_limit_up_pct_lowers_probability(
        self,
        trade_date: date,
    ) -> None:
        """With a higher limit-up threshold (20% vs 10%), same price is further from limit."""
        data_10pct = {"STOCK": _approaching_limit_up_df(limit_up_pct=0.10)}
        data_20pct = {"STOCK": _approaching_limit_up_df(limit_up_pct=0.20)}

        config_10 = LimitUpConfig(name="limit_up", limit_up_pct=0.10)
        config_20 = LimitUpConfig(name="limit_up", limit_up_pct=0.20)

        s10 = LimitUp(config=config_10)
        s20 = LimitUp(config=config_20)

        sig_10 = s10.generate_signals(trade_date, data_10pct)
        sig_20 = s20.generate_signals(trade_date, data_20pct)

        # Same price, but 20% limit is further away → lower probability
        assert sig_10[0]["confidence"] >= sig_20[0]["confidence"]


# ─── Signal Generation — Edge Cases ─────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for limit-up probability strategy."""

    def test_empty_market_data_returns_empty(
        self, strategy: LimitUp, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_missing_required_columns_skipped(
        self, strategy: LimitUp, trade_date: date
    ) -> None:
        incomplete = pl.DataFrame({"close": [10.0] * 30, "volume": [1.0] * 30})
        signals = strategy.generate_signals(trade_date, {"STOCK_BAD": incomplete})
        assert signals == []

    def test_insufficient_data_skipped(
        self, strategy: LimitUp, trade_date: date
    ) -> None:
        """DataFrame with fewer than 2 rows cannot compute returns, skipped."""
        tiny = pl.DataFrame({
            "close": [10.0],
            "volume": [1_000_000.0],
            "turnover": [0.05],
            "bid_ask_spread": [0.001],
        })
        signals = strategy.generate_signals(trade_date, {"STOCK_TINY": tiny})
        assert signals == []

    def test_mixed_valid_and_invalid(
        self,
        strategy: LimitUp,
        trade_date: date,
        approaching_data: dict[str, pl.DataFrame],
    ) -> None:
        mixed = dict(approaching_data)
        mixed["STOCK_BAD"] = pl.DataFrame({"close": [1.0] * 30})
        signals = strategy.generate_signals(trade_date, mixed)
        bad_signals = [s for s in signals if s["fund_code"] == "STOCK_BAD"]
        assert len(bad_signals) == 0

    def test_normal_trading_low_probability(
        self,
        strategy: LimitUp,
        trade_date: date,
        normal_data: dict[str, pl.DataFrame],
    ) -> None:
        """Normal trading should yield low limit-up probability."""
        signals = strategy.generate_signals(trade_date, normal_data)
        if signals:
            assert signals[0]["confidence"] < 0.50


# ─── Strategy Identity and Validation ───────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation checks."""

    def test_name_matches_config(self, strategy: LimitUp) -> None:
        assert strategy.name == "limit_up"

    def test_validate_passes_for_valid_config(self, strategy: LimitUp) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_strategy_is_frozen(self, strategy: LimitUp) -> None:
        with pytest.raises(Exception):
            strategy.config.limit_up_pct = 0.20  # type: ignore[misc]

    def test_validate_catches_invalid_limit_up_pct(self) -> None:
        """Negative limit_up_pct is rejected by model_validator."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LimitUpConfig(name="limit_up", limit_up_pct=0.0)

    def test_validate_catches_volume_threshold_zero(self) -> None:
        """Zero volume_spike_threshold is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LimitUpConfig(name="limit_up", volume_spike_threshold=0.0)


# ─── Limit-up Probability Computation ───────────────────────────────────────────


class TestLimitUpProbability:
    """Verify the limit-up probability computation logic."""

    def test_volume_spike_increases_probability(
        self,
        trade_date: date,
    ) -> None:
        """Higher volume spike ratio → higher limit-up probability."""
        data_low_vol = {
            "STOCK": _make_stock_df(
                close_prices=[10.0] * 57 + [10.5, 10.7, 10.9],
                volumes=[1_000_000.0] * 60,
                turnover=0.12,
                bid_ask_spread=0.0003,
            )
        }
        data_high_vol = {
            "STOCK": _make_stock_df(
                close_prices=[10.0] * 57 + [10.5, 10.7, 10.9],
                volumes=[1_000_000.0] * 57 + [3_000_000.0, 5_000_000.0, 8_000_000.0],
                turnover=0.12,
                bid_ask_spread=0.0003,
            )
        }
        config = LimitUpConfig(name="limit_up")
        s = LimitUp(config=config)
        sig_low = s.generate_signals(trade_date, data_low_vol)
        sig_high = s.generate_signals(trade_date, data_high_vol)
        assert sig_high[0]["confidence"] >= sig_low[0]["confidence"]

    def test_high_turnover_increases_probability(
        self,
        trade_date: date,
    ) -> None:
        """Higher turnover → higher limit-up probability (speculative activity)."""
        data_low = {"STOCK": _make_stock_df(turnover=0.02, bid_ask_spread=0.001)}
        data_high = {"STOCK": _make_stock_df(turnover=0.15, bid_ask_spread=0.001)}
        config = LimitUpConfig(name="limit_up")
        s = LimitUp(config=config)
        sig_low = s.generate_signals(trade_date, data_low)
        sig_high = s.generate_signals(trade_date, data_high)
        assert sig_high[0]["confidence"] >= sig_low[0]["confidence"]

    def test_tight_spread_increases_probability(
        self,
        trade_date: date,
    ) -> None:
        """Tighter bid-ask spread (more liquidity) → higher probability."""
        data_wide = {"STOCK": _make_stock_df(turnover=0.05, bid_ask_spread=0.02)}
        data_tight = {"STOCK": _make_stock_df(turnover=0.05, bid_ask_spread=0.0001)}
        config = LimitUpConfig(name="limit_up")
        s = LimitUp(config=config)
        sig_wide = s.generate_signals(trade_date, data_wide)
        sig_tight = s.generate_signals(trade_date, data_tight)
        assert sig_tight[0]["confidence"] >= sig_wide[0]["confidence"]

    def test_probability_bounded_zero_to_one(
        self,
        strategy: LimitUp,
        trade_date: date,
    ) -> None:
        """Probability must always be in [0, 1]."""
        # Extreme case: very close to limit up
        extreme = {
            "STOCK": _make_stock_df(
                close_prices=[10.0] * 57 + [10.8, 10.9, 10.99],
                volumes=[1_000_000.0] * 57 + [10_000_000.0] * 3,
                turnover=0.30,
                bid_ask_spread=0.0001,
            )
        }
        signals = strategy.generate_signals(trade_date, extreme)
        for s in signals:
            assert 0.0 <= s["confidence"] <= 1.0, f"Probability out of bounds: {s['confidence']}"
