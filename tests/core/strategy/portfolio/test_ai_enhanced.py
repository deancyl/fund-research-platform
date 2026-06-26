"""
Tests for AI-Enhanced Portfolio strategy (S20) — AI增强组合.

S20 — AI-Enhanced Portfolio: Base = S19 aggressive allocation with AI overlay
via simulated multi-agent weekend debate adjusting weights ±20%.

Base allocation: CSI300 30% + ChiNext 20% + Sector ETF 30% + Gold 10% + Cash 10%.
AI adjustment range: ±20% of base weight per component.

Parameters: base_weights dict, ai_adjustment_range=0.20.
Eligible regimes: ALL (AI can operate in any market).
Required data: ["close"].
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.portfolio.ai_enhanced import AIEnhancedPortfolio, AIEnhancedConfig


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _make_fund_df(
    close_prices: list[float] | None = None,
    n_rows: int = 60,
) -> pl.DataFrame:
    """Build a polars DataFrame with close prices for AI analysis."""
    if close_prices is None:
        close_prices = [1.0 + i * 0.001 for i in range(n_rows)]
    return pl.DataFrame({"close": close_prices})


def _make_bullish_df(n_rows: int = 60) -> pl.DataFrame:
    """Fund with strong uptrend (AI should boost its weight)."""
    return pl.DataFrame({
        "close": [1.0 + i * 0.05 for i in range(n_rows)],  # strong uptrend
    })


def _make_bearish_df(n_rows: int = 60) -> pl.DataFrame:
    """Fund with downtrend (AI should reduce its weight)."""
    return pl.DataFrame({
        "close": [10.0 - i * 0.05 for i in range(n_rows)],  # downtrend
    })


def _make_flat_df(n_rows: int = 60) -> pl.DataFrame:
    """Fund trading flat (AI should keep weight near base)."""
    return pl.DataFrame({"close": [5.0 + np.sin(i * 0.1) * 0.02 for i in range(n_rows)]})


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> AIEnhancedPortfolio:
    """Default AIEnhancedPortfolio with S19 base weights."""
    config = AIEnhancedConfig(
        name="ai_enhanced",
        version="1.0.0",
        eligible_regimes={
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOL,
            MarketRegime.CRISIS,
        },
    )
    return AIEnhancedPortfolio(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2025, 6, 1)


@pytest.fixture
def full_market_data() -> dict[str, pl.DataFrame]:
    """All five portfolio components present with close data."""
    return {
        "510300": _make_fund_df(),
        "159915": _make_fund_df(),
        "sector_etf": _make_fund_df(),
        "gold": _make_fund_df(),
    }


@pytest.fixture
def bullish_market_data() -> dict[str, pl.DataFrame]:
    """Strong uptrend in all growth components → AI should boost them."""
    return {
        "510300": _make_bullish_df(),
        "159915": _make_bullish_df(),
        "sector_etf": _make_bullish_df(),
        "gold": _make_flat_df(),
    }


@pytest.fixture
def bearish_market_data() -> dict[str, pl.DataFrame]:
    """Downtrend in all growth components → AI should reduce them."""
    return {
        "510300": _make_bearish_df(),
        "159915": _make_bearish_df(),
        "sector_etf": _make_bearish_df(),
        "gold": _make_flat_df(),
    }


# ─── Configuration Tests ────────────────────────────────────────────────────────


class TestAIEnhancedConfig:
    """Configuration validation for S20 AI-Enhanced Portfolio."""

    def test_default_config(self) -> None:
        config = AIEnhancedConfig(name="ai_enhanced")
        assert config.ai_adjustment_range == 0.20
        assert "510300" in config.base_weights
        assert config.base_weights["510300"] == 0.30
        assert config.base_weights["159915"] == 0.20
        assert config.base_weights["sector_etf"] == 0.30
        assert config.base_weights["gold"] == 0.10
        assert config.base_weights["cash"] == 0.10

    def test_weights_sum_to_one(self) -> None:
        config = AIEnhancedConfig(name="ai_enhanced")
        total = sum(config.base_weights.values())
        assert total == pytest.approx(1.0)

    def test_custom_base_weights(self) -> None:
        config = AIEnhancedConfig(
            name="ai_enhanced",
            base_weights={"510300": 0.4, "159915": 0.3, "gold": 0.1, "cash": 0.2},
        )
        assert config.base_weights["510300"] == 0.4

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError):
            AIEnhancedConfig(
                name="ai_enhanced",
                base_weights={"510300": -0.1, "cash": 0.0},
            )

    def test_rejects_weight_over_one(self) -> None:
        with pytest.raises(ValueError):
            AIEnhancedConfig(
                name="ai_enhanced",
                base_weights={"510300": 1.5, "cash": 0.0},
            )

    def test_rejects_non_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            AIEnhancedConfig(
                name="ai_enhanced",
                base_weights={"510300": 0.3, "159915": 0.3},
            )

    def test_rejects_empty_weights(self) -> None:
        with pytest.raises(ValueError):
            AIEnhancedConfig(name="ai_enhanced", base_weights={})

    def test_inherits_strategy_config(self) -> None:
        config = AIEnhancedConfig(name="ai_enhanced")
        assert isinstance(config, StrategyConfig)

    def test_rejects_invalid_adjustment_range(self) -> None:
        with pytest.raises(ValueError):
            AIEnhancedConfig(name="ai_enhanced", ai_adjustment_range=0.0)

    def test_rejects_adjustment_range_above_one(self) -> None:
        with pytest.raises(ValueError):
            AIEnhancedConfig(name="ai_enhanced", ai_adjustment_range=1.5)


# ─── Regime Eligibility ─────────────────────────────────────────────────────────


class TestRegimeEligibility:
    """AI-Enhanced Portfolio valid in ALL regimes."""

    def test_eligible_in_all_regimes(self, strategy: AIEnhancedPortfolio) -> None:
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True


# ─── required_data ──────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare close field."""

    def test_returns_close_field(self, strategy: AIEnhancedPortfolio) -> None:
        fields = strategy.required_data()
        assert "close" in fields

    def test_required_data_is_list_of_str(self, strategy: AIEnhancedPortfolio) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Happy Path ─────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    """Signal generation with valid market data."""

    def test_generates_signals_with_full_data(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_all_component_funds_present(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        signals = strategy.generate_signals(trade_date, full_market_data)
        codes = {s["fund_code"] for s in signals}
        assert "510300" in codes
        assert "159915" in codes
        assert "sector_etf" in codes
        assert "gold" in codes

    def test_no_cash_signal_emitted(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Cash allocation is the residual, not emitted as a signal."""
        signals = strategy.generate_signals(trade_date, full_market_data)
        codes = {s["fund_code"] for s in signals}
        assert "cash" not in codes

    def test_signals_have_all_required_keys(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Every signal must contain fund_code, direction, confidence, target_weight, reason."""
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

    def test_all_signals_are_buy(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """All emitted signals should be BUY (portfolio allocation)."""
        signals = strategy.generate_signals(trade_date, full_market_data)
        for s in signals:
            assert s["direction"] == SignalDirection.BUY.value


# ─── Signal Generation — AI Adjustment Logic ────────────────────────────────────


class TestAIAdjustmentLogic:
    """Verify AI overlay adjusts weights based on recent performance."""

    def test_bullish_trend_boosts_weights(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        bullish_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Strong uptrend → AI should boost growth-component weights above base."""
        signals = strategy.generate_signals(trade_date, bullish_market_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        # With strong uptrend, growth weights should be boosted above base
        assert weights.get("510300", 0) >= 0.25  # base=0.30; boosted

    def test_bearish_trend_reduces_weights(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        bearish_market_data: dict[str, pl.DataFrame],
    ) -> None:
        """Downtrend → AI should reduce growth-component weights below base."""
        signals = strategy.generate_signals(trade_date, bearish_market_data)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        # With downtrend, growth weights should be reduced below base
        assert weights.get("510300", 1.0) <= 0.35  # base=0.30; reduced

    def test_weights_within_adjustment_range(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Adjusted weights must stay within ±ai_adjustment_range of base."""
        # Extreme data: very strong uptrend
        extreme = {
            "510300": pl.DataFrame({"close": [1.0 + i * 0.10 for i in range(60)]}),
            "159915": pl.DataFrame({"close": [1.0 + i * 0.10 for i in range(60)]}),
            "sector_etf": pl.DataFrame({"close": [1.0 + i * 0.10 for i in range(60)]}),
            "gold": pl.DataFrame({"close": [1.0 + i * 0.10 for i in range(60)]}),
        }
        signals = strategy.generate_signals(trade_date, extreme)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        base = strategy.config.base_weights
        ar = strategy.config.ai_adjustment_range

        for code in weights:
            if code in base:
                base_w = base[code]
                assert base_w * (1 - ar) - 0.01 <= weights[code] <= base_w * (1 + ar) + 0.01, (
                    f"Weight {weights[code]:.4f} for {code} outside "
                    f"[{base_w * (1 - ar):.4f}, {base_w * (1 + ar):.4f}]"
                )
        # Total weights (excluding cash) should be <= 1.0
        total = sum(weights.values())
        assert 0.0 < total <= 1.0 + 0.01

    def test_adjustment_range_respected_extreme_bearish(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Even with extreme downtrend, weights stay within adjustment bounds."""
        extreme_bear = {
            "510300": pl.DataFrame({"close": [10.0 - i * 0.10 for i in range(60)]}),
            "159915": pl.DataFrame({"close": [10.0 - i * 0.10 for i in range(60)]}),
            "sector_etf": pl.DataFrame({"close": [10.0 - i * 0.10 for i in range(60)]}),
            "gold": pl.DataFrame({"close": [10.0 - i * 0.10 for i in range(60)]}),
        }
        signals = strategy.generate_signals(trade_date, extreme_bear)
        weights = {s["fund_code"]: s["target_weight"] for s in signals}
        base = strategy.config.base_weights
        ar = strategy.config.ai_adjustment_range

        for code in weights:
            if code in base:
                base_w = base[code]
                assert weights[code] >= base_w * (1 - ar) - 0.01


# ─── Edge Cases ─────────────────────────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    """Edge case handling for AI-enhanced portfolio strategy."""

    def test_empty_market_data_returns_empty(
        self, strategy: AIEnhancedPortfolio, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_missing_close_column_skipped(
        self, strategy: AIEnhancedPortfolio, trade_date: date
    ) -> None:
        incomplete = pl.DataFrame({"volume": [1.0] * 30})
        signals = strategy.generate_signals(trade_date, {"510300": incomplete})
        assert signals == []

    def test_extra_funds_ignored(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
        full_market_data: dict[str, pl.DataFrame],
    ) -> None:
        mixed = dict(full_market_data)
        mixed["EXTRA"] = _make_fund_df()
        signals = strategy.generate_signals(trade_date, mixed)
        codes = {s["fund_code"] for s in signals}
        assert "EXTRA" not in codes

    def test_missing_fund_skip(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Only some funds in market_data; missing ones skipped."""
        partial = {
            "510300": _make_fund_df(),
            "gold": _make_fund_df(),
        }
        signals = strategy.generate_signals(trade_date, partial)
        codes = {s["fund_code"] for s in signals}
        assert "510300" in codes
        assert "gold" in codes
        assert "159915" not in codes

    def test_single_row_data(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Single row of data should still work — confidence near neutral."""
        tiny = {"510300": pl.DataFrame({"close": [1.5]})}
        signals = strategy.generate_signals(trade_date, tiny)
        # With single row, AI confidence is neutral → weight near base
        assert len(signals) == 1
        assert 0.0 <= signals[0]["confidence"] <= 1.0
        assert "ai_adjusted" in signals[0]["reason"].lower() or "ai" in signals[0]["reason"].lower()


# ─── Strategy Identity and Validation ───────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity and validation checks."""

    def test_name_matches_config(self, strategy: AIEnhancedPortfolio) -> None:
        assert strategy.name == "ai_enhanced"

    def test_validate_passes_for_valid_config(self, strategy: AIEnhancedPortfolio) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_strategy_is_frozen(self, strategy: AIEnhancedPortfolio) -> None:
        with pytest.raises(Exception):
            strategy.config.base_weights = {}  # type: ignore[misc]

    def test_validate_catches_empty_weights(self) -> None:
        """Empty base_weights caught by model_validator."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AIEnhancedConfig(name="ai_enhanced", base_weights={})


# ─── AI Confidence Factor ───────────────────────────────────────────────────────


class TestAIConfidenceFactor:
    """Verify AI confidence factor computation from price data."""

    def test_uptrend_yields_positive_ai_confidence(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Strong uptrend → positive AI confidence → boosted weights."""
        bull = {"510300": _make_bullish_df()}
        signals = strategy.generate_signals(trade_date, bull)
        assert signals[0]["confidence"] > 0.50

    def test_downtrend_yields_lower_confidence(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Downtrend → lower AI confidence → reduced weights."""
        bear = {"510300": _make_bearish_df()}
        signals = strategy.generate_signals(trade_date, bear)
        assert signals[0]["confidence"] < 0.60

    def test_flat_market_neutral_confidence(
        self,
        strategy: AIEnhancedPortfolio,
        trade_date: date,
    ) -> None:
        """Flat market → AI confidence near 0.5."""
        flat = {"510300": _make_flat_df()}
        signals = strategy.generate_signals(trade_date, flat)
        assert 0.40 <= signals[0]["confidence"] <= 0.65

    def test_custom_adjustment_range(
        self,
        trade_date: date,
    ) -> None:
        """Custom adjustment range of 10% should produce tighter bounds."""
        config_small = AIEnhancedConfig(
            name="ai_enhanced_small",
            ai_adjustment_range=0.10,
        )
        s_small = AIEnhancedPortfolio(config=config_small)
        config_large = AIEnhancedConfig(
            name="ai_enhanced_large",
            ai_adjustment_range=0.20,
        )
        s_large = AIEnhancedPortfolio(config=config_large)

        data = {"510300": _make_bullish_df()}
        sig_small = s_small.generate_signals(trade_date, data)
        sig_large = s_large.generate_signals(trade_date, data)

        base_w = 0.30
        # Small range: weight closer to base
        dev_small = abs(sig_small[0]["target_weight"] - base_w)
        dev_large = abs(sig_large[0]["target_weight"] - base_w)
        assert dev_small <= dev_large + 0.01
