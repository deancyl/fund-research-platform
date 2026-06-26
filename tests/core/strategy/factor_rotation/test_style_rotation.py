"""
Tests for Style Rotation strategy (S12).

S12 — Style Rotation: Predict large/small × value/growth factor direction
using macro indicators (47 metrics). Quarterly rebalance.

Win rate: 63.6% quarterly, annual excess: +6.52%.
Eligible: ALL. Required data: ["close", "pe", "pb", "roe", "market_cap"].
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.factor_rotation.style_rotation import (
    StyleQuadrant,
    StyleRotation,
    StyleRotationConfig,
)


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _make_macro_df(**indicators: float) -> pl.DataFrame:
    """Build a polars DataFrame with macro indicator columns.
    Each kwarg becomes a column; one row of data.
    """
    return pl.DataFrame([{k: v for k, v in indicators.items()}])


def _make_fund_df(
    pe: float,
    pb: float,
    roe: float,
    market_cap: float,
    n_rows: int = 60,
) -> pl.DataFrame:
    """Build a polars DataFrame with fund-level style columns."""
    return pl.DataFrame({
        "close": [1.0] * n_rows,
        "pe": [pe] * n_rows,
        "pb": [pb] * n_rows,
        "roe": [roe] * n_rows,
        "market_cap": [market_cap] * n_rows,
    })


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> StyleRotation:
    config = StyleRotationConfig(
        name="style_rotation",
        version="1.0.0",
        eligible_regimes={r for r in MarketRegime},
        macro_lookback=12,
    )
    return StyleRotation(config=config)


@pytest.fixture
def trade_date() -> date:
    return date(2026, 6, 30)


@pytest.fixture
def expansion_macro() -> dict[str, pl.DataFrame]:
    """Strong expansion: high PMI, low CPI → favors growth styles."""
    return {
        "macro": _make_macro_df(
            pmi=54.0, cpi=2.0, m2_growth=12.0,
            industrial_production=8.5, retail_sales=6.0,
            interest_rate_10y=3.2, credit_spread=0.8,
        ),
    }


@pytest.fixture
def contraction_macro() -> dict[str, pl.DataFrame]:
    """Contraction: low PMI, high CPI → favors value styles."""
    return {
        "macro": _make_macro_df(
            pmi=47.0, cpi=4.5, m2_growth=5.0,
            industrial_production=2.0, retail_sales=1.5,
            interest_rate_10y=4.5, credit_spread=2.0,
        ),
    }


@pytest.fixture
def four_fund_data() -> dict[str, pl.DataFrame]:
    """Four funds, one per style quadrant."""
    return {
        "LARGE_VALUE": _make_fund_df(pe=8.0, pb=1.0, roe=0.15, market_cap=5000.0),
        "LARGE_GROWTH": _make_fund_df(pe=35.0, pb=5.0, roe=0.20, market_cap=6000.0),
        "SMALL_VALUE": _make_fund_df(pe=10.0, pb=1.2, roe=0.12, market_cap=500.0),
        "SMALL_GROWTH": _make_fund_df(pe=40.0, pb=6.0, roe=0.18, market_cap=400.0),
    }


@pytest.fixture
def two_fund_data() -> dict[str, pl.DataFrame]:
    """Two funds: one large-value, one small-growth."""
    return {
        "LV": _make_fund_df(pe=9.0, pb=1.1, roe=0.14, market_cap=8000.0),
        "SG": _make_fund_df(pe=45.0, pb=7.0, roe=0.22, market_cap=300.0),
    }


# ─── Configuration Tests ────────────────────────────────────────────────────────


class TestStyleRotationConfig:
    def test_default_config(self) -> None:
        config = StyleRotationConfig(name="style_rotation")
        assert config.macro_lookback == 12
        assert config.style_quadrants == {
            "large_value": 0.0,
            "large_growth": 0.0,
            "small_value": 0.0,
            "small_growth": 0.0,
        }
        assert config.max_position_pct == 0.20

    def test_custom_macro_lookback(self) -> None:
        config = StyleRotationConfig(name="style_rotation", macro_lookback=6)
        assert config.macro_lookback == 6

    def test_rejects_invalid_macro_lookback(self) -> None:
        with pytest.raises(ValueError):
            StyleRotationConfig(name="style_rotation", macro_lookback=0)

    def test_rejects_negative_macro_lookback(self) -> None:
        with pytest.raises(ValueError):
            StyleRotationConfig(name="style_rotation", macro_lookback=-5)

    def test_inherits_strategy_config(self) -> None:
        config = StyleRotationConfig(name="style_rotation")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ─────────────────────────────────────────────────────────


class TestRegimeEligibility:
    def test_eligible_in_all_regimes(self, strategy: StyleRotation) -> None:
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True


# ─── required_data ──────────────────────────────────────────────────────────────


class TestRequiredData:
    def test_returns_all_required_fields(self, strategy: StyleRotation) -> None:
        fields = set(strategy.required_data())
        expected = {"close", "pe", "pb", "roe", "market_cap"}
        assert fields == expected

    def test_required_data_is_list_of_str(self, strategy: StyleRotation) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Quadrant Classification ────────────────────────────────────────────────────


class TestQuadrantClassification:
    def test_large_value_classified(self, strategy: StyleRotation) -> None:
        q = strategy.classify_quadrant(pe=8.0, pb=1.0, roe=0.15, market_cap=5000.0)
        assert q == StyleQuadrant.LARGE_VALUE

    def test_large_growth_classified(self, strategy: StyleRotation) -> None:
        q = strategy.classify_quadrant(pe=35.0, pb=5.0, roe=0.20, market_cap=6000.0)
        assert q == StyleQuadrant.LARGE_GROWTH

    def test_small_value_classified(self, strategy: StyleRotation) -> None:
        q = strategy.classify_quadrant(pe=10.0, pb=1.2, roe=0.12, market_cap=500.0)
        assert q == StyleQuadrant.SMALL_VALUE

    def test_small_growth_classified(self, strategy: StyleRotation) -> None:
        q = strategy.classify_quadrant(pe=40.0, pb=6.0, roe=0.18, market_cap=400.0)
        assert q == StyleQuadrant.SMALL_GROWTH

    def test_classify_uses_thresholds(self) -> None:
        """Custom thresholds should affect classification."""
        config = StyleRotationConfig(
            name="style_rotation",
            market_cap_threshold=2000.0,
            value_score_threshold=0.5,
        )
        s = StyleRotation(config=config)
        # market_cap 3000 > 2000 → large; low PE/PB + high ROE → value
        q = s.classify_quadrant(pe=8.0, pb=1.0, roe=0.18, market_cap=3000.0)
        assert q == StyleQuadrant.LARGE_VALUE


# ─── Signal Generation — Happy Path ─────────────────────────────────────────────


class TestGenerateSignalsHappyPath:
    def test_generates_signals_with_valid_data(
        self,
        strategy: StyleRotation,
        trade_date: date,
        expansion_macro: dict[str, pl.DataFrame],
        four_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        data = {**expansion_macro, **four_fund_data}
        signals = strategy.generate_signals(trade_date, data)
        assert len(signals) >= 1

    def test_expansion_favors_growth(
        self,
        strategy: StyleRotation,
        trade_date: date,
        expansion_macro: dict[str, pl.DataFrame],
        two_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        """In expansion (high PMI, low CPI), growth styles should be preferred."""
        data = {**expansion_macro, **two_fund_data}
        signals = strategy.generate_signals(trade_date, data)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        # SG (small growth) should get BUY, LV (large value) should not
        assert "SG" in buy_codes

    def test_contraction_favors_value(
        self,
        strategy: StyleRotation,
        trade_date: date,
        contraction_macro: dict[str, pl.DataFrame],
        two_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        """In contraction (low PMI, high CPI), value styles should be preferred."""
        data = {**contraction_macro, **two_fund_data}
        signals = strategy.generate_signals(trade_date, data)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        assert "LV" in buy_codes

    def test_signal_structure(
        self,
        strategy: StyleRotation,
        trade_date: date,
        expansion_macro: dict[str, pl.DataFrame],
        four_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        data = {**expansion_macro, **four_fund_data}
        signals = strategy.generate_signals(trade_date, data)
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in {e.value for e in SignalDirection}
            assert isinstance(s["confidence"], float)
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], float)
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0

    def test_favored_quadrant_gets_buy(
        self,
        strategy: StyleRotation,
        trade_date: date,
        four_fund_data: dict[str, pl.DataFrame],
    ) -> None:
        """Regardless of macro, exactly one quadrant is predicted best."""
        # Force macro to expansion state
        macro = {"macro": _make_macro_df(pmi=55.0, cpi=1.5, m2_growth=15.0)}
        data = {**macro, **four_fund_data}
        signals = strategy.generate_signals(trade_date, data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        # At least one fund should be BUY
        assert len(buy_signals) >= 1
        # Funds not in the favored quadrant get TRIM
        trim_signals = [s for s in signals if s["direction"] == SignalDirection.TRIM.value]
        # At least some should be trimmed
        assert len(trim_signals) >= 1


# ─── Signal Generation — Edge Cases ─────────────────────────────────────────────


class TestGenerateSignalsEdgeCases:
    def test_empty_market_data(self, strategy: StyleRotation, trade_date: date) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_no_macro_key(self, strategy: StyleRotation, trade_date: date) -> None:
        data = {"FUND_A": _make_fund_df(pe=15.0, pb=2.0, roe=0.12, market_cap=1000.0)}
        signals = strategy.generate_signals(trade_date, data)
        assert signals == []

    def test_missing_required_fund_columns(
        self, strategy: StyleRotation, trade_date: date, expansion_macro: dict[str, pl.DataFrame]
    ) -> None:
        data = {**expansion_macro, "FUND_BAD": pl.DataFrame({"close": [1.0] * 30})}
        signals = strategy.generate_signals(trade_date, data)
        bad_signals = [s for s in signals if s["fund_code"] == "FUND_BAD"]
        assert len(bad_signals) == 0

    def test_empty_macro_dataframe(
        self, strategy: StyleRotation, trade_date: date
    ) -> None:
        data = {
            "macro": pl.DataFrame(),
            "FUND_A": _make_fund_df(pe=15.0, pb=2.0, roe=0.12, market_cap=1000.0),
        }
        signals = strategy.generate_signals(trade_date, data)
        assert signals == []

    def test_single_fund_generates_signal(
        self,
        strategy: StyleRotation,
        trade_date: date,
        expansion_macro: dict[str, pl.DataFrame],
    ) -> None:
        data = {**expansion_macro, "FUND_A": _make_fund_df(pe=15.0, pb=2.0, roe=0.12, market_cap=1000.0)}
        signals = strategy.generate_signals(trade_date, data)
        assert len(signals) >= 1

    def test_macro_only_no_funds(
        self, strategy: StyleRotation, trade_date: date, expansion_macro: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, expansion_macro)
        assert signals == []


# ─── Strategy Identity ──────────────────────────────────────────────────────────


class TestStrategyIdentity:
    def test_name_matches_config(self, strategy: StyleRotation) -> None:
        assert strategy.name == "style_rotation"

    def test_validate_passes(self, strategy: StyleRotation) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_strategy_is_frozen(self, strategy: StyleRotation) -> None:
        with pytest.raises(Exception):
            strategy.config.macro_lookback = 99  # type: ignore[misc]
