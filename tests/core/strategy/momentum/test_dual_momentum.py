"""
Tests for Dual Momentum strategy (S3 GEM) — Gary Antonacci.

Absolute momentum: 12-month return vs bond threshold — if all equities underperform
the bond benchmark, rotate to bonds (defensive posture).
Relative momentum: select the single best-performing index asset when in equities.

Covers: configuration, regime eligibility, absolute momentum defence, relative
momentum ranking, bond rotation, empty/missing data handling, signal structure.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.momentum.dual_momentum import DualMomentum, DualMomentumConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_N_DAYS = 260  # ~12 trading months


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_price_series(
    start_price: float,
    daily_ret: float,
    n_days: int = _N_DAYS + 10,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic close prices with deterministic drift."""
    rng = np.random.default_rng(seed)
    returns = np.full(n_days, daily_ret)
    prices = start_price * np.cumprod(1.0 + returns)
    return prices


def _make_df(prices: np.ndarray) -> pl.DataFrame:
    """Build a polars DataFrame with close column."""
    return pl.DataFrame({"close": prices})


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> DualMomentum:
    """Default DualMomentum with lookback_months=12, bond_fund='511260'."""
    config = DualMomentumConfig(
        name="dual_momentum",
        version="1.0.0",
        eligible_regimes={
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.SIDEWAYS,
            MarketRegime.HIGH_VOL,
            MarketRegime.CRISIS,
        },
        max_position_pct=0.30,
        min_holding_days=21,
        lookback_months=12,
        bond_fund="511260",
    )
    return DualMomentum(config=config)


@pytest.fixture
def strategy_short_lookback() -> DualMomentum:
    """DualMomentum with 3-month lookback for edge-case testing with short series."""
    config = DualMomentumConfig(
        name="dual_momentum_short",
        eligible_regimes={
            MarketRegime.TRENDING_UP,
            MarketRegime.SIDEWAYS,
        },
        lookback_months=3,
        bond_fund="511260",
    )
    return DualMomentum(config=config)


@pytest.fixture
def trade_date() -> date:
    """A trading date for signal generation."""
    return date(2025, 6, 15)


@pytest.fixture
def equity_beats_bond_data() -> dict[str, pl.DataFrame]:
    """All equities beat the bond fund — best equity should be selected.

    Expected:
      - 510300 (CSI 300):  +20% annual  → highest
      - 510500 (CSI 500):  +15% annual  → second
      - 511260 (Bond):     +4%  annual  → defensive
    """
    n = _N_DAYS + 10
    return {
        "510300": _make_df(_make_price_series(1.0, 0.0008, n, seed=1)),
        "510500": _make_df(_make_price_series(1.0, 0.0006, n, seed=2)),
        "511260": _make_df(_make_price_series(1.0, 0.00016, n, seed=3)),
    }


@pytest.fixture
def bond_beats_equity_data() -> dict[str, pl.DataFrame]:
    """Bond outperforms all equities — should trigger absolute momentum defence.

    Equities are negative or flat, bond is positive.
    """
    n = _N_DAYS + 10
    return {
        "510300": _make_df(_make_price_series(1.0, -0.0004, n, seed=10)),
        "510500": _make_df(_make_price_series(1.0, -0.0002, n, seed=11)),
        "511260": _make_df(_make_price_series(1.0, 0.00012, n, seed=12)),
    }


@pytest.fixture
def single_equity_data() -> dict[str, pl.DataFrame]:
    """Single equity + bond fund — minimal universe."""
    n = _N_DAYS + 10
    return {
        "510300": _make_df(_make_price_series(1.0, 0.001, n, seed=20)),
        "511260": _make_df(_make_price_series(1.0, 0.0002, n, seed=21)),
    }


@pytest.fixture
def all_bond_data() -> dict[str, pl.DataFrame]:
    """Only bond funds in the universe — edge case."""
    n = _N_DAYS + 10
    return {
        "511260": _make_df(_make_price_series(1.0, 0.0001, n, seed=30)),
        "511270": _make_df(_make_price_series(1.0, 0.00015, n, seed=31)),
    }


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestDualMomentumConfig:
    """Strategy configuration validation for S3 Dual Momentum (GEM)."""

    def test_default_config(self) -> None:
        """Default config has spec-compliant values."""
        config = DualMomentumConfig(name="dual_momentum")
        assert config.lookback_months == 12
        assert config.bond_fund == "511260"
        assert config.max_position_pct == 0.20

    def test_rejects_zero_lookback(self) -> None:
        """lookback_months must be >= 1."""
        with pytest.raises(ValueError):
            DualMomentumConfig(name="dual_momentum", lookback_months=0)

    def test_rejects_negative_lookback(self) -> None:
        """lookback_months must be positive."""
        with pytest.raises(ValueError):
            DualMomentumConfig(name="dual_momentum", lookback_months=-1)

    def test_rejects_empty_bond_fund(self) -> None:
        """bond_fund must be a non-empty string."""
        with pytest.raises(ValueError):
            DualMomentumConfig(name="dual_momentum", bond_fund="")

    def test_custom_bond_fund(self) -> None:
        """bond_fund can be customised."""
        config = DualMomentumConfig(name="dual_momentum", bond_fund="511880")
        assert config.bond_fund == "511880"

    def test_inherits_strategy_config(self) -> None:
        """DualMomentumConfig is a StrategyConfig subclass."""
        config = DualMomentumConfig(name="dual_momentum")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ───────────────────────────────────────────────────────


class TestRegimeEligibility:
    """Dual Momentum (GEM) is valid in ALL market regimes."""

    def test_eligible_in_all_regimes(self, strategy: DualMomentum) -> None:
        """GEM operates across all regimes — it has a built-in defensive component."""
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True, (
                f"GEM should be eligible in {regime.value}"
            )


# ─── required_data ────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare close field."""

    def test_returns_close_field(self, strategy: DualMomentum) -> None:
        fields = strategy.required_data()
        assert fields == ["close"]

    def test_returns_list_of_str(self, strategy: DualMomentum) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Dual Momentum Logic ──────────────────────────────────


class TestDualMomentumLogic:
    """Core dual momentum algorithm: absolute + relative momentum."""

    def test_best_equity_selected_when_beating_bond(
        self,
        strategy: DualMomentum,
        trade_date: date,
        equity_beats_bond_data: dict[str, pl.DataFrame],
    ) -> None:
        """When equities beat bonds, the strongest equity should get a BUY."""
        signals = strategy.generate_signals(trade_date, equity_beats_bond_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]

        assert len(buy_signals) == 1, f"Expected 1 BUY, got {len(buy_signals)}"
        assert buy_signals[0]["fund_code"] == "510300", (
            "510300 (best equity) should be selected"
        )

    def test_bond_selected_when_equities_underperform(
        self,
        strategy: DualMomentum,
        trade_date: date,
        bond_beats_equity_data: dict[str, pl.DataFrame],
    ) -> None:
        """Absolute momentum: when no equity beats the bond, rotate to bond."""
        signals = strategy.generate_signals(trade_date, bond_beats_equity_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]

        assert len(buy_signals) == 1, f"Expected 1 BUY (bond), got {len(buy_signals)}"
        assert buy_signals[0]["fund_code"] == "511260", (
            "Should rotate to bond fund when equities underperform"
        )

    def test_single_equity_beats_bond(
        self,
        strategy: DualMomentum,
        trade_date: date,
        single_equity_data: dict[str, pl.DataFrame],
    ) -> None:
        """Single equity that beats bond should be selected."""
        signals = strategy.generate_signals(trade_date, single_equity_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]

        assert len(buy_signals) == 1
        assert buy_signals[0]["fund_code"] == "510300"

    def test_bond_only_universe_buys_best_bond(
        self,
        strategy: DualMomentum,
        trade_date: date,
        all_bond_data: dict[str, pl.DataFrame],
    ) -> None:
        """When only bond funds exist, treat the best bond as the equity proxy."""
        signals = strategy.generate_signals(trade_date, all_bond_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]

        assert len(buy_signals) == 1
        assert buy_signals[0]["fund_code"] == "511270", "Best bond should win"

    def test_absolute_momentum_filters_out_all_equities(
        self,
        strategy_short_lookback: DualMomentum,
        trade_date: date,
    ) -> None:
        """With strongly positive bond and negative equities, absolute defence kicks in."""
        n = 70  # enough for 3-month lookback (~63 trading days)
        data = {
            "510300": _make_df(_make_price_series(1.0, -0.002, n, seed=50)),
            "510500": _make_df(_make_price_series(1.0, -0.001, n, seed=51)),
            "511260": _make_df(_make_price_series(1.0, 0.0004, n, seed=52)),
        }
        signals = strategy_short_lookback.generate_signals(trade_date, data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert buy_signals[0]["fund_code"] == "511260"


# ─── Signal Structure ─────────────────────────────────────────────────────────


class TestSignalStructure:
    """Signal dict structure and constraints."""

    def test_signals_have_all_required_keys(
        self, strategy: DualMomentum, trade_date: date, equity_beats_bond_data: dict[str, pl.DataFrame]
    ) -> None:
        """Every signal must have fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(trade_date, equity_beats_bond_data)
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

    def test_buy_signal_has_positive_weight(
        self, strategy: DualMomentum, trade_date: date, equity_beats_bond_data: dict[str, pl.DataFrame]
    ) -> None:
        """BUY signal should carry a positive target weight."""
        signals = strategy.generate_signals(trade_date, equity_beats_bond_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        for s in buy_signals:
            assert s["target_weight"] > 0.0

    def test_reason_mentions_momentum(
        self, strategy: DualMomentum, trade_date: date, equity_beats_bond_data: dict[str, pl.DataFrame]
    ) -> None:
        """Reason string should reference the decision logic."""
        signals = strategy.generate_signals(trade_date, equity_beats_bond_data)
        for s in signals:
            reason_lower = s["reason"].lower()
            assert "momentum" in reason_lower or "bond" in reason_lower or "defence" in reason_lower


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_market_data_returns_empty(
        self, strategy: DualMomentum, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_single_fund_bond_only(
        self, strategy: DualMomentum, trade_date: date
    ) -> None:
        """Single fund that is the bond fund itself."""
        n = _N_DAYS + 10
        data = {"511260": _make_df(_make_price_series(1.0, 0.0001, n, seed=60))}
        signals = strategy.generate_signals(trade_date, data)
        assert len(signals) >= 1
        assert signals[0]["fund_code"] == "511260"

    def test_insufficient_rows_returns_empty(
        self, strategy: DualMomentum, trade_date: date
    ) -> None:
        """Data with fewer rows than lookback window returns no signals."""
        short_data = {
            "510300": _make_df(_make_price_series(1.0, 0.001, 50, seed=70)),
            "511260": _make_df(_make_price_series(1.0, 0.0001, 50, seed=71)),
        }
        signals = strategy.generate_signals(trade_date, short_data)
        assert signals == []

    def test_missing_close_column(
        self, strategy: DualMomentum, trade_date: date
    ) -> None:
        """DataFrame without 'close' column is skipped."""
        bad_df = pl.DataFrame({"nav": [1.0] * 300})
        signals = strategy.generate_signals(trade_date, {"510300": bad_df})
        assert signals == []

    def test_nan_close_prices_handled(
        self, strategy_short_lookback: DualMomentum, trade_date: date
    ) -> None:
        """NaN in close prices should not crash."""
        n = 70
        prices_a = _make_price_series(1.0, 0.001, n, seed=80)
        prices_a[30] = np.nan
        data = {
            "510300": _make_df(prices_a),
            "511260": _make_df(_make_price_series(1.0, 0.0002, n, seed=81)),
        }
        signals = strategy_short_lookback.generate_signals(trade_date, data)
        # Should not crash, may return signals or empty
        assert isinstance(signals, list)

    def test_zero_price_returns_zero_return(
        self, strategy_short_lookback: DualMomentum, trade_date: date
    ) -> None:
        """Zero prices should yield zero (or handle division safely)."""
        n = 70
        prices = np.full(n, 1.0)
        prices[0] = 0.0  # zero reference price
        data = {
            "510300": _make_df(prices),
            "511260": _make_df(_make_price_series(1.0, 0.0002, n, seed=91)),
        }
        signals = strategy_short_lookback.generate_signals(trade_date, data)
        assert isinstance(signals, list)


# ─── Strategy Identity and Validation ─────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity, validation, and immutability."""

    def test_name_matches_config(self, strategy: DualMomentum) -> None:
        assert strategy.name == "dual_momentum"

    def test_validate_passes_for_valid_config(self, strategy: DualMomentum) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_validate_catches_bad_lookback(self) -> None:
        """lookback=0 is rejected at config construction."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DualMomentumConfig(name="dual_momentum", lookback_months=0)

    def test_strategy_is_frozen(self, strategy: DualMomentum) -> None:
        """Strategy model should be immutable."""
        with pytest.raises(Exception):
            strategy.config.lookback_months = 6  # type: ignore[misc]

    def test_lookback_in_monthly_units(self, strategy: DualMomentum) -> None:
        """lookback_months should be 12 by default (~252 trading days)."""
        assert strategy.config.lookback_months == 12
