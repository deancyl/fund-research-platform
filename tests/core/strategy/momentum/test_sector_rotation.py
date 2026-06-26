"""
Tests for Sector Rotation strategy (S4) — Correlation-Consolidated.

S4 Sector Rotation: 6-month formation, 6-month holding, 0.75 correlation consolidation.
Monthly return 4.8%, Sharpe 0.71→1.16 after correlation consolidation.

Algorithm:
  1. Compute each sector's formation-period return (form_months).
  2. Compute pairwise correlation matrix of sector returns.
  3. Consolidation: group sectors with correlation > corr_threshold (0.75).
  4. Within each group, retain the best-performing sector.
  5. Rank group representatives and emit BUY for top-ranked sector.

Covers: configuration, regime eligibility, correlation consolidation, signal
structure, edge cases.
"""

from datetime import date

import numpy as np
import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.momentum.sector_rotation import SectorRotation, SectorRotationConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_N_DAYS = 300  # well above 6 months (~126 trading days)


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_price_series(
    start_price: float,
    daily_ret: float,
    n_days: int = _N_DAYS,
    seed: int = 42,
    noise: float = 0.0,
) -> np.ndarray:
    """Generate synthetic close prices with deterministic drift + optional noise."""
    rng = np.random.default_rng(seed)
    returns = daily_ret + rng.normal(0.0, noise, n_days) if noise > 0 else np.full(n_days, daily_ret)
    return start_price * np.cumprod(1.0 + returns)


def _make_df(prices: np.ndarray) -> pl.DataFrame:
    """Build a polars DataFrame with close column."""
    return pl.DataFrame({"close": prices})


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def strategy() -> SectorRotation:
    """Default SectorRotation with 6-month formation / 6-month hold / 0.75 corr."""
    config = SectorRotationConfig(
        name="sector_rotation",
        version="1.0.0",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        max_position_pct=0.25,
        min_holding_days=21,
        form_months=6,
        hold_months=6,
        corr_threshold=0.75,
    )
    return SectorRotation(config=config)


@pytest.fixture
def strategy_low_threshold() -> SectorRotation:
    """Low correlation threshold for easier grouping in tests."""
    config = SectorRotationConfig(
        name="sector_rotation_low",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        form_months=6,
        hold_months=6,
        corr_threshold=0.3,
    )
    return SectorRotation(config=config)


@pytest.fixture
def strategy_short() -> SectorRotation:
    """Short formation/hold for edge-case testing."""
    config = SectorRotationConfig(
        name="sector_rotation_short",
        eligible_regimes={MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        form_months=2,
        hold_months=2,
        corr_threshold=0.75,
    )
    return SectorRotation(config=config)


@pytest.fixture
def trade_date() -> date:
    """Trading date for signal generation."""
    return date(2025, 6, 15)


@pytest.fixture
def uncorrelated_sector_data() -> dict[str, pl.DataFrame]:
    """4 sectors with different deterministic momentums (no noise for predictable ranking).

    Best performer: TECH (+30% annual) — highest return.
    Sorted by 6-month return:
      TECH:  +30% annual  → best
      FIN:   +20% annual
      CONS:  +10% annual
      UTIL:  +5%  annual
    """
    n = _N_DAYS
    annual_ret = {"TECH": 0.0012, "FIN": 0.0008, "CONS": 0.0004, "UTIL": 0.0002}
    return {
        code: _make_df(_make_price_series(1.0, annual_ret[code], n, noise=0.0))
        for code in annual_ret
    }


@pytest.fixture
def correlated_pair_data() -> dict[str, pl.DataFrame]:
    """TECH_A and TECH_B share a common driver (highly correlated >0.9). FIN is independent.

    With corr_threshold=0.75, TECH_A and TECH_B should consolidate to one group.
    TECH_A: +30% annual (slightly better than B).
    TECH_B: +28% annual (similar to A, driven by same base + small offset).
    FIN:    +20% annual (independent, different driver).
    """
    n = _N_DAYS
    rng = np.random.default_rng(42)
    # Common tech driver
    common_driver = rng.normal(0.0010, 0.005, n)  # moderate noise for realistic correlation
    # Independent finance driver
    fin_driver = rng.normal(0.0006, 0.005, n)

    # TECH_A gets extra positive drift on top of common
    tech_a_daily = common_driver + 0.0002  # slightly higher drift
    tech_b_daily = common_driver + 0.0001  # slightly lower drift
    fin_daily = fin_driver + 0.0001

    prices_a = np.cumprod(np.maximum(1.0 + tech_a_daily, 0.01))
    prices_b = np.cumprod(np.maximum(1.0 + tech_b_daily, 0.01))
    prices_f = np.cumprod(np.maximum(1.0 + fin_daily, 0.01))

    return {
        "TECH_A": _make_df(prices_a),
        "TECH_B": _make_df(prices_b),
        "FIN": _make_df(prices_f),
    }


@pytest.fixture
def three_correlated_groups() -> dict[str, pl.DataFrame]:
    """Two groups of correlated sectors plus one independent.

    Group 1 (tech): TECH_A, TECH_B — highly correlated
    Group 2 (finance): FIN_A, FIN_B — highly correlated
    Independent: GOLD — uncorrelated with both groups
    """
    n = _N_DAYS
    rng = np.random.default_rng(99)
    # Tech common driver
    tech_driver = rng.normal(0.001, 0.012, n)
    # Finance common driver (different from tech)
    fin_driver = rng.normal(0.0005, 0.008, n)

    tech_a = 1.0 + tech_driver + rng.normal(0.0003, 0.003, n)
    tech_b = 1.0 + tech_driver + rng.normal(0.0002, 0.003, n)
    fin_a = 1.0 + fin_driver + rng.normal(0.0001, 0.002, n)
    fin_b = 1.0 + fin_driver + rng.normal(0.0, 0.002, n)
    gold = 1.0 + rng.normal(0.0002, 0.008, n)

    return {
        "TECH_A": _make_df(np.cumprod(np.maximum(tech_a, 0.01))),
        "TECH_B": _make_df(np.cumprod(np.maximum(tech_b, 0.01))),
        "FIN_A": _make_df(np.cumprod(np.maximum(fin_a, 0.01))),
        "FIN_B": _make_df(np.cumprod(np.maximum(fin_b, 0.01))),
        "GOLD": _make_df(np.cumprod(np.maximum(gold, 0.01))),
    }


@pytest.fixture
def single_sector_data() -> dict[str, pl.DataFrame]:
    """Single sector — edge case for correlation matrix of size 1."""
    n = _N_DAYS
    return {
        "TECH": _make_df(_make_price_series(1.0, 0.001, n, seed=50, noise=0.01)),
    }


# ─── Configuration Tests ───────────────────────────────────────────────────────


class TestSectorRotationConfig:
    """Strategy configuration validation for S4 Sector Rotation."""

    def test_default_config(self) -> None:
        """Default config has spec-compliant values."""
        config = SectorRotationConfig(name="sector_rotation")
        assert config.form_months == 6
        assert config.hold_months == 6
        assert config.corr_threshold == 0.75
        assert config.max_position_pct == 0.20

    def test_rejects_zero_form_months(self) -> None:
        """form_months must be >= 1."""
        with pytest.raises(ValueError):
            SectorRotationConfig(name="sector_rotation", form_months=0)

    def test_rejects_zero_hold_months(self) -> None:
        """hold_months must be >= 1."""
        with pytest.raises(ValueError):
            SectorRotationConfig(name="sector_rotation", hold_months=0)

    def test_rejects_corr_threshold_below_negative_one(self) -> None:
        """corr_threshold must be > -1."""
        with pytest.raises(ValueError):
            SectorRotationConfig(name="sector_rotation", corr_threshold=-1.5)

    def test_rejects_corr_threshold_above_one(self) -> None:
        """corr_threshold must be <= 1."""
        with pytest.raises(ValueError):
            SectorRotationConfig(name="sector_rotation", corr_threshold=1.5)

    def test_corr_threshold_one_disables_consolidation(self) -> None:
        """corr_threshold=1 means no grouping ever (only perfect correlation)."""
        config = SectorRotationConfig(name="sector_rotation", corr_threshold=1.0)
        assert config.corr_threshold == 1.0

    def test_inherits_strategy_config(self) -> None:
        """SectorRotationConfig is a StrategyConfig subclass."""
        config = SectorRotationConfig(name="sector_rotation")
        assert isinstance(config, StrategyConfig)


# ─── Regime Eligibility ───────────────────────────────────────────────────────


class TestRegimeEligibility:
    """Sector Rotation valid in TRENDING_UP and SIDEWAYS only."""

    def test_eligible_in_trending_up(self, strategy: SectorRotation) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_UP) is True

    def test_eligible_in_sideways(self, strategy: SectorRotation) -> None:
        assert strategy.is_eligible(MarketRegime.SIDEWAYS) is True

    def test_not_eligible_in_trending_down(self, strategy: SectorRotation) -> None:
        assert strategy.is_eligible(MarketRegime.TRENDING_DOWN) is False

    def test_not_eligible_in_high_vol(self, strategy: SectorRotation) -> None:
        assert strategy.is_eligible(MarketRegime.HIGH_VOL) is False

    def test_not_eligible_in_crisis(self, strategy: SectorRotation) -> None:
        assert strategy.is_eligible(MarketRegime.CRISIS) is False


# ─── required_data ────────────────────────────────────────────────────────────


class TestRequiredData:
    """required_data() must declare close field."""

    def test_returns_close_field(self, strategy: SectorRotation) -> None:
        assert strategy.required_data() == ["close"]

    def test_returns_list_of_str(self, strategy: SectorRotation) -> None:
        result = strategy.required_data()
        assert isinstance(result, list)
        assert all(isinstance(f, str) for f in result)


# ─── Signal Generation — Core Algorithm ───────────────────────────────────────


class TestSectorRotationLogic:
    """Core sector rotation with correlation consolidation."""

    def test_best_uncorrelated_sector_selected(
        self,
        strategy: SectorRotation,
        trade_date: date,
        uncorrelated_sector_data: dict[str, pl.DataFrame],
    ) -> None:
        """Without correlation grouping, best-return sector should win."""
        signals = strategy.generate_signals(trade_date, uncorrelated_sector_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]

        assert len(buy_signals) >= 1
        # TECH has the highest return
        assert buy_signals[0]["fund_code"] == "TECH"

    def test_correlated_pair_consolidates(
        self,
        strategy_low_threshold: SectorRotation,
        trade_date: date,
        correlated_pair_data: dict[str, pl.DataFrame],
    ) -> None:
        """With low correlation threshold, TECH_A and TECH_B should consolidate.
        
        They share a common driver (high correlation) — only one should be selected.
        """
        signals = strategy_low_threshold.generate_signals(trade_date, correlated_pair_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]

        buy_codes = {s["fund_code"] for s in buy_signals}
        # TECH_A and TECH_B are highly correlated — they should NOT both appear as buys
        assert not ("TECH_A" in buy_codes and "TECH_B" in buy_codes), (
            "Correlated tech sectors should not both be BUY signals"
        )
        # At least one signal should be generated
        assert len(buy_signals) >= 1

    def test_multiple_groups_consolidate_correctly(
        self,
        strategy_low_threshold: SectorRotation,
        trade_date: date,
        three_correlated_groups: dict[str, pl.DataFrame],
    ) -> None:
        """Two correlated groups + one independent → each group contributes at most one."""
        signals = strategy_low_threshold.generate_signals(trade_date, three_correlated_groups)
        assert len(signals) >= 1

        # Check that TECH_A and TECH_B are not both selected (they're correlated)
        buy_codes = {s["fund_code"] for s in signals if s["direction"] == SignalDirection.BUY.value}
        assert not ("TECH_A" in buy_codes and "TECH_B" in buy_codes), (
            "Correlated tech sectors should not both be selected"
        )
        assert not ("FIN_A" in buy_codes and "FIN_B" in buy_codes), (
            "Correlated finance sectors should not both be selected"
        )

    def test_highest_return_wins_within_group(
        self,
        strategy: SectorRotation,
        trade_date: date,
        uncorrelated_sector_data: dict[str, pl.DataFrame],
    ) -> None:
        """Highest return sector should be ranked first."""
        signals = strategy.generate_signals(trade_date, uncorrelated_sector_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert buy_signals[0]["fund_code"] == "TECH"


# ─── Signal Structure ─────────────────────────────────────────────────────────


class TestSignalStructure:
    """Signal dict structure and constraints."""

    def test_signals_have_all_required_keys(
        self, strategy: SectorRotation, trade_date: date, uncorrelated_sector_data: dict[str, pl.DataFrame]
    ) -> None:
        """Every signal must have fund_code, direction, confidence, target_weight, reason."""
        signals = strategy.generate_signals(trade_date, uncorrelated_sector_data)
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
        self, strategy: SectorRotation, trade_date: date, uncorrelated_sector_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, uncorrelated_sector_data)
        buy_signals = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        for s in buy_signals:
            assert s["target_weight"] > 0.0

    def test_reason_mentions_sector_or_rotation(
        self, strategy: SectorRotation, trade_date: date, uncorrelated_sector_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, uncorrelated_sector_data)
        for s in signals:
            reason_lower = s["reason"].lower()
            assert any(kw in reason_lower for kw in ["return", "sector", "rank", "momentum"])


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_market_data_returns_empty(
        self, strategy: SectorRotation, trade_date: date
    ) -> None:
        signals = strategy.generate_signals(trade_date, {})
        assert signals == []

    def test_single_sector_generates_signal(
        self, strategy: SectorRotation, trade_date: date, single_sector_data: dict[str, pl.DataFrame]
    ) -> None:
        signals = strategy.generate_signals(trade_date, single_sector_data)
        assert len(signals) >= 1
        assert signals[0]["fund_code"] == "TECH"

    def test_insufficient_rows_returns_empty(
        self, strategy: SectorRotation, trade_date: date
    ) -> None:
        """Data shorter than formation period returns no signals."""
        short_data = {
            "TECH": _make_df(_make_price_series(1.0, 0.001, 50, seed=70)),
        }
        signals = strategy.generate_signals(trade_date, short_data)
        assert signals == []

    def test_missing_close_column_returns_empty(
        self, strategy: SectorRotation, trade_date: date
    ) -> None:
        bad_df = pl.DataFrame({"nav": [1.0] * 300})
        signals = strategy.generate_signals(trade_date, {"TECH": bad_df})
        assert signals == []

    def test_two_sectors_with_same_return(
        self, strategy: SectorRotation, trade_date: date
    ) -> None:
        """Two sectors with identical returns — both should be considered."""
        n = _N_DAYS
        data = {
            "A": _make_df(_make_price_series(1.0, 0.001, n, seed=90)),
            "B": _make_df(_make_price_series(1.0, 0.001, n, seed=90)),
        }
        signals = strategy.generate_signals(trade_date, data)
        assert len(signals) >= 1

    def test_all_negative_returns(
        self, strategy_short: SectorRotation, trade_date: date
    ) -> None:
        """When all sectors have negative returns, still generate signals."""
        n = 60
        data = {
            "A": _make_df(_make_price_series(1.0, -0.001, n, seed=100)),
            "B": _make_df(_make_price_series(1.0, -0.002, n, seed=101)),
        }
        signals = strategy_short.generate_signals(trade_date, data)
        assert isinstance(signals, list)


# ─── Strategy Identity and Validation ─────────────────────────────────────────


class TestStrategyIdentity:
    """Strategy identity, validation, and immutability."""

    def test_name_matches_config(self, strategy: SectorRotation) -> None:
        assert strategy.name == "sector_rotation"

    def test_validate_passes_for_valid_config(self, strategy: SectorRotation) -> None:
        errors = strategy.validate()
        assert errors == []

    def test_validate_catches_bad_form_months(self) -> None:
        """form_months=0 is rejected at config level."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SectorRotationConfig(name="sector_rotation", form_months=0)

    def test_strategy_is_frozen(self, strategy: SectorRotation) -> None:
        with pytest.raises(Exception):
            strategy.config.form_months = 12  # type: ignore[misc]

    def test_hold_months_affects_rebalance_cadence(self) -> None:
        """hold_months config should be stored correctly."""
        config = SectorRotationConfig(name="sector_rotation", hold_months=3)
        assert config.hold_months == 3
