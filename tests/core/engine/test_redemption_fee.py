"""
Tests for src/core/engine/redemption_fee.py — 2026 statutory redemption fee.

Covers:
  - Four-tier holding-period fee schedule: <7d, 7-30d, 30-180d, >=180d
  - ETF on-exchange → always zero fee
  - Exempt categories (INDEX_FUND, MONEY_MARKET) → contract-based rates
  - FIFO lot-level holding period calculation
  - enforce_min_hold() warning levels
"""

from datetime import date

import pytest

from src.core.data.schema import (
    FundCategory,
    FundChannel,
    FundTradingProfile,
    PositionLot,
)
from src.core.engine.redemption_fee import (
    HoldingWarning,
    RedemptionFeeCalculator,
)


@pytest.fixture
def calc() -> RedemptionFeeCalculator:
    return RedemptionFeeCalculator()


@pytest.fixture
def otc_equity_profile() -> FundTradingProfile:
    """Standard OTC equity fund — statutory minimum fees apply."""
    return FundTradingProfile(fund_code="005827", channel=FundChannel.OTC_OPEN_END)


@pytest.fixture
def etf_profile() -> FundTradingProfile:
    """Exchange-traded fund — zero redemption fee."""
    return FundTradingProfile(fund_code="510300", channel=FundChannel.ETF_ON_EXCHANGE)


# ─── Statutory Fee Schedule Tests ────────────────────────────────────────────


class TestStatutoryFeeSchedule:
    """Verify the 2026 statutory redemption fee tiers."""

    @pytest.mark.parametrize(
        "holding_days, expected_rate",
        [
            (1, 0.015),    # < 7 days → 1.5%
            (3, 0.015),
            (6, 0.015),
            (7, 0.010),    # 7-30 days → 1.0%
            (15, 0.010),
            (29, 0.010),
            (30, 0.005),   # 30-180 days → 0.5%
            (60, 0.005),
            (179, 0.005),
            (180, 0.0),    # >= 180 days → 0%
            (365, 0.0),
        ],
    )
    def test_fee_tiers(
        self,
        calc: RedemptionFeeCalculator,
        otc_equity_profile: FundTradingProfile,
        holding_days: int,
        expected_rate: float,
    ) -> None:
        """Each holding period maps to the correct statutory rate."""
        fee_amount, rate = calc.calculate(
            fund_profile=otc_equity_profile,
            fund_category=FundCategory.EQUITY,
            holding_days=holding_days,
            redemption_amount=10000.0,
        )
        assert rate == expected_rate
        assert fee_amount == pytest.approx(10000.0 * expected_rate, rel=1e-6)


class TestETFFees:
    """ETF secondary-market trades never incur redemption fees."""

    def test_etf_always_zero(
        self, calc: RedemptionFeeCalculator, etf_profile: FundTradingProfile
    ) -> None:
        """ETF: any holding period → 0 fee."""
        for days in [1, 7, 30, 180, 365]:
            fee_amount, rate = calc.calculate(
                fund_profile=etf_profile,
                fund_category=FundCategory.INDEX_FUND,
                holding_days=days,
                redemption_amount=10000.0,
            )
            assert rate == 0.0, f"ETF fee should be 0 for {days} days"
            assert fee_amount == 0.0


class TestExemptCategories:
    """INDEX_FUND and MONEY_MARKET use contract schedule, not statutory min."""

    def test_index_fund_with_contract(
        self, calc: RedemptionFeeCalculator
    ) -> None:
        """Index fund: contract says >=7d → 0.1%, not statutory 1.0%."""
        profile = FundTradingProfile(
            fund_code="160119",
            channel=FundChannel.OTC_OPEN_END,
            redemption_fee_schedule={(0, 7): 0.005, (7, 365): 0.001},
        )
        fee_amount, rate = calc.calculate(
            fund_profile=profile,
            fund_category=FundCategory.INDEX_FUND,
            holding_days=15,
            redemption_amount=10000.0,
        )
        assert rate == 0.001  # contract rate, not statutory 0.010


class TestFIFOHoldingPeriod:
    """FIFO holding period is computed from the earliest lot."""

    def test_holding_days_from_earliest_lot(
        self, calc: RedemptionFeeCalculator
    ) -> None:
        """When multiple lots exist, FIFO uses the earliest purchase_date."""
        lots = [
            PositionLot(
                purchase_date=date(2026, 6, 1),
                shares=1000.0,
                purchase_nav=1.5,
                cost_amount=1500.0,
            ),
            PositionLot(
                purchase_date=date(2026, 6, 15),
                shares=500.0,
                purchase_nav=1.6,
                cost_amount=800.0,
            ),
        ]
        current = date(2026, 6, 20)
        # Oldest lot: June 1 → 19 days holding → 7-30 day tier → 1.0%
        fee_amount, rate = calc.calculate_for_lots(
            fund_profile=FundTradingProfile(
                fund_code="005827", channel=FundChannel.OTC_OPEN_END
            ),
            fund_category=FundCategory.EQUITY,
            lots=lots,
            shares_to_redeem=500.0,  # only consume part of oldest lot
            current_nav=2.0,
            current_date=current,
        )
        # 500 shares from June 1 lot → 19 days → 1.0% statutory
        assert rate == 0.010


class TestEnforceMinHold:
    """Warning levels for short holding periods."""

    def test_less_than_7_days_rejected(
        self, calc: RedemptionFeeCalculator
    ) -> None:
        """Holding < 7 days → REJECT warning."""
        result = calc.enforce_min_hold(
            holding_days=5,
            channel=FundChannel.OTC_OPEN_END,
        )
        assert result == HoldingWarning.REJECT

    def test_between_7_and_30_days_warned(
        self, calc: RedemptionFeeCalculator
    ) -> None:
        """Holding 7-30 days → WARN."""
        result = calc.enforce_min_hold(
            holding_days=15,
            channel=FundChannel.OTC_OPEN_END,
        )
        assert result == HoldingWarning.WARN

    def test_over_30_days_pass(
        self, calc: RedemptionFeeCalculator
    ) -> None:
        """Holding >= 30 days → PASS."""
        result = calc.enforce_min_hold(
            holding_days=60,
            channel=FundChannel.OTC_OPEN_END,
        )
        assert result == HoldingWarning.PASS

    def test_etf_always_pass(
        self, calc: RedemptionFeeCalculator
    ) -> None:
        """ETF: any holding period → PASS (no redemption fee)."""
        for days in [1, 3, 7, 15, 60]:
            result = calc.enforce_min_hold(
                holding_days=days,
                channel=FundChannel.ETF_ON_EXCHANGE,
            )
            assert result == HoldingWarning.PASS, f"ETF should PASS for {days}d"
