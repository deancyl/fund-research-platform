"""
Tests for src/core/data/schema.py — FundChannel, PositionLot, FundPosition.

Covers: happy path creation, FIFO ordering invariant, redemption fee schedule lookup,
fund category exemption logic, DateTime validation.
"""

from datetime import date, time
import pytest

# Will fail until schema.py is implemented — this is the RED phase
from src.core.data.schema import (
    FundCategory,
    FundChannel,
    FundTradingProfile,
    PositionLot,
    FundPosition,
)


class TestFundChannel:
    def test_four_channels_exist(self) -> None:
        """All four trading channels must be defined."""
        members = {m.value for m in FundChannel}
        assert members == {"ETF_ON_EXCHANGE", "OTC_OPEN_END", "OTC_ETF_FEEDER", "QDII"}

    def test_channel_is_str_enum(self) -> None:
        """FundChannel must be a StrEnum so values are directly usable as strings."""
        assert isinstance(FundChannel.ETF_ON_EXCHANGE.value, str)


class TestFundCategory:
    def test_exempt_categories_exist(self) -> None:
        """Categories exempt from statutory redemption fee minimums."""
        members = {m.value for m in FundCategory}
        assert "INDEX_FUND" in members
        assert "MONEY_MARKET" in members


class TestPositionLot:
    def test_create_lot(self) -> None:
        """Happy path: create a valid PositionLot."""
        lot = PositionLot(
            purchase_date=date(2026, 3, 1),
            shares=1000.0,
            purchase_nav=1.5,
            cost_amount=1500.0,
        )
        assert lot.shares == 1000.0
        assert lot.purchase_nav == 1.5
        assert lot.cost_amount == 1500.0

    def test_shares_must_be_positive(self) -> None:
        """Shares must be > 0."""
        with pytest.raises(ValueError):
            PositionLot(
                purchase_date=date(2026, 3, 1),
                shares=0.0,
                purchase_nav=1.5,
                cost_amount=0.0,
            )

    def test_purchase_nav_must_be_positive(self) -> None:
        """NAV must be > 0."""
        with pytest.raises(ValueError):
            PositionLot(
                purchase_date=date(2026, 3, 1),
                shares=100.0,
                purchase_nav=0.0,
                cost_amount=150.0,
            )


class TestFundPosition:
    def test_create_position(self) -> None:
        """Happy path: create a valid FundPosition with multiple lots."""
        lot1 = PositionLot(
            purchase_date=date(2026, 3, 1),
            shares=1000.0,
            purchase_nav=1.5,
            cost_amount=1500.0,
        )
        lot2 = PositionLot(
            purchase_date=date(2026, 5, 1),
            shares=500.0,
            purchase_nav=1.8,
            cost_amount=900.0,
        )
        pos = FundPosition(
            fund_code="005827",
            fund_name="易方达蓝筹精选",
            category=FundCategory.EQUITY,
            channel=FundChannel.OTC_OPEN_END,
            lots=[lot2, lot1],  # intentionally out of order
            current_nav=2.0,
            total_shares=1500.0,
            market_value=3000.0,
            weight_pct=0.15,
        )
        # CRITICAL: lots must be sorted by purchase_date (FIFO invariant)
        assert pos.lots[0].purchase_date < pos.lots[1].purchase_date
        assert pos.lots[0].purchase_date == date(2026, 3, 1)
        assert pos.lots[1].purchase_date == date(2026, 5, 1)

    def test_weight_pct_range(self) -> None:
        """weight_pct must be between 0 and 1."""
        lot = PositionLot(
            purchase_date=date(2026, 3, 1),
            shares=100.0,
            purchase_nav=1.0,
            cost_amount=100.0,
        )
        with pytest.raises(ValueError):
            FundPosition(
                fund_code="005827",
                fund_name="test",
                category=FundCategory.EQUITY,
                channel=FundChannel.OTC_OPEN_END,
                lots=[lot],
                current_nav=1.0,
                total_shares=100.0,
                market_value=100.0,
                weight_pct=1.5,
            )

    def test_total_shares_matches_lots(self) -> None:
        """total_shares must equal sum of lot shares."""
        lot1 = PositionLot(
            purchase_date=date(2026, 3, 1),
            shares=1000.0,
            purchase_nav=1.5,
            cost_amount=1500.0,
        )
        lot2 = PositionLot(
            purchase_date=date(2026, 5, 1),
            shares=500.0,
            purchase_nav=1.8,
            cost_amount=900.0,
        )
        with pytest.raises(ValueError):
            FundPosition(
                fund_code="005827",
                fund_name="test",
                category=FundCategory.EQUITY,
                channel=FundChannel.OTC_OPEN_END,
                lots=[lot1, lot2],
                current_nav=2.0,
                total_shares=999.0,  # wrong — should be 1500
                market_value=3000.0,
                weight_pct=0.15,
            )


class TestFundTradingProfile:
    def test_otc_profile_defaults(self) -> None:
        """Default OTC fund profile has correct settlement delay."""
        profile = FundTradingProfile(
            fund_code="005827",
            channel=FundChannel.OTC_OPEN_END,
        )
        assert profile.cutoff_time == time(15, 0)
        assert profile.settlement_delay_days == 4

    def test_etf_profile_defaults(self) -> None:
        """ETF profile has zero settlement delay."""
        profile = FundTradingProfile(
            fund_code="510300",
            channel=FundChannel.ETF_ON_EXCHANGE,
        )
        assert profile.settlement_delay_days == 0
        assert profile.redemption_fee_schedule == {}

    def test_qdii_profile_defaults(self) -> None:
        """QDII profile has extended settlement delay."""
        profile = FundTradingProfile(
            fund_code="164824",
            channel=FundChannel.QDII,
        )
        assert profile.settlement_delay_days >= 7
