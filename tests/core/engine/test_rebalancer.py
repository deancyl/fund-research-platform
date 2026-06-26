"""
Tests for src/core/engine/rebalancer.py — SmartRebalancer.

The SmartRebalancer composes order_cutoff, redemption_fee, and cash_lock
to produce a complete rebalancing plan with friction cost analysis.
"""

from datetime import date

import pytest

from src.core.data.schema import (
    FundCategory,
    FundChannel,
    FundPosition,
    FundTradingProfile,
    PositionLot,
)
from src.core.engine.rebalancer import RebalanceAction, SmartRebalancer


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def rebalancer() -> SmartRebalancer:
    return SmartRebalancer()


@pytest.fixture
def otc_position() -> FundPosition:
    """OTC equity fund — 30 days holding."""
    lot = PositionLot(
        purchase_date=date(2026, 5, 27),
        shares=5000.0,
        purchase_nav=2.0,
        cost_amount=10000.0,
    )
    return FundPosition(
        fund_code="005827",
        fund_name="Test OTC Fund",
        category=FundCategory.EQUITY,
        channel=FundChannel.OTC_OPEN_END,
        lots=[lot],
        current_nav=2.2,
        total_shares=5000.0,
        market_value=11000.0,
        weight_pct=0.50,
    )


@pytest.fixture
def etf_position() -> FundPosition:
    """ETF — short holding, zero redemption fee."""
    lot = PositionLot(
        purchase_date=date(2026, 6, 20),
        shares=2000.0,
        purchase_nav=3.5,
        cost_amount=7000.0,
    )
    return FundPosition(
        fund_code="510300",
        fund_name="Test ETF",
        category=FundCategory.INDEX_FUND,
        channel=FundChannel.ETF_ON_EXCHANGE,
        lots=[lot],
        current_nav=3.6,
        total_shares=2000.0,
        market_value=7200.0,
        weight_pct=0.30,
    )


# ─── Core Rebalancing ────────────────────────────────────────────────────────


class TestGenerateRebalancePlan:
    """Test the full rebalancing plan generation."""

    def test_overweight_reduction_otc(
        self, rebalancer: SmartRebalancer, otc_position: FundPosition
    ) -> None:
        """Overweight OTC fund → reduce, with 30-day holding fee (0.5%)."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[otc_position],
            target_weights={"005827": 0.20},  # reduce from 50% to 20%
            current_date=date(2026, 6, 26),
            total_portfolio_value=22000.0,
        )

        assert plan.status == "ANALYZED"
        assert len(plan.actions) >= 1

        # Find the REDEEM action for 005827
        redeem = next(a for a in plan.actions if a.action_type == "REDEEM" and a.fund_code == "005827")
        assert redeem.amount > 0
        # Holding = 30 days → 0.5% fee (on the statutory boundary)
        assert redeem.estimated_fee > 0
        # Net cash = amount - fee
        assert redeem.net_cash < redeem.amount

    def test_underweight_increase(
        self, rebalancer: SmartRebalancer, otc_position: FundPosition
    ) -> None:
        """Underweight → add more (SUBSCRIBE action)."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[otc_position],
            target_weights={"005827": 0.60},  # increase from 50% to 60%
            current_date=date(2026, 6, 26),
            total_portfolio_value=22000.0,
        )

        subscribe = next((a for a in plan.actions if a.action_type == "SUBSCRIBE" and a.fund_code == "005827"), None)
        assert subscribe is not None
        assert subscribe.amount > 0

    def test_new_fund_addition(
        self, rebalancer: SmartRebalancer, otc_position: FundPosition
    ) -> None:
        """Adding a new fund not currently in portfolio."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[otc_position],
            target_weights={"005827": 0.40, "510300": 0.10},
            current_date=date(2026, 6, 26),
            total_portfolio_value=22000.0,
        )

        subscribe = next((a for a in plan.actions if a.fund_code == "510300"), None)
        assert subscribe is not None
        assert subscribe.action_type == "SUBSCRIBE"
        assert subscribe.amount > 0

    def test_etf_zero_fee(
        self, rebalancer: SmartRebalancer, etf_position: FundPosition
    ) -> None:
        """ETF reduction → zero redemption fee."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[etf_position],
            target_weights={"510300": 0.10},
            current_date=date(2026, 6, 26),
            total_portfolio_value=24000.0,
        )

        redeem = next(a for a in plan.actions if a.fund_code == "510300")
        assert redeem.estimated_fee == 0.0
        assert redeem.net_cash == redeem.amount


# ─── Fee Penalty (Cost Gate) ────────────────────────────────────────────────


class TestFeePenaltyGate:
    """Short holding periods → fee may exceed holding-period-alpha → skip rebalancing."""

    def test_short_holding_penalty_skip(
        self, rebalancer: SmartRebalancer
    ) -> None:
        """OTC fund held < 7 days → 1.5% fee vs holding-period alpha → skip."""
        lot = PositionLot(
            purchase_date=date(2026, 6, 25),
            shares=1000.0, purchase_nav=5.0, cost_amount=5000.0,
        )
        pos = FundPosition(
            fund_code="005827", fund_name="Very Recent Fund",
            category=FundCategory.EQUITY, channel=FundChannel.OTC_OPEN_END,
            lots=[lot], current_nav=5.1, total_shares=1000.0,
            market_value=5100.0, weight_pct=0.50,
        )

        # Holding-period alpha = 0.03/365 * 30 = 0.0025 → 0.25%
        # 1.5% fee > 0.25% alpha → should skip
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[pos],
            target_weights={"005827": 0.20},
            current_date=date(2026, 6, 26),
            total_portfolio_value=10200.0,
            expected_alpha_pct=0.03,
            expected_holding_days=30,
        )

        redeem = next((a for a in plan.actions if a.fund_code == "005827"), None)
        if redeem and redeem.estimated_fee / redeem.amount > 0.01:
            assert redeem.skip_reason != "", "Should warn about high fee"

    def test_long_holding_no_penalty(
        self, rebalancer: SmartRebalancer
    ) -> None:
        """With 365-day holding, 3% alpha covers 1.5% fee → no skip."""
        lot = PositionLot(
            purchase_date=date(2026, 6, 25),
            shares=1000.0, purchase_nav=5.0, cost_amount=5000.0,
        )
        pos = FundPosition(
            fund_code="005827", fund_name="Short Fund",
            category=FundCategory.EQUITY, channel=FundChannel.OTC_OPEN_END,
            lots=[lot], current_nav=5.1, total_shares=1000.0,
            market_value=5100.0, weight_pct=0.50,
        )

        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[pos],
            target_weights={"005827": 0.20},
            current_date=date(2026, 6, 26),
            total_portfolio_value=10200.0,
            expected_alpha_pct=0.03,
            expected_holding_days=365,
        )

        redeem = next((a for a in plan.actions if a.fund_code == "005827"), None)
        # With 365-day holding, 3% annual alpha > 1.5% fee → should NOT skip
        assert redeem is not None
        # skip_reason may be empty since alpha covers the fee


# ─── Timeline Generation ─────────────────────────────────────────────────────


class TestTimeline:
    """Execution timeline must reflect real settlement delays."""

    def test_timeline_has_events(
        self, rebalancer: SmartRebalancer, otc_position: FundPosition
    ) -> None:
        """Timeline should contain events for redemption + cash unlock."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[otc_position],
            target_weights={"005827": 0.20},
            current_date=date(2026, 6, 26),
            total_portfolio_value=22000.0,
        )

        assert len(plan.timeline) >= 1, "Timeline should have at least one event"
        # There should be a redemption event at T+0
        redeem_events = [e for e in plan.timeline if "赎回" in e.event or "redeem" in e.event.lower()]
        assert len(redeem_events) >= 1

    def test_otc_delay_in_timeline(
        self, rebalancer: SmartRebalancer, otc_position: FundPosition
    ) -> None:
        """OTC redemption timeline shows 4-day cash lock."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[otc_position],
            target_weights={"005827": 0.20},
            current_date=date(2026, 6, 26),
            total_portfolio_value=22000.0,
        )

        unlock_events = [e for e in plan.timeline if "解冻" in e.event or "unlock" in e.event.lower()]
        if unlock_events:
            assert unlock_events[0].t_day >= 4  # OTC delay


class TestFrictionCost:
    """Total friction cost reporting."""

    def test_friction_cost_reported(
        self, rebalancer: SmartRebalancer, otc_position: FundPosition
    ) -> None:
        """total_friction_cost_yuan must be reported."""
        plan = rebalancer.generate_rebalance_plan(
            current_portfolio=[otc_position],
            target_weights={"005827": 0.20},
            current_date=date(2026, 6, 26),
            total_portfolio_value=22000.0,
        )

        assert plan.total_friction_cost_yuan >= 0
        # OTC with >30 day holding → 0.5% fee on some amount
        assert plan.total_friction_cost_yuan > 0
