"""
2026 statutory redemption fee calculator with FIFO lot-level tracking.

China's revised "Publicly Offered Securities Investment Fund Sales Fee Management
Regulations" (公开募集证券投资基金销售费用管理规定) took full effect in 2026.

Key rules:
  - OTC open-end funds: <7d → 1.5%, 7-30d → 1.0%, 30-180d → 0.5%, ≥180d → 0%.
  - ETFs traded on exchange: always 0 redemption fee.
  - Exempt categories (INDEX_FUND, MONEY_MARKET) use contract-based schedules.
  - All redemption fees are paid INTO the fund (全额计入基金财产), acting as a
    penalty on short-term trading that benefits remaining unitholders.

FIFO lot tracking: when only part of a position is sold, lots are consumed in
purchase-date order. The holding period of each lot determines its fee rate.
"""

from datetime import date
from enum import StrEnum

from src.core.data.schema import FundCategory, FundChannel, FundTradingProfile, PositionLot


# ─── Holding Warning ─────────────────────────────────────────────────────────


class HoldingWarning(StrEnum):
    """Warning levels for short holding periods before redemption."""

    PASS = "PASS"
    WARN = "WARN"  # 7-30 days — 1.0% fee applies
    REJECT = "REJECT"  # <7 days — 1.5% penalty fee


# ─── Statutory Fee Schedule ──────────────────────────────────────────────────


# 2026 statutory minimum redemption fee rates for OTC equity/mixed funds.
# Format: (lo_days, hi_days) → rate
_STATUTORY_SCHEDULE: dict[tuple[int, int], float] = {
    (0, 7): 0.015,       # < 7 days → 1.5% punitive
    (7, 30): 0.010,      # 7-30 days → 1.0% (2026 new tier)
    (30, 180): 0.005,    # 30-180 days → 0.5%
    (180, 10_000_000): 0.0,  # ≥ 180 days → 0%
}

# Categories exempt from statutory minimum — use contract schedule instead.
_EXEMPT_CATEGORIES: frozenset[FundCategory] = frozenset({
    FundCategory.INDEX_FUND,
    FundCategory.SPECIAL_BOND,
    FundCategory.MONEY_MARKET,
})


# ─── Calculator ──────────────────────────────────────────────────────────────


class RedemptionFeeCalculator:
    """Calculate redemption fees under 2026 China fund regulations."""

    def calculate(
        self,
        fund_profile: FundTradingProfile,
        fund_category: FundCategory,
        holding_days: int,
        redemption_amount: float,
    ) -> tuple[float, float]:
        """
        Calculate redemption fee for a single redemption.

        Returns:
            (fee_amount_cny, fee_rate) — both 0.0 for ETFs and >=180d holdings.
        """
        # ETF on-exchange: never a redemption fee
        if fund_profile.channel == FundChannel.ETF_ON_EXCHANGE:
            return 0.0, 0.0

        # Determine which schedule to use
        if fund_category in _EXEMPT_CATEGORIES and fund_profile.redemption_fee_schedule:
            rate = self._lookup_fee(fund_profile.redemption_fee_schedule, holding_days)
        else:
            rate = self._lookup_fee(_STATUTORY_SCHEDULE, holding_days)

        return redemption_amount * rate, rate

    def calculate_for_lots(
        self,
        fund_profile: FundTradingProfile,
        fund_category: FundCategory,
        lots: list[PositionLot],
        shares_to_redeem: float,
        current_nav: float,
        current_date: date,
    ) -> tuple[float, float]:
        """
        Calculate redemption fee when consuming lots in FIFO order.

        When only part of a position is sold, lots are drawn from earliest
        purchase to latest. Each lot's holding period determines its fee rate.

        Returns:
            (total_fee_cny, blended_rate)
        """
        if fund_profile.channel == FundChannel.ETF_ON_EXCHANGE:
            return 0.0, 0.0

        remaining = shares_to_redeem
        total_fee = 0.0

        for lot in lots:
            if remaining <= 0:
                break

            shares_from_this_lot = min(lot.shares, remaining)
            lot_value = shares_from_this_lot * current_nav
            holding_days = (current_date - lot.purchase_date).days

            fee_amount, _rate = self.calculate(
                fund_profile=fund_profile,
                fund_category=fund_category,
                holding_days=holding_days,
                redemption_amount=lot_value,
            )
            total_fee += fee_amount
            remaining -= shares_from_this_lot

        # Blended rate: total fee / total redemption value
        total_value = shares_to_redeem * current_nav
        blended_rate = total_fee / total_value if total_value > 0 else 0.0
        return total_fee, blended_rate

    def enforce_min_hold(
        self,
        holding_days: int,
        channel: FundChannel,
    ) -> HoldingWarning:
        """
        Check if the holding period triggers a warning or rejection.

        ETF trades always PASS (no redemption fee mechanism).
        OTC funds: <7d → REJECT, 7-30d → WARN, ≥30d → PASS.
        """
        match channel:
            case FundChannel.ETF_ON_EXCHANGE:
                return HoldingWarning.PASS
            case _:
                if holding_days < 7:
                    return HoldingWarning.REJECT
                if holding_days < 30:
                    return HoldingWarning.WARN
                return HoldingWarning.PASS

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _lookup_fee(
        schedule: dict[tuple[int, int], float],
        holding_days: int,
    ) -> float:
        """Find the rate for a given holding period from a schedule."""
        for (lo, hi), rate in sorted(schedule.items()):
            if lo <= holding_days < hi:
                return rate
        return 0.0

    @property
    def statutory_schedule(self) -> dict[tuple[int, int], float]:
        """Expose the statutory schedule for transparency."""
        return dict(_STATUTORY_SCHEDULE)
