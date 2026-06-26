"""
Data schema for the fund research platform — Layer 1 boundary types.

All types use Pydantic v2 with frozen=True for immutability.
FundChannel and FundCategory are StrEnum for ergonomic string comparison.

Key design decisions:
  - PositionLot is the core FIFO tracking unit — every purchase is recorded as a lot.
  - FundPosition.lots is sorted by purchase_date on construction (FIFO invariant).
  - FundTradingProfile encapsulates all China-specific trading rules per fund.
"""

from datetime import date, time
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator


# ─── Enums ───────────────────────────────────────────────────────────────────


class FundChannel(StrEnum):
    """Trading channel — determines which execution pipeline is used."""

    ETF_ON_EXCHANGE = "ETF_ON_EXCHANGE"  # Exchange-traded fund (secondary market)
    OTC_OPEN_END = "OTC_OPEN_END"  # Off-exchange mutual fund (subscription/redemption)
    OTC_ETF_FEEDER = "OTC_ETF_FEEDER"  # ETF feeder fund (off-exchange)
    QDII = "QDII"  # Cross-border QDII fund


class FundCategory(StrEnum):
    """Fund category — affects redemption fee exemptions."""

    EQUITY = "EQUITY"  # Stock/mixed fund — statutory minimum fees apply
    INDEX_FUND = "INDEX_FUND"  # Index fund — contract-based fees (usually lower)
    SPECIAL_BOND = "SPECIAL_BOND"  # Special bond fund — contract-based
    MONEY_MARKET = "MONEY_MARKET"  # Money market — typically zero redemption fee


# ─── Position Tracking ──────────────────────────────────────────────────────


class PositionLot(BaseModel, frozen=True):
    """
    A single purchase batch — the atomic unit for FIFO redemption fee calculation.

    Each time a user buys a fund, a new PositionLot is created. When selling,
    lots are consumed in FIFO order (earliest purchase first) to determine
    the holding period and associated redemption fee rate.
    """

    purchase_date: date = Field(description="Date the shares were purchased (T日)")
    shares: float = Field(gt=0, description="Number of shares purchased in this lot")
    purchase_nav: float = Field(gt=0, description="NAV (净值) at time of purchase")
    cost_amount: float = Field(ge=0, description="Total cost in CNY (shares × purchase_nav)")


# ─── Fund Profile ────────────────────────────────────────────────────────────


class FundTradingProfile(BaseModel, frozen=True):
    """
    Per-fund trading characteristics extracted from the fund contract.

    Default values reflect statutory minimums for Chinese OTC open-end funds.
    ETF profiles override these with exchange-specific rules.

    redemption_fee_schedule uses the format: {(lo_days, hi_days): rate}
    e.g. {(0, 7): 0.015, (7, 30): 0.010, (30, 180): 0.005, (180, inf): 0.0}
    """

    fund_code: str = Field(min_length=6, max_length=6, description="6-digit fund code")
    channel: FundChannel = Field(description="Trading channel (ETF/OTC/Feeder/QDII)")

    cutoff_time: time = Field(
        default=time(15, 0),
        description="T日 15:00 is the order cutoff. After this → T+1 execution.",
    )
    redemption_fee_schedule: dict[tuple[int, int], float] = Field(
        default_factory=dict,
        description="Contract-specific redemption fee schedule {(lo,hi): rate}",
    )
    settlement_delay_days: int = Field(
        default=0,
        ge=0,
        description="Trading days until redeemed cash becomes available for new purchases",
    )
    max_subscription_per_day_yuan: float | None = Field(
        default=None,
        ge=0,
        description="Daily subscription limit in CNY (None = no limit)",
    )
    is_suspended: bool = Field(
        default=False,
        description="Whether subscription/redemption is currently suspended",
    )

    # Class-level default settlement delays — overridden by channel
    _SETTLEMENT_DELAYS: ClassVar[dict[FundChannel, int]] = {
        FundChannel.ETF_ON_EXCHANGE: 0,  # T+0 available for buying, T+1 for withdrawal
        FundChannel.OTC_OPEN_END: 4,  # T+3 to T+5 (conservative: 4)
        FundChannel.OTC_ETF_FEEDER: 3,  # T+2 to T+3 (conservative: 3)
        FundChannel.QDII: 8,  # T+6 to T+10 (midpoint: 8)
    }

    @model_validator(mode="after")
    def _apply_channel_defaults(self) -> "FundTradingProfile":
        """Apply channel-specific settlement delay if not explicitly overridden."""
        if self.settlement_delay_days == 0:
            # model_extra is not available — use object.__setattr__ workaround
            channel_delay = self._SETTLEMENT_DELAYS.get(self.channel, 4)
            if channel_delay != 0:
                object.__setattr__(self, "settlement_delay_days", channel_delay)
        return self


# ─── Fund Position ──────────────────────────────────────────────────────────


class FundPosition(BaseModel, frozen=True):
    """
    A user's position in a single fund, composed of one or more PositionLots.

    Construction invariant: lots are sorted by purchase_date (FIFO order).
    This guarantees that redemption always consumes the oldest lots first,
    which is required for accurate holding-period-based fee calculation.
    """

    fund_code: str = Field(min_length=6, max_length=6)
    fund_name: str = Field(min_length=1)
    category: FundCategory = Field(description="Fund category for fee exemption checks")
    channel: FundChannel = Field(description="Trading channel")
    lots: list[PositionLot] = Field(min_length=1, description="Purchase batches (FIFO sorted on construction)")
    current_nav: float = Field(gt=0, description="Latest known NAV (净值)")
    total_shares: float = Field(gt=0, description="Sum of all lot shares")
    market_value: float = Field(ge=0, description="total_shares × current_nav")
    weight_pct: float = Field(ge=0, le=1, description="Portfolio weight (0.0-1.0)")

    @model_validator(mode="after")
    def _sort_lots_fifo(self) -> "FundPosition":
        """Sort lots by purchase_date to enforce FIFO ordering."""
        sorted_lots = sorted(self.lots, key=lambda lot: lot.purchase_date)
        if sorted_lots != self.lots:
            object.__setattr__(self, "lots", sorted_lots)
        return self

    @model_validator(mode="after")
    def _validate_shares_consistency(self) -> "FundPosition":
        """Confirm total_shares matches the sum of lot shares."""
        expected = sum(lot.shares for lot in self.lots)
        if abs(self.total_shares - expected) > 0.001:
            msg = (
                f"total_shares ({self.total_shares}) does not match "
                f"sum of lot shares ({expected})"
            )
            raise ValueError(msg)
        return self
