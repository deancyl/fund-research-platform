"""
Cash lock manager — simulates fund settlement delays in China's OTC market.

Real-world microstructure:
  - ETF sold on exchange: cash available same day for new ETF purchases (T+0).
  - OTC equity fund redemption: cash arrives T+3 to T+5 (4 days, conservative).
  - ETF feeder fund redemption: cash arrives T+2 to T+3 (3 days, conservative).
  - QDII fund redemption: cash arrives T+6 to T+10 (8 days, midpoint).
  - Money market fund: cash arrives T+1 to T+2 (1 day).

Without this simulation, a backtest effectively grants an infinite free
bridge loan to rotation strategies, inflating returns by 30-50%.

Implementation uses collections.deque for O(1) enqueue/dequeue of locked
cash entries. Per the system contract, ALL operations are pure in-memory;
no database queries inside lock/unlock. The deques are kept in the manager
instance and never flushed to SQLite during the backtest loop.
"""

from collections import deque
from dataclasses import dataclass

from src.core.data.schema import FundChannel


# ─── Locked Cash Entry ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LockedCash:
    """A single locked cash entry — immutable once created."""

    amount: float
    unlock_day: int   # trading day index when cash becomes available
    fund_channel: FundChannel  # determines settlement delay
    redeem_day: int   # trading day index when redemption was submitted


# ─── Settlement Delays — injectable, extensible ─────────────────────────────


# Default settlement delays mapped by FundChannel.
# External code can provide custom delays via the contructor `delay_override`.
_DEFAULT_DELAYS: dict[FundChannel, int] = {
    FundChannel.ETF_ON_EXCHANGE: 0,
    FundChannel.OTC_OPEN_END: 4,
    FundChannel.OTC_ETF_FEEDER: 3,
    FundChannel.QDII: 8,
}


# ─── Manager ─────────────────────────────────────────────────────────────────


class CashLockManager:
    """
    Simulates settlement delay for redeemed fund cash.

    Settlement delays are determined by FundChannel. A custom delay override
    can be injected for per-fund contract deviations (e.g., some OTC funds
    advertise T+2 instead of T+4).

    Attributes:
        available_cash: Cash immediately available for new purchases.
    """

    def __init__(
        self,
        initial_cash: float,
        delay_override: dict[FundChannel, int] | None = None,
    ) -> None:
        self.available_cash: float = initial_cash
        self._locked: deque[LockedCash] = deque()
        self._delays: dict[FundChannel, int] = (
            {**_DEFAULT_DELAYS, **(delay_override or {})}
        )

    # ── Public API ───────────────────────────────────────────────────────

    def lock(
        self, amount: float, channel: FundChannel, current_day: int
    ) -> None:
        """
        Freeze cash upon redemption submission.

        Args:
            amount: Redemption proceeds in CNY.
            channel: FundChannel enum — determines settlement delay.
            current_day: Trading day index (0-based).
        """
        delay = self._delays.get(channel, 4)  # fallback to OTC default
        unlock_day = current_day + delay

        entry = LockedCash(
            amount=amount,
            unlock_day=unlock_day,
            fund_channel=channel,
            redeem_day=current_day,
        )
        self._locked.append(entry)
        self.available_cash -= amount

    def unlock_daily(self, current_day: int) -> float:
        """
        Release all cash whose unlock_day <= current_day.

        Must be called at the start of each trading day.

        Returns:
            Total amount unlocked this day.
        """
        unlocked = 0.0
        while self._locked and self._locked[0].unlock_day <= current_day:
            entry = self._locked.popleft()
            unlocked += entry.amount

        self.available_cash += unlocked
        return unlocked

    def get_buying_power(self) -> float:
        """Return cash currently available for new purchases."""
        return self.available_cash

    def get_locked_summary(self) -> dict[str, float | int]:
        """Return a summary of currently locked cash."""
        if not self._locked:
            return {"locked_total": 0.0, "pending_items": 0}

        total = sum(entry.amount for entry in self._locked)
        next_unlock = self._locked[0].unlock_day
        max_lock = max(
            entry.unlock_day - entry.redeem_day for entry in self._locked
        )

        return {
            "locked_total": total,
            "pending_items": len(self._locked),
            "next_unlock_day": next_unlock,
            "max_lock_days": max_lock,
        }

    @property
    def settlement_delays(self) -> dict[FundChannel, int]:
        """Expose active settlement delay table (read-only)."""
        return dict(self._delays)
