"""
Transaction ledger engine — record, reconstruct, and backtest portfolio states.
Supports PRD v0.4.0: FIFO lot tracking, time-travel reconstruction, fee calculation.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime, time
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class TradeDirection(StrEnum):
    SUBSCRIBE = "SUBSCRIBE"
    REDEMPTION = "REDEMPTION"
    DIVIDEND_REINVEST = "DIVIDEND_REINVEST"


class TransactionLedgerEntry(BaseModel, frozen=True):
    """A single transaction in the adjustment ledger."""
    tx_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(description="Precise timestamp for 15:00 cutoff")
    fund_code: str = Field(min_length=6, max_length=6)
    direction: TradeDirection
    amount: float | None = Field(default=None, ge=0, description="Amount in CNY (for SUBSCRIBE)")
    shares: float | None = Field(default=None, ge=0, description="Shares (for REDEMPTION)")
    nav_applied: float = Field(gt=0, description="NAV at execution")
    fee_charged: float = Field(default=0.0, ge=0)


class LedgerEngine:
    """
    Transaction ledger with FIFO lot tracking and time-travel reconstruction.

    Usage:
        engine = LedgerEngine()
        engine.record_subscribe("005827", datetime(2026,1,15,10,0), 100000.0, nav=1.80)
        engine.record_redeem("005827", datetime(2026,6,28,10,0), 5000.0, nav=1.85)
        state = engine.reconstruct(date(2026,7,1))  # All holdings at that date
    """

    CUTOFF: time = time(15, 0)

    def __init__(self) -> None:
        self._ledger: list[TransactionLedgerEntry] = []
        self._lots: dict[str, deque[dict]] = {}  # fund_code -> FIFO lot queue

    # ── Record transactions ─────────────────────────────────────────────

    def record_subscribe(
        self, fund_code: str, ts: datetime, amount: float, nav: float, fee_rate: float = 0.0015
    ) -> TransactionLedgerEntry:
        """Record a simulated subscription."""
        net = amount / (1 + fee_rate)
        shares = net / nav
        fee = amount - net
        entry = TransactionLedgerEntry(
            timestamp=ts, fund_code=fund_code, direction=TradeDirection.SUBSCRIBE,
            amount=amount, shares=round(shares, 2), nav_applied=nav, fee_charged=round(fee, 2),
        )
        self._ledger.append(entry)
        # Add to FIFO lot queue
        if fund_code not in self._lots:
            self._lots[fund_code] = deque()
        self._lots[fund_code].append({
            "purchase_date": ts.date(), "shares": round(shares, 2),
            "purchase_nav": nav, "cost_amount": amount,
        })
        return entry

    def record_redeem(
        self, fund_code: str, ts: datetime, shares: float, nav: float
    ) -> TransactionLedgerEntry:
        """Record a simulated redemption with FIFO fee calculation."""
        fee = self._calc_redeem_fee(fund_code, shares, ts.date())
        gross = shares * nav
        net = gross - fee

        entry = TransactionLedgerEntry(
            timestamp=ts, fund_code=fund_code, direction=TradeDirection.REDEMPTION,
            shares=shares, amount=round(net, 2), nav_applied=nav, fee_charged=round(fee, 2),
        )
        self._ledger.append(entry)

        # Consume lots in FIFO order
        remaining = shares
        if fund_code in self._lots:
            while remaining > 0 and self._lots[fund_code]:
                lot = self._lots[fund_code][0]
                consume = min(lot["shares"], remaining)
                lot["shares"] -= consume
                remaining -= consume
                if lot["shares"] <= 0.001:
                    self._lots[fund_code].popleft()
        return entry

    # ── Reconstruction ──────────────────────────────────────────────────

    def reconstruct(self, as_of: date) -> dict[str, dict]:
        """Reconstruct all holdings at a given date by replaying the ledger."""
        result: dict[str, dict] = {}
        for code, lot_q in self._lots.items():
            active = [l for l in lot_q if l["purchase_date"] <= as_of]
            total_shares = sum(l["shares"] for l in active)
            total_cost = sum(l["cost_amount"] for l in active)
            if total_shares > 0:
                result[code] = {
                    "fund_code": code,
                    "total_shares": round(total_shares, 2),
                    "avg_cost": round(total_cost / total_shares, 4),
                    "lots": [dict(l) for l in active],
                    "holding_days_min": min((as_of - l["purchase_date"]).days for l in active) if active else 0,
                }
        return result

    def get_ledger(self) -> list[TransactionLedgerEntry]:
        return list(self._ledger)

    # ── Internal ────────────────────────────────────────────────────────

    def _calc_redeem_fee(self, fund_code: str, shares: float, as_of: date) -> float:
        """Calculate redemption fee using statutory schedule + FIFO lot aging."""
        if fund_code not in self._lots:
            return 0.0
        remaining = shares
        total_fee = 0.0
        for lot in self._lots[fund_code]:
            if remaining <= 0:
                break
            consume = min(lot["shares"], remaining)
            holding_days = (as_of - lot["purchase_date"]).days
            rate = self._statutory_fee_rate(holding_days)
            total_fee += consume * lot["purchase_nav"] * rate
            remaining -= consume
        return round(total_fee, 2)

    @staticmethod
    def _statutory_fee_rate(holding_days: int) -> float:
        if holding_days < 7: return 0.015
        if holding_days < 30: return 0.0075
        if holding_days < 365: return 0.005
        return 0.0
