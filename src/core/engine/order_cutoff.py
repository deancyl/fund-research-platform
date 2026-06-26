"""
15:00 subscription/redemption cutoff validator.

China OTC mutual funds use forward-pricing (unknown-price / 未知价):
  - Applications before 15:00 on trading day T → executed at T's closing NAV.
  - Applications after 15:00 (even 15:00:01) → deferred to T+1 trading day.

This module enforces that constraint in the backtesting engine.
It also detects look-ahead bias: post-cutoff signals that incorrectly
assume same-day NAV execution.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, time


# ─── Trading Calendar Interface ─────────────────────────────────────────────


class TradingCalendar(ABC):
    """Abstract trading calendar — implement with China-specific holidays."""

    @abstractmethod
    def is_trading_day(self, d: date) -> bool:
        """Return True if date d is a trading day."""
        ...

    @abstractmethod
    def next_trading_day(self, d: date) -> date:
        """Return the next trading day on or after d."""
        ...


# ─── Simple Default Calendar ────────────────────────────────────────────────


class SimpleTradingCalendar(TradingCalendar):
    """Weekday-only calendar (no holiday awareness)."""

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5

    def next_trading_day(self, d: date) -> date:
        nxt = d
        while not self.is_trading_day(nxt):
            nxt = date(nxt.year, nxt.month, nxt.day) + timedelta(days=1)
            nxt = date(nxt.year, nxt.month, nxt.day)
        return nxt


# ─── Constants ──────────────────────────────────────────────────────────────

_CUTOFF: time = time(15, 0)


# ─── Validator ──────────────────────────────────────────────────────────────


class OrderCutoffValidator:
    """Validate that subscription/redemption orders respect the 15:00 cutoff."""

    CUTOFF: time = _CUTOFF

    def __init__(self, calendar: TradingCalendar | None = None) -> None:
        self._calendar = calendar or SimpleTradingCalendar()

    # ── Public API ──────────────────────────────────────────────────────

    def resolve_execution_date(self, signal_datetime: datetime) -> date:
        """
        Determine the actual NAV date for a signal generated at signal_datetime.

        Rules:
          1. Before 15:00 on a trading day → same-day NAV.
          2. After 15:00 or on a non-trading day → next trading day's NAV.

        This is the core function that prevents look-ahead bias:
        an AI agent generating a BUY signal at 21:00 must use T+1's NAV, not T's.
        """
        signal_date = signal_datetime.date()
        signal_time = signal_datetime.time()

        if signal_time <= self.CUTOFF and self._calendar.is_trading_day(signal_date):
            return signal_date

        # After cutoff or non-trading day → defer
        candidate = signal_date
        if signal_time > self.CUTOFF:
            candidate = candidate + timedelta(days=1)

        return self._calendar.next_trading_day(candidate)

    def validate_backtest_order(
        self,
        signal_datetime: datetime,
        nav_publish_datetime: datetime,
        execution_date: date,
    ) -> bool:
        """
        Detect look-ahead bias in backtest orders.

        Returns False (rejected) if:
          - Signal was after 15:00 but claims same-day execution.
          - Signal was before 15:00 but after the NAV was already published
            AND still claimed same-day execution.
            (If NAV is already published, you know the price — this is
             forward-pricing violation.)

        Returns True if the order timeline is physically possible.
        """
        signal_time = signal_datetime.time()

        # Rule 1: Post-cutoff signal → cannot be same-day execution
        if signal_time > self.CUTOFF and execution_date == signal_datetime.date():
            return False

        # Rule 2: Signal after NAV published and claiming same-day → reject
        # (NAV is disclosed, so you know the price — forward-pricing broken)
        if signal_datetime > nav_publish_datetime and execution_date == signal_datetime.date():
            return False

        return True
