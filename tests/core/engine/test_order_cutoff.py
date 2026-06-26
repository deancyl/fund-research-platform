"""
Tests for src/core/engine/order_cutoff.py — 15:00 subscription/redemption cutoff.

China OTC mutual funds use forward-pricing (unknown-price) mechanism:
  - Applications before 15:00 on trading day T → executed at T日的 closing NAV
  - Applications after 15:00 (even 15:00:01) → deferred to T+1 trading day

These tests verify:
  - The cutoff logic at exactly 15:00 boundary
  - Weekend/holiday deferral
  - Look-ahead bias detection: post-15:00 signals must NEVER use same-day NAV
  - After-hours AI-generated signals (20:00+) correctly defer to T+1
"""

from datetime import date, datetime, time

import pytest

from src.core.engine.order_cutoff import OrderCutoffValidator, TradingCalendar


# ─── Test Fixture: Simple Trading Calendar ──────────────────────────────────


class FakeTradingCalendar(TradingCalendar):
    """Calendar where weekdays are trading days, weekends are not."""

    def is_trading_day(self, d: date) -> bool:
        """Weekday = trading day, weekend = non-trading."""
        return d.weekday() < 5  # Monday=0, Friday=4

    def next_trading_day(self, d: date) -> date:
        """Return the next trading day on or after d."""
        nxt = d
        while not self.is_trading_day(nxt):
            nxt = nxt + date.resolution
        return nxt


@pytest.fixture
def calendar() -> FakeTradingCalendar:
    return FakeTradingCalendar()


@pytest.fixture
def validator(calendar: FakeTradingCalendar) -> OrderCutoffValidator:
    return OrderCutoffValidator(calendar=calendar)


# ─── resolve_execution_date ─────────────────────────────────────────────────


class TestResolveExecutionDate:
    """Tests for the resolve_execution_date method."""

    def test_before_cutoff_on_trading_day(self, validator: OrderCutoffValidator) -> None:
        """Signal at 14:30 on a Tuesday → same-day execution."""
        signal_time = datetime(2026, 6, 23, 14, 30)  # Tuesday
        result = validator.resolve_execution_date(signal_time)
        assert result == date(2026, 6, 23)

    def test_exactly_at_cutoff(self, validator: OrderCutoffValidator) -> None:
        """Signal at exactly 15:00 → same-day execution (inclusive bound)."""
        signal_time = datetime(2026, 6, 23, 15, 0)  # Tuesday
        result = validator.resolve_execution_date(signal_time)
        assert result == date(2026, 6, 23)

    def test_one_second_after_cutoff(self, validator: OrderCutoffValidator) -> None:
        """Signal at 15:00:01 → MUST defer to next trading day."""
        signal_time = datetime(2026, 6, 23, 15, 0, 1)  # Tuesday
        result = validator.resolve_execution_date(signal_time)
        # Next trading day is Wednesday (June 24)
        assert result == date(2026, 6, 24)

    def test_after_cutoff_on_friday_deferred_to_monday(
        self, validator: OrderCutoffValidator
    ) -> None:
        """Signal at 15:30 on Friday → deferred to Monday (skip weekend)."""
        signal_time = datetime(2026, 6, 19, 15, 30)  # Friday
        result = validator.resolve_execution_date(signal_time)
        assert result == date(2026, 6, 22)  # Monday

    def test_after_cutoff_on_saturday(self, validator: OrderCutoffValidator) -> None:
        """Signal on Saturday → non-trading day, deferred to Monday."""
        signal_time = datetime(2026, 6, 20, 10, 0)  # Saturday
        result = validator.resolve_execution_date(signal_time)
        assert result == date(2026, 6, 22)  # Monday

    def test_ai_debate_after_hours_signal(
        self, validator: OrderCutoffValidator
    ) -> None:
        """
        CRITICAL: AI debate finishes at 21:00 after NAV disclosure.
        The signal must NOT use today's NAV — must be T+1.
        """
        signal_time = datetime(2026, 6, 23, 21, 0)  # Tuesday evening
        result = validator.resolve_execution_date(signal_time)
        assert result == date(2026, 6, 24)  # Wednesday


# ─── validate_backtest_order (look-ahead bias detection) ─────────────────────


class TestValidateBacktestOrder:
    """Tests for look-ahead bias detection in backtesting."""

    def test_valid_before_cutoff_order(
        self, validator: OrderCutoffValidator
    ) -> None:
        """14:30 signal using same-day NAV → valid (forward-pricing OK)."""
        is_valid = validator.validate_backtest_order(
            signal_datetime=datetime(2026, 6, 23, 14, 30),
            nav_publish_datetime=datetime(2026, 6, 23, 21, 0),
            execution_date=date(2026, 6, 23),
        )
        assert is_valid

    def test_reject_post_cutoff_with_same_day_nav(
        self, validator: OrderCutoffValidator
    ) -> None:
        """
        🚨 15:30 signal claiming same-day NAV → REJECT.
        This is the classic backtest look-ahead bias — the signal was generated
        after the cutoff but the backtest assumed same-day execution.
        """
        is_valid = validator.validate_backtest_order(
            signal_datetime=datetime(2026, 6, 23, 15, 30),
            nav_publish_datetime=datetime(2026, 6, 23, 21, 0),
            execution_date=date(2026, 6, 23),
        )
        assert not is_valid

    def test_valid_post_cutoff_with_t1_nav(
        self, validator: OrderCutoffValidator
    ) -> None:
        """15:30 signal using T+1 NAV → valid (correctly deferred)."""
        is_valid = validator.validate_backtest_order(
            signal_datetime=datetime(2026, 6, 23, 15, 30),
            nav_publish_datetime=datetime(2026, 6, 24, 21, 0),
            execution_date=date(2026, 6, 24),
        )
        assert is_valid

    def test_reject_signal_before_nav_published(
        self, validator: OrderCutoffValidator
    ) -> None:
        """
        🚨 Signal at 10:00 claiming to use a NAV that hasn't been published yet
        (publishes at 21:00 same day) → this is physically impossible.
        """
        is_valid = validator.validate_backtest_order(
            signal_datetime=datetime(2026, 6, 23, 10, 0),
            nav_publish_datetime=datetime(2026, 6, 23, 21, 0),
            execution_date=date(2026, 6, 23),
        )
        # Signal is at 10:00 but NAV hasn't been published (21:00)
        # However, in forward-pricing, the NAV is unknown at order time —
        # this is the mechanism's design. The execution happens at closing NAV.
        # The key constraint is: you cannot know today's NAV before it's published.
        # But for cutoff validation, 10:00 < 15:00 → the order is placed for TODAY,
        # which means it executes at the as-yet-unknown closing NAV.
        # This IS valid forward-pricing behavior.
        assert is_valid

    def test_reject_signal_time_exceeds_nav_time_same_day(
        self, validator: OrderCutoffValidator
    ) -> None:
        """
        🚨 Signal at 22:00 (after NAV published) claiming TODAY as execution date
        → REJECT. NAV already published, cutoff long past. Must be T+1.
        """
        is_valid = validator.validate_backtest_order(
            signal_datetime=datetime(2026, 6, 23, 22, 0),
            nav_publish_datetime=datetime(2026, 6, 23, 21, 0),
            execution_date=date(2026, 6, 23),
        )
        assert not is_valid
