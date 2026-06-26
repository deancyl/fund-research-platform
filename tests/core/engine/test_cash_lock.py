"""
Tests for src/core/engine/cash_lock.py — CashLockManager.

The CashLockManager simulates the real-world settlement delay in China's
OTC fund market: when you redeem a fund, the cash is NOT immediately available
for new purchases. The delay varies by fund channel:

  - ETF (on-exchange):  0 trading days (T+0 available for buying)
  - OTC open-end:       4 trading days (T+3 to T+5)
  - ETF feeder (OTC):   3 trading days (T+2 to T+3)
  - QDII:               8 trading days (T+6 to T+10)

Without this simulation, backtests give an unrealistic "infinite free bridge loan"
to multi-asset rotation strategies.
"""

from src.core.data.schema import FundChannel
from src.core.engine.cash_lock import CashLockManager


class TestCashLockManager:
    """Core lock/unlock cycle tests."""

    def test_initial_cash_fully_available(self) -> None:
        mgr = CashLockManager(initial_cash=100000.0)
        assert mgr.get_buying_power() == 100000.0
        assert mgr.available_cash == 100000.0

    def test_lock_etf_immediately_available(self) -> None:
        """ETF sale: cash locked for 0 days → available immediately."""
        mgr = CashLockManager(initial_cash=50000.0)
        mgr.lock(amount=10000.0, channel=FundChannel.ETF_ON_EXCHANGE, current_day=0)
        mgr.unlock_daily(current_day=0)
        assert mgr.get_buying_power() == 50000.0

    def test_lock_otc_reduces_available(self) -> None:
        """OTC redemption: cash locked for 4 days, unavailable for buying."""
        mgr = CashLockManager(initial_cash=100000.0)
        mgr.lock(amount=20000.0, channel=FundChannel.OTC_OPEN_END, current_day=0)
        assert mgr.get_buying_power() == 80000.0

    def test_otc_unlock_after_delay(self) -> None:
        """Cash unlocks exactly after 4 trading days."""
        mgr = CashLockManager(initial_cash=100000.0)
        mgr.lock(amount=20000.0, channel=FundChannel.OTC_OPEN_END, current_day=0)

        for day in range(3):
            unlocked = mgr.unlock_daily(current_day=day + 1)
            assert mgr.get_buying_power() == 80000.0, f"Day {day+1}: buying power wrong"

        unlocked = mgr.unlock_daily(current_day=4)
        assert unlocked == 20000.0
        assert mgr.get_buying_power() == 100000.0

    def test_qdii_long_lock(self) -> None:
        """QDII: cash locked for 8 trading days."""
        mgr = CashLockManager(initial_cash=100000.0)
        mgr.lock(amount=15000.0, channel=FundChannel.QDII, current_day=0)
        mgr.unlock_daily(current_day=7)
        assert mgr.get_buying_power() == 85000.0
        unlocked = mgr.unlock_daily(current_day=8)
        assert unlocked == 15000.0

    def test_multiple_staggered_locks(self) -> None:
        """Multiple redemptions at different times → staggered unlocks."""
        mgr = CashLockManager(initial_cash=200000.0)
        mgr.lock(amount=30000.0, channel=FundChannel.OTC_OPEN_END, current_day=0)
        mgr.lock(amount=20000.0, channel=FundChannel.OTC_OPEN_END, current_day=2)
        mgr.unlock_daily(current_day=3)
        assert mgr.get_buying_power() == 150000.0
        unlocked = mgr.unlock_daily(current_day=4)
        assert unlocked == 30000.0
        unlocked = mgr.unlock_daily(current_day=6)
        assert unlocked == 20000.0

    def test_etf_feeder_delay(self) -> None:
        """ETF feeder fund: 3-day lock."""
        mgr = CashLockManager(initial_cash=50000.0)
        mgr.lock(amount=10000.0, channel=FundChannel.OTC_ETF_FEEDER, current_day=0)
        assert mgr.get_buying_power() == 40000.0
        mgr.unlock_daily(current_day=2)
        assert mgr.get_buying_power() == 40000.0
        unlocked = mgr.unlock_daily(current_day=3)
        assert unlocked == 10000.0

    def test_custom_delay_override(self) -> None:
        """delay_override allows per-environment tuning."""
        mgr = CashLockManager(
            initial_cash=50000.0,
            delay_override={FundChannel.OTC_OPEN_END: 2},  # fast-track OTC
        )
        mgr.lock(amount=10000.0, channel=FundChannel.OTC_OPEN_END, current_day=0)
        # Day 1: still locked (delay=2)
        mgr.unlock_daily(current_day=1)
        assert mgr.get_buying_power() == 40000.0
        # Day 2: unlocks
        unlocked = mgr.unlock_daily(current_day=2)
        assert unlocked == 10000.0

    def test_lock_summary(self) -> None:
        """get_locked_summary reports correctly."""
        mgr = CashLockManager(initial_cash=100000.0)
        mgr.lock(amount=10000.0, channel=FundChannel.OTC_OPEN_END, current_day=0)
        mgr.lock(amount=20000.0, channel=FundChannel.QDII, current_day=0)

        summary = mgr.get_locked_summary()
        assert summary["locked_total"] == 30000.0
        assert summary["pending_items"] == 2
        assert summary["next_unlock_day"] == 4
        assert summary["max_lock_days"] == 8

    def test_empty_lock_summary(self) -> None:
        """get_locked_summary when nothing is locked."""
        mgr = CashLockManager(initial_cash=50000.0)
        summary = mgr.get_locked_summary()
        assert summary["locked_total"] == 0
        assert summary["pending_items"] == 0
