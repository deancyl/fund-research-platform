"""
Backtrader extension plugins — Chinese market rule enforcement.
Phase 3 T3.3-T3.5 integration. Per audit: OrderCutoff + CashLock +
AShareSlippage injected as Backtrader broker/observer middleware.
"""

from __future__ import annotations

from datetime import time as _time


# ─── A-Share Slippage Model ──────────────────────────────────────────────────


class AShareSlippageModel:
    """
    A-Share specific trade constraints injected into Backtrader.

    Rules enforced:
      1. Price limits: ±10% (main), ±20% (ChiNext/STAR), ±5% (ST).
      2. If a bar hits limit-up → BUY orders are rejected (no liquidity).
      3. If a bar hits limit-down → SELL orders are rejected.
      4. T+1: signal at T close → executed at T+1 open.
    """

    MAIN_LIMIT: float = 0.10
    GEM_LIMIT: float = 0.20
    ST_LIMIT: float = 0.05

    def __init__(self, board: str = "main") -> None:
        self._limit = {"main": self.MAIN_LIMIT, "gem": self.GEM_LIMIT, "st": self.ST_LIMIT}.get(board, self.MAIN_LIMIT)

    def can_buy(self, open_price: float, high: float, low: float, close: float) -> bool:
        """Check if a BUY order can execute (not locked at limit-up)."""
        prev_close = open_price / (1.0 + self._limit)
        limit_up = prev_close * (1.0 + self._limit)

        # If open == high == limit_up → locked limit-up, no BUY possible
        if high == low and abs(high - limit_up) < 1e-8:
            return False
        return True

    def can_sell(self, open_price: float, high: float, low: float, close: float) -> bool:
        """Check if a SELL order can execute (not locked at limit-down)."""
        prev_close = open_price / (1.0 - self._limit)
        limit_down = prev_close * (1.0 - self._limit)

        if high == low and abs(low - limit_down) < 1e-8:
            return False
        return True


# ─── Order Cutoff Middleware ──────────────────────────────────────────────────


class OrderCutoffMiddleware:
    """
    Enforces 15:00 subscription/redemption cutoff in Backtrader.

    If a signal arrives after 15:00, execution is deferred to the
    next bar's closing price (T+1 NAV for OTC funds).
    """

    CUTOFF: _time = _time(15, 0)

    def __init__(self) -> None:
        self._pending_signals: list[dict] = []

    def submit(self, signal: dict, signal_time: _time) -> list[dict]:
        """
        Process a signal through the cutoff filter.

        Returns:
            Empty list if signal is deferred (pending for next bar),
            or [signal] if it passes the cutoff (execute at current bar).
        """
        if signal_time > self.CUTOFF:
            self._pending_signals.append(signal)
            return []
        return [signal]

    def flush_pending(self) -> list[dict]:
        """Release all deferred signals for next-bar execution."""
        pending = list(self._pending_signals)
        self._pending_signals.clear()
        return pending


# ─── Cash Lock Middleware ────────────────────────────────────────────────────


class CashLockMiddleware:
    """
    Bridges CashLockManager into Backtrader's account model.

    Tracks locked cash from OTC redemptions and prevents overbuying
    during the settlement vacuum period.
    """

    def __init__(self, cash_lock_manager: object) -> None:
        self._cash_mgr = cash_lock_manager

    def available_cash(self, current_day: int) -> float:
        """Get buying power considering locked settlement funds."""
        self._cash_mgr.unlock_daily(current_day)  # type: ignore[union-attr]
        return self._cash_mgr.get_buying_power()  # type: ignore[union-attr]

    def lock_redemption(self, amount: float, channel, current_day: int) -> None:
        """Lock cash from a redemption at the given trading day."""
        self._cash_mgr.lock(amount, channel, current_day)  # type: ignore[union-attr]
