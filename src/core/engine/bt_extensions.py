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
    A-Share directional liquidity freeze model (v0.3.1 audit fix).

    Limit-up (high==low==close >= limit): BUY is blocked (no sellers), SELL is allowed.
    Limit-down (high==low==close <= limit): SELL is blocked (no buyers), BUY is allowed.

    This is NOT symmetric — liquidity freezes in ONE direction only.
    """

    MAIN_LIMIT: float = 0.10
    GEM_LIMIT: float = 0.20
    ST_LIMIT: float = 0.05

    def __init__(self, board: str = "main") -> None:
        self._limit = {"main": self.MAIN_LIMIT, "gem": self.GEM_LIMIT, "st": self.ST_LIMIT}.get(board, self.MAIN_LIMIT)

    def can_execute(self, open_price: float, high: float, low: float, close: float, is_buy: bool) -> bool:
        """
        Directional liquidity check.

        Returns False if the order CANNOT execute (must be queued/deferred).
        """
        prev_close = open_price / (1.0 + self._limit)
        limit_up = prev_close * (1.0 + self._limit)
        limit_down = prev_close * (1.0 - self._limit)

        locked_up = high == low and abs(close - limit_up) < 1e-8
        locked_down = high == low and abs(close - limit_down) < 1e-8

        if locked_up and is_buy:
            return False  # Limit-up: no sellers → BUY blocked
        if locked_down and not is_buy:
            return False  # Limit-down: no buyers → SELL blocked
        return True

    def can_buy(self, *args) -> bool:
        """Legacy API: delegate to can_execute."""
        return self.can_execute(*args, is_buy=True)

    def can_sell(self, *args) -> bool:
        """Legacy API: delegate to can_execute."""
        return self.can_execute(*args, is_buy=False)


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
