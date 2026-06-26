"""
Backtesting engine stubs — VectorBT + Backtrader integration points.
Phase 3 T3.1-T3.2. Provides the architecture for plugging in real engines.

VectorBT: vectorized backtest for fast factor screening (no CashLockManager).
Backtrader: event-driven engine with full China market rules (T+1, limits).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ─── VectorBT Stub ──────────────────────────────────────────────────────────


@dataclass
class VectorBTResult:
    """Results from a VectorBT backtest run."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    params: dict[str, float] = field(default_factory=dict)


class VectorBTEngine:
    """
    Vectorized backtesting engine — for fast parameter screening.

    Per SYSTEM-CONTRACT §9: VectorBT does NOT apply CashLockManager or
    redemption fees. It's used only for factor IC computation and
    rough parameter scanning.

    In production, install `vectorbt` and replace this stub.
    """

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        initial_capital: float = 100000.0,
    ) -> VectorBTResult:
        """
        Vectorized backtest from price and signal arrays.

        Args:
            prices: (n,) array of close prices.
            signals: (n,) array of target weights (0-1).
            initial_capital: Starting capital in CNY.
        """
        n = len(prices)
        if n < 2 or not signals.any():
            return VectorBTResult(0.0, 0.0, 0.0, 0.0, 0)

        # Simple vectorized PnL
        returns = np.diff(prices) / prices[:-1]
        weights = signals[:-1]
        strategy_returns = returns * weights
        cumulative = np.cumprod(1.0 + strategy_returns)
        total_return = float(cumulative[-1] - 1.0) if len(cumulative) > 0 else 0.0

        mu = float(np.mean(strategy_returns))
        sigma = float(np.std(strategy_returns, ddof=1))
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 1e-12 else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / peak
        max_dd = float(abs(np.min(dd)))

        win_rate = float(np.mean(strategy_returns > 0)) if len(strategy_returns) > 0 else 0.0
        n_trades = int(np.sum(np.diff(weights) != 0))

        return VectorBTResult(
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            n_trades=n_trades,
        )


# ─── Backtrader Stub ────────────────────────────────────────────────────────


@dataclass
class BacktraderResult:
    """Results from an event-driven backtest with full China market rules."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_fees: float
    n_trades: int
    final_value: float


class BacktraderEngine:
    """
    Event-driven backtesting engine — for production-grade validation.

    Per SYSTEM-CONTRACT §9: THIS is the only engine where CashLockManager,
    2026 redemption fees, FIFO lot tracking, and T+1/price limit rules
    are fully enforced.

    In production, install `backtrader` and connect to the core engine
    modules (CashLockManager, RedemptionFeeCalculator, OrderCutoffValidator).
    """

    def __init__(self) -> None:
        pass

    def run(
        self,
        prices: np.ndarray,
        strategy_signals: list[dict],
        initial_capital: float = 100000.0,
        commission_pct: float = 0.00025,
        stamp_duty_pct: float = 0.0005,
    ) -> BacktraderResult:
        """
        Event-driven backtest with simplified cost model.

        In full production, this delegates to CashLockManager and
        RedemptionFeeCalculator for China-specific rules.
        """
        n = len(prices)
        if n < 2 or not strategy_signals:
            return BacktraderResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, initial_capital)

        cash = initial_capital
        position = 0.0
        trade_returns: list[float] = []
        total_fees = 0.0
        n_trades = 0

        signal_idx = 0
        for i in range(1, n):
            # Check for signal at this bar
            if signal_idx < len(strategy_signals):
                sig = strategy_signals[signal_idx]
                target_weight = sig.get("target_weight", 0.0)
                target_value = cash * target_weight
                current_value = position * prices[i]
                diff = target_value - current_value

                if abs(diff) > 0:
                    fee = abs(diff) * (commission_pct + stamp_duty_pct)
                    total_fees += fee
                    cash -= diff + fee
                    position += diff / prices[i]
                    n_trades += 1

                signal_idx += 1

            # Daily return
            if n_trades > 0:
                portfolio_value = cash + position * prices[i]
                if i > 1:
                    prev_value = cash + position * prices[i - 1]
                    if prev_value > 0:
                        trade_returns.append(portfolio_value / prev_value - 1.0)

        final_value = cash + position * prices[-1]
        total_return = (final_value - initial_capital) / initial_capital

        ret_arr = np.array(trade_returns, dtype=np.float64) if trade_returns else np.array([0.0])
        mu = float(np.mean(ret_arr))
        sigma = float(np.std(ret_arr, ddof=1))
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 1e-12 else 0.0

        cum_vals = np.cumprod(1.0 + ret_arr) if len(ret_arr) > 0 else np.array([1.0])
        peak = np.maximum.accumulate(cum_vals)
        max_dd = float(abs(np.min((cum_vals - peak) / peak)))

        win_rate = float(np.mean(ret_arr > 0)) if len(ret_arr) > 0 else 0.0

        return BacktraderResult(
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            total_fees=round(total_fees, 2),
            n_trades=n_trades,
            final_value=round(final_value, 2),
        )
