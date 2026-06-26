"""
Risk metrics — pure NumPy implementations.

Functions accept daily return arrays (not price arrays except where noted).
All annualized at 252 trading days.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS: int = 252
RISK_FREE_RATE: float = 0.02  # Chinese 10-year govt bond approx


# ─── Sharpe Ratio ────────────────────────────────────────────────────────────


def sharpe_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio."""
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    mu = float(np.mean(excess))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-12:
        return float("nan")
    return (mu / sigma) * np.sqrt(TRADING_DAYS)


# ─── Sortino Ratio ──────────────────────────────────────────────────────────


def sortino_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    mu = float(np.mean(excess))
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if mu > 0 else 0.0
    sigma_d = float(np.std(downside, ddof=1))
    if sigma_d < 1e-12:
        return float("nan")
    return (mu / sigma_d) * np.sqrt(TRADING_DAYS)


# ─── Max Drawdown ───────────────────────────────────────────────────────────


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum drawdown as a positive fraction (0.0 = no drawdown)."""
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / peak
    return float(abs(np.min(dd)))


# ─── VaR / CVaR ─────────────────────────────────────────────────────────────


def var_95(returns: np.ndarray) -> float:
    """Historical VaR at 95% confidence. Returns negative value (loss)."""
    return float(np.percentile(returns, 5))


def cvar_95(returns: np.ndarray) -> float:
    """Historical CVaR (Expected Shortfall) at 95%."""
    var = var_95(returns)
    tail = returns[returns <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


# ─── Win Rate ───────────────────────────────────────────────────────────────


def win_rate(returns: np.ndarray) -> float:
    """Fraction of periods with positive return."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


# ─── Profit Factor ──────────────────────────────────────────────────────────


def profit_factor(returns: np.ndarray) -> float:
    """Gross profit / gross loss. > 1 is profitable."""
    gains = float(np.sum(returns[returns > 0]))
    losses = float(abs(np.sum(returns[returns < 0])))
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# ─── Calmar Ratio ───────────────────────────────────────────────────────────


def calmar_ratio(prices: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Annualized return / max drawdown."""
    if len(prices) < 2:
        return 0.0
    total_return = float(prices[-1] / prices[0] - 1)
    years = len(prices) / TRADING_DAYS
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    mdd = max_drawdown(prices)
    if mdd < 1e-12:
        return float("inf") if ann_return > 0 else 0.0
    return (ann_return - rf) / mdd
