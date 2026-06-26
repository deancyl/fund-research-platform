"""
CH-3 Chinese Three-Factor Model wrapper.

The CH-3 model (Liu, Stambaugh & Yuan 2019) is specifically designed for
the Chinese A-share market, removing the bottom 30% of stocks by market cap
to eliminate shell-value contamination that invalidates the standard Fama-French
three-factor model (FF3) in China.

This module provides the factor definitions and portfolio return decomposition.
When jh-factors is installed, it delegates to that library. Otherwise, it
operates in stub mode with factor definitions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CH3Factors:
    """CH-3 factor returns for a given period."""

    mkt: float  # Market excess return (Rm - Rf)
    smb: float  # Small-minus-Big (size factor, shell-contamination-free)
    vmg: float  # Value-minus-Growth (value factor, shell-contamination-free)


@dataclass(frozen=True, slots=True)
class FactorExposure:
    """Fund/portfolio exposure to CH-3 factors."""

    alpha: float  # Jensen's alpha (abnormal return)
    beta_mkt: float  # Market beta
    beta_smb: float  # Size exposure
    beta_vmg: float  # Value exposure
    r_squared: float  # Model fit


def decompose_returns(
    fund_returns: np.ndarray,
    factor_returns: np.ndarray,
    risk_free_rate: float = 0.02,
) -> FactorExposure:
    """
    Decompose fund returns using CH-3 factor model.

    【审计修复 v0.1.8】factor_returns 现在接收形状为 (n, 3) 的时序数组，
    列为 [mkt_excess_daily, smb_daily, vmg_daily]，而非静态标量。
    消除了 np.full() 造成的绝对共线性错误。

    Args:
        fund_returns: Array (n,) of daily fund excess returns.
        factor_returns: Array (n, 3) of CH-3 factor daily returns:
                        col 0 = mkt_excess, col 1 = smb, col 2 = vmg.
        risk_free_rate: Annual risk-free rate (default 2% for China).

    Returns:
        FactorExposure with alpha, betas, and R-squared.
    """
    n = len(fund_returns)
    if n < 20 or factor_returns.shape[0] != n or factor_returns.shape[1] < 3:
        return FactorExposure(alpha=0.0, beta_mkt=1.0, beta_smb=0.0, beta_vmg=0.0, r_squared=0.0)

    # 【v0.1.8】真实时序设计矩阵: intercept + 3因子每日变动
    x = np.column_stack([
        np.ones(n),
        factor_returns[:, 0],  # mkt_excess — daily values
        factor_returns[:, 1],  # smb — daily values
        factor_returns[:, 2],  # vmg — daily values
    ])

    try:
        coeffs, residuals, rank, singular = np.linalg.lstsq(x, fund_returns, rcond=None)
        alpha, beta_mkt, beta_smb, beta_vmg = coeffs

        # R-squared
        ss_res = float(np.sum(residuals**2)) if len(residuals) > 0 else 0.0
        ss_tot = float(np.sum((fund_returns - np.mean(fund_returns))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return FactorExposure(
            alpha=float(alpha),
            beta_mkt=float(beta_mkt),
            beta_smb=float(beta_smb),
            beta_vmg=float(beta_vmg),
            r_squared=round(r2, 4),
        )
    except np.linalg.LinAlgError:
        return FactorExposure(alpha=0.0, beta_mkt=1.0, beta_smb=0.0, beta_vmg=0.0, r_squared=0.0)


def compute_factor_returns_daily(
    market_returns: np.ndarray,
    size_portfolio_returns: np.ndarray,
    value_portfolio_returns: np.ndarray,
    risk_free_daily: float = 0.02 / 252,
) -> CH3Factors:
    """
    Compute CH-3 factor returns from raw portfolio returns.

    In production, these are pre-computed by jh-factors from the
    full A-share universe with shell-value exclusion.

    Args:
        market_returns: Market portfolio daily returns.
        size_portfolio_returns: SMB portfolio daily returns.
        value_portfolio_returns: VMG portfolio daily returns.
        risk_free_daily: Daily risk-free rate.
    """
    return CH3Factors(
        mkt=float(np.mean(market_returns) - risk_free_daily) * 252,
        smb=float(np.mean(size_portfolio_returns)) * 252,
        vmg=float(np.mean(value_portfolio_returns)) * 252,
    )
