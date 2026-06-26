"""
Monte Carlo block bootstrap for strategy robustness testing.
Phase 3 T3.12. Pure numpy — no external dependencies.

Block bootstrap preserves serial correlation in financial returns,
unlike i.i.d. bootstrap which underestimates drawdown risk by 7-23%.
"""

import numpy as np


def block_bootstrap(
    returns: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 20,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Block bootstrap simulation preserving autocorrelation.

    Returns dict with arrays of shape (n_simulations,):
      - sharpe: annualized Sharpe ratios
      - max_dd: maximum drawdown ratios
      - annual_return: annualized returns
      - annual_vol: annualized volatilities

    Per audit: block_size should be at least the autocorrelation decay length.
    Typical: 20 for daily data (1 month), 60 for weekly.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))

    sharpes = np.empty(n_simulations, dtype=np.float64)
    max_dds = np.empty(n_simulations, dtype=np.float64)
    ann_rets = np.empty(n_simulations, dtype=np.float64)
    ann_vols = np.empty(n_simulations, dtype=np.float64)

    for sim in range(n_simulations):
        # Sample blocks with replacement
        sampled_blocks = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot = np.concatenate([returns[b:b + block_size] for b in sampled_blocks])[:n]

        mu = float(np.mean(boot))
        sigma = float(np.std(boot, ddof=1))

        ann_rets[sim] = mu * 252
        ann_vols[sim] = sigma * np.sqrt(252)
        sharpes[sim] = (mu / sigma) * np.sqrt(252) if sigma > 1e-12 else 0.0

        # Max drawdown on bootstrapped returns
        prices = np.cumprod(1.0 + boot)
        peak = np.maximum.accumulate(prices)
        dd = (prices - peak) / peak
        max_dds[sim] = float(abs(np.min(dd)))

    return {
        "sharpe": sharpes,
        "max_dd": max_dds,
        "annual_return": ann_rets,
        "annual_vol": ann_vols,
    }


def bootstrap_confidence_interval(
    values: np.ndarray, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile-based confidence interval from bootstrap distribution."""
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "ci_lower": float(np.percentile(values, alpha * 100)),
        "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
        "std": float(np.std(values, ddof=1)),
    }


def monte_carlo_summary(
    returns: np.ndarray,
    n_simulations: int = 1000,
    block_size: int = 20,
) -> dict[str, dict[str, float]]:
    """
    Full Monte Carlo summary with confidence intervals.

    Returns dict with 'sharpe', 'max_dd', 'annual_return' each containing
    median, mean, ci_lower, ci_upper, std.
    """
    result = block_bootstrap(returns, n_simulations, block_size)
    return {
        "sharpe": bootstrap_confidence_interval(result["sharpe"]),
        "max_dd": bootstrap_confidence_interval(result["max_dd"]),
        "annual_return": bootstrap_confidence_interval(result["annual_return"]),
    }
