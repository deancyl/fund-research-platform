"""
Backtest overfitting detection — CPCV + DSR (Lopez de Prado 2014-2018).

CPCV: Combinatorial Purged Cross-Validation — generates C(N,K) OOS paths.
DSR:  Deflated Sharpe Ratio — adjusts for multiple testing in strategy selection.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def combinatorial_purged_cv(
    returns: np.ndarray,
    n_splits: int = 6,
    n_test_groups: int = 2,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
) -> list[dict[str, object]]:
    """
    Generate C(N,K) train/test pairs with purge and embargo.

    Purge removes data near the boundary to prevent label overlap.
    Embargo removes a window after the training set to prevent information leak.

    Returns list of dicts with 'train_idx', 'test_idx', 'oos_sharpe'.
    """
    t = len(returns)
    if t < n_splits * 20:
        return []

    group_indices = np.array_split(np.arange(t), n_splits)
    combos = list(combinations(range(n_splits), n_test_groups))
    results: list[dict[str, object]] = []

    for combo in combos:
        test_idx = np.concatenate([group_indices[i] for i in combo])
        train_idx = np.setdiff1d(np.arange(t), test_idx)

        # 【v0.1.8 审计修复】逐连续测试窗口 Purge + Embargo
        # 识别测试集中每个连续区间，对其左边界 Purge、右边界 Embargo
        purge_n = max(1, int(t * purge_pct / n_splits))
        embargo_n = max(1, int(t * embargo_pct / n_splits))

        test_sorted = np.sort(test_idx)
        # Find contiguous test windows
        boundaries = np.where(np.diff(test_sorted) > 1)[0] + 1
        windows = np.split(test_sorted, boundaries)

        # 对每个连续测试窗口，从训练集中剔除边界附近的点
        mask = np.ones(len(train_idx), dtype=bool)
        for win in windows:
            win_start, win_end = win[0], win[-1]
            # Purge: 剔除测试窗口左侧 purge_n 个训练点
            mask &= ~((train_idx >= win_start - purge_n) & (train_idx < win_start))
            # Embargo: 剔除测试窗口右侧 embargo_n 个训练点
            mask &= ~((train_idx > win_end) & (train_idx <= win_end + embargo_n))

        train_idx = train_idx[mask]

        if len(train_idx) < 30 or len(test_idx) < 10:
            continue

        train_r = returns[train_idx]
        test_r = returns[test_idx]

        mu = float(np.mean(test_r))
        sigma = float(np.std(test_r, ddof=1))
        oos_sr = float((mu / sigma) * np.sqrt(252)) if sigma > 1e-12 else 0.0

        results.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "oos_sharpe": oos_sr,
        })

    return results


def compute_pbo(
    cpcv_results: list[dict[str, object]],
    in_sample_sharpes: list[float],
) -> float:
    """
    Probability of Backtest Overfitting.

    PBO = fraction of paths where the IS-best strategy ranks below median in OOS.
    """
    n = len(cpcv_results)
    if n < 2:
        return 1.0

    is_best_idx = int(np.argmax(in_sample_sharpes))
    oos_rankings = np.argsort([-float(r["oos_sharpe"]) for r in cpcv_results])  # type: ignore[arg-type]
    oos_rank = int(np.where(oos_rankings == is_best_idx)[0][0])

    return float(oos_rank > n / 2)  # simplified: 1 or 0 per path


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Corrects for the fact that searching over M strategies inflates the
    maximum observed Sharpe ratio even under the null (no skill).

    DSR > 0.95 → statistically significant at 5% level.
    """
    euler = 0.5772156649
    log_m = np.log(n_trials)

    e_max = np.sqrt(2 * log_m) - (np.log(log_m) + np.log(4 * np.pi) - 2 * euler) / (2 * np.sqrt(2 * log_m))

    sr_se = np.sqrt(
        (1.0 / n_observations)
        * (1.0 + 0.5 * observed_sr**2 - skewness * observed_sr + (kurtosis - 3.0) / 4.0 * observed_sr**2)
    )

    # 【v0.1.8 审计修复】使用 norm.cdf 将 z-score 映射为 [0,1] 置信概率
    z_score = (observed_sr - e_max) / sr_se
    return float(stats.norm.cdf(z_score))


def minimum_track_record_length(observed_sr: float, alpha: float = 0.05) -> int:
    """Minimum number of observations needed for statistical significance."""
    z = float(stats.norm.ppf(1 - alpha / 2))
    return int(np.ceil((z / observed_sr) ** 2 * (1 + 0.5 * observed_sr**2))) if observed_sr > 0 else 999999
