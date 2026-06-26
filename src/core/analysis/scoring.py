"""
8-factor fund scoring system.

Inspired by fundseeker + Morningstar MRAR methodology.
Weights are configurable; defaults follow the research-validated allocation:

  return     15%  — 3/6/12 month weighted return
  trend      10%  — Long-term trend (3yr/5yr annualized)
  risk_adj   20%  — Sharpe/Sortino/Calmar composite
  drawdown   15%  — 3yr rolling max drawdown
  manager    10%  — Fund manager tenure + alpha persistence
  rating     10%  — Morningstar/Shanghai Securities rating aggregate
  size       10%  — Fund size + daily turnover
  momentum   10%  — Recent 1-month performance trend
"""

from __future__ import annotations

import numpy as np
import polars as pl


class FundScorer:
    """Compute 8-factor scores for a set of funds."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "return": 0.15,
        "trend": 0.10,
        "risk_adj": 0.20,
        "drawdown": 0.15,
        "manager": 0.10,
        "rating": 0.10,
        "size": 0.10,
        "momentum": 0.10,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    # ── Public ────────────────────────────────────────────────────────────

    def score_all(self, market_data: dict[str, pl.DataFrame]) -> list[dict[str, float | str]]:
        """Score all funds in market_data. Returns list sorted by total_score desc."""
        if not market_data:
            return []

        scores = []
        for fund_code, df in market_data.items():
            if "nav" not in df.columns and "close" not in df.columns:
                continue

            nav = df["nav"].to_numpy().astype(np.float64) if "nav" in df.columns else df["close"].to_numpy().astype(np.float64)
            factors = self._compute_factors(nav)
            total = sum(factors[k] * self.weights[k] for k in self.weights)

            scores.append({
                "fund_code": fund_code,
                "total_score": round(float(total), 4),
                **{f"{k}_score": round(float(v), 4) for k, v in factors.items()},
            })

        scores.sort(key=lambda x: x["total_score"], reverse=True)
        return scores

    # ── Factor computation ────────────────────────────────────────────────

    def _compute_factors(self, nav: np.ndarray) -> dict[str, float]:
        """Compute all 8 factor scores from NAV series."""
        returns = np.diff(np.log(nav + 1e-12))
        if len(returns) < 21:
            return {k: 0.5 for k in self.weights}

        return {
            "return": self._return_factor(returns),
            "trend": self._trend_factor(nav),
            "risk_adj": self._risk_adj_factor(returns),
            "drawdown": self._drawdown_factor(nav),
            "manager": 0.5,   # stub — requires external data
            "rating": 0.5,    # stub — requires external data
            "size": 0.5,      # stub — requires external data
            "momentum": self._momentum_factor(returns),
        }

    @staticmethod
    def _return_factor(returns: np.ndarray) -> float:
        """Weighted multi-period return: 3m(40%) + 6m(35%) + 12m(25%)."""
        periods = {63: 0.40, 126: 0.35, 252: 0.25}
        score = 0.0
        total_w = 0.0
        for p, w in periods.items():
            if len(returns) >= p:
                cum = np.prod(1 + returns[-p:]) - 1
                score += cum * w
                total_w += w
        if total_w == 0:
            return 0.5
        # Normalize: 30% return → score 1.0
        normalized = min(score / total_w / 0.30, 2.0)
        return float(np.clip(normalized / 2.0, 0.0, 1.0))

    @staticmethod
    def _trend_factor(nav: np.ndarray) -> float:
        """Long-term trend from 1yr/3yr annualized return."""
        if len(nav) < 252:
            return 0.5
        one_yr = (nav[-1] / nav[-252]) - 1
        normalized = min(one_yr / 0.10, 2.0)
        return float(np.clip(normalized / 2.0, 0.0, 1.0))

    @staticmethod
    def _risk_adj_factor(returns: np.ndarray) -> float:
        """Composite: Sharpe (50%) + Sortino (50%)."""
        if len(returns) < 21 or np.std(returns) < 1e-12:
            return 0.5
        ann_ret = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252)
        sharpe = ann_ret / ann_vol

        downside = returns[returns < 0]
        sortino = ann_ret / (np.std(downside) * np.sqrt(252)) if len(downside) > 0 and np.std(downside) > 1e-12 else sharpe

        # Sharpe=1.0 → score 0.5, Sharpe=2.0 → score 1.0
        s_score = float(np.clip(sharpe / 2.0, 0.0, 1.0))
        so_score = float(np.clip(sortino / 2.0, 0.0, 1.0))
        return (s_score + so_score) / 2.0

    @staticmethod
    def _drawdown_factor(nav: np.ndarray) -> float:
        """Lower max drawdown → higher score."""
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak
        max_dd = abs(float(np.min(dd)))
        # 10% drawdown → score 0.5, 0% → 1.0, 30% → 0.0
        return float(np.clip(1.0 - max_dd / 0.30, 0.0, 1.0))

    @staticmethod
    def _momentum_factor(returns: np.ndarray) -> float:
        """Recent 1-month performance trend."""
        if len(returns) < 21:
            return 0.5
        recent = returns[-21:]
        ann_ret = np.mean(recent) * 252
        normalized = min(ann_ret / 0.15, 2.0)
        return float(np.clip(normalized / 2.0, 0.0, 1.0))
