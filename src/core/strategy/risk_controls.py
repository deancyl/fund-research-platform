"""
Market risk controls — CrowdingMonitor and LotTurnoverController.
Phase 5 gap fill. Pure math, zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── Crowding Monitor ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CrowdingSignal:
    strategy_name: str
    signal_concentration: float
    ic_trend: float
    turnover_zscore: float
    crowding_score: float
    warning_level: str  # GREEN / YELLOW / RED


class CrowdingMonitor:
    """
    Detects strategy crowding — when too many strategies target the same assets.

    Three dimensions:
      1. Signal concentration — how many strategies point to the same target.
      2. Factor IC decay — is the factor's predictive power declining?
      3. Turnover anomaly — is turnover significantly above historical norm?
    """

    def __init__(
        self,
        concentration_weight: float = 0.40,
        ic_weight: float = 0.35,
        turnover_weight: float = 0.25,
    ) -> None:
        self._cw = concentration_weight
        self._iw = ic_weight
        self._tw = turnover_weight

    def evaluate(
        self,
        signal_targets: list[str],
        ic_series: np.ndarray,
        current_turnover: float,
        historical_turnover: np.ndarray,
        strategy_name: str = "default",
    ) -> CrowdingSignal:
        """Compute composite crowding score."""
        concentration = self.compute_concentration(signal_targets)
        ic_trend = self.compute_ic_trend(ic_series, window=100)
        turnover_z = self.compute_turnover_z(current_turnover, historical_turnover)

        # Normalize each component to [0, 1]
        c_risk = (1.0 - concentration) * self._cw  # lower concentration → more risk
        i_risk = max(0.0, min(1.0, -ic_trend * 10.0)) * self._iw
        t_risk = max(0.0, min(1.0, turnover_z / 3.0)) * self._tw

        score = c_risk + i_risk + t_risk

        if score < 0.3:
            level = "GREEN"
        elif score < 0.6:
            level = "YELLOW"
        else:
            level = "RED"

        return CrowdingSignal(
            strategy_name=strategy_name,
            signal_concentration=concentration,
            ic_trend=ic_trend,
            turnover_zscore=turnover_z,
            crowding_score=round(score, 4),
            warning_level=level,
        )

    @staticmethod
    def compute_concentration(targets: list[str]) -> float:
        """Ratio of unique targets to total signals. 0.25 = very crowded, 1.0 = perfect diversity."""
        if not targets:
            return 1.0
        return len(set(targets)) / len(targets)

    @staticmethod
    def compute_ic_trend(ic_series: np.ndarray, window: int = 100) -> float:
        """Rolling window IC trend. Positive = improving, negative = decaying."""
        if len(ic_series) < window:
            return 0.0
        rolling = np.convolve(ic_series, np.ones(window) / window, mode="valid")
        if len(rolling) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(rolling)), rolling, 1)[0])

    @staticmethod
    def compute_turnover_z(
        current: float, historical: np.ndarray
    ) -> float:
        """Z-score of current turnover vs historical distribution."""
        mu = float(np.mean(historical))
        sigma = float(np.std(historical, ddof=1))
        if sigma < 1e-12:
            return 0.0
        return (current - mu) / sigma


# ─── Lot Turnover Controller ─────────────────────────────────────────────────


class LotTurnoverController:
    """
    Low-Turnover Control (LoT) — prevents excessive rebalancing.

    Dynamically adjusts the rebalancing threshold based on regime
    transition probability: high probability → lower threshold → more active.
    """

    def __init__(self, base_threshold: float = 0.05) -> None:
        self._base = base_threshold

    def should_rebalance(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        transition_prob: float,
    ) -> bool:
        """Determine if rebalancing is warranted given turnover cost."""
        if not target_weights or not current_weights:
            return False

        turnover = self._compute_turnover(target_weights, current_weights)
        threshold = self.compute_dynamic_threshold(transition_prob)
        return turnover > threshold

    def compute_dynamic_threshold(self, transition_prob: float) -> float:
        """
        LoT dynamic threshold: lower threshold when regime change is likely.

        transition_prob = 0.0 → threshold = base (conservative, don't trade)
        transition_prob = 1.0 → threshold = base × 0.2 (aggressive, regime shifted)
        """
        factor = max(0.2, 1.0 - transition_prob)
        return self._base * factor

    @staticmethod
    def _compute_turnover(
        target: dict[str, float], current: dict[str, float]
    ) -> float:
        """Compute portfolio turnover ratio."""
        all_keys = set(target) | set(current)
        if not all_keys:
            return 0.0
        diffs = [abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in all_keys]
        return sum(diffs) / 2.0  # divide by 2 because each trade counted twice
