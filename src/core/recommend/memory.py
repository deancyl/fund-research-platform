"""
3-Phase Memory System (Light Sleep / REM Sleep / Deep Sleep).
Phase 6 T6.9. Pure computation — no LLM dependencies.

Light Sleep (daily): Record decisions → short-term calibration data.
REM Sleep (weekly): Cluster outcomes → generate actionable lessons.
Deep Sleep (monthly): Recalibrate factor weights → prune decaying factors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np


@dataclass
class DecisionRecord:
    decision_id: str
    timestamp: date
    verdict: str
    factor_score: float
    confidence: float
    expected_return: float
    actual_return: float | None = None
    settled: bool = False


@dataclass
class ClusterInsight:
    pattern: str
    win_rate: float
    sample_count: int
    recommendation: str


@dataclass
class FactorRecalibration:
    factor_name: str
    old_weight: float
    new_weight: float
    reason: str


class MemorySystem:
    """
    Three-phase memory derived from openInvest Dreaming + FinAgent Reflection.

    Light Sleep (daily): Append decisions, settle 7-day outcomes.
    REM Sleep (weekly): Cluster by market conditions, extract insights.
    Deep Sleep (monthly): Check factor IC decay, propose weight adjustments.
    """

    def __init__(self) -> None:
        self._decisions: list[DecisionRecord] = []
        self._insights: list[ClusterInsight] = []
        self._recalibrations: list[FactorRecalibration] = []
        self._last_rem: date | None = None
        self._last_deep: date | None = None

    # ── Light Sleep (daily) ──────────────────────────────────────────────

    def record_decision(
        self, decision_id: str, verdict: str, factor_score: float,
        confidence: float, expected_return: float, ts: date | None = None,
    ) -> None:
        self._decisions.append(DecisionRecord(
            decision_id=decision_id, timestamp=ts or date.today(),
            verdict=verdict, factor_score=factor_score,
            confidence=confidence, expected_return=expected_return,
        ))

    def settle_decisions(self, outcomes: dict[str, float]) -> None:
        """Settle decisions with 7-day actual returns."""
        for d in self._decisions:
            if d.settled:
                continue
            actual = outcomes.get(d.decision_id)
            if actual is not None:
                d.actual_return = actual
                d.settled = True

    def light_sleep_stats(self) -> dict[str, float]:
        """Short-term calibration statistics."""
        settled = [d for d in self._decisions if d.settled]
        if not settled:
            return {}
        correct = sum(1 for d in settled if (d.expected_return > 0) == (d.actual_return and d.actual_return > 0))
        return {
            "total": len(settled),
            "direction_accuracy": correct / len(settled),
            "avg_confidence": float(np.mean([d.confidence for d in settled])),
        }

    # ── REM Sleep (weekly) ───────────────────────────────────────────────

    def rem_sleep(self, current_date: date) -> list[ClusterInsight]:
        """Cluster decisions by market condition and extract insights."""
        if len(self._decisions) < 20:
            return []

        settled = [d for d in self._decisions if d.settled]
        if len(settled) < 10:
            return []

        # Cluster by verdict type
        by_verdict: dict[str, list[DecisionRecord]] = defaultdict(list)
        for d in settled:
            by_verdict[d.verdict].append(d)

        insights: list[ClusterInsight] = []
        for verdict, group in by_verdict.items():
            correct = sum(1 for d in group if d.actual_return and (d.expected_return > 0) == (d.actual_return > 0))
            wr = correct / len(group)
            if wr > 0.65:
                insights.append(ClusterInsight(
                    pattern=f"verdict={verdict}", win_rate=wr,
                    sample_count=len(group),
                    recommendation=f"Continue trusting {verdict} signals (WR={wr:.0%})",
                ))
            elif wr < 0.35:
                insights.append(ClusterInsight(
                    pattern=f"verdict={verdict}", win_rate=wr,
                    sample_count=len(group),
                    recommendation=f"Consider reducing confidence on {verdict} (WR={wr:.0%})",
                ))

        self._insights.extend(insights)
        self._last_rem = current_date
        return insights

    # ── Deep Sleep (monthly) ─────────────────────────────────────────────

    def deep_sleep(
        self, factor_weights: dict[str, float],
        factor_ic_trends: dict[str, float], current_date: date,
    ) -> list[FactorRecalibration]:
        """Recalibrate factor weights based on IC trends."""
        recalibrations: list[FactorRecalibration] = []
        for name, ic_trend in factor_ic_trends.items():
            old_w = factor_weights.get(name, 0.10)
            if ic_trend < -0.1:
                new_w = max(0.01, old_w * 0.5)
                recalibrations.append(FactorRecalibration(name, old_w, new_w, f"IC decay {ic_trend:.3f} → halved"))
            elif ic_trend > 0.1:
                new_w = min(0.30, old_w * 1.2)
                recalibrations.append(FactorRecalibration(name, old_w, new_w, f"IC improving {ic_trend:.3f} → boosted"))
        self._recalibrations.extend(recalibrations)
        self._last_deep = current_date
        return recalibrations

    @property
    def total_decisions(self) -> int:
        return len(self._decisions)

    @property
    def insights(self) -> list[ClusterInsight]:
        return list(self._insights)

    @property
    def recalibrations(self) -> list[FactorRecalibration]:
        return list(self._recalibrations)
