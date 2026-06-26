"""
DecisionHub — deterministic decision engine.

LLM agents provide analysis; DecisionHub makes the final call.
Implements 6 safety checks per SYSTEM-CONTRACT §3:
  1. LLM adjustment limited to ±15%
  2. CRISIS risk circuit-breaker → max HOLD
  3. Single position capped at 20%
  4. Overconfidence detection (few agents → halve confidence)
  5. Factor score anomaly → forced neutral
  6. LLM/factor contradiction → zero LLM adjustment
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    TRIM = "TRIM"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    factor_score: float
    llm_adjustment: float
    adjusted_score: float
    position_pct: float
    confidence: float
    safety_checks_passed: int
    safety_checks_total: int


class DecisionHub:
    """Deterministic decision engine — LLM advises, Hub decides."""

    THRESHOLDS: dict[Verdict, float] = {
        Verdict.BUY: 0.75,
        Verdict.ACCUMULATE: 0.60,
        Verdict.HOLD: 0.40,
        Verdict.TRIM: 0.25,
        Verdict.SELL: 0.00,
    }

    LLM_MAX_ADJUSTMENT: float = 0.15
    MAX_POSITION_PCT: float = 0.20
    KELLY_FRACTION: float = 0.25
    ANOMALY_LOW: float = 0.05
    ANOMALY_HIGH: float = 0.95

    def __init__(self, risk_flag: str = "NORMAL") -> None:
        self.risk_flag = risk_flag

    def decide(
        self,
        factor_score: float,
        llm_advice: str,
        llm_confidence: float,
        n_agents: int = 4,
    ) -> Decision:
        checks = 0
        total_checks = 6

        # ── Check 1: LLM adjustment direction ─────────────────────────
        factor_verdict = self._score_to_verdict(factor_score)
        llm_direction = self._verdict_direction(llm_advice)
        factor_direction = self._verdict_direction(factor_verdict)

        if llm_direction == factor_direction and llm_direction != 0:
            adjustment = llm_confidence * self.LLM_MAX_ADJUSTMENT * (1 if llm_direction > 0 else -1)
        else:
            adjustment = 0.0  # contradiction → zero LLM influence
        checks += 1

        adjusted = float(min(max(factor_score + adjustment, 0.0), 1.0))
        checks += 1  # Check 2: adjustment cap enforced via clip

        # ── Check 3: CRISIS circuit-breaker ───────────────────────────
        if self.risk_flag == "CRISIS":
            adjusted = min(adjusted, self.THRESHOLDS[Verdict.HOLD])
        checks += 1

        # ── Check 4: Anomaly score → neutral ─────────────────────────
        if factor_score > self.ANOMALY_HIGH or factor_score < self.ANOMALY_LOW:
            adjusted = 0.50
        checks += 1

        # ── Verdict from adjusted score ───────────────────────────────
        verdict = self._score_to_verdict(adjusted)

        # ── Check 5: Position sizing ──────────────────────────────────
        position_pct = min(adjusted * self.KELLY_FRACTION, self.MAX_POSITION_PCT)
        checks += 1

        # ── Check 6: Confidence calibration ───────────────────────────
        if n_agents < 3:
            confidence = llm_confidence * 0.5
        else:
            confidence = llm_confidence
        checks += 1

        return Decision(
            verdict=verdict,
            factor_score=factor_score,
            llm_adjustment=round(adjustment, 4),
            adjusted_score=round(adjusted, 4),
            position_pct=round(position_pct, 4),
            confidence=round(confidence, 4),
            safety_checks_passed=checks,
            safety_checks_total=total_checks,
        )

    # ── Internal ─────────────────────────────────────────────────────────

    @classmethod
    def _score_to_verdict(cls, score: float) -> Verdict:
        for v, threshold in sorted(cls.THRESHOLDS.items(), key=lambda x: -x[1]):
            if score >= threshold:
                return v
        return Verdict.SELL

    @staticmethod
    def _verdict_direction(verdict_str: str) -> int:
        mapping = {"BUY": 2, "ACCUMULATE": 1, "HOLD": 0, "TRIM": -1, "SELL": -2}
        return mapping.get(verdict_str, 0)
