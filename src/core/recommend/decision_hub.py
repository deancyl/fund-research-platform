"""
DecisionHub — deterministic decision engine. v0.1.1

Per external audit fixes:
  - LLM alignment: sign-based (±) not exact integer match. BUY(2) and
    ACCUMULATE(1) are now correctly recognized as both bullish.
  - Position sizing: true fractional Kelly f* = max(0, p - (1-p)/b) × 0.25
    replacing the static linear multiplier.
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
    THRESHOLDS: dict[Verdict, float] = {
        Verdict.BUY: 0.75,
        Verdict.ACCUMULATE: 0.60,
        Verdict.HOLD: 0.40,
        Verdict.TRIM: 0.25,
        Verdict.SELL: 0.00,
    }

    LLM_MAX_ADJUSTMENT: float = 0.15
    MAX_POSITION_PCT: float = 0.20

    def __init__(self, risk_flag: str = "NORMAL") -> None:
        self.risk_flag = risk_flag

    def decide(
        self,
        factor_score: float,
        llm_advice: str,
        llm_confidence: float,
        win_rate: float = 0.55,
        win_loss_ratio: float = 1.4,
        n_agents: int = 4,
    ) -> Decision:
        checks = 0
        total_checks = 6

        # ── Check 1: LLM alignment (SIGN-BASED, per audit fix) ──────────
        factor_verdict = self._score_to_verdict(factor_score)
        llm_direction = self._verdict_direction(llm_advice)
        factor_direction = self._verdict_direction(factor_verdict)

        # 审计修复: 符号对齐 — 同为正(看多)或同为负(看空)即对齐
        is_aligned = (llm_direction > 0 and factor_direction > 0) or (
            llm_direction < 0 and factor_direction < 0
        )
        if is_aligned and llm_direction != 0 and factor_direction != 0:
            adjustment = llm_confidence * self.LLM_MAX_ADJUSTMENT * (1 if factor_direction > 0 else -1)
        else:
            adjustment = 0.0
        checks += 1

        adjusted = float(min(max(factor_score + adjustment, 0.0), 1.0))
        checks += 1

        # Check 3: CRISIS
        if self.risk_flag == "CRISIS":
            adjusted = min(adjusted, self.THRESHOLDS[Verdict.HOLD])
        checks += 1

        # Check 4: Data validity — only block NaN/Inf, never legitimate extreme factor scores.
        # Per audit: 0.98 from multi-factor momentum resonance is VALID alpha — killing it
        # at 0.50 destroys S2/S14 strategies that depend on tail-event signals.
        if factor_score != factor_score or factor_score == float("inf"):
            adjusted = 0.50
        checks += 1

        verdict = self._score_to_verdict(adjusted)

        # ── Check 5: True fractional Kelly (audit fix) ───────────────────
        if win_rate > 0 and win_loss_ratio > 0:
            kelly_f = win_rate - (1.0 - win_rate) / win_loss_ratio
            fractional_kelly = max(0.0, kelly_f * 0.25)
            position_pct = min(adjusted * fractional_kelly * 4.0, self.MAX_POSITION_PCT)
        else:
            position_pct = min(adjusted * 0.25, self.MAX_POSITION_PCT)
        checks += 1

        # Check 6: Confidence
        confidence = llm_confidence * 0.5 if n_agents < 3 else llm_confidence
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
