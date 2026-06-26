"""
Agent pipeline orchestration (T6.1) — ties together all Phase 6 components.

Pipeline: Scoring → Debate → DecisionHub → Memory → Output
Each stage can fail independently; pipeline never crashes on component failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.core.analysis.scoring import FundScorer
from src.core.recommend.debate_engine import (
    AgentRole,
    DebateEngine,
    StubAgent,
)
from src.core.recommend.decision_hub import DecisionHub, Verdict
from src.core.recommend.memory import MemorySystem


@dataclass
class RecommendationOutput:
    """Structured output consumed by TUI and Web adapters."""

    fund_code: str
    verdict: str
    confidence: float
    factor_score: float
    position_pct: float
    debate_rounds: int
    debate_status: str
    audit_warnings: list[str] = field(default_factory=list)
    memory_insight: str = ""


class AgentPipeline:
    """
    Full recommendation pipeline — all stages, graceful failure isolation.

    Usage:
        pipeline = AgentPipeline()
        result = pipeline.run(market_data, fund_code="005827")
        # result.verdict → "BUY" / "HOLD" / ...
    """

    def __init__(self) -> None:
        self._scorer = FundScorer()
        self._debater = DebateEngine()
        self._hub = DecisionHub()
        self._memory = MemorySystem()

    def run(
        self,
        market_data: dict,
        fund_code: str = "",
        current_date: date | None = None,
    ) -> RecommendationOutput:
        """
        Execute the full pipeline for a given fund.

        Each stage wraps exceptions — a failure in stage N does not block N+1.
        """
        warnings: list[str] = []

        # ── Stage 1: Factor scoring ──────────────────────────────────
        factor_score = 0.5
        try:
            scores = self._scorer.score_all(market_data)
            target = next((s for s in scores if s["fund_code"] == fund_code), None)
            factor_score = target["total_score"] if target else 0.5
        except Exception:
            warnings.append("scoring_failed")

        # ── Stage 2: Multi-agent debate ──────────────────────────────
        debate_result = None
        try:
            debate_result = self._debater.run_debate(
                macro_agent=StubAgent(AgentRole.MACRO, "neutral", 5.0),
                quant_agent=StubAgent(AgentRole.QUANT, "neutral", 5.0),
                risk_agent=StubAgent(AgentRole.RISK, "ok", 5.0),
                market_context=market_data,
                factor_score=factor_score,
            )
        except Exception:
            warnings.append("debate_failed")

        # ── Stage 3: DecisionHub ─────────────────────────────────────
        decision = None
        try:
            llm_advice = debate_result.cio_verdict if debate_result else "HOLD"
            llm_conf = debate_result.cio_confidence if debate_result else 0.5

            decision = self._hub.decide(
                factor_score=factor_score,
                llm_advice=llm_advice,
                llm_confidence=llm_conf,
            )
        except Exception:
            warnings.append("decision_failed")
            decision = self._hub.decide(factor_score=factor_score, llm_advice="HOLD", llm_confidence=0.5)

        # ── Stage 4: Memory recording ─────────────────────────────────
        memory_insight = ""
        try:
            import uuid
            did = str(uuid.uuid4())[:8]
            self._memory.record_decision(
                decision_id=did,
                verdict=decision.verdict.value,
                factor_score=factor_score,
                confidence=decision.confidence,
                expected_return=0.05 if decision.verdict in (Verdict.BUY, Verdict.ACCUMULATE) else 0.0,
                ts=current_date,
            )
            stats = self._memory.light_sleep_stats()
            if stats:
                memory_insight = f"Historical accuracy: {stats.get('direction_accuracy', 0):.1%}"
        except Exception:
            pass  # memory failure is non-blocking

        # ── Output ────────────────────────────────────────────────────
        return RecommendationOutput(
            fund_code=fund_code,
            verdict=decision.verdict.value if decision else "HOLD",
            confidence=decision.confidence if decision else 0.5,
            factor_score=factor_score,
            position_pct=decision.position_pct if decision else 0.0,
            debate_rounds=debate_result.rounds_completed if debate_result else 0,
            debate_status=debate_result.status.value if debate_result else "N/A",
            audit_warnings=warnings,
            memory_insight=memory_insight,
        )


def run_pipeline_for_fund(
    market_data: dict,
    fund_code: str,
    current_date: date | None = None,
) -> RecommendationOutput:
    """Convenience function: run full pipeline for a single fund."""
    pipeline = AgentPipeline()
    return pipeline.run(market_data, fund_code, current_date)
