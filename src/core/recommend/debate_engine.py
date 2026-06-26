"""
AI Decision Engine — Multi-Agent debate system with audit trail.

Architecture (inspired by openInvest + FinAgent):
  Round 1: Macro → Quant → Risk (parallel, isolated information)
  Round 2+: Cross-challenge (each sees the other's output)
  Convergence: Same signal + strength < 1.0 for 2 consecutive rounds
  CIO: Synthesizes debate + factor scores → final recommendation

HallucinationAuditor: Post-hoc validation of AI decisions against
actual outcomes after 7/30/90 days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any


class AgentRole(StrEnum):
    MACRO = "macro_strategist"
    QUANT = "quant_analyst"
    RISK = "risk_officer"
    CIO = "cio"


class ConvergenceStatus(StrEnum):
    CONVERGED = "converged"
    MAX_ROUNDS = "max_rounds_reached"
    DEADLOCK = "deadlock"


@dataclass(frozen=True, slots=True)
class AgentStatement:
    """A single agent's analysis in one debate round."""
    role: AgentRole
    round_num: int
    signal: str
    strength: float
    key_data: dict[str, float] = field(default_factory=dict)
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "round": self.round_num,
            "signal": self.signal,
            "strength": self.strength,
            "key_data": self.key_data,
            "narrative": self.narrative,
        }


@dataclass(frozen=True, slots=True)
class DebateResult:
    """Complete debate outcome."""
    status: ConvergenceStatus
    rounds_completed: int
    statements: list[AgentStatement]
    cio_verdict: str
    cio_confidence: float
    factor_score: float
    llm_adjustment: float


class DebateEngine:
    """
    Multi-agent debate orchestrator.

    In production, each agent calls an LLM API with role-specific prompts.
    For now, it provides the protocol and convergence logic — agents can be
    implemented as LLM-calling plugins or rule-based stubs.
    """

    MAX_ROUNDS: int = 5
    CONVERGENCE_WINDOW: int = 2

    def __init__(self) -> None:
        self._history: list[DebateResult] = []

    def run_debate(
        self,
        macro_agent: "BaseAgent",
        quant_agent: "BaseAgent",
        risk_agent: "BaseAgent",
        market_context: dict[str, Any],
        factor_score: float = 0.5,
    ) -> DebateResult:
        """Execute the full debate protocol."""
        statements: list[AgentStatement] = []
        round_num = 1

        # Round 1: Parallel independent analysis (isolated)
        macro_stmt = macro_agent.analyze(market_context, round_num=round_num)
        quant_stmt = quant_agent.analyze(market_context, round_num=round_num)
        risk_stmt = risk_agent.analyze(market_context, round_num=round_num)
        statements.extend([macro_stmt, quant_stmt, risk_stmt])

        # Rounds 2+: Cross-challenge
        prev_quant = quant_stmt
        prev_risk = risk_stmt
        converged = self._check_convergence(quant_stmt, risk_stmt)

        while not converged and round_num < self.MAX_ROUNDS:
            round_num += 1
            cross_context = {**market_context, "peer_output": prev_risk.to_dict()}
            new_quant = quant_agent.analyze(cross_context, round_num=round_num)

            cross_context = {**market_context, "peer_output": prev_quant.to_dict()}
            new_risk = risk_agent.analyze(cross_context, round_num=round_num)

            statements.extend([new_quant, new_risk])
            converged = self._check_convergence(new_quant, new_risk)
            prev_quant, prev_risk = new_quant, new_risk

        # CIO synthesis
        cio_stmt = self._cio_synthesize(statements, factor_score)

        status = ConvergenceStatus.CONVERGED if converged else ConvergenceStatus.MAX_ROUNDS

        result = DebateResult(
            status=status,
            rounds_completed=round_num,
            statements=statements,
            cio_verdict=cio_stmt.signal,
            cio_confidence=cio_stmt.strength / 10.0,
            factor_score=factor_score,
            llm_adjustment=0.0,
        )
        self._history.append(result)
        return result

    def _check_convergence(self, q: AgentStatement, r: AgentStatement) -> bool:
        """Converged if signals align AND strength gap < 1.0."""
        same_direction = self._direction(q.signal) == self._direction(r.signal)
        small_gap = abs(q.strength - r.strength) < 1.0
        return same_direction and small_gap

    def _cio_synthesize(
        self, statements: list[AgentStatement], factor_score: float
    ) -> AgentStatement:
        """CIO synthesizes debate into final verdict (stub — uses factor score)."""
        if factor_score >= 0.75:
            signal, strength = "BUY", 8.0
        elif factor_score >= 0.60:
            signal, strength = "ACCUMULATE", 6.0
        elif factor_score >= 0.40:
            signal, strength = "HOLD", 5.0
        elif factor_score >= 0.25:
            signal, strength = "TRIM", 3.0
        else:
            signal, strength = "SELL", 1.0

        return AgentStatement(
            role=AgentRole.CIO,
            round_num=0,  # CIO runs after debate
            signal=signal,
            strength=strength,
            key_data={"factor_score": factor_score},
            narrative=f"CIO synthesis: factor={factor_score:.2f} → {signal}",
        )

    @staticmethod
    def _direction(signal: str) -> int:
        mapping = {
            "BUY": 2, "ACCUMULATE": 1, "HOLD": 0, "TRIM": -1, "SELL": -2,
            "bullish": 1, "bearish": -1, "neutral": 0,
            "risk_on": 1, "risk_off": -1,
            "ok": 1, "concerned": -1, "high_risk": -2,
        }
        return mapping.get(signal, 0)


# ─── Agent Protocol ─────────────────────────────────────────────────────────


class BaseAgent:
    """Protocol for debate agents. Override analyze() with role-specific logic."""

    role: AgentRole

    def analyze(self, context: dict[str, Any], round_num: int = 1) -> AgentStatement:
        """Analyze market context and produce a statement."""
        raise NotImplementedError


class StubAgent(BaseAgent):
    """Stub agent for testing — returns a fixed signal."""

    def __init__(self, role: AgentRole, default_signal: str = "neutral", default_strength: float = 5.0) -> None:
        self.role = role
        self._signal = default_signal
        self._strength = default_strength

    def analyze(self, context: dict[str, Any], round_num: int = 1) -> AgentStatement:
        return AgentStatement(
            role=self.role,
            round_num=round_num,
            signal=self._signal,
            strength=self._strength,
            narrative=f"Stub {self.role.value} analysis (round {round_num})",
        )


# ─── Hallucination Auditor ──────────────────────────────────────────────────


class AuditVerdict(StrEnum):
    PASS = "PASS"
    OVERCONFIDENT = "OVERCONFIDENT"
    HALLUCINATION = "HALLUCINATION"
    FACTUAL_ERROR = "FACTUAL_ERROR"
    PENDING = "PENDING"


@dataclass
class AuditRecord:
    decision_id: str
    timestamp: datetime
    agent_role: str
    verdict: str
    expected_return: float
    actual_return_7d: float | None = None
    actual_return_30d: float | None = None
    result_7d: AuditVerdict = AuditVerdict.PENDING
    result_30d: AuditVerdict = AuditVerdict.PENDING


class HallucinationAuditor:
    """Post-hoc audit of AI decisions against actual outcomes."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def record(self, decision_id: str, agent_role: str, verdict: str, expected_return: float) -> None:
        self.records.append(AuditRecord(
            decision_id=decision_id,
            timestamp=datetime.now(),
            agent_role=agent_role,
            verdict=verdict,
            expected_return=expected_return,
        ))

    def settle(self, actual_returns: dict[str, float]) -> None:
        """Settle pending records against actual returns."""
        for r in self.records:
            if r.result_30d != AuditVerdict.PENDING:
                continue
            actual = actual_returns.get(r.decision_id)
            if actual is None:
                continue
            r.actual_return_30d = actual
            r.result_30d = self._evaluate(r.expected_return, actual)

    def report(self) -> dict[str, float]:
        """Generate audit report: pass rate, hallucination rate per agent."""
        settled = [r for r in self.records if r.result_30d != AuditVerdict.PENDING]
        if not settled:
            return {"total_decisions": 0}

        total = len(settled)
        passed = sum(1 for r in settled if r.result_30d == AuditVerdict.PASS)
        return {
            "total_decisions": total,
            "pass_rate": passed / total,
            "hallucination_rate": sum(1 for r in settled if r.result_30d == AuditVerdict.HALLUCINATION) / total,
        }

    @staticmethod
    def _evaluate(expected: float, actual: float) -> AuditVerdict:
        direction_correct = (expected > 0) == (actual > 0)
        if not direction_correct:
            return AuditVerdict.HALLUCINATION
        error = abs(expected - actual) / max(abs(actual), 0.001)
        if error > 0.5:
            return AuditVerdict.OVERCONFIDENT
        return AuditVerdict.PASS
