"""TDD tests for debate engine and hallucination auditor."""
import pytest

from src.core.recommend.debate_engine import (
    AgentRole,
    AuditVerdict,
    BaseAgent,
    ConvergenceStatus,
    DebateEngine,
    HallucinationAuditor,
    StubAgent,
)


class TestDebateEngine:
    def test_converges_when_agents_agree(self) -> None:
        engine = DebateEngine()
        quant = StubAgent(AgentRole.QUANT, "bullish", 7.0)
        risk = StubAgent(AgentRole.RISK, "ok", 7.0)  # same direction, strength close
        macro = StubAgent(AgentRole.MACRO, "risk_on", 6.0)

        result = engine.run_debate(macro, quant, risk, {}, factor_score=0.8)
        assert result.status == ConvergenceStatus.CONVERGED
        assert result.rounds_completed <= 2  # should converge quickly

    def test_does_not_converge_when_opposed(self) -> None:
        engine = DebateEngine()
        quant = StubAgent(AgentRole.QUANT, "bullish", 8.0)
        risk = StubAgent(AgentRole.RISK, "high_risk", 8.0)
        macro = StubAgent(AgentRole.MACRO, "neutral", 5.0)

        result = engine.run_debate(macro, quant, risk, {}, factor_score=0.5)
        assert result.status == ConvergenceStatus.MAX_ROUNDS
        assert result.rounds_completed == engine.MAX_ROUNDS

    def test_cio_verdict_follows_factor_score(self) -> None:
        engine = DebateEngine()
        agents = [StubAgent(AgentRole.MACRO), StubAgent(AgentRole.QUANT), StubAgent(AgentRole.RISK)]
        result = engine.run_debate(*agents, {}, factor_score=0.85)
        assert result.cio_verdict == "BUY"

        result = engine.run_debate(*agents, {}, factor_score=0.15)
        assert result.cio_verdict == "SELL"

    def test_statements_recorded(self) -> None:
        engine = DebateEngine()
        result = engine.run_debate(
            StubAgent(AgentRole.MACRO), StubAgent(AgentRole.QUANT), StubAgent(AgentRole.RISK),
            {}, factor_score=0.5,
        )
        assert len(result.statements) >= 3


class TestHallucinationAuditor:
    def test_record_and_settle(self) -> None:
        auditor = HallucinationAuditor()
        auditor.record("d1", "cio", "BUY", 0.10)
        auditor.record("d2", "cio", "SELL", -0.05)

        auditor.settle({"d1": 0.12, "d2": 0.03})
        report = auditor.report()
        assert report["total_decisions"] == 2

    def test_hallucination_on_wrong_direction(self) -> None:
        auditor = HallucinationAuditor()
        auditor.record("d1", "quant", "BUY", 0.10)
        auditor.settle({"d1": -0.05})  # expected +10%, actual -5%
        report = auditor.report()
        assert report["hallucination_rate"] == 1.0

    def test_empty_report(self) -> None:
        auditor = HallucinationAuditor()
        report = auditor.report()
        assert report["total_decisions"] == 0
