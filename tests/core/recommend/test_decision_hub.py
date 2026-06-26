"""
TDD tests for DecisionHub — the deterministic decision engine.
LLM provides analysis, DecisionHub makes the call.
"""

import pytest

from src.core.recommend.decision_hub import DecisionHub, Verdict


class TestBaseDecision:
    """Happy path: factor scores with no LLM adjustment."""

    def test_strong_buy(self) -> None:
        hub = DecisionHub()
        result = hub.decide(factor_score=0.85, llm_advice="BUY", llm_confidence=0.9)
        assert result.verdict == Verdict.BUY
        assert result.position_pct > 0

    def test_mid_hold(self) -> None:
        hub = DecisionHub()
        result = hub.decide(factor_score=0.50, llm_advice="HOLD", llm_confidence=0.6)
        assert result.verdict == Verdict.HOLD

    def test_weak_sell(self) -> None:
        hub = DecisionHub()
        result = hub.decide(factor_score=0.15, llm_advice="SELL", llm_confidence=0.8)
        assert result.verdict == Verdict.SELL


class TestLLMAdjustmentCap:
    """LLM can only shift the score by ±15%."""

    def test_llm_cannot_override_sell_to_buy(self) -> None:
        """Score 0.20 → max adjustment +0.15 = 0.35, still below BUY threshold (0.75)."""
        hub = DecisionHub()
        result = hub.decide(factor_score=0.20, llm_advice="BUY", llm_confidence=1.0)
        assert result.verdict != Verdict.BUY
        assert result.verdict != Verdict.ACCUMULATE

    def test_llm_can_upgrade_hold_to_accumulate(self) -> None:
        """Score 0.55 + max 0.15 = 0.70 → crosses ACCUMULATE threshold (0.60)."""
        hub = DecisionHub()
        result = hub.decide(factor_score=0.55, llm_advice="ACCUMULATE", llm_confidence=1.0)
        assert result.verdict in (Verdict.ACCUMULATE, Verdict.HOLD)


class TestSafetyChecks:
    """Six safety checks must gate all decisions."""

    def test_crisis_limits_to_hold(self) -> None:
        """CRISIS mode → max verdict is HOLD regardless of score."""
        hub = DecisionHub(risk_flag="CRISIS")
        result = hub.decide(factor_score=0.95, llm_advice="BUY", llm_confidence=1.0)
        assert result.verdict != Verdict.BUY
        assert result.verdict != Verdict.ACCUMULATE

    def test_llm_contradiction_zeroes_adjustment(self) -> None:
        """LLM says SELL but factor says BUY → adjustment = 0."""
        hub = DecisionHub()
        result = hub.decide(factor_score=0.80, llm_advice="SELL", llm_confidence=0.9)
        assert result.llm_adjustment == 0.0

    def test_overconfidence_capped(self) -> None:
        """When only 1 agent active, confidence is halved."""
        hub = DecisionHub()
        result = hub.decide(factor_score=0.80, llm_advice="BUY", llm_confidence=0.95, n_agents=1)
        assert result.confidence <= 0.5

    def test_anomaly_score_resets_to_neutral(self) -> None:
        """Factor score > 0.95 or < 0.05 → forced to 0.50."""
        hub = DecisionHub()
        result = hub.decide(factor_score=0.99, llm_advice="BUY", llm_confidence=0.8)
        assert result.verdict == Verdict.HOLD


class TestPositionSizing:
    """Position sizing never exceeds limits."""

    def test_position_capped(self) -> None:
        hub = DecisionHub()
        result = hub.decide(factor_score=0.90, llm_advice="BUY", llm_confidence=1.0)
        assert result.position_pct <= 0.20
