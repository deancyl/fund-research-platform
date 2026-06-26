"""Smoke tests for agent_core.py — info isolation, calibration, fact anchoring, cross-validation."""
import pytest

from src.core.recommend.agent_core import (
    AgentRole,
    AgentStatement,
    ConfidenceCalibrator,
    CrossValidationResult,
    DualLLMValidator,
    filter_context,
    validate_fact_anchoring,
    validate_isolation,
)


class TestInfoIsolation:
    def test_macro_sees_only_macro(self) -> None:
        ctx = {"vix": 20.0, "pmi": 50.5, "close": 3.5, "portfolio_weights": 0.6}
        filtered = filter_context(ctx, AgentRole.MACRO)
        assert "vix" in filtered
        assert "close" not in filtered
        assert "portfolio_weights" not in filtered

    def test_cio_sees_everything(self) -> None:
        ctx = {"vix": 20.0, "close": 3.5, "portfolio_weights": 0.6}
        filtered = filter_context(ctx, AgentRole.CIO)
        assert len(filtered) == 3

    def test_isolation_violation_detected(self) -> None:
        assert not validate_isolation(AgentRole.MACRO, {"vix", "close"})  # close not allowed
        assert validate_isolation(AgentRole.MACRO, {"vix", "pmi"})


class TestCalibration:
    def test_cold_start_regresses_to_mean(self) -> None:
        cal = ConfidenceCalibrator(min_samples=50)
        result = cal.calibrate(0.95)
        assert result < 0.95  # regressed toward 0.5

    def test_with_data_converges(self) -> None:
        cal = ConfidenceCalibrator(min_samples=10)
        for _ in range(50):
            cal.record(0.80, 0.75)
            cal.record(0.40, 0.30)
        calibrated = cal.calibrate(0.80)
        assert 0.0 <= calibrated <= 1.0


class TestFactAnchoring:
    def test_clean_statement(self) -> None:
        stmt = AgentStatement("quant", "bullish", 7.0, {"pe": 14.4, "rsi": 55.0}, "test")
        data = {"pe": 14.4, "rsi": 55.0}
        issues = validate_fact_anchoring(stmt, data)
        assert issues == []

    def test_hallucination_detected(self) -> None:
        stmt = AgentStatement("macro", "risk_on", 8.0, {"vix": 25.0, "fake_metric": 99.0}, "test")
        data = {"vix": 25.0}
        issues = validate_fact_anchoring(stmt, data)
        assert len(issues) >= 1
        assert "HALLUCINATION" in issues[0]


class TestDualLLMValidator:
    def test_fully_consensus(self) -> None:
        r = DualLLMValidator.validate("BUY", 0.9, "BUY", 0.85)
        assert r.verdict_match
        assert r.confidence_multiplier == 1.0

    def test_direction_match_only(self) -> None:
        r = DualLLMValidator.validate("BUY", 0.9, "ACCUMULATE", 0.8)
        assert r.direction_match
        assert not r.verdict_match
        assert r.confidence_multiplier == 0.7

    def test_contradiction_forces_hold(self) -> None:
        r = DualLLMValidator.validate("BUY", 0.9, "SELL", 0.8)
        assert not r.direction_match
        assert r.merged_verdict == "HOLD"
