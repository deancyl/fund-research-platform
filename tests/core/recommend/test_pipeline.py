"""Tests for agent pipeline orchestration."""
import polars as pl
import numpy as np

from src.core.recommend.pipeline import AgentPipeline, RecommendationOutput


def _make_market_data(n: int = 252) -> dict:
    rng = np.random.default_rng(1)
    nav = np.cumprod(1 + rng.normal(0.0005, 0.015, n)) * 2.0
    return {
        "005827": pl.DataFrame({"nav": nav.tolist(), "close": nav.tolist()}),
        "510300": pl.DataFrame({"nav": (nav * 1.1).tolist(), "close": (nav * 1.1).tolist()}),
    }


class TestAgentPipeline:
    def test_run_returns_output(self) -> None:
        pipeline = AgentPipeline()
        result = pipeline.run(_make_market_data(), "005827")
        assert isinstance(result, RecommendationOutput)
        assert result.fund_code == "005827"
        assert result.verdict in ("BUY", "ACCUMULATE", "HOLD", "TRIM", "SELL")
        assert 0 <= result.confidence <= 1

    def test_empty_market_data_does_not_crash(self) -> None:
        pipeline = AgentPipeline()
        result = pipeline.run({}, "005827")
        assert result.verdict == "HOLD"

    def test_pipeline_graceful_unknown_fund(self) -> None:
        pipeline = AgentPipeline()
        result = pipeline.run(_make_market_data(), "999999")
        assert result.factor_score == 0.5  # default for unknown fund
