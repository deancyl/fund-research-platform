"""TDD tests for 8-factor fund scoring system."""
import numpy as np
import polars as pl
import pytest

from src.core.analysis.scoring import FundScorer


@pytest.fixture
def scorer() -> FundScorer:
    return FundScorer()


@pytest.fixture
def sample_nav_data() -> dict[str, pl.DataFrame]:
    """5 funds, 252 days of NAV with different characteristics."""
    rng = np.random.default_rng(42)
    n = 252
    funds = {
        "F1_HIGH_RETURN": pl.DataFrame({
            "date": pl.date_range(pl.date(2025,6,1), pl.date(2026,6,1), "1d", eager=True)[:n],
            "nav": (np.cumprod(1 + rng.normal(0.001, 0.01, n)) * 2.0).tolist(),
            "close": (np.cumprod(1 + rng.normal(0.001, 0.01, n)) * 2.0).tolist(),
        }),
        "F2_LOW_VOL": pl.DataFrame({
            "date": pl.date_range(pl.date(2025,6,1), pl.date(2026,6,1), "1d", eager=True)[:n],
            "nav": (np.cumprod(1 + rng.normal(0.0003, 0.005, n)) * 1.5).tolist(),
            "close": (np.cumprod(1 + rng.normal(0.0003, 0.005, n)) * 1.5).tolist(),
        }),
        "F3_HIGH_DRAWDOWN": pl.DataFrame({
            "date": pl.date_range(pl.date(2025,6,1), pl.date(2026,6,1), "1d", eager=True)[:n],
            "nav": (np.cumprod(1 + rng.normal(-0.002, 0.015, n)) * 1.8).tolist(),
            "close": (np.cumprod(1 + rng.normal(-0.002, 0.015, n)) * 1.8).tolist(),
        }),
        "F4_MOMENTUM": pl.DataFrame({
            "date": pl.date_range(pl.date(2025,6,1), pl.date(2026,6,1), "1d", eager=True)[:n],
            "nav": (np.cumprod(1 + rng.normal(0.002, 0.008, n)) * 2.5).tolist(),
            "close": (np.cumprod(1 + rng.normal(0.002, 0.008, n)) * 2.5).tolist(),
        }),
        "F5_FLAT": pl.DataFrame({
            "date": pl.date_range(pl.date(2025,6,1), pl.date(2026,6,1), "1d", eager=True)[:n],
            "nav": np.full(n, 1.0).tolist(),
            "close": np.full(n, 1.0).tolist(),
        }),
    }
    return funds


class TestFundScorer:
    def test_scores_all_funds(self, scorer: FundScorer, sample_nav_data) -> None:
        scores = scorer.score_all(sample_nav_data)
        assert len(scores) == 5
        for s in scores:
            assert "fund_code" in s
            assert "total_score" in s
            assert 0 <= s["total_score"] <= 1

    def test_high_return_fund_scores_higher(self, scorer: FundScorer, sample_nav_data) -> None:
        scores = scorer.score_all(sample_nav_data)
        scored = {s["fund_code"]: s["total_score"] for s in scores}
        # F4 (momentum) should outscore F5 (flat)
        assert scored["F4_MOMENTUM"] > scored["F5_FLAT"]

    def test_eight_factors_computed(self, scorer: FundScorer, sample_nav_data) -> None:
        scores = scorer.score_all(sample_nav_data)
        s = scores[0]
        for factor in ["return_score", "trend_score", "risk_adj_score", "drawdown_score",
                        "manager_score", "rating_score", "size_score", "momentum_score"]:
            assert factor in s, f"Missing factor: {factor}"

    def test_empty_data(self, scorer: FundScorer) -> None:
        scores = scorer.score_all({})
        assert scores == []

    def test_single_fund(self, scorer: FundScorer, sample_nav_data) -> None:
        single = {"F1_HIGH_RETURN": sample_nav_data["F1_HIGH_RETURN"]}
        scores = scorer.score_all(single)
        assert len(scores) == 1
        assert scores[0]["fund_code"] == "F1_HIGH_RETURN"

    def test_custom_weights(self) -> None:
        custom = FundScorer(
            weights={
                "return": 0.30, "trend": 0.05, "risk_adj": 0.30,
                "drawdown": 0.15, "manager": 0.05, "rating": 0.05,
                "size": 0.05, "momentum": 0.05,
            }
        )
        assert sum(custom.weights.values()) == pytest.approx(1.0)
