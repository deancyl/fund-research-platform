"""TDD tests for multi-period Brinson attribution with Carino smoothing."""
import numpy as np
import pytest

from src.core.analysis.attribution import (
    brinson_attribution,
    multi_period_brinson,
)


@pytest.fixture
def daily_data() -> list[dict]:
    """3-day data: portfolio overweights tech, tech outperforms."""
    return [
        {
            "portfolio_weights": {"tech": 0.6, "bond": 0.4},
            "benchmark_weights": {"tech": 0.5, "bond": 0.5},
            "portfolio_returns": {"tech": 0.02, "bond": 0.001},
            "benchmark_returns": {"tech": 0.01, "bond": 0.002},
            "benchmark_total_return": 0.006,
        },
        {
            "portfolio_weights": {"tech": 0.6, "bond": 0.4},
            "benchmark_weights": {"tech": 0.5, "bond": 0.5},
            "portfolio_returns": {"tech": 0.015, "bond": 0.001},
            "benchmark_returns": {"tech": 0.012, "bond": 0.001},
            "benchmark_total_return": 0.0065,
        },
        {
            "portfolio_weights": {"tech": 0.7, "bond": 0.3},
            "benchmark_weights": {"tech": 0.5, "bond": 0.5},
            "portfolio_returns": {"tech": 0.01, "bond": 0.002},
            "benchmark_returns": {"tech": 0.008, "bond": 0.002},
            "benchmark_total_return": 0.005,
        },
    ]


class TestMultiPeriodBrinson:
    def test_returns_expected_keys(self, daily_data: list[dict]) -> None:
        result = multi_period_brinson(daily_data)
        for key in ("allocation_effect", "selection_effect", "interaction_effect", "active_return", "periods"):
            assert key in result

    def test_active_return_close_to_sum(self, daily_data: list[dict]) -> None:
        result = multi_period_brinson(daily_data)
        total = result["allocation_effect"] + result["selection_effect"] + result["interaction_effect"]
        # Carino smoothing preserves additive property approximately
        assert abs(result["active_return"] - total) < 0.01

    def test_single_period_fallback(self) -> None:
        """Single period: Carino k=1, should match single-period Brinson."""
        sp = {
            "portfolio_weights": {"a": 0.6}, "benchmark_weights": {"a": 0.5},
            "portfolio_returns": {"a": 0.1}, "benchmark_returns": {"a": 0.08},
            "benchmark_total_return": 0.08,
        }
        mp = multi_period_brinson([sp])
        single = brinson_attribution(
            portfolio_weights=sp["portfolio_weights"],
            benchmark_weights=sp["benchmark_weights"],
            portfolio_returns=sp["portfolio_returns"],
            benchmark_returns=sp["benchmark_returns"],
            benchmark_total_return=sp["benchmark_total_return"],
        )
        assert abs(mp["active_return"] - single["active_return"]) < 0.01

    def test_empty_list(self) -> None:
        result = multi_period_brinson([])
        assert result["periods"] == 0
