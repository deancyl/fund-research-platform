"""Smoke tests for attribution and validation modules."""
import numpy as np

from src.core.analysis.attribution import brinson_attribution
from src.core.engine.validation import (
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    minimum_track_record_length,
)


class TestBrinson:
    def test_basic_attribution(self) -> None:
        result = brinson_attribution(
            portfolio_weights={"tech": 0.6, "finance": 0.4},
            benchmark_weights={"tech": 0.5, "finance": 0.5},
            portfolio_returns={"tech": 0.15, "finance": 0.05},
            benchmark_returns={"tech": 0.12, "finance": 0.06},
            benchmark_total_return=0.09,
        )
        assert "allocation_effect" in result
        assert abs(result["active_return"] - (result["allocation_effect"] + result["selection_effect"] + result["interaction_effect"])) < 1e-10


class TestCPCV:
    def test_generates_paths(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0005, 0.015, 500).astype(np.float64)
        paths = combinatorial_purged_cv(returns, n_splits=5, n_test_groups=2)
        assert len(paths) > 0

    def test_empty_short_data(self) -> None:
        returns = np.array([0.01, -0.02], dtype=np.float64)
        paths = combinatorial_purged_cv(returns)
        assert paths == []


class TestDSR:
    def test_dsr_positive_for_good_sr(self) -> None:
        # 5 trials, 252 days, SR=1.5 → modest correction, DSR still positive
        dsr = deflated_sharpe_ratio(1.5, 5, 252)
        assert dsr > 0

    def test_min_track_record(self) -> None:
        t = minimum_track_record_length(1.0)
        assert t > 0
