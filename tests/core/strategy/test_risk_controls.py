"""TDD tests for CrowdingMonitor + LotTurnoverController."""
import numpy as np
import pytest

from src.core.strategy.risk_controls import CrowdingMonitor, LotTurnoverController


@pytest.fixture
def ic_series() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(0.03, 0.08, 200).astype(np.float64)


@pytest.fixture
def decaying_ic() -> np.ndarray:
    """IC that decays from 0.06 to -0.02."""
    return np.linspace(0.06, -0.02, 200, dtype=np.float64)


class TestCrowdingMonitor:
    def test_signal_concentration_perfect(self) -> None:
        cm = CrowdingMonitor()
        # All strategies point to different targets → concentration = 1.0
        targets = ["A", "B", "C", "D"]
        assert cm.compute_concentration(targets) == 1.0

    def test_signal_concentration_crowded(self) -> None:
        cm = CrowdingMonitor()
        # All strategies point to same target → highly concentrated
        targets = ["A", "A", "A", "A"]
        assert cm.compute_concentration(targets) == 0.25

    def test_ic_decay_detected(self, decaying_ic: np.ndarray) -> None:
        cm = CrowdingMonitor()
        trend = cm.compute_ic_trend(decaying_ic, window=100)
        assert trend < 0  # IC is decaying

    def test_ic_improving(self, ic_series: np.ndarray) -> None:
        cm = CrowdingMonitor()
        # Reverse to make IC improving
        improving = ic_series[::-1]
        trend = cm.compute_ic_trend(improving, window=100)
        assert trend > 0

    def test_crowding_score_high_when_concentrated(self) -> None:
        cm = CrowdingMonitor()
        # Extremely crowded (4→1) + strongly decaying IC + high turnover
        score = cm.evaluate(
            signal_targets=["A", "A", "A", "A"],  # all same → 0.25 concentration
            ic_series=np.linspace(0.08, -0.04, 200),  # steep decay
            current_turnover=0.30,
            historical_turnover=np.array([0.05, 0.06, 0.04, 0.05, 0.06], dtype=np.float64),
        )
        # c_risk=(1-0.25)*0.4=0.3, strong IC decay + high turnover → score > 0.5
        assert score.crowding_score > 0.4


class TestLotTurnoverController:
    def test_rebalance_when_above_threshold(self) -> None:
        ltc = LotTurnoverController(base_threshold=0.05)
        # Large turnover → should rebalance
        assert ltc.should_rebalance(
            target_weights={"A": 0.5, "B": 0.5},
            current_weights={"A": 0.2, "B": 0.8},
            transition_prob=0.1,
        )

    def test_skip_when_below_threshold(self) -> None:
        ltc = LotTurnoverController(base_threshold=0.20)
        # Small turnover → skip
        assert not ltc.should_rebalance(
            target_weights={"A": 0.51, "B": 0.49},
            current_weights={"A": 0.50, "B": 0.50},
            transition_prob=0.1,
        )

    def test_dynamic_threshold(self) -> None:
        ltc = LotTurnoverController(base_threshold=0.05)
        # High transition probability → lower threshold → more likely to rebalance
        dynamic = ltc.compute_dynamic_threshold(transition_prob=0.9)
        assert dynamic < 0.05

    def test_empty_weights_no_crash(self) -> None:
        ltc = LotTurnoverController()
        assert not ltc.should_rebalance({}, {}, 0.1)
