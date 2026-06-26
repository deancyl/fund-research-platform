"""Tests for memory.py + monte_carlo.py."""
import numpy as np
import pytest

from src.core.recommend.memory import MemorySystem
from src.core.engine.monte_carlo import block_bootstrap, bootstrap_confidence_interval, monte_carlo_summary


class TestMemorySystem:
    def test_record_and_settle(self) -> None:
        ms = MemorySystem()
        ms.record_decision("d1", "BUY", 0.80, 0.90, 0.05)
        ms.record_decision("d2", "SELL", 0.20, 0.70, -0.03)
        ms.settle_decisions({"d1": 0.06, "d2": -0.02})
        stats = ms.light_sleep_stats()
        assert stats["total"] == 2
        assert stats["direction_accuracy"] == 1.0  # both correct direction

    def test_rem_sleep_empty(self) -> None:
        ms = MemorySystem()
        insights = ms.rem_sleep(ms._last_rem or __import__("datetime").date.today())
        assert insights == []

    def test_deep_sleep_recalibrates(self) -> None:
        ms = MemorySystem()
        recal = ms.deep_sleep(
            factor_weights={"momentum": 0.15, "value": 0.10},
            factor_ic_trends={"momentum": -0.15, "value": 0.12},
            current_date=__import__("datetime").date.today(),
        )
        assert len(recal) >= 2


class TestMonteCarlo:
    def test_bootstrap_output_shape(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0005, 0.015, 500).astype(np.float64)
        result = block_bootstrap(returns, n_simulations=100, block_size=20)
        assert len(result["sharpe"]) == 100
        assert len(result["max_dd"]) == 100

    def test_ci_properties(self) -> None:
        values = np.linspace(0, 1, 1000, dtype=np.float64)
        ci = bootstrap_confidence_interval(values)
        assert ci["ci_lower"] < ci["median"] < ci["ci_upper"]

    def test_monte_carlo_summary(self) -> None:
        rng = np.random.default_rng(2)
        returns = rng.normal(0.0005, 0.015, 252).astype(np.float64)
        summary = monte_carlo_summary(returns, n_simulations=50, block_size=20)
        assert "sharpe" in summary
        assert "ci_upper" in summary["sharpe"]
