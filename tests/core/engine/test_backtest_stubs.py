"""Smoke tests for backtesting engine stubs."""
import numpy as np
from src.core.engine.backtest_stubs import BacktraderEngine, VectorBTEngine


class TestVectorBT:
    def test_run_with_signals(self) -> None:
        rng = np.random.default_rng(1)
        prices = np.cumprod(1 + rng.normal(0.0005, 0.015, 252)) * 3.0
        signals = np.zeros(252, dtype=np.float64)
        signals[63:] = 0.5  # half-position after 3 months

        engine = VectorBTEngine()
        result = engine.run(prices, signals)
        assert result.n_trades > 0
        assert result.sharpe_ratio != 0.0

    def test_no_signals_no_trades(self) -> None:
        engine = VectorBTEngine()
        result = engine.run(np.array([3.0, 3.1, 3.2]), np.zeros(3))
        assert result.n_trades == 0


class TestBacktrader:
    def test_run_basic(self) -> None:
        rng = np.random.default_rng(2)
        prices = np.cumprod(1 + rng.normal(0.0005, 0.015, 252)) * 3.0
        signals = [{"target_weight": 0.5} for _ in range(252)]

        engine = BacktraderEngine()
        result = engine.run(prices, signals)
        assert result.final_value > 0
        assert result.n_trades > 0

    def test_fees_accumulate(self) -> None:
        engine = BacktraderEngine()
        prices = np.array([3.0, 3.1, 3.2, 3.3, 3.4], dtype=np.float64)
        signals = [{"target_weight": 1.0}, {"target_weight": 0.0}, {"target_weight": 0.5}, {"target_weight": 0.0}, {"target_weight": 0.0}]
        result = engine.run(prices, signals)
        assert result.total_fees > 0
