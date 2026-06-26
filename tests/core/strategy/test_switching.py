"""Tests for regime detection and strategy switching."""
import numpy as np
import pytest

from src.core.strategy.base import MarketRegime, StrategyConfig
from src.core.strategy.mean_reversion.grid_hurst import GridHurstStrategy
from src.core.strategy.switching import RegimeDetector, StrategySwitcher


class TestRegimeDetector:
    def test_trend_up_detected(self) -> None:
        detector = RegimeDetector()
        # Strong uptrend
        close = np.linspace(3.0, 4.0, 100) + np.random.default_rng(1).normal(0, 0.01, 100)
        result = detector.detect(close)
        assert result.regime in (MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS)
        assert 0 <= result.confidence <= 1

    def test_trend_down_detected(self) -> None:
        detector = RegimeDetector()
        close = np.linspace(4.0, 3.0, 100) + np.random.default_rng(2).normal(0, 0.01, 100)
        result = detector.detect(close)
        assert result.regime in (MarketRegime.TRENDING_DOWN, MarketRegime.SIDEWAYS)

    def test_sideways_detected(self) -> None:
        detector = RegimeDetector()
        close = np.full(100, 3.5) + np.random.default_rng(3).normal(0, 0.005, 100)
        result = detector.detect(close)
        assert result.regime == MarketRegime.SIDEWAYS

    def test_hurst_returned(self) -> None:
        detector = RegimeDetector()
        close = np.random.default_rng(4).normal(0, 0.01, 100).cumsum() + 3.0
        result = detector.detect(close)
        assert 0 <= result.hurst <= 1


class TestStrategySwitcher:
    def test_eligible_strategies(self) -> None:
        sw = StrategySwitcher()
        grid = GridHurstStrategy()
        sw.register("grid_hurst", grid)
        assert "grid_hurst" in sw.eligible_strategies(MarketRegime.SIDEWAYS)
        assert "grid_hurst" not in sw.eligible_strategies(MarketRegime.TRENDING_UP)

    def test_position_scale(self) -> None:
        sw = StrategySwitcher()
        assert sw.position_scale(MarketRegime.TRENDING_UP) == 1.0
        assert sw.position_scale(MarketRegime.CRISIS) == 0.0

    def test_allocate_returns_weights(self) -> None:
        sw = StrategySwitcher()
        sw.register("pe_pb", GridHurstStrategy(config=StrategyConfig(name="test", eligible_regimes={MarketRegime.SIDEWAYS})))
        alloc = sw.allocate(MarketRegime.SIDEWAYS, 100000.0, {})
        assert sum(alloc.values()) <= 70000  # 70% scale for sideways
