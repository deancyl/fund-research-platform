"""TDD tests for risk_metrics.py — Sharpe/Sortino/Calmar/VaR/MaxDD/etc."""
import numpy as np
import pytest

from src.core.analysis.risk_metrics import (
    calmar_ratio,
    cvar_95,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    var_95,
    win_rate,
)


@pytest.fixture
def up_returns() -> np.ndarray:
    """100 days of mostly positive returns with variation."""
    rng = np.random.default_rng(99)
    return rng.normal(0.003, 0.005, 100).astype(np.float64)


@pytest.fixture
def mixed_returns() -> np.ndarray:
    """Mixed gains and losses."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.015, 252).astype(np.float64)


@pytest.fixture
def down_returns() -> np.ndarray:
    """Mostly negative returns."""
    rng = np.random.default_rng(77)
    return rng.normal(-0.003, 0.005, 100).astype(np.float64)


class TestSharpeRatio:
    def test_positive_sharpe(self, up_returns: np.ndarray) -> None:
        sr = sharpe_ratio(up_returns)
        assert sr > 0

    def test_negative_sharpe(self, down_returns: np.ndarray) -> None:
        sr = sharpe_ratio(down_returns)
        assert sr < 0

    def test_zero_vol_returns_nan(self) -> None:
        flat = np.full(50, 0.0, dtype=np.float64)
        sr = sharpe_ratio(flat)
        assert np.isnan(sr)


class TestSortinoRatio:
    def test_sortino_and_sharpe_computed(self, mixed_returns: np.ndarray) -> None:
        """Both ratios should be finite numbers."""
        sr = sharpe_ratio(mixed_returns)
        so = sortino_ratio(mixed_returns)
        assert not np.isnan(sr)
        assert not np.isnan(so)


class TestMaxDrawdown:
    def test_positive_trend(self) -> None:
        prices = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        assert max_drawdown(prices) == 0.0

    def test_drawdown_detected(self) -> None:
        prices = np.array([100, 90, 95, 85, 100], dtype=np.float64)
        mdd = max_drawdown(prices)
        assert mdd > 0.10  # at least 10% drawdown


class TestVaRCVaR:
    def test_var_95_range(self, mixed_returns: np.ndarray) -> None:
        v = var_95(mixed_returns)
        assert v < 0  # VaR is a loss (negative)

    def test_cvar_worse_than_var(self, mixed_returns: np.ndarray) -> None:
        v = var_95(mixed_returns)
        cv = cvar_95(mixed_returns)
        # CVaR <= VaR (more negative = worse)
        assert cv <= v


class TestWinRate:
    def test_up_mostly_positive(self, up_returns: np.ndarray) -> None:
        assert win_rate(up_returns) > 0.6

    def test_down_mostly_negative(self, down_returns: np.ndarray) -> None:
        assert win_rate(down_returns) < 0.4


class TestProfitFactor:
    def test_profitable(self, up_returns: np.ndarray) -> None:
        pf = profit_factor(up_returns)
        assert pf > 1

    def test_unprofitable(self, down_returns: np.ndarray) -> None:
        pf = profit_factor(down_returns)
        assert pf < 1


class TestCalmarRatio:
    def test_no_drawdown(self) -> None:
        prices = np.linspace(100, 200, 252)
        cr = calmar_ratio(prices)
        assert cr > 0
