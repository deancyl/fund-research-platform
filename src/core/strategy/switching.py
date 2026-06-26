"""
Market regime detection and strategy switching engine.

Detects 5 market regimes using a combination of:
  - ADX (trend strength)
  - Hurst exponent (trend vs mean-reversion)
  - Volatility percentile (stress detection)
  - Correlation (crisis clustering)

Then maps each regime to eligible strategies with position scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from src.core.analysis.indicators import sma_tdx
from src.core.strategy.base import MarketRegime


# ─── Regime Detector ────────────────────────────────────────────────────────


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float   # 0-1
    adx: float
    hurst: float
    volatility_percentile: float
    correlation: float
    details: dict[str, float] = field(default_factory=dict)


class RegimeDetector:
    """Market regime classifier using ADX + Hurst + Volatility."""

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        hurst_trend_threshold: float = 0.60,
        hurst_mr_threshold: float = 0.40,
        vol_crisis_percentile: float = 0.90,
        vol_high_percentile: float = 0.70,
        corr_crisis_threshold: float = 0.70,
    ) -> None:
        self.adx_trend = adx_trend_threshold
        self.hurst_trend = hurst_trend_threshold
        self.hurst_mr = hurst_mr_threshold
        self.vol_crisis = vol_crisis_percentile
        self.vol_high = vol_high_percentile
        self.corr_crisis = corr_crisis_threshold

    def detect(
        self,
        close: np.ndarray,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        benchmark_close: np.ndarray | None = None,
        historical_volatility: np.ndarray | None = None,
    ) -> RegimeResult:
        """Detect current market regime from price data."""
        # ADX (simplified from ATR-based)
        adx_val = self._compute_adx_simple(close)

        # Hurst exponent
        hurst_val = self._hurst_rs(close)

        # Volatility percentile (current vs historical)
        recent_vol = np.std(np.diff(np.log(close[-20:] + 1e-12), axis=0)) if len(close) >= 20 else 0.01
        if historical_volatility is not None and len(historical_volatility) > 20:
            vol_percentile = float(np.mean(recent_vol > historical_volatility))
        else:
            vol_percentile = 0.5

        # Correlation with benchmark (crisis proxy)
        if benchmark_close is not None and len(benchmark_close) >= 20:
            corr = float(np.corrcoef(close[-20:], benchmark_close[-20:])[0, 1])
            corr = 0.0 if np.isnan(corr) else corr
        else:
            corr = 0.3

        # Price vs MA60 direction
        ma60 = sma_tdx(close, n_weight=60, m_weight=1) if len(close) >= 60 else np.full_like(close, close[-1])
        price_above_ma = float(close[-1]) > float(ma60[-1])

        # Classify
        regime, confidence = self._classify(adx_val, hurst_val, vol_percentile, corr, price_above_ma)

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            adx=adx_val,
            hurst=hurst_val,
            volatility_percentile=vol_percentile,
            correlation=corr,
        )

    def _classify(
        self, adx: float, hurst: float, vol_pct: float, corr: float, above_ma: bool
    ) -> tuple[MarketRegime, float]:
        # Crisis: high volatility + high correlation
        if vol_pct > self.vol_crisis and corr > self.corr_crisis:
            return MarketRegime.CRISIS, 0.90

        # High vol
        if vol_pct > self.vol_high:
            return MarketRegime.HIGH_VOL, 0.75

        # Trending
        if adx > self.adx_trend and hurst > self.hurst_trend:
            if above_ma:
                return MarketRegime.TRENDING_UP, 0.80
            return MarketRegime.TRENDING_DOWN, 0.80

        # Default: sideways
        return MarketRegime.SIDEWAYS, 0.65

    @staticmethod
    def _compute_adx_simple(close: np.ndarray, period: int = 14) -> float:
        """Simplified ADX from price direction changes."""
        if len(close) < period + 1:
            return 15.0
        diffs = np.diff(close[-period - 1:])
        up_moves = np.sum(diffs > 0)
        down_moves = np.sum(diffs < 0)
        total = up_moves + down_moves
        if total == 0:
            return 0.0
        return min(float(abs(up_moves - down_moves) / total * 100), 100.0)

    @staticmethod
    def _hurst_rs(prices: np.ndarray) -> float:
        """R/S Hurst estimator (simplified)."""
        n = len(prices)
        if n < 32:
            return 0.5
        returns = np.diff(np.log(prices + 1e-12))
        if np.std(returns) < 1e-12:
            return 0.0
        k = max(n // 4, 16)
        segment = returns[-k:]
        mean = np.mean(segment)
        cumulative = np.cumsum(segment - mean)
        r = float(np.max(cumulative) - np.min(cumulative))
        s = float(np.std(segment))
        if s < 1e-12:
            return 0.5
        rs = r / s
        return float(np.clip(np.log(rs) / np.log(k), 0.0, 1.0))


# ─── Strategy Switching Engine ──────────────────────────────────────────────


class StrategySwitcher:
    """Maps detected regime to eligible strategies with position scaling."""

    REGIME_SCALE: dict[MarketRegime, float] = {
        MarketRegime.TRENDING_UP: 1.0,
        MarketRegime.TRENDING_DOWN: 0.3,
        MarketRegime.SIDEWAYS: 0.7,
        MarketRegime.HIGH_VOL: 0.4,
        MarketRegime.CRISIS: 0.0,
    }

    def __init__(self, strategy_registry: dict[str, "BaseStrategy"] | None = None) -> None:  # noqa: F821
        self._registry = strategy_registry or {}

    def eligible_strategies(self, regime: MarketRegime) -> list[str]:
        """Return strategy names eligible in the given regime."""
        return [
            name for name, s in self._registry.items()
            if s.is_eligible(regime)
        ]

    def position_scale(self, regime: MarketRegime) -> float:
        """Return position scaling factor for the regime."""
        return self.REGIME_SCALE.get(regime, 0.5)

    def allocate(
        self,
        regime: MarketRegime,
        capital: float,
        strategy_signals: dict[str, list[dict]],
    ) -> dict[str, float]:
        """Allocate capital across eligible strategies."""
        eligible = self.eligible_strategies(regime)
        if not eligible:
            return {}

        scale = self.position_scale(regime)
        deployable = capital * scale

        # Equal weight among eligible strategies (simplified)
        weight = 1.0 / len(eligible) if eligible else 0.0
        return {name: deployable * weight for name in eligible}

    def register(self, name: str, strategy: "BaseStrategy") -> None:  # noqa: F821
        """Register a strategy in the switcher."""
        self._registry[name] = strategy
