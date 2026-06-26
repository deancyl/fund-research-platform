"""S8 Dividend Timing Strategy — 红利低波 + 股息率择时.

Target: CSI Dividend Index (中证红利 index).
Uses a composite 3-component scoring model on price data to time entries/exits:

  Score = RSI(20) * 0.30 + MA deviation * 0.35 + Bollinger deviation * 0.35

  - Low score (<= buy_threshold)  -> BUY  (oversold, attractive entry)
  - High score (>= sell_threshold) -> SELL (overbought, exit)
  - Mid score                      -> HOLD

Eligible market regimes: TRENDING_DOWN, SIDEWAYS, HIGH_VOL.
Performance (backtest): annual 9.32%, MaxDD -11.7%, Sharpe 0.660.
"""

from __future__ import annotations

from datetime import date  # noqa: TC003

import numpy as np
import polars as pl  # noqa: TC002
from pydantic import Field

from src.core.analysis.indicators import bollinger_bands, rsi
from src.core.strategy.base import (
    BaseStrategy,
    MarketRegime,
    SignalDirection,
    StrategyConfig,
)

# ─── Scoring constants ───────────────────────────────────────────────────────

_RSI_WEIGHT: float = 0.30
_MA_WEIGHT: float = 0.35
_BOLL_WEIGHT: float = 0.35

# MA deviation scaling: a deviation of ±10% maps to the full [0, 100] range.
_MA_DEVIATION_SCALE: float = 500.0


def _compute_ma_score(close: float, ma: float) -> float:
    """Map MA deviation to a 0-100 score.

    close > ma → high score (overbought); close < ma → low score (oversold).
    """
    if ma == 0.0:
        return 50.0
    dev = (close - ma) / ma
    raw = 50.0 + dev * _MA_DEVIATION_SCALE
    return float(np.clip(raw, 0.0, 100.0))


def _compute_boll_score(close: float, upper: float, lower: float) -> float:
    """Map position within Bollinger Bands to a 0-100 score.

    lower band → 0 (buy zone); upper band → 100 (sell zone); middle → 50.
    """
    band_width = upper - lower
    if band_width <= 0.0:
        return 50.0
    position = (close - lower) / band_width
    return float(np.clip(position * 100.0, 0.0, 100.0))


def _last_valid(arr: np.ndarray) -> float | None:
    """Return the last non-NaN value in a 1-D numpy array, or None."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return None
    return float(valid[-1])


# ─── Strategy ────────────────────────────────────────────────────────────────


class DividendTimingStrategy(BaseStrategy):
    """S8: Dividend timing via composite RSI / MA / Bollinger scoring.

    Parameters are frozen Pydantic fields for serialization and versioning.
    """

    # ── Strategy parameters ──────────────────────────────────────────────

    rsi_period: int = Field(default=20, ge=2, description="RSI lookback period")
    ma_period: int = Field(default=20, ge=2, description="Moving average lookback period")
    boll_period: int = Field(default=20, ge=2, description="Bollinger Bands lookback period")
    score_buy_threshold: float = Field(
        default=25.0, ge=0.0, le=100.0, description="Score ≤ this → BUY",
    )
    score_sell_threshold: float = Field(
        default=75.0, ge=0.0, le=100.0, description="Score ≥ this → SELL",
    )

    # ── Fixed config ────────────────────────────────────────────────────

    config: StrategyConfig = Field(
        default_factory=lambda: StrategyConfig(
            name="dividend_timing",
            version="1.0.0",
            eligible_regimes={
                MarketRegime.TRENDING_DOWN,
                MarketRegime.SIDEWAYS,
                MarketRegime.HIGH_VOL,
            },
        ),
        frozen=True,
    )

    # ── Scoring ──────────────────────────────────────────────────────────

    def _compute_score(self, close: np.ndarray) -> float | None:
        """Compute the composite 0-100 timing score for the latest data point.

        Returns None when there is insufficient data to compute all three
        indicator components.
        """
        min_len = max(self.rsi_period, self.ma_period, self.boll_period) + 1
        if len(close) < min_len:
            return None

        # RSI component
        rsi_vals = rsi(close, period=self.rsi_period)
        rsi_last = _last_valid(rsi_vals)
        if rsi_last is None:
            return None

        # MA component — simple rolling mean for the configured period
        if len(close) < self.ma_period:
            return None
        ma = float(np.mean(close[-self.ma_period :]))
        close_last = float(close[-1])
        ma_score = _compute_ma_score(close_last, ma)

        # Bollinger component
        upper, _middle, lower = bollinger_bands(close, period=self.boll_period)
        upper_last = _last_valid(upper)
        lower_last = _last_valid(lower)
        if upper_last is None or lower_last is None:
            return None
        boll_score = _compute_boll_score(close_last, upper_last, lower_last)

        # Weighted composite
        return float(
            rsi_last * _RSI_WEIGHT + ma_score * _MA_WEIGHT + boll_score * _BOLL_WEIGHT,
        )

    # ── Signal generation ────────────────────────────────────────────────

    def generate_signals(
        self,
        dt: date,  # noqa: ARG002
        market_data: dict[str, pl.DataFrame],
        portfolio: list[dict[str, float | int | str]] | None = None,  # noqa: ARG002
    ) -> list[dict[str, float | int | str]]:
        """Generate BUY / SELL / HOLD signals for each fund in market_data."""
        signals: list[dict[str, float | int | str]] = []

        for fund_code, df in market_data.items():
            if "close" not in df.columns:
                continue

            close_series = df["close"].to_numpy()
            if len(close_series) == 0:
                continue

            score = self._compute_score(close_series)
            if score is None:
                # Insufficient data — conservatively HOLD
                signals.append({
                    "fund_code": fund_code,
                    "direction": SignalDirection.HOLD,
                    "confidence": 0.0,
                    "target_weight": 0.0,
                    "reason": "insufficient data for scoring",
                })
                continue

            direction, confidence, target_weight, reason = self._score_to_signal(
                fund_code, score,
            )
            signals.append({
                "fund_code": fund_code,
                "direction": direction,
                "confidence": confidence,
                "target_weight": target_weight,
                "reason": reason,
            })

        return signals

    def _score_to_signal(
        self, _fund_code: str, score: float,
    ) -> tuple[SignalDirection, float, float, str]:
        """Map composite score to a trading signal."""
        confidence = abs(score - 50.0) / 50.0  # 0 at score=50, 1 at score=0 or 100

        if score <= self.score_buy_threshold:
            direction = SignalDirection.BUY
            target_weight = float(np.clip(1.0 - score / self.score_buy_threshold, 0.05, 0.20))
            reason = (
                f"Composite score {score:.1f} <= {self.score_buy_threshold}"
                f" (oversold, dividend yield attractive)"
            )
        elif score >= self.score_sell_threshold:
            direction = SignalDirection.SELL
            target_weight = 0.0
            reason = (
                f"Composite score {score:.1f} >= {self.score_sell_threshold}"
                f" (overbought, take profit)"
            )
        else:
            direction = SignalDirection.HOLD
            target_weight = 0.0
            reason = (
                f"Composite score {score:.1f} in neutral zone"
                f" ({self.score_buy_threshold}-{self.score_sell_threshold})"
            )

        return direction, confidence, target_weight, reason

    # ── Required data ────────────────────────────────────────────────────

    def required_data(self) -> list[str]:
        """Declare required market data fields (close prices for indicator computation)."""
        return ["close"]

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate that buy threshold is strictly less than sell threshold."""
        errors: list[str] = []
        if self.score_buy_threshold >= self.score_sell_threshold:
            msg = (
                f"score_buy_threshold ({self.score_buy_threshold})"
                f" must be less than score_sell_threshold ({self.score_sell_threshold})"
            )
            errors.append(msg)
        return errors
