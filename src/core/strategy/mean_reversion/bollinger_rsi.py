"""Bollinger + RSI Composite Strategy (S9) — 布林带 + RSI 复合信号.

Target: range-bound (SIDEWAYS) markets.
Win rate: 68-70% in sideways conditions.

Algorithm:
  1. Compute Bollinger Bands(boll_period, boll_std) on close prices.
  2. Compute RSI(rsi_period) on close prices.
  3. BUY when RSI < rsi_oversold AND price < lower band (double-confirmation).
  4. SELL when RSI > rsi_overbought AND price > upper band.
  5. HOLD otherwise — single confirmation is not enough.

Parameters:
  - boll_period (default=20): Bollinger Band SMA period
  - boll_std (default=2.0): Standard deviation multiplier
  - rsi_period (default=14): RSI lookback period
  - rsi_oversold (default=30): RSI oversold threshold
  - rsi_overbought (default=70): RSI overbought threshold
"""

from __future__ import annotations

from datetime import date  # noqa: TC003 — runtime parameter type

import numpy as np
import polars as pl  # noqa: TC002 — runtime DataFrame access

from src.core.analysis.indicators import bollinger_bands, rsi
from src.core.strategy.base import (
    BaseStrategy,
    MarketRegime,
    SignalDirection,
    StrategyConfig,
)

# ─── Constants ───────────────────────────────────────────────────────────────

_MIN_LOOKBACK_FACTOR: float = 1.5
_MIN_CONFIDENCE: float = 0.10
_MAX_CONFIDENCE: float = 0.95


class BollingerRsiStrategy(BaseStrategy):
    """Bollinger Band + RSI composite mean-reversion strategy.

    Generates signals only when BOTH Bollinger Band position and RSI
    agree on overbought/oversold — an AND gate that filters out false
    positives. Designed for range-bound (SIDEWAYS) market regimes.

    Attributes:
        boll_period: SMA window for Bollinger Bands.
        boll_std: Standard deviation multiplier for band width.
        rsi_period: RSI lookback period.
        rsi_oversold: RSI level below which the asset is oversold.
        rsi_overbought: RSI level above which the asset is overbought.
    """

    boll_period: int = 20
    boll_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    config: StrategyConfig = StrategyConfig(
        name="bollinger_rsi",
        version="1.0.0",
        eligible_regimes={MarketRegime.SIDEWAYS},
        max_position_pct=0.20,
        min_holding_days=1,
    )

    # ─── Abstract Interface ───────────────────────────────────────────────

    def generate_signals(
        self,
        dt: date,  # noqa: ARG002 — reserved for future signal stamping
        market_data: dict[str, pl.DataFrame],
        portfolio: list | None = None,  # noqa: ARG002
    ) -> list[dict[str, float | int | str]]:
        """Generate trading signals based on Bollinger+RSI composite.

        For each fund:
          1. Extract close prices.
          2. Compute Bollinger Bands (upper, middle, lower).
          3. Compute RSI.
          4. Check composite conditions:
             - RSI < oversold AND close < lower_band → BUY
             - RSI > overbought AND close > upper_band → SELL
             - Otherwise → HOLD
        """
        if not market_data:
            return []

        signals: list[dict[str, float | int | str]] = []
        min_period = max(self.boll_period, self.rsi_period)
        min_data_points = max(int(min_period * _MIN_LOOKBACK_FACTOR), min_period + 1)

        for fund_code, df in market_data.items():
            if "close" not in df.columns:
                continue

            close = df["close"].to_numpy()
            if len(close) < min_data_points:
                continue

            # Check for all-NaN input
            valid_mask = ~np.isnan(close)
            if np.sum(valid_mask) < min_data_points:
                continue

            # Compute indicators
            upper, middle, lower = bollinger_bands(
                close, period=self.boll_period, std_mult=self.boll_std,
            )
            rsi_values = rsi(close, period=self.rsi_period)

            # Use the last valid value (both Bollinger and RSI may have NaN warm-up)
            last_idx = -1
            if np.isnan(rsi_values[last_idx]) or np.isnan(lower[last_idx]):
                # Walk back to find the most recent valid indicator pair
                found_valid = False
                for idx in range(len(rsi_values) - 1, -1, -1):
                    if not np.isnan(rsi_values[idx]) and not np.isnan(lower[idx]):
                        last_idx = idx
                        found_valid = True
                        break
                if not found_valid:
                    continue

            last_close = float(close[last_idx])
            last_rsi = float(rsi_values[last_idx])
            last_upper = float(upper[last_idx])
            last_lower = float(lower[last_idx])
            last_middle = float(middle[last_idx])

            direction: SignalDirection
            confidence: float
            target_weight: float
            reason: str

            if last_rsi < self.rsi_oversold and last_close < last_lower:
                # Double-confirmed oversold BUY
                direction = SignalDirection.BUY
                # Confidence: how far below lower band + how low RSI is
                band_signal = (last_lower - last_close) / (last_lower + 1e-12)
                rsi_signal = (self.rsi_oversold - last_rsi) / max(self.rsi_oversold, 1e-12)
                confidence = min(0.5 + 0.25 * band_signal + 0.25 * rsi_signal, _MAX_CONFIDENCE)
                confidence = max(confidence, _MIN_CONFIDENCE)
                target_weight = self.config.max_position_pct * confidence
                reason = (
                    f"RSI={last_rsi:.1f}<{self.rsi_oversold} AND "
                    f"close={last_close:.3f}<lower={last_lower:.3f} "
                    f"(middle={last_middle:.3f}) → oversold BUY"
                )

            elif last_rsi > self.rsi_overbought and last_close > last_upper:
                # Double-confirmed overbought SELL
                direction = SignalDirection.SELL
                band_signal = (last_close - last_upper) / (last_upper + 1e-12)
                rsi_signal = (last_rsi - self.rsi_overbought) / (
                    (100.0 - self.rsi_overbought) + 1e-12
                )
                confidence = min(0.5 + 0.25 * band_signal + 0.25 * rsi_signal, _MAX_CONFIDENCE)
                confidence = max(confidence, _MIN_CONFIDENCE)
                target_weight = 0.0
                reason = (
                    f"RSI={last_rsi:.1f}>{self.rsi_overbought} AND "
                    f"close={last_close:.3f}>upper={last_upper:.3f} "
                    f"(middle={last_middle:.3f}) → overbought SELL"
                )

            else:
                # No double-confirmation → HOLD
                direction = SignalDirection.HOLD
                confidence = 0.0
                target_weight = 0.0
                conditions: list[str] = []
                if last_rsi < self.rsi_oversold:
                    conditions.append(f"RSI oversold ({last_rsi:.1f}) but inside band")
                elif last_rsi > self.rsi_overbought:
                    conditions.append(f"RSI overbought ({last_rsi:.1f}) but inside band")
                elif last_close < last_lower:
                    conditions.append("below lower band but RSI not oversold")
                elif last_close > last_upper:
                    conditions.append("above upper band but RSI not overbought")
                else:
                    conditions.append("no extreme signal")
                reason = (
                    f"RSI={last_rsi:.1f} close={last_close:.3f} "
                    f"bands=({last_lower:.3f},{last_middle:.3f},{last_upper:.3f}) → "
                    + " & ".join(conditions)
                )

            signals.append({
                "fund_code": fund_code,
                "direction": direction.value,
                "confidence": round(confidence, 4),
                "target_weight": round(target_weight, 4),
                "reason": reason,
            })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields."""
        return ["close"]

    def validate(self) -> list[str]:
        """Validate strategy parameters."""
        errors: list[str] = []
        if self.boll_period < 2:
            errors.append(f"boll_period must be >= 2, got {self.boll_period}")
        if self.boll_std <= 0.0:
            errors.append(f"boll_std must be > 0, got {self.boll_std}")
        if self.rsi_period < 2:
            errors.append(f"rsi_period must be >= 2, got {self.rsi_period}")
        if self.rsi_oversold >= self.rsi_overbought:
            errors.append(
                f"rsi_oversold ({self.rsi_oversold}) must be < rsi_overbought ({self.rsi_overbought})"
            )
        return errors
