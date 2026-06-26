"""S7 RSI Mean Reversion + Fixed Income Strategy — RSI均值回归 + 固收.

Logic:
  - RSI(14) < 35 → oversold → BUY equity ETF (buy the dip).
  - RSI(14) > 70 → overbought → SELL equity ETF (take profit).
  - 35 <= RSI(14) <= 70 → neutral → rotate into bond ETF (defensive fixed income).

Performance (backtest): annual 6.00%, Vol 2.88%, MaxDD -2.64%, Win rate 70%+.
Eligible regimes: ALL.
"""

from __future__ import annotations

from datetime import date  # noqa: TC003

import numpy as np
import polars as pl  # noqa: TC002
from pydantic import Field

from src.core.analysis.indicators import rsi
from src.core.strategy.base import (
    BaseStrategy,
    MarketRegime,
    SignalDirection,
    StrategyConfig,
)


def _last_valid(arr: np.ndarray) -> float | None:
    """Return the last non-NaN value in a 1-D numpy array, or None."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return None
    return float(valid[-1])


class RsiFixedIncomeStrategy(BaseStrategy):
    """S7: RSI mean reversion with fixed income rotation.

    When RSI drops below the oversold threshold, the strategy buys the equity
    ETF (oversold → mean reversion expected). When RSI rises above the
    overbought threshold, it sells. In the neutral zone it rotates into a bond
    ETF for capital preservation.

    Parameters are frozen Pydantic fields for serialization and versioning.
    """

    # ── Strategy parameters ──────────────────────────────────────────────

    rsi_period: int = Field(default=14, ge=2, description="RSI lookback period")
    rsi_oversold: float = Field(
        default=35.0, ge=0.0, le=100.0, description="RSI below this → oversold → BUY",
    )
    rsi_overbought: float = Field(
        default=70.0, ge=0.0, le=100.0, description="RSI above this → overbought → SELL",
    )
    bond_fund_code: str = Field(
        default="511260", min_length=1, description="Bond ETF code for fixed income rotation",
    )

    # ── Fixed config ────────────────────────────────────────────────────

    config: StrategyConfig = Field(
        default_factory=lambda: StrategyConfig(
            name="rsi_fixed_income",
            version="1.0.0",
            eligible_regimes={r for r in MarketRegime},
        ),
        frozen=True,
    )

    # ── Signal generation ────────────────────────────────────────────────

    def generate_signals(
        self,
        dt: date,  # noqa: ARG002
        market_data: dict[str, pl.DataFrame],
        portfolio: list[dict[str, float | int | str]] | None = None,  # noqa: ARG002
    ) -> list[dict[str, float | int | str]]:
        """Generate BUY / SELL / HOLD signals for each fund in market_data.

        For each equity fund, compute RSI(14) and:
          - RSI < rsi_oversold  → BUY equity ETF (oversold entry).
          - RSI > rsi_overbought → SELL equity ETF (overbought exit).
          - Otherwise            → BUY bond ETF (rotate to fixed income).
        """
        signals: list[dict[str, float | int | str]] = []

        for fund_code, df in market_data.items():
            if "close" not in df.columns:
                continue

            close_series = df["close"].to_numpy()
            if len(close_series) == 0:
                continue

            rsi_vals = rsi(close_series, period=self.rsi_period)
            rsi_last = _last_valid(rsi_vals)

            if rsi_last is None:
                # Insufficient data — conservatively HOLD
                signals.append({
                    "fund_code": fund_code,
                    "direction": SignalDirection.HOLD,
                    "confidence": 0.0,
                    "target_weight": 0.0,
                    "reason": "insufficient data for RSI computation",
                })
                continue

            signal = self._rsi_to_signal(fund_code, float(rsi_last))
            signals.append(signal)

        return signals

    def _rsi_to_signal(
        self, fund_code: str, rsi_value: float,
    ) -> dict[str, float | int | str]:
        """Map RSI value to a trading signal dict.

        - Oversold (RSI < threshold) → BUY equity ETF.
        - Overbought (RSI > threshold) → SELL equity ETF.
        - Neutral zone → BUY bond ETF (rotate to fixed income).
        """
        if rsi_value < self.rsi_oversold:
            # Oversold: buy the dip on equity.
            oversold_depth = (self.rsi_oversold - rsi_value) / self.rsi_oversold
            confidence = float(np.clip(oversold_depth, 0.0, 1.0))
            target_weight = float(
                np.clip(oversold_depth * self.config.max_position_pct, 0.05, self.config.max_position_pct),
            )
            return {
                "fund_code": fund_code,
                "direction": SignalDirection.BUY,
                "confidence": confidence,
                "target_weight": target_weight,
                "reason": (
                    f"RSI {rsi_value:.1f} < {self.rsi_oversold}"
                    f" (oversold → buy equity {fund_code})"
                ),
            }

        if rsi_value > self.rsi_overbought:
            # Overbought: take profit, sell equity.
            overbought_excess = (rsi_value - self.rsi_overbought) / (100.0 - self.rsi_overbought)
            confidence = float(np.clip(overbought_excess, 0.0, 1.0))
            return {
                "fund_code": fund_code,
                "direction": SignalDirection.SELL,
                "confidence": confidence,
                "target_weight": 0.0,
                "reason": (
                    f"RSI {rsi_value:.1f} > {self.rsi_overbought}"
                    f" (overbought → sell equity {fund_code})"
                ),
            }

        # Neutral zone (35 <= RSI <= 70): rotate to bond ETF.
        distance_from_center = abs(rsi_value - 50.0)
        max_distance = max(self.rsi_oversold, 100.0 - self.rsi_overbought, 1e-9)
        confidence = float(np.clip(1.0 - distance_from_center / max_distance, 0.0, 1.0))
        target_weight = float(
            np.clip(confidence * self.config.max_position_pct, 0.05, self.config.max_position_pct),
        )
        return {
            "fund_code": self.bond_fund_code,
            "direction": SignalDirection.BUY,
            "confidence": confidence,
            "target_weight": target_weight,
            "reason": (
                f"RSI {rsi_value:.1f} in [{self.rsi_oversold}, {self.rsi_overbought}]"
                f" (neutral → rotate to bond {self.bond_fund_code})"
            ),
        }

    # ── Required data ────────────────────────────────────────────────────

    def required_data(self) -> list[str]:
        """Declare required market data fields (close prices for RSI)."""
        return ["close"]

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate that oversold threshold < overbought threshold."""
        errors: list[str] = []
        if self.rsi_oversold >= self.rsi_overbought:
            msg = (
                f"rsi_oversold ({self.rsi_oversold}) must be less than"
                f" rsi_overbought ({self.rsi_overbought})"
            )
            errors.append(msg)
        return errors
