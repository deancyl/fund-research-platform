"""
52-Week High Momentum Strategy (S5) — Buy ETFs trading near their 52-week high.

Strategy: Select ETFs whose current price is within a configurable proximity
percentage of their 52-week trailing high. The strategy exploits the momentum
anomaly where assets near highs tend to continue trending upward.

Key parameters:
  - lookback_weeks: trailing window in weeks for computing the high (default: 52)
  - proximity_pct: price must be ≥ this fraction of the 52w high to qualify (default: 0.95)

Eligible regimes: TRENDING_UP only (needs trending market for momentum to persist).

Required data: ["close"]
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_TRADING_DAYS_PER_WEEK: int = 5
_REQUIRED_FIELDS: list[str] = ["close"]
_MIN_REQUIRED_ROWS: int = 52 * _TRADING_DAYS_PER_WEEK  # at minimum need 52 weeks


# ─── Configuration ────────────────────────────────────────────────────────────


class FiftyTwoWeekHighConfig(StrategyConfig):
    """Configuration for 52-Week High Momentum strategy (S5).

    Extends StrategyConfig with lookback window and proximity threshold
    for identifying ETFs near their trailing high.
    """

    lookback_weeks: int = Field(
        default=52,
        ge=1,
        description="Trailing lookback window in weeks for computing 52-week high",
    )
    proximity_pct: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum price proximity to 52-week high to qualify (e.g., 0.95 = within 5%)",
    )
    top_n: int = Field(
        default=3,
        ge=1,
        description="Maximum number of ETFs to select for BUY signals",
    )
    max_position_pct: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Maximum position as fraction of portfolio",
    )

    @model_validator(mode="after")
    def _check_proximity_range(self) -> "FiftyTwoWeekHighConfig":
        """Ensure proximity_pct is in (0, 1]."""
        if not (0.0 < self.proximity_pct <= 1.0):
            raise ValueError(f"proximity_pct must be in (0, 1], got {self.proximity_pct}")
        return self


# ─── Strategy ─────────────────────────────────────────────────────────────────


class FiftyTwoWeekHigh(BaseStrategy):
    """52-Week High Momentum strategy — buy ETFs near their 52-week high.

    Algorithm per generate_signals():
      1. Filter ETFs with sufficient price history (≥ lookback_weeks * 5 days).
      2. For each ETF, compute the 52-week high from trailing close prices.
      3. Compute proximity = current_price / 52_week_high.
      4. Rank ETFs by proximity descending.
      5. Emit BUY for top_n ETFs whose proximity ≥ proximity_pct.
      6. Remaining qualifying ETFs get HOLD (not far from high).
      7. Non-qualifying ETFs get TRIM.
    """

    config: FiftyTwoWeekHighConfig = Field(
        default_factory=lambda: FiftyTwoWeekHighConfig(name="fiftytwo_week_high"),
        description="52-week high momentum strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate trading signals for the given date.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with close prices.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts sorted by proximity to 52-week high (descending).
        """
        if not market_data:
            return []

        lookback_days = self.config.lookback_weeks * _TRADING_DAYS_PER_WEEK

        # 1. Compute proximity to 52-week high for each valid ETF
        proximities: list[tuple[str, float]] = []
        for code, df in market_data.items():
            if not self._has_close_column(df) or len(df) < lookback_days:
                continue
            prox = self._compute_high_proximity(df, lookback_days)
            if prox >= 0.0:
                proximities.append((code, prox))

        if not proximities:
            return []

        # 2. Sort by proximity descending (closer to high = higher rank)
        proximities.sort(key=lambda x: x[1], reverse=True)

        # 3. Emit signals
        signals: list[dict[str, float | int | str]] = []
        threshold = self.config.proximity_pct
        top_n = self.config.top_n
        base_weight = 1.0 / top_n

        for rank, (code, proximity) in enumerate(proximities):
            # Confidence: normalized proximity
            if len(proximities) > 1:
                prox_min = proximities[-1][1]
                prox_max = proximities[0][1]
                prox_range = prox_max - prox_min if prox_max != prox_min else 1.0
                confidence = float(np.clip((proximity - prox_min) / prox_range, 0.0, 1.0))
            else:
                confidence = 0.7

            if rank < top_n and proximity >= threshold:
                direction = SignalDirection.BUY.value
                target_weight = base_weight
                reason = (
                    f"Rank #{rank + 1}/{len(proximities)} — "
                    f"proximity to 52w high={proximity:.4f}, "
                    f"threshold={threshold:.2f}"
                )
            elif proximity >= threshold:
                direction = SignalDirection.HOLD.value
                target_weight = 0.0
                reason = (
                    f"Rank #{rank + 1}/{len(proximities)} — "
                    f"near 52w high ({proximity:.4f}) but outside top {top_n}"
                )
            else:
                direction = SignalDirection.TRIM.value
                target_weight = 0.0
                reason = (
                    f"Rank #{rank + 1}/{len(proximities)} — "
                    f"proximity to 52w high={proximity:.4f} below threshold={threshold:.2f}"
                )

            signals.append({
                "fund_code": code,
                "direction": direction,
                "confidence": confidence,
                "target_weight": target_weight,
                "reason": reason,
            })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields for the data layer to prefetch."""
        return _REQUIRED_FIELDS

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors: list[str] = []
        cfg = self.config
        if cfg.lookback_weeks < 1:
            errors.append("lookback_weeks must be >= 1")
        if not (0.0 < cfg.proximity_pct <= 1.0):
            errors.append(
                f"proximity_pct must be in (0, 1], got {cfg.proximity_pct}"
            )
        if cfg.top_n < 1:
            errors.append("top_n must be >= 1")
        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")
        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_close_column(df: pl.DataFrame) -> bool:
        """Check that DataFrame has the close column."""
        return "close" in df.columns

    @staticmethod
    def _compute_high_proximity(df: pl.DataFrame, lookback_days: int) -> float:
        """Compute current price as fraction of trailing high.

        Args:
            df: DataFrame with close column of length ≥ lookback_days.
            lookback_days: Number of trading days to look back.

        Returns:
            Proximity ratio: current_price / trailing_high in [0, ∞).
            Returns 0.0 if the high price is zero or negative.
        """
        close = df["close"].to_numpy()
        n = len(close)
        trailing_window = close[n - lookback_days:]
        trailing_high = float(np.nanmax(trailing_window))
        current_price = float(close[-1])

        if trailing_high <= 0.0 or np.isnan(trailing_high):
            return 0.0

        return current_price / trailing_high
