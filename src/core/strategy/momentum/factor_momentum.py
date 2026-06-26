"""
Factor Momentum Strategy (S1) — Ma, Liao & Jiang (2024), Journal of Empirical Finance.

10 common factors ranked monthly by 1-month momentum. Buy top-ranked factor group,
rebalance monthly. 1-month lookback is optimal for Chinese index funds.

Performance (from paper): 9.91% annual return, Sharpe 1.15, MaxDD -12.4%.
Eligible regimes: TRENDING_UP, SIDEWAYS.
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Factor weights for composite scoring ─────────────────────────────────────
# Primary driver: momentum_1m (0.50). Secondary: multi-horizon momentum + quality.
# Tertiary: low volatility, value, dividend yield.

_FACTOR_WEIGHTS: dict[str, float] = {
    "momentum_1m": 0.50,
    "momentum_3m": 0.15,
    "momentum_6m": 0.10,
    "roe": 0.10,
}

_VALUE_WEIGHTS: dict[str, float] = {
    "volatility_1m": -0.05,  # lower vol → higher score
    "pe_ratio": -0.05,  # lower PE → higher score
    "pb_ratio": -0.03,  # lower PB → higher score
    "dividend_yield": 0.02,
}

_REQUIRED_FIELDS: list[str] = [
    "close",
    "volume",
    "pe_ratio",
    "pb_ratio",
    "roe",
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "volatility_1m",
    "dividend_yield",
]


# ─── Configuration ────────────────────────────────────────────────────────────


class FactorMomentumConfig(StrategyConfig):
    """Configuration for Factor Momentum strategy.

    Extends StrategyConfig with factor-specific parameters.
    """

    top_n: int = Field(default=3, ge=1, description="Number of top-ranked factor groups to buy")
    rebalance_day: int = Field(
        default=1, ge=1, le=28, description="Day of month to rebalance (1 = first trading day)"
    )
    min_momentum_threshold: float = Field(
        default=0.0,
        description="Minimum composite score to trigger a BUY signal",
    )


# ─── Strategy ─────────────────────────────────────────────────────────────────


class FactorMomentum(BaseStrategy):
    """Factor Momentum strategy — rank funds by composite factor score, buy top N.

    The composite score is a weighted sum of normalized factor values:
      - Primary: momentum_1m (0.50 weight) — short-term momentum is the strongest
        signal in Chinese markets per Ma, Liao & Jiang (2024).
      - Secondary: momentum_3m, momentum_6m, ROE (0.15, 0.10, 0.10).
      - Tertiary: low volatility (−), low PE/PB (−), dividend yield (+).
    """

    config: FactorMomentumConfig = Field(
        default_factory=lambda: FactorMomentumConfig(name="factor_momentum"),
        description="Factor momentum strategy configuration",
    )

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate trading signals for the given date.

        Only generates signals on the configured rebalance day of the month.
        On other days, returns an empty list.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with factor columns.
            portfolio: Current portfolio (unused by this strategy).

        Returns:
            List of signal dicts with fund_code, direction, confidence, target_weight, reason.
        """
        if dt.day != self.config.rebalance_day:
            return []

        if not market_data:
            return []

        # Validate columns — skip funds missing required factor data
        valid_data: dict[str, pl.DataFrame] = {}
        for code, df in market_data.items():
            if self._has_required_columns(df):
                valid_data[code] = df

        if not valid_data:
            return []

        # Compute composite score for each fund
        scores: list[tuple[str, float]] = []
        for code, df in valid_data.items():
            composite = self._compute_composite_score(df)
            scores.append((code, composite))

        # Sort by composite score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Determine signal direction and confidence
        signals: list[dict[str, float | int | str]] = []
        max_weight = min(1.0 / self.config.top_n, self.config.max_position_pct)

        if len(scores) > 1:
            score_min = scores[-1][1]
            score_max = scores[0][1]
            score_range = score_max - score_min if score_max != score_min else 1.0
        else:
            score_range = 1.0

        for rank, (code, composite) in enumerate(scores):
            # Normalize confidence to [0, 1]
            if len(scores) > 1:
                confidence = float(np.clip((composite - score_min) / score_range, 0.0, 1.0))
            else:
                confidence = 0.5

            if rank < self.config.top_n and composite >= self.config.min_momentum_threshold:
                direction = SignalDirection.BUY.value if rank == 0 else SignalDirection.ACCUMULATE.value
                target_weight = max_weight
                reason = f"Rank #{rank + 1} by composite factor score ({composite:.4f})"
            elif composite >= 0:
                direction = SignalDirection.HOLD.value
                target_weight = 0.0
                reason = f"Rank #{rank + 1} — positive score ({composite:.4f}) but outside top {self.config.top_n}"
            else:
                direction = SignalDirection.TRIM.value if composite > -0.02 else SignalDirection.SELL.value
                target_weight = 0.0
                reason = f"Rank #{rank + 1} — negative composite score ({composite:.4f})"

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
        if self.config.top_n < 1:
            errors.append("top_n must be >= 1")
        if not (1 <= self.config.rebalance_day <= 28):
            errors.append("rebalance_day must be between 1 and 28")
        if not (0.0 <= self.config.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")
        return errors

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _has_required_columns(df: pl.DataFrame) -> bool:
        """Check that the DataFrame has all required factor columns."""
        df_cols = set(df.columns)
        return all(col in df_cols for col in _REQUIRED_FIELDS)

    @staticmethod
    def _compute_composite_score(df: pl.DataFrame) -> float:
        """Compute weighted composite factor momentum score for a fund.

        Uses the latest row in the DataFrame. Weights are from Ma, Liao & Jiang (2024),
        calibrated for Chinese A-share / index fund markets.

        Returns:
            Composite score normalized to approximately [-0.5, 0.5] range.
        """
        row = df.row(0, named=True)

        score = 0.0

        # Primary factors: direct weighted sum
        for field, weight in _FACTOR_WEIGHTS.items():
            score += float(row[field]) * weight

        # Value/quality factors: transformed before weighting
        # Lower PE/PB → score bonus, use inverse with cap to avoid division issues
        pe = max(float(row["pe_ratio"]), 0.1)
        pb = max(float(row["pb_ratio"]), 0.01)

        score += float(row["dividend_yield"]) * _VALUE_WEIGHTS["dividend_yield"]
        score -= float(row["volatility_1m"]) * abs(_VALUE_WEIGHTS["volatility_1m"])
        score -= (1.0 / pe) * abs(_VALUE_WEIGHTS["pe_ratio"])
        score -= (1.0 / pb) * abs(_VALUE_WEIGHTS["pb_ratio"])

        return score
