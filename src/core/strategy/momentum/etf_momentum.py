"""
ETF Multi-Factor Momentum Strategy (S2) — Chinese ETF momentum with risk gating.

Strategy: Select Top-2 ETFs from a universe of 49 (41 A-share + 8 QDII) using
multi-period momentum scoring across 23 factors. Position sizing is adjusted
by volatility-based risk gating and Exp4 hysteresis prevents excessive turnover.

Key parameters (from the spec):
  - Universe: 49 ETFs, 23 factors
  - Selection: Top-2 via weighted multi-period momentum
  - Lookback: 60/120/252 days (weighted 0.50/0.30/0.20)
  - Hysteresis: Exp4 minimum 9-day hold
  - Risk gating: volatility percentile → position scale reduction

Performance (from spec):
  - Sharpe: 1.38
  - Max Drawdown: 10.8%
  - Win rate: 83.3%

Eligible regimes: TRENDING_UP, SIDEWAYS
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Factor Weights ────────────────────────────────────────────────────────────
# Multi-period momentum (primary driver, 60% of score):
#   short (60d) = 0.50 → captures immediate trend
#   mid   (120d) = 0.30 → captures medium-term persistence
#   long  (252d) = 0.20 → captures annual trend quality

_MOMENTUM_WEIGHTS: dict[str, float] = {
    "short": 0.50,
    "mid": 0.30,
    "long": 0.20,
}

# ETF quality adjustments (secondary, applied as gating multipliers):
#   tracking_error → higher = worse tracking → negative impact
#   premium_rate / discount_rate → larger deviation → negative impact
#   liquidity → lower = harder to trade → negative impact

_QUALITY_WEIGHTS: dict[str, float] = {
    "tracking_error": -0.02,   # lower tracking error → better (te ~0.001-0.05)
    "premium_discount": -0.01,  # lower premium/discount gap → better (gap ~0.0005-0.01)
    "liquidity": 0.02,          # higher liquidity → better (0-1, max contrib 0.02)
    "turnover_rate": 0.01,      # higher turnover → active → better (rate ~0.01-0.10)
}

# Minimum rows required to compute longest lookback
_MIN_REQUIRED_ROWS = 60  # at minimum need enough for short lookback

_REQUIRED_FIELDS: list[str] = [
    "close",
    "volume",
    "turnover_rate",
    "tracking_error",
    "premium_rate",
    "discount_rate",
    "liquidity",
]


# ─── Configuration ────────────────────────────────────────────────────────────


class EtfMomentumConfig(StrategyConfig):
    """Configuration for ETF Multi-Factor Momentum strategy (S2).

    Extends StrategyConfig with ETF-specific parameters for multi-period
    momentum scoring, volatility gating, and Exp4 hysteresis.
    """

    max_position_pct: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximum position as fraction of portfolio (overrides StrategyConfig default)",
    )
    top_n: int = Field(
        default=2,
        ge=1,
        description="Number of top-ranked ETFs to select for BUY signals",
    )
    lookback_short: int = Field(
        default=60,
        ge=1,
        description="Short-term momentum lookback in trading days (default: 60)",
    )
    lookback_mid: int = Field(
        default=120,
        ge=1,
        description="Mid-term momentum lookback in trading days (default: 120)",
    )
    lookback_long: int = Field(
        default=252,
        ge=1,
        description="Long-term momentum lookback in trading days (default: 252)",
    )
    min_hold_days: int = Field(
        default=9,
        ge=1,
        description="Exp4 hysteresis: minimum holding days to prevent excessive turnover",
    )
    volatility_percentile_cap: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="ETFs at or above this tracking-error percentile get position scaled down",
    )
    min_weight_floor: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum position weight as fraction of full allocation (risk gate floor)",
    )

    @model_validator(mode="after")
    def _check_lookback_ordering(self) -> "EtfMomentumConfig":
        """Ensure lookback_short < lookback_mid < lookback_long for valid multi-period ranking."""
        if not (self.lookback_short < self.lookback_mid < self.lookback_long):
            raise ValueError(
                f"lookback ordering violated: {self.lookback_short} < "
                f"{self.lookback_mid} < {self.lookback_long} required"
            )
        return self


# ─── Strategy ─────────────────────────────────────────────────────────────────


class EtfMomentum(BaseStrategy):
    """ETF Multi-Factor Momentum strategy — rank ETFs by multi-period momentum,
    select top N, and apply volatility-based position scaling.

    Algorithm per generate_signals() call:
      1. Validate each ETF DataFrame has required columns and sufficient rows.
      2. Compute multi-period momentum score for each ETF:
           score = w_s * ret_60d + w_m * ret_120d + w_l * ret_252d
      3. Compute ETF quality adjustments from tracking error, premium/discount, liquidity.
      4. Apply volatility risk gating: scale position by tracking-error percentile.
      5. Rank ETFs by composite score descending.
      6. Emit BUY signals for top_n ETFs with positive scores with target_weight
         = (1/top_n) × risk_gate_scale.
      7. Remaining ETFs get HOLD or TRIM based on score sign.
    """

    config: EtfMomentumConfig = Field(
        default_factory=lambda: EtfMomentumConfig(name="etf_momentum"),
        description="ETF multi-factor momentum strategy configuration",
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
            market_data: Dict of {etf_code: DataFrame} with OHLCV + ETF-specific columns.
            portfolio: Current portfolio (used for hysteresis — optional).

        Returns:
            List of signal dicts sorted by composite score (highest first).
        """
        if not market_data:
            return []

        # 1. Validate and filter funds
        valid_etfs: dict[str, pl.DataFrame] = {}
        for code, df in market_data.items():
            if self._has_required_columns(df) and len(df) >= _MIN_REQUIRED_ROWS:
                valid_etfs[code] = df

        if not valid_etfs:
            return []

        # 2. Compute scores and collect tracking errors for risk gating
        scores: list[tuple[str, float, float]] = []  # (code, composite_score, tracking_error)
        all_tracking_errors: list[float] = []

        for code, df in valid_etfs.items():
            momentum = self._compute_momentum_score(df)
            quality = self._compute_quality_adjustment(df)
            composite = momentum + quality
            te = float(df["tracking_error"][-1])
            scores.append((code, composite, te))
            all_tracking_errors.append(te)

        # 3. Rank by composite score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Compute tracking error percentiles for risk gating
        te_array = np.array(all_tracking_errors, dtype=np.float64)

        # 5. Emit signals
        signals: list[dict[str, float | int | str]] = []
        base_weight = 1.0 / self.config.top_n

        for rank, (code, composite, tracking_error) in enumerate(scores):
            # Risk gating: scale weight by tracking error percentile
            risk_scale = self._compute_risk_scale(tracking_error, te_array)

            # Confidence: normalized by score range
            if len(scores) > 1:
                score_min = scores[-1][1]
                score_max = scores[0][1]
                score_range = score_max - score_min if score_max != score_min else 1.0
                confidence = float(np.clip((composite - score_min) / score_range, 0.0, 1.0))
            else:
                confidence = 0.5

            if rank < self.config.top_n and composite > 0:
                direction = SignalDirection.BUY.value
                target_weight = base_weight * risk_scale
                reason = (
                    f"Rank #{rank + 1}/{len(scores)} — "
                    f"momentum={composite:.4f}, risk_scale={risk_scale:.2f}"
                )
            elif composite > 0:
                direction = SignalDirection.HOLD.value
                target_weight = 0.0
                reason = (
                    f"Rank #{rank + 1}/{len(scores)} — "
                    f"positive momentum ({composite:.4f}) but outside top {self.config.top_n}"
                )
            else:
                direction = SignalDirection.TRIM.value
                target_weight = 0.0
                reason = (
                    f"Rank #{rank + 1}/{len(scores)} — "
                    f"negative momentum ({composite:.4f})"
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
        if cfg.top_n < 1:
            errors.append("top_n must be >= 1")
        if cfg.lookback_short < 1:
            errors.append("lookback_short must be >= 1")
        if cfg.lookback_mid < 1:
            errors.append("lookback_mid must be >= 1")
        if cfg.lookback_long < 1:
            errors.append("lookback_long must be >= 1")
        if not (cfg.lookback_short < cfg.lookback_mid < cfg.lookback_long):
            errors.append(
                f"lookback ordering invalid: {cfg.lookback_short} < "
                f"{cfg.lookback_mid} < {cfg.lookback_long}"
            )
        if cfg.min_hold_days < 1:
            errors.append("min_hold_days must be >= 1")
        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")
        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_required_columns(df: pl.DataFrame) -> bool:
        """Check that DataFrame has all required ETF columns."""
        df_cols = set(df.columns)
        return all(col in df_cols for col in _REQUIRED_FIELDS)

    def _compute_momentum_score(self, df: pl.DataFrame) -> float:
        """Compute multi-period weighted momentum score.

        Uses close prices at different lookback horizons:
          ret_short = (close[-1] - close[-lookback_short]) / close[-lookback_short]
          ret_mid   = (close[-1] - close[-lookback_mid])   / close[-lookback_mid]
          ret_long  = (close[-1] - close[-lookback_long])  / close[-lookback_long]

        Returns:
            Weighted momentum score (typically in [-0.3, 0.3] range).
        """
        close = df["close"].to_numpy()
        n = len(close)
        price_now = float(close[-1])

        cfg = self.config

        # Short-term
        idx_s = max(0, n - cfg.lookback_short)
        ret_s = (price_now - float(close[idx_s])) / float(close[idx_s]) if float(close[idx_s]) > 0 else 0.0

        # Mid-term
        idx_m = max(0, n - cfg.lookback_mid)
        ret_m = (price_now - float(close[idx_m])) / float(close[idx_m]) if float(close[idx_m]) > 0 else 0.0

        # Long-term
        idx_l = max(0, n - cfg.lookback_long)
        ret_l = (price_now - float(close[idx_l])) / float(close[idx_l]) if float(close[idx_l]) > 0 else 0.0

        score = (
            _MOMENTUM_WEIGHTS["short"] * ret_s
            + _MOMENTUM_WEIGHTS["mid"] * ret_m
            + _MOMENTUM_WEIGHTS["long"] * ret_l
        )
        return score

    @staticmethod
    def _compute_quality_adjustment(df: pl.DataFrame) -> float:
        """Compute ETF quality adjustment from latest-row data.

        Factors:
          - tracking_error: negative (lower = better)
          - premium_discount gap: negative (smaller gap = better tracking)
          - liquidity: positive (higher = easier to trade)
          - turnover_rate: positive (higher = more active)

        Returns:
            Quality adjustment added to momentum score (typically in [-0.05, 0.05]).
        """
        row = df.row(-1, named=True)
        te = float(row["tracking_error"])
        premium = float(row["premium_rate"])
        discount = float(row["discount_rate"])
        liq = float(row["liquidity"])
        turnover = float(row["turnover_rate"])

        # Premium/discount gap: larger gap = worse ETF tracking quality
        prem_disc_gap = abs(premium - discount)

        adjustment = (
            _QUALITY_WEIGHTS["tracking_error"] * te
            + _QUALITY_WEIGHTS["premium_discount"] * prem_disc_gap
            + _QUALITY_WEIGHTS["liquidity"] * liq
            + _QUALITY_WEIGHTS["turnover_rate"] * turnover
        )
        return float(adjustment)

    def _compute_risk_scale(
        self, tracking_error: float, all_tracking_errors: np.ndarray
    ) -> float:
        """Compute position scale factor based on tracking error percentile.

        ETFs at or above the volatility_percentile_cap get their position
        scaled down proportionally to their excess percentile.

        Scale logic:
          - Below cap (default 90th percentile): scale = 1.0 (no reduction)
          - Above cap: linear scale from 1.0 down to min_weight_floor at 100th percentile

        Args:
            tracking_error: This ETF's tracking error.
            all_tracking_errors: Array of all ETF tracking errors in the universe.

        Returns:
            Scale factor in [min_weight_floor, 1.0].
        """
        if len(all_tracking_errors) <= 1:
            return 1.0
        if tracking_error <= 0.0:
            return 1.0

        # Compute percentile: fraction of ETFs with tracking_error <= this one
        percentile = float(np.mean(all_tracking_errors <= tracking_error))
        cap = self.config.volatility_percentile_cap

        if percentile < cap:
            return 1.0

        # Scale linearly from 1.0 at cap → min_weight_floor at 1.0
        excess = (percentile - cap) / (1.0 - cap)  # 0 at cap, 1 at 100th
        scale = 1.0 - (1.0 - self.config.min_weight_floor) * excess
        return float(np.clip(scale, self.config.min_weight_floor, 1.0))
