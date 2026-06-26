"""
AI-Enhanced Portfolio Strategy (S20) — AI增强组合.

Strategy: Base allocation mirrors S19 aggressive portfolio (CSI300 30%,
ChiNext 20%, Sector ETF 30%, Gold 10%, Cash 10%). An AI overlay simulates
a multi-agent weekend debate to adjust each component's weight by up to
±ai_adjustment_range based on recent price trends.

AI overlay logic:
  1. For each fund (except cash), compute a simulated AI confidence factor
     from recent close price momentum using tanh normalization.
  2. adjusted_weight = base_weight × (1 + ai_confidence × ai_adjustment_range)
  3. Weights are capped so total non-cash allocation ≤ 1.0 (cash absorbs residual).
  4. Cash is the residual buffer — not emitted as a signal.

The AI confidence factor simulates a multi-agent debate output:
  - Positive recent returns → bullish consensus → weight boost
  - Negative recent returns → bearish consensus → weight reduction
  - Flat returns → neutral → weight near base

Key parameters:
  - base_weights: dict[str, float] mapping fund codes to base allocation
  - ai_adjustment_range: maximum ±adjustment for any component (default: 0.20)

Eligible regimes: ALL.
Required data: ["close"].
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["close"]

# Trend normalization: 5% 5-day return → tanh(1) → strong signal
_AI_TREND_NORMALIZER: float = 0.05
_AI_LOOKBACK_DAYS: int = 5

_CASH_KEY: str = "cash"

_DEFAULT_BASE_WEIGHTS: dict[str, float] = {
    "510300": 0.30,
    "159915": 0.20,
    "sector_etf": 0.30,
    "gold": 0.10,
    _CASH_KEY: 0.10,
}


# ─── Configuration ────────────────────────────────────────────────────────────


class AIEnhancedConfig(StrategyConfig):
    """Configuration for AI-Enhanced Portfolio strategy (S20).

    Defines base allocation weights and the maximum AI adjustment range per component.
    """

    base_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_BASE_WEIGHTS),
        description="Base allocation weights: {fund_code: weight}. Must include 'cash'.",
    )
    ai_adjustment_range: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="Maximum ±adjustment for any component weight (e.g., 0.20 = ±20%)",
    )
    max_position_pct: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximum position per component after AI adjustment",
    )

    @model_validator(mode="after")
    def _check_weights(self) -> "AIEnhancedConfig":
        """Validate base_weights: non-empty, sum to ~1.0, cash present, all in [0,1]."""
        bw = self.base_weights
        if not bw:
            raise ValueError("base_weights must not be empty")

        for code, w in bw.items():
            if w < 0.0 or w > 1.0:
                raise ValueError(
                    f"base_weights['{code}']={w} must be in [0, 1]"
                )

        total = sum(bw.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"base_weights must sum to ~1.0, got {total:.4f}"
            )

        if _CASH_KEY not in bw:
            raise ValueError(
                "base_weights must include 'cash' key for residual allocation"
            )

        return self

    @property
    def _non_cash_weights(self) -> dict[str, float]:
        """Return base weights excluding cash."""
        return {k: v for k, v in self.base_weights.items() if k != _CASH_KEY}

    @property
    def _cash_weight(self) -> float:
        """Return base cash weight."""
        return self.base_weights.get(_CASH_KEY, 0.0)


# ─── Strategy ─────────────────────────────────────────────────────────────────


class AIEnhancedPortfolio(BaseStrategy):
    """AI-Enhanced Portfolio — base allocation with simulated AI overlay.

    Algorithm per generate_signals():
      1. For each fund in base_weights (excluding cash):
         a. Compute AI confidence factor from recent close returns.
         b. adjusted_weight = base_weight × (1 + ai_confidence × ai_adjustment_range)
         c. Clamp to [0, max_position_pct].
      2. If total adjusted non-cash weights exceed 1.0, scale down proportionally
         to ensure cash residual ≥ 0.
      3. Emit BUY signals for each fund with its AI-adjusted weight.
      4. Cash absorbs the residual — no signal emitted for it.
    """

    config: AIEnhancedConfig = Field(
        default_factory=lambda: AIEnhancedConfig(name="ai_enhanced"),
        description="AI-enhanced portfolio strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate AI-adjusted portfolio allocation signals.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with close prices.
            portfolio: Current portfolio (unused).

        Returns:
            List of BUY signal dicts for each available fund with AI-adjusted weights.
        """
        if not market_data:
            return []

        cfg = self.config
        non_cash = cfg._non_cash_weights
        signals: list[dict[str, float | int | str]] = []

        # Step 1: Compute AI-adjusted weights for each available fund
        raw_adjusted: list[tuple[str, float, float]] = []  # (code, weight, confidence)

        for code, base_weight in non_cash.items():
            if code not in market_data:
                continue
            df = market_data[code]
            if not self._has_close_column(df):
                continue

            ai_confidence = self._compute_ai_confidence(df)
            adjusted_weight = self._apply_ai_overlay(base_weight, ai_confidence)

            raw_adjusted.append((code, adjusted_weight, ai_confidence))

        if not raw_adjusted:
            return []

        # Step 2: Normalize if total non-cash exceeds 1.0 (cash can't be negative)
        total_raw = sum(w for _, w, _ in raw_adjusted)
        if total_raw > 1.0:
            scale = 1.0 / total_raw
        else:
            scale = 1.0

        # Step 3: Emit signals
        for code, raw_w, ai_conf in raw_adjusted:
            final_weight = float(np.clip(raw_w * scale, 0.0, cfg.max_position_pct))

            # AI confidence mapped to [0, 1] for signal confidence
            mapped_confidence = float(np.clip((ai_conf + 1.0) / 2.0, 0.0, 1.0))

            signals.append({
                "fund_code": code,
                "direction": SignalDirection.BUY.value,
                "confidence": mapped_confidence,
                "target_weight": final_weight,
                "reason": (
                    f"AI-enhanced allocation — base={base_weight:.0%}, "
                    f"ai_confidence={ai_conf:+.3f}, "
                    f"ai_adjusted={final_weight:.0%}"
                ),
            })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields for the data layer to prefetch."""
        return _REQUIRED_FIELDS

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors: list[str] = []
        cfg = self.config

        if not cfg.base_weights:
            errors.append("base_weights must not be empty")

        for code, w in cfg.base_weights.items():
            if w < 0.0 or w > 1.0:
                errors.append(f"base_weights['{code}']={w} must be in [0, 1]")

        total = sum(cfg.base_weights.values())
        if not (0.99 <= total <= 1.01):
            errors.append(
                f"base_weights must sum to ~1.0, got {total:.4f}"
            )

        if _CASH_KEY not in cfg.base_weights:
            errors.append(
                "base_weights must include 'cash' key for residual allocation"
            )

        if not (0.0 < cfg.ai_adjustment_range <= 1.0):
            errors.append(
                f"ai_adjustment_range must be in (0, 1], got {cfg.ai_adjustment_range}"
            )

        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")

        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_close_column(df: pl.DataFrame) -> bool:
        """Check that DataFrame has the close column."""
        return "close" in df.columns

    @staticmethod
    def _compute_ai_confidence(df: pl.DataFrame) -> float:
        """Compute simulated AI confidence factor from recent close trend.

        Simulates a multi-agent weekend debate output:
          - Computes the 5-day return from close prices.
          - Normalizes via tanh to produce a [-1, 1] signal.

        tanh(return / 0.05) maps:
          +5% return → tanh(1) ≈ 0.76 (strong bullish consensus)
          -5% return → tanh(-1) ≈ -0.76 (strong bearish consensus)
          0% return → tanh(0) = 0.00 (neutral)

        Uses up to _AI_LOOKBACK_DAYS of lookback. If insufficient data, returns 0.0.

        Returns:
            AI confidence factor in [-1, 1].
        """
        close_arr = df["close"].to_numpy().astype(np.float64)
        n = len(close_arr)

        if n < 2:
            return 0.0

        # Use the last N days for the trend, but at minimum compare last 2
        lookback = min(_AI_LOOKBACK_DAYS, n - 1)
        if lookback < 1:
            lookback = 1

        current = float(close_arr[-1])
        previous = float(close_arr[-1 - lookback])

        if previous <= 1e-12:
            return 0.0

        period_return = (current - previous) / previous

        # Normalize via tanh to get AI confidence in [-1, 1]
        ai_confidence = float(np.tanh(period_return / _AI_TREND_NORMALIZER))
        return ai_confidence

    def _apply_ai_overlay(self, base_weight: float, ai_confidence: float) -> float:
        """Adjust base weight using AI confidence factor.

        Args:
            base_weight: Base allocation weight for this component.
            ai_confidence: AI confidence factor in [-1, 1].

        Returns:
            Adjusted weight: base_weight × (1 + ai_confidence × ai_adjustment_range).
        """
        ar = self.config.ai_adjustment_range
        adjusted = base_weight * (1.0 + ai_confidence * ar)
        return float(np.clip(adjusted, 0.0, self.config.max_position_pct))
