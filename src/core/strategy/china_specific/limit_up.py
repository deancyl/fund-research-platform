"""
Limit-up Probability Strategy (S17) — 量化打板概率模型.

Strategy: Monitor order-by-order data to estimate the probability of hitting
the daily limit-up. This strategy is marked HIGH_RISK per 2026 regulations
and is NOT recommended for retail investors. Designed for informational
display only with mandatory risk warnings.

Logic:
  1. Compute limit-up proximity: how close is current close to the limit-up price?
  2. Detect volume spike: is current volume significantly above trailing average?
  3. Analyze turnover: high turnover signals speculative frenzy.
  4. Evaluate bid-ask spread: tight spread suggests buying pressure near limit.
  5. Combine into a weighted probability score.
  6. ALWAYS include a risk_warning field per 2026 regulations.

Key parameters:
  - limit_up_pct: daily limit percentage (default: 0.10 for main board)
  - volume_spike_threshold: multiple of average volume to flag as spike (default: 3.0)

Eligible regimes: TRENDING_UP only (informational display, strong warning).
Required data: ["close", "volume", "turnover", "bid_ask_spread"]
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["close", "volume", "turnover", "bid_ask_spread"]

# Probability weights for the four signal components
_W_PROXIMITY: float = 0.35
_W_VOLUME: float = 0.25
_W_TURNOVER: float = 0.20
_W_SPREAD: float = 0.20

# Normalization caps
_MAX_TURNOVER_FOR_SCORE: float = 0.20  # 20% turnover = full score
_MAX_SPREAD_FOR_SCORE: float = 0.01  # 1% spread = zero tightness score
_VOLUME_TRAILING_WINDOW: int = 20  # lookback for average volume computation

_RISK_WARNING_TEXT: str = (
    "HIGH_RISK: 量化打板模型仅作为信息展示参考。2026年监管新规下打板策略"
    "风险极高，不推荐散户使用。本信号不代表投资建议。"
    " (Limit-up chasing is HIGH RISK under 2026 regulations. "
    "Not recommended for retail investors. For informational display only.)"
)

_CONFIDENCE_THRESHOLD_HIGH: float = 0.60


# ─── Configuration ────────────────────────────────────────────────────────────


class LimitUpConfig(StrategyConfig):
    """Configuration for Limit-up Probability strategy (S17).

    Extends StrategyConfig with limit-up percentage and volume spike threshold.
    """

    limit_up_pct: float = Field(
        default=0.10,
        gt=0.0,
        description="Daily limit-up percentage (e.g., 0.10 for main board, 0.20 for ChiNext)",
    )
    volume_spike_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="Volume multiple of trailing average to flag as spike (e.g., 3.0 = 3x avg)",
    )
    eligible_regimes: set[MarketRegime] = Field(
        default_factory=lambda: {MarketRegime.TRENDING_UP},
        description="Limit-up chasing only meaningful in trending-up markets",
    )


# ─── Strategy ─────────────────────────────────────────────────────────────────


class LimitUp(BaseStrategy):
    """Limit-up Probability strategy — estimates chance of hitting daily limit.

    Algorithm per generate_signals():
      1. For each fund, verify all four required columns exist and >= 2 rows.
      2. Compute limit-up proximity from last two close prices.
      3. Compute volume spike ratio (last volume / trailing average volume).
      4. Extract latest turnover and bid-ask spread.
      5. Combine into a weighted probability score in [0, 1].
      6. Emit ACCUMULATE if probability >= threshold, else HOLD.
      7. ALWAYS include risk_warning field and strong warning in reason.
    """

    config: LimitUpConfig = Field(
        default_factory=lambda: LimitUpConfig(name="limit_up"),
        description="Limit-up probability strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate limit-up probability signals with mandatory risk warnings.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with close, volume,
                         turnover, and bid_ask_spread columns.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts sorted by limit-up probability descending.
            Every signal includes a risk_warning field.
        """
        if not market_data:
            return []

        signals: list[dict[str, float | int | str]] = []

        for code, df in market_data.items():
            if not self._has_required_columns(df):
                continue
            if df.height < 2:
                continue

            probability = self._compute_limit_up_probability(df)

            if probability >= _CONFIDENCE_THRESHOLD_HIGH:
                direction = SignalDirection.ACCUMULATE.value
                weight = 0.05  # small allocation for informational tracking only
            else:
                direction = SignalDirection.HOLD.value
                weight = 0.0

            signals.append({
                "fund_code": code,
                "direction": direction,
                "confidence": float(np.clip(probability, 0.0, 1.0)),
                "target_weight": float(weight),
                "reason": (
                    f"Limit-up probability={probability:.1%} — "
                    f"{_RISK_WARNING_TEXT}"
                ),
                "risk_warning": _RISK_WARNING_TEXT,
            })

        # Sort by probability descending
        signals.sort(key=lambda s: s["confidence"], reverse=True)
        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields for the data layer to prefetch."""
        return _REQUIRED_FIELDS

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors: list[str] = []
        cfg = self.config
        if cfg.limit_up_pct <= 0.0:
            errors.append("limit_up_pct must be positive")
        if cfg.volume_spike_threshold <= 0.0:
            errors.append("volume_spike_threshold must be positive")
        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_required_columns(df: pl.DataFrame) -> bool:
        """Check that DataFrame has all four required columns."""
        df_cols = set(df.columns)
        return all(col in df_cols for col in _REQUIRED_FIELDS)

    def _compute_limit_up_probability(self, df: pl.DataFrame) -> float:
        """Compute weighted limit-up probability from the latest data row.

        Components:
          - proximity: (current_close - prev_close) / (prev_close * limit_up_pct),
            clamped [0,1] — what fraction of the daily limit has been used.
          - volume_factor: current_volume / avg_trailing_volume, scaled by threshold
          - turnover_factor: min(turnover / _MAX_TURNOVER_FOR_SCORE, 1.0)
          - spread_tightness: max(0, 1 - bid_ask_spread / _MAX_SPREAD_FOR_SCORE)

        Returns:
            Probability in [0, 1].
        """
        cfg = self.config

        # ── 1. Limit-up proximity ───────────────────────────────────
        # Measures how much of the daily limit the stock has already used.
        # proximity = (current_close - prev_close) / (prev_close * limit_up_pct)
        # 0% = no movement toward limit, 100% = exactly at limit-up.
        close_arr = df["close"].to_numpy().astype(np.float64)
        prev_close = float(close_arr[-2])
        current_close = float(close_arr[-1])

        if prev_close <= 1e-12:
            proximity = 0.0
        else:
            max_gain = prev_close * cfg.limit_up_pct
            actual_gain = current_close - prev_close
            proximity = float(np.clip(actual_gain / max_gain, 0.0, 1.0)) if max_gain > 0 else 0.0

        # ── 2. Volume spike ratio ──────────────────────────────────
        vol_arr = df["volume"].to_numpy().astype(np.float64)
        current_vol = float(vol_arr[-1])

        # Trailing average (excluding the last row)
        if len(vol_arr) > 1:
            trailing_vol = vol_arr[:-1]
            lookback = min(len(trailing_vol), _VOLUME_TRAILING_WINDOW)
            avg_vol = float(np.mean(trailing_vol[-lookback:]))
        else:
            avg_vol = max(current_vol, 1.0)

        if avg_vol > 1e-12:
            vol_ratio = current_vol / avg_vol
        else:
            vol_ratio = 1.0

        volume_factor = float(np.clip(vol_ratio / cfg.volume_spike_threshold, 0.0, 1.0))

        # ── 3. Turnover factor ─────────────────────────────────────
        turnover_arr = df["turnover"].to_numpy().astype(np.float64)
        turnover = float(turnover_arr[-1])
        turnover_factor = float(np.clip(turnover / _MAX_TURNOVER_FOR_SCORE, 0.0, 1.0))

        # ── 4. Bid-ask spread tightness ────────────────────────────
        spread_arr = df["bid_ask_spread"].to_numpy().astype(np.float64)
        spread = float(spread_arr[-1])
        spread_tightness = float(np.clip(1.0 - spread / _MAX_SPREAD_FOR_SCORE, 0.0, 1.0))

        # ── Combine ────────────────────────────────────────────────
        probability = (
            _W_PROXIMITY * proximity
            + _W_VOLUME * volume_factor
            + _W_TURNOVER * turnover_factor
            + _W_SPREAD * spread_tightness
        )

        return float(np.clip(probability, 0.0, 1.0))
