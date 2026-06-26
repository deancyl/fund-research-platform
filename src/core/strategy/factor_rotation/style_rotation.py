"""
Style Rotation Strategy (S12) — 风格轮动.

Predicts large/small × value/growth factor direction using macro indicators
(up to 47 metrics). Quarterly rebalance.

Win rate: 63.6% quarterly, annual excess: +6.52%.
Eligible: ALL regimes. Required data: ["close", "pe", "pb", "roe", "market_cap"].
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

import polars as pl
from pydantic import Field

from src.core.strategy.base import (
    BaseStrategy,
    MarketRegime,
    SignalDirection,
    StrategyConfig,
)

# ─── Style Quadrant Enum ────────────────────────────────────────────────────────


class StyleQuadrant(StrEnum):
    """Four style quadrants: large/small × value/growth."""

    LARGE_VALUE = "large_value"
    LARGE_GROWTH = "large_growth"
    SMALL_VALUE = "small_value"
    SMALL_GROWTH = "small_growth"


# ─── Constants ──────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["close", "pe", "pb", "roe", "market_cap"]

_QUADRANT_LABELS: dict[StyleQuadrant, str] = {
    StyleQuadrant.LARGE_VALUE: "大盘价值",
    StyleQuadrant.LARGE_GROWTH: "大盘成长",
    StyleQuadrant.SMALL_VALUE: "小盘价值",
    StyleQuadrant.SMALL_GROWTH: "小盘成长",
}

_DEFAULT_MARKET_CAP_THRESHOLD: float = 1000.0
_DEFAULT_VALUE_SCORE_THRESHOLD: float = 0.3

_STYLE_CONFIDENCE: float = 0.636
_TRIM_CONFIDENCE: float = 0.55
_DEFAULT_STYLE_WEIGHT: float = 0.20

# ─── Macro indicator normalization parameters ───────────────────────────────────
# Each entry: (neutral_value, scale) — contribution = (value - neutral) / scale.
# Growth indicators: higher = expansion → favors growth styles.
# Stress indicators: higher = contraction/stress → favors value styles.

_MACRO_INDICATOR_PARAMS: dict[str, tuple[float, float, str]] = {
    # (neutral, scale, category: "growth" | "stress")
    "pmi": (50.0, 10.0, "growth"),
    "pmi_new_orders": (50.0, 10.0, "growth"),
    "pmi_production": (50.0, 10.0, "growth"),
    "cpi": (3.0, 2.0, "stress"),
    "cpi_core": (2.0, 1.5, "stress"),
    "ppi": (0.0, 5.0, "stress"),
    "m2_growth": (10.0, 5.0, "growth"),
    "social_financing": (10.0, 10.0, "growth"),
    "loan_growth": (10.0, 5.0, "growth"),
    "industrial_production": (5.0, 5.0, "growth"),
    "fixed_asset_investment": (5.0, 5.0, "growth"),
    "retail_sales": (5.0, 5.0, "growth"),
    "export_growth": (5.0, 10.0, "growth"),
    "import_growth": (5.0, 10.0, "growth"),
    "interest_rate_10y": (3.5, 2.0, "stress"),
    "interest_rate_1y": (2.0, 1.5, "stress"),
    "credit_spread": (1.0, 2.0, "stress"),
    "fx_reserves": (31000.0, 5000.0, "growth"),
    "usd_cny": (7.0, 1.0, "stress"),
    "housing_price_index": (100.0, 10.0, "stress"),
}


# ─── Configuration ──────────────────────────────────────────────────────────────


class StyleRotationConfig(StrategyConfig):
    """Configuration for Style Rotation strategy (S12).

    Extends StrategyConfig with macro lookback window, quadrant weights,
    and classification thresholds.
    """

    macro_lookback: int = Field(
        default=12,
        ge=1,
        description="Number of months of macro data to use for composite score",
    )
    style_quadrants: dict[str, float] = Field(
        default_factory=lambda: {
            "large_value": 0.0,
            "large_growth": 0.0,
            "small_value": 0.0,
            "small_growth": 0.0,
        },
        description="Style quadrant allocation weights (computed at runtime)",
    )
    market_cap_threshold: float = Field(
        default=_DEFAULT_MARKET_CAP_THRESHOLD,
        gt=0,
        description="Market cap threshold: above = large-cap, below = small-cap",
    )
    value_score_threshold: float = Field(
        default=_DEFAULT_VALUE_SCORE_THRESHOLD,
        gt=0,
        description="Value score threshold: above = value, below = growth",
    )
    eligible_regimes: set[MarketRegime] = Field(
        default_factory=lambda: {r for r in MarketRegime},
        description="Style rotation operates in all market regimes",
    )


# ─── Strategy ───────────────────────────────────────────────────────────────────


class StyleRotation(BaseStrategy):
    """Style Rotation strategy — macro-driven quadrant rotation.

    Uses macro indicators to predict which style quadrant (large/small ×
    value/growth) will outperform, then rotates the portfolio accordingly.

    Algorithm per generate_signals():
      1. Extract macro data from market_data["macro"].
      2. Compute a composite macro score from available indicators.
      3. Determine favored style quadrant(s) from the composite score.
      4. Classify each fund into a style quadrant (PE/PB/ROE + market cap).
      5. Emit BUY for funds in favored quadrant(s), TRIM for others.
    """

    config: StyleRotationConfig = Field(
        default_factory=lambda: StyleRotationConfig(name="style_rotation"),
        description="Style rotation strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate trading signals based on macro-driven style rotation.

        Args:
            dt: Trading date.
            market_data: Must contain 'macro' key with indicator columns,
                         plus DataFrames for each fund with PE/PB/ROE/market_cap.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts, one per fund with sufficient data.
        """
        if not market_data or "macro" not in market_data:
            return []

        macro_df = market_data["macro"]
        if macro_df.is_empty():
            return []

        composite = self._compute_macro_composite(macro_df)
        favored = self._favored_quadrants(composite)

        signals: list[dict[str, float | int | str]] = []
        max_weight = self.config.max_position_pct

        for code, df in market_data.items():
            if code == "macro":
                continue
            if not self._has_style_columns(df):
                continue

            quadrant = self._classify_quadrant_from_df(df)
            label = _QUADRANT_LABELS[quadrant]

            if quadrant in favored:
                signals.append({
                    "fund_code": code,
                    "direction": SignalDirection.BUY.value,
                    "confidence": _STYLE_CONFIDENCE,
                    "target_weight": max_weight,
                    "reason": (
                        f"{label} quadrant favored "
                        f"(macro_composite={composite:+.3f}), "
                        f"pe={float(df['pe'][-1]):.1f} pb={float(df['pb'][-1]):.2f}"
                    ),
                })
            else:
                signals.append({
                    "fund_code": code,
                    "direction": SignalDirection.TRIM.value,
                    "confidence": _TRIM_CONFIDENCE,
                    "target_weight": 0.0,
                    "reason": (
                        f"{label} quadrant not favored "
                        f"(macro_composite={composite:+.3f})"
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
        if cfg.macro_lookback < 1:
            errors.append("macro_lookback must be >= 1")
        if cfg.market_cap_threshold <= 0:
            errors.append("market_cap_threshold must be > 0")
        if cfg.value_score_threshold <= 0:
            errors.append("value_score_threshold must be > 0")
        return errors

    # ── Public Helpers ─────────────────────────────────────────────────────

    def classify_quadrant(
        self,
        pe: float,
        pb: float,
        roe: float,
        market_cap: float,
    ) -> StyleQuadrant:
        """Classify a fund into a style quadrant based on its metrics.

        Size classification: market_cap >= threshold → large, else → small.
        Style classification: value_score = (1/pb + roe/pe) / 2.
          value_score >= threshold → value, else → growth.

        Args:
            pe: Price-to-earnings ratio.
            pb: Price-to-book ratio.
            roe: Return on equity (decimal, e.g. 0.15 = 15%).
            market_cap: Market capitalization in 100M CNY.

        Returns:
            The corresponding StyleQuadrant.
        """
        is_large = market_cap >= self.config.market_cap_threshold
        value_score = (1.0 / pb + roe / pe) / 2.0
        is_value = value_score >= self.config.value_score_threshold

        if is_large and is_value:
            return StyleQuadrant.LARGE_VALUE
        if is_large and not is_value:
            return StyleQuadrant.LARGE_GROWTH
        if not is_large and is_value:
            return StyleQuadrant.SMALL_VALUE
        return StyleQuadrant.SMALL_GROWTH

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_style_columns(df: pl.DataFrame) -> bool:
        """Check that DataFrame has the required style classification columns."""
        required = {"pe", "pb", "roe", "market_cap"}
        return required.issubset(df.columns)

    def _classify_quadrant_from_df(self, df: pl.DataFrame) -> StyleQuadrant:
        """Classify fund quadrant from the latest row of a DataFrame."""
        row = df.row(-1, named=True)
        return self.classify_quadrant(
            pe=float(row["pe"]),
            pb=float(row["pb"]),
            roe=float(row["roe"]),
            market_cap=float(row["market_cap"]),
        )

    @staticmethod
    def _compute_macro_composite(macro_df: pl.DataFrame) -> float:
        """Compute a composite macro score from available indicators.

        For each indicator with a known normalization, compute:
            contribution = (value - neutral) / scale
        Growth indicators contribute positively; stress indicators negatively.

        Returns:
            Composite score: positive = favors growth, negative = favors value.
        """
        row = macro_df.row(0, named=True)
        growth_contributions: list[float] = []
        stress_contributions: list[float] = []

        for col, (neutral, scale, category) in _MACRO_INDICATOR_PARAMS.items():
            if col not in macro_df.columns:
                continue
            raw = float(row[col])
            normalized = (raw - neutral) / scale
            if category == "growth":
                growth_contributions.append(normalized)
            else:
                stress_contributions.append(normalized)

        growth_score = (
            sum(growth_contributions) / len(growth_contributions)
            if growth_contributions
            else 0.0
        )
        stress_score = (
            sum(stress_contributions) / len(stress_contributions)
            if stress_contributions
            else 0.0
        )

        return growth_score - stress_score

    @staticmethod
    def _favored_quadrants(composite: float) -> set[StyleQuadrant]:
        """Determine favored style quadrant(s) from macro composite score.

        Thresholds:
          composite >  1.0 → strongly favors growth (both large and small)
          0 < composite <= 1.0 → mildly favors growth (small growth)
          -1.0 < composite <= 0 → mildly favors value (large value)
          composite <= -1.0 → strongly favors value (both large and small)
        """
        if composite > 1.0:
            return {StyleQuadrant.LARGE_GROWTH, StyleQuadrant.SMALL_GROWTH}
        if composite > 0.0:
            return {StyleQuadrant.SMALL_GROWTH}
        if composite > -1.0:
            return {StyleQuadrant.LARGE_VALUE}
        return {StyleQuadrant.LARGE_VALUE, StyleQuadrant.SMALL_VALUE}
