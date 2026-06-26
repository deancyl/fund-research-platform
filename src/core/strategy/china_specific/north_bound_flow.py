"""
North-bound Flow Following Strategy (S16) — 北向资金流向跟踪.

Logic: North-bound capital net inflow 3 consecutive days → BUY;
net outflow 3 consecutive days → SELL; otherwise → HOLD.

Eligible regimes: TRENDING_UP, SIDEWAYS (flow signals most reliable in directional/stable markets).
Required data: ["close", "north_bound_flow"].

Strategy is signal-only — it does not allocate weights. The portfolio layer
interprets BUY/SELL/HOLD to adjust positions.
"""

from __future__ import annotations

from datetime import date

import polars as pl
from pydantic import Field

from src.core.strategy.base import (
    BaseStrategy,
    MarketRegime,
    SignalDirection,
    StrategyConfig,
)

# ─── Constants ──────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["close", "north_bound_flow"]

_FLOW_INFLOW_CONFIDENCE: float = 0.72
_FLOW_OUTFLOW_CONFIDENCE: float = 0.72
_FLOW_HOLD_CONFIDENCE: float = 0.35
_DEFAULT_WEIGHT: float = 0.15


# ─── Configuration ──────────────────────────────────────────────────────────────


class NorthBoundFlowConfig(StrategyConfig):
    """Configuration for North-bound Flow Following strategy (S16).

    Extends StrategyConfig with consecutive-days threshold and flow threshold.
    """

    consecutive_days: int = Field(
        default=3,
        ge=1,
        description="Number of consecutive days of same-direction flow to trigger signal",
    )
    flow_threshold: float = Field(
        default=0.0,
        description="Absolute flow threshold: |flow| > threshold counts as directional",
    )
    eligible_regimes: set[MarketRegime] = Field(
        default_factory=lambda: {MarketRegime.TRENDING_UP, MarketRegime.SIDEWAYS},
        description="Flow signals most reliable in trending-up or sideways markets",
    )


# ─── Strategy ───────────────────────────────────────────────────────────────────


class NorthBoundFlow(BaseStrategy):
    """North-bound Flow Following strategy.

    Monitors north-bound capital (北向资金) net flow direction. When flow is
    consistently directional for `consecutive_days`, emits a trading signal.

    Algorithm per generate_signals():
      1. For each fund, check that 'north_bound_flow' column exists.
      2. Extract the last `consecutive_days` rows.
      3. If all rows have flow > +threshold → BUY (net persistent inflow).
      4. If all rows have flow < -threshold → SELL (net persistent outflow).
      5. Otherwise → HOLD.
      6. Funds with insufficient data are skipped silently.
    """

    config: NorthBoundFlowConfig = Field(
        default_factory=lambda: NorthBoundFlowConfig(name="north_bound_flow"),
        description="North-bound flow strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate trading signals based on north-bound flow direction.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with 'north_bound_flow' column.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts, one per fund with sufficient data.
        """
        if not market_data:
            return []

        n = self.config.consecutive_days
        threshold = self.config.flow_threshold
        signals: list[dict[str, float | int | str]] = []

        for code, df in market_data.items():
            if not self._has_flow_column(df):
                continue

            direction = self._classify_flow(df, n, threshold)
            if direction is None:
                continue

            if direction == "inflow":
                sig_dir = SignalDirection.BUY.value
                conf = _FLOW_INFLOW_CONFIDENCE
                weight = _DEFAULT_WEIGHT
                reason = f"北向资金连续{n}日净流入, flow_threshold={threshold}"
            elif direction == "outflow":
                sig_dir = SignalDirection.SELL.value
                conf = _FLOW_OUTFLOW_CONFIDENCE
                weight = 0.0
                reason = f"北向资金连续{n}日净流出, flow_threshold={threshold}"
            else:
                sig_dir = SignalDirection.HOLD.value
                conf = _FLOW_HOLD_CONFIDENCE
                weight = 0.0
                reason = f"北向资金流向不明确: 不满足连续{n}日同向条件"

            signals.append({
                "fund_code": code,
                "direction": sig_dir,
                "confidence": conf,
                "target_weight": weight,
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
        if cfg.consecutive_days < 1:
            errors.append("consecutive_days must be >= 1")
        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_flow_column(df: pl.DataFrame) -> bool:
        """Check that DataFrame has the required north_bound_flow column."""
        return "north_bound_flow" in df.columns

    @staticmethod
    def _classify_flow(
        df: pl.DataFrame,
        consecutive_days: int,
        threshold: float,
    ) -> str | None:
        """Classify flow direction from the last N rows.

        Args:
            df: DataFrame with 'north_bound_flow' column.
            consecutive_days: Number of consecutive same-direction rows required.
            threshold: Minimum absolute flow value to count as directional.

        Returns:
            'inflow', 'outflow', 'neutral', or None if insufficient data.
        """
        n_rows = df.height
        if n_rows < consecutive_days:
            return None

        flows = df["north_bound_flow"].tail(consecutive_days).to_list()
        flows_float = [float(f) for f in flows]

        all_inflow = all(f > threshold for f in flows_float)
        all_outflow = all(f < -threshold for f in flows_float)

        if all_inflow:
            return "inflow"
        if all_outflow:
            return "outflow"
        return "neutral"
