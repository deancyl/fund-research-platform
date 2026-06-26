"""
Aggressive ETF Portfolio Strategy (S19) — Dynamic factor momentum overlay.

Strategy: Base allocation of CSI300 (30%), ChiNext (20%), Sector ETF (30%),
Gold (10%), and Cash (10%). Growth-component weights are dynamically adjusted
using a factor momentum overlay based on 1-month and 3-month momentum signals.

Momentum overlay logic:
  - Composite momentum = 0.5 × momentum_1m + 0.5 × momentum_3m
  - Positive momentum → boost weight up to momentum_max_boost above base
  - Negative momentum → reduce weight down to momentum_max_cut below base
  - Gold and Cash are defensive anchors — their weights are not momentum-adjusted

The residual (to reach 100%) goes to cash with no explicit BUY signal.

Key parameters:
  - csi300_code / chinext_code / sector_code / gold_code: Fund codes
  - csi300_weight / chinext_weight / sector_weight / gold_weight / cash_weight
  - momentum_window: lookback for momentum factor signal
  - momentum_max_boost: maximum weight increase from momentum (default: 0.10)
  - momentum_max_cut: maximum weight decrease from momentum (default: 0.10)

Eligible regimes: TRENDING_UP, SIDEWAYS.
Required data: ["close", "momentum_1m", "momentum_3m"]
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["close", "momentum_1m", "momentum_3m"]


# ─── Configuration ────────────────────────────────────────────────────────────


class AggressivePortfolioConfig(StrategyConfig):
    """Configuration for Aggressive ETF Portfolio strategy (S19).

    Defines fund codes, base allocation weights, and momentum overlay parameters.
    """

    # Fund codes
    csi300_code: str = Field(
        default="510300",
        min_length=1,
        description="Fund code for CSI300 ETF component",
    )
    chinext_code: str = Field(
        default="159915",
        min_length=1,
        description="Fund code for ChiNext ETF component",
    )
    sector_code: str = Field(
        default="512100",
        min_length=1,
        description="Fund code for sector ETF component",
    )
    gold_code: str = Field(
        default="518880",
        min_length=1,
        description="Fund code for gold ETF component (defensive anchor)",
    )

    # Base allocation weights
    csi300_weight: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Base allocation weight for CSI300",
    )
    chinext_weight: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Base allocation weight for ChiNext",
    )
    sector_weight: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Base allocation weight for sector ETF",
    )
    gold_weight: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Base allocation weight for gold (defensive anchor)",
    )
    cash_weight: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Base allocation weight for cash (residual, no signal emitted)",
    )

    # Momentum overlay parameters
    momentum_window: int = Field(
        default=1,
        ge=1,
        description="Number of periods for momentum smoothing (1 = use raw signal)",
    )
    momentum_max_boost: float = Field(
        default=0.10,
        ge=0.0,
        le=0.50,
        description="Maximum weight increase from positive momentum overlay",
    )
    momentum_max_cut: float = Field(
        default=0.10,
        ge=0.0,
        le=0.50,
        description="Maximum weight decrease from negative momentum overlay",
    )
    max_position_pct: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Maximum position per component after momentum adjustment",
    )

    @model_validator(mode="after")
    def _check_weights_sum(self) -> "AggressivePortfolioConfig":
        """Ensure base allocation weights sum to 1.0."""
        total = (
            self.csi300_weight
            + self.chinext_weight
            + self.sector_weight
            + self.gold_weight
            + self.cash_weight
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Base allocation weights must sum to ~1.0, got {total:.4f}"
            )
        return self

    @property
    def _growth_allocation(self) -> dict[str, float]:
        """Return base weights for growth components (subject to momentum overlay)."""
        return {
            self.csi300_code: self.csi300_weight,
            self.chinext_code: self.chinext_weight,
            self.sector_code: self.sector_weight,
        }

    @property
    def _defensive_allocation(self) -> dict[str, float]:
        """Return base weights for defensive components (fixed)."""
        return {
            self.gold_code: self.gold_weight,
        }


# ─── Strategy ─────────────────────────────────────────────────────────────────


class AggressivePortfolio(BaseStrategy):
    """Aggressive ETF Portfolio — dynamic factor momentum overlay.

    Algorithm per generate_signals():
      1. For each growth ETF (CSI300, ChiNext, Sector):
         a. Extract momentum_1m and momentum_3m from latest row.
         b. Compute composite momentum: (m1 + m3) / 2.
         c. Compute momentum adjustment: composite * momentum_multiplier, clamped.
         d. Adjusted weight = base_weight + adjustment.
      2. Gold ETF gets fixed base weight (no momentum adjustment).
      3. Cash absorbs the residual (not emitted as a signal).
      4. All ETFs get BUY signal with their adjusted weights.
      5. ETFs with negative momentum may get lower confidence.
    """

    config: AggressivePortfolioConfig = Field(
        default_factory=lambda: AggressivePortfolioConfig(name="aggressive_portfolio"),
        description="Aggressive ETF portfolio strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate dynamic allocation signals with momentum overlay.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with close, momentum_1m,
                         and momentum_3m columns.
            portfolio: Current portfolio (unused).

        Returns:
            List of BUY signal dicts for each available ETF with adjusted weights.
        """
        if not market_data:
            return []

        signals: list[dict[str, float | int | str]] = []
        cfg = self.config

        # 1. Process growth components with momentum overlay
        growth_map = cfg._growth_allocation
        defense_map = cfg._defensive_allocation

        for code, base_weight in growth_map.items():
            if code not in market_data:
                continue
            df = market_data[code]
            if not self._has_required_columns(df):
                continue

            momentum = self._compute_composite_momentum(df)
            adjusted_weight = self._apply_momentum_overlay(base_weight, momentum)

            signals.append({
                "fund_code": code,
                "direction": SignalDirection.BUY.value,
                "confidence": float(np.clip(0.5 + momentum, 0.0, 1.0)),
                "target_weight": float(adjusted_weight),
                "reason": (
                    f"Growth component — base={base_weight:.0%}, "
                    f"momentum={momentum:+.4f}, "
                    f"adjusted={adjusted_weight:.0%}"
                ),
            })

        # 2. Process defensive components with fixed weights
        for code, weight in defense_map.items():
            if code not in market_data:
                continue
            df = market_data[code]
            if not self._has_required_columns(df):
                continue

            signals.append({
                "fund_code": code,
                "direction": SignalDirection.BUY.value,
                "confidence": 0.80,
                "target_weight": float(weight),
                "reason": f"Defensive anchor — fixed allocation ({weight:.0%})",
            })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields for the data layer to prefetch."""
        return _REQUIRED_FIELDS

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors: list[str] = []
        cfg = self.config

        if not cfg.csi300_code.strip():
            errors.append("csi300_code must not be empty")
        if not cfg.chinext_code.strip():
            errors.append("chinext_code must not be empty")
        if not cfg.sector_code.strip():
            errors.append("sector_code must not be empty")
        if not cfg.gold_code.strip():
            errors.append("gold_code must not be empty")

        total = (
            cfg.csi300_weight
            + cfg.chinext_weight
            + cfg.sector_weight
            + cfg.gold_weight
            + cfg.cash_weight
        )
        if not (0.99 <= total <= 1.01):
            errors.append(
                f"Base allocation weights must sum to ~1.0, got {total:.4f}"
            )
        if not (0.0 <= cfg.momentum_max_boost <= 0.50):
            errors.append("momentum_max_boost must be in [0, 0.50]")
        if not (0.0 <= cfg.momentum_max_cut <= 0.50):
            errors.append("momentum_max_cut must be in [0, 0.50]")
        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")

        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_required_columns(df: pl.DataFrame) -> bool:
        """Check that DataFrame has all required columns."""
        df_cols = set(df.columns)
        return all(col in df_cols for col in _REQUIRED_FIELDS)

    @staticmethod
    def _compute_composite_momentum(df: pl.DataFrame) -> float:
        """Compute composite momentum from latest 1m and 3m signals.

        Uses equal weighting: (momentum_1m + momentum_3m) / 2.

        Returns:
            Composite momentum score (e.g., 0.05 = 5% positive momentum).
        """
        row = df.row(-1, named=True)
        mom_1m = float(row["momentum_1m"])
        mom_3m = float(row["momentum_3m"])
        return (mom_1m + mom_3m) / 2.0

    def _apply_momentum_overlay(self, base_weight: float, momentum: float) -> float:
        """Adjust base weight using factor momentum overlay.

        Positive momentum → boost weight (up to momentum_max_boost).
        Negative momentum → reduce weight (up to momentum_max_cut).

        Args:
            base_weight: Base allocation weight for this component.
            momentum: Composite momentum score.

        Returns:
            Adjusted weight clamped to [0, max_position_pct].
        """
        cfg = self.config

        if momentum > 0:
            # Scale momentum to [0, 1] range for boost
            boost_factor = min(momentum / 0.20, 1.0)
            adjustment = boost_factor * cfg.momentum_max_boost
        elif momentum < 0:
            # Scale momentum to [0, 1] range for cut
            cut_factor = min(abs(momentum) / 0.20, 1.0)
            adjustment = -cut_factor * cfg.momentum_max_cut
        else:
            adjustment = 0.0

        adjusted = base_weight + adjustment
        return float(np.clip(adjusted, 0.0, cfg.max_position_pct))
