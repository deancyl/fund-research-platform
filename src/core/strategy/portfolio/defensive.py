"""
3-Layer Defensive Portfolio Strategy (S18) — Fixed low-risk allocation.

Strategy: Fixed allocation across three defensive asset classes:
  - Dividend ETF (dividend_code): 40% — stable income from high-dividend stocks
  - Bond ETF (bond_code): 40% — capital preservation via government/corporate bonds
  - Gold ETF (gold_code): 20% — inflation hedge and crisis protection

The allocation is static and rebalanced annually. This is a passive strategy
designed for capital preservation with modest returns in any market regime.

Key parameters:
  - dividend_code: Fund code for the dividend ETF (default: "510880")
  - bond_code: Fund code for the bond ETF (default: "511260")
  - gold_code: Fund code for the gold ETF (default: "518880")
  - dividend_weight / bond_weight / gold_weight: Fixed allocation weights

Eligible regimes: ALL (defensive strategy works in any market).
Required data: ["close"]
"""

from datetime import date

import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["close"]


# ─── Configuration ────────────────────────────────────────────────────────────


class DefensivePortfolioConfig(StrategyConfig):
    """Configuration for 3-Layer Defensive Portfolio strategy (S18).

    Defines the three ETF fund codes and their fixed allocation weights.
    """

    dividend_code: str = Field(
        default="510880",
        min_length=1,
        description="Fund code for the dividend ETF component",
    )
    bond_code: str = Field(
        default="511260",
        min_length=1,
        description="Fund code for the bond ETF component",
    )
    gold_code: str = Field(
        default="518880",
        min_length=1,
        description="Fund code for the gold ETF component",
    )
    dividend_weight: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Fixed allocation weight for dividend ETF (default: 40%)",
    )
    bond_weight: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Fixed allocation weight for bond ETF (default: 40%)",
    )
    gold_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Fixed allocation weight for gold ETF (default: 20%)",
    )
    max_position_pct: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Maximum position per component (matches largest default weight)",
    )

    @model_validator(mode="after")
    def _check_weights_sum(self) -> "DefensivePortfolioConfig":
        """Ensure the three allocation weights sum to 1.0."""
        total = self.dividend_weight + self.bond_weight + self.gold_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Allocation weights must sum to ~1.0, got {total:.4f}"
            )
        return self

    @property
    def _allocation(self) -> dict[str, float]:
        """Return the fixed allocation mapping: {fund_code: weight}."""
        return {
            self.dividend_code: self.dividend_weight,
            self.bond_code: self.bond_weight,
            self.gold_code: self.gold_weight,
        }


# ─── Strategy ─────────────────────────────────────────────────────────────────


class DefensivePortfolio(BaseStrategy):
    """3-Layer Defensive Portfolio — fixed allocation across defensive assets.

    Algorithm per generate_signals():
      1. Look up the three target fund codes from config.
      2. For each fund present in market_data, emit a BUY signal with its
         fixed allocation weight.
      3. Missing funds are simply skipped (no signal for them).
      4. Extra funds in market_data beyond the three targets are ignored.

    This is a passive, static strategy. The "annual rebalance" is enforced
    by the scheduling layer — this strategy always returns the same fixed
    allocation regardless of current holdings or market conditions.
    """

    config: DefensivePortfolioConfig = Field(
        default_factory=lambda: DefensivePortfolioConfig(name="defensive_portfolio"),
        description="3-layer defensive portfolio strategy configuration",
    )

    # ── Abstract Interface Implementation ──────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,
    ) -> list[dict[str, float | int | str]]:
        """Generate fixed allocation signals.

        Args:
            dt: Trading date.
            market_data: Dict of {fund_code: DataFrame} with close prices.
            portfolio: Current portfolio (unused — static allocation).

        Returns:
            List of BUY signal dicts for each available target ETF.
        """
        signals: list[dict[str, float | int | str]] = []
        allocation = self.config._allocation

        for code, weight in allocation.items():
            if code not in market_data:
                continue
            df = market_data[code]
            if not self._has_close_column(df):
                continue

            component_name = self._component_name(code)

            signals.append({
                "fund_code": code,
                "direction": SignalDirection.BUY.value,
                "confidence": 0.90,
                "target_weight": float(weight),
                "reason": f"Fixed defensive allocation — {component_name} ({weight:.0%})",
            })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields for the data layer to prefetch."""
        return _REQUIRED_FIELDS

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors: list[str] = []
        cfg = self.config

        if not cfg.dividend_code.strip():
            errors.append("dividend_code must not be empty")
        if not cfg.bond_code.strip():
            errors.append("bond_code must not be empty")
        if not cfg.gold_code.strip():
            errors.append("gold_code must not be empty")

        total = cfg.dividend_weight + cfg.bond_weight + cfg.gold_weight
        if not (0.99 <= total <= 1.01):
            errors.append(
                f"Allocation weights must sum to ~1.0, got {total:.4f}"
            )
        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")

        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_close_column(df: pl.DataFrame) -> bool:
        """Check that DataFrame has the close column."""
        return "close" in df.columns

    def _component_name(self, code: str) -> str:
        """Return a human-readable name for a fund code."""
        cfg = self.config
        if code == cfg.dividend_code:
            return "dividend ETF"
        elif code == cfg.bond_code:
            return "bond ETF"
        elif code == cfg.gold_code:
            return "gold ETF"
        return "unknown component"
