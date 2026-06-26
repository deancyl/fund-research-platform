"""
Strategy base class — foundation for all 20 strategies in the platform.

Each strategy implements:
  - generate_signals(): produce trading signals for a given date.
  - required_data(): declare what market data the strategy needs (for prefetch).
  - validate(): self-check parameters and data consistency.

Strategies are Pydantic models (frozen) for parameter serialization and
version tracking. The `name` attribute is the unique strategy identifier.

Extension: new strategies subclass BaseStrategy and override the abstract
methods. The strategy registry (in __init__.py) auto-discovers subclasses.
"""

from abc import ABC, abstractmethod
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


# ─── Signal Types ────────────────────────────────────────────────────────────


class SignalDirection(StrEnum):
    """Direction of a trading signal."""

    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    TRIM = "TRIM"
    SELL = "SELL"


class MarketRegime(StrEnum):
    """Market regime classification — drives strategy eligibility."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"


# ─── Strategy Config & Context ─────────────────────────────────────────────


class StrategyConfig(BaseModel, frozen=True):
    """Serializable strategy parameters — versioned for reproducibility."""

    name: str = Field(description="Unique strategy identifier")
    version: str = Field(default="1.0.0", description="Semantic version")
    eligible_regimes: set[MarketRegime] = Field(
        default_factory=lambda: {r for r in MarketRegime},
        description="Market regimes this strategy can operate in",
    )
    max_position_pct: float = Field(
        default=0.20, ge=0, le=1.0, description="Maximum position as fraction of portfolio"
    )
    min_holding_days: int = Field(
        default=1, ge=1, description="Minimum holding period to avoid penalty fees"
    )


# ─── Strategy Context ──────────────────────────────────────────────────────


class StrategyContext(BaseModel, frozen=True):
    """
    Full execution context passed to strategy.generate_signals().

    Per audit: strategies need more than just a date — they need cash,
    existing positions, and regime awareness to avoid overbuying.
    """

    current_date: date = Field(description="Trading date for signal generation")
    available_cash: float = Field(default=0.0, ge=0, description="Buying power in CNY")
    existing_positions: dict[str, float] = Field(
        default_factory=dict, description="{fund_code: current_shares}"
    )
    market_regime: MarketRegime = Field(
        default=MarketRegime.SIDEWAYS, description="Detected market regime from Layer 5"
    )
    total_portfolio_value: float = Field(default=0.0, ge=0)


# ─── Strategy Base ──────────────────────────────────────────────────────────


class BaseStrategy(ABC, BaseModel):
    """
    Abstract base for all trading strategies.

    Subclasses must override:
      - generate_signals()
      - required_data()
      - validate()

    Strategy parameters are frozen Pydantic fields — serializable to JSON/YAML
    for version tracking and reproduction.
    """

    model_config = ConfigDict(frozen=True)

    config: StrategyConfig = Field(
        default_factory=StrategyConfig,
        description="Strategy metadata and risk constraints",
    )

    # ── Abstract Interface ───────────────────────────────────────────────

    @abstractmethod
    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, "polars.DataFrame"],  # noqa: F821
        portfolio: "list[FundPosition] | None" = None,  # noqa: F821
    ) -> list[dict[str, float | int | str]]:
        """
        Generate trading signals for a given date.

        Args:
            dt: The trading date for which to generate signals.
            market_data: Dict of {fund_code: DataFrame} with historical OHLCV/NAV data.
            portfolio: Current portfolio positions (optional).

        Returns:
            List of signal dicts with keys:
              - fund_code: str
              - direction: SignalDirection value
              - confidence: float (0.0-1.0)
              - target_weight: float (proposed position weight)
              - reason: str (human-readable rationale)
        """
        ...

    @abstractmethod
    def required_data(self) -> list[str]:
        """
        Declare required data fields for the data layer to prefetch.

        Returns: list of field names like ['nav', 'close', 'volume', 'pe'].
        """
        ...

    def validate(self) -> list[str]:  # noqa: B027 — default no-issue
        """
        Self-validate parameters and constraints.

        Returns: list of validation error messages (empty = valid).
        Override in subclasses with strategy-specific checks.
        """
        return []

    # ── Common Utilities ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Convenience accessor for strategy name."""
        return self.config.name

    def is_eligible(self, regime: MarketRegime) -> bool:
        """Check if the strategy can operate in the given market regime."""
        return regime in self.config.eligible_regimes
