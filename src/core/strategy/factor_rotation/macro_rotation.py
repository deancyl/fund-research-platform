"""Macro 4-Regime Rotation Strategy (S11) — PMI + CPI driven asset allocation.

4 macro regimes mapped to optimal asset classes:
  衰退(Recession): PMI < threshold, CPI < threshold → bonds
  复苏(Recovery): PMI >= threshold, CPI < threshold → stocks
  过热(Overheating): PMI >= threshold, CPI >= threshold → commodities
  滞涨(Stagflation): PMI < threshold, CPI >= threshold → cash

Performance: annual excess +16.1%, Info Ratio 1.78, Monthly win rate 66%.
Eligible in ALL market regimes (it drives allocation, not timing).
"""

from __future__ import annotations

from datetime import date  # noqa: TC003
from enum import StrEnum

import polars as pl  # noqa: TC002
from pydantic import Field

from src.core.strategy.base import BaseStrategy, SignalDirection, StrategyConfig

# ─── Macro Regime Enum ────────────────────────────────────────────────────────


class MacroRegime(StrEnum):
    """Four macro regimes based on PMI and CPI thresholds."""

    RECESSION = "RECESSION"
    RECOVERY = "RECOVERY"
    OVERHEATING = "OVERHEATING"
    STAGFLATION = "STAGFLATION"


# ─── Regime-to-asset mapping ──────────────────────────────────────────────────

_REGIME_ASSET_MAP: dict[MacroRegime, str] = {
    MacroRegime.RECESSION: "bond_fund",
    MacroRegime.RECOVERY: "stock_fund",
    MacroRegime.OVERHEATING: "commodity_fund",
    MacroRegime.STAGFLATION: "cash_fund",
}

_REGIME_LABELS: dict[MacroRegime, str] = {
    MacroRegime.RECESSION: "衰退",
    MacroRegime.RECOVERY: "复苏",
    MacroRegime.OVERHEATING: "过热",
    MacroRegime.STAGFLATION: "滞涨",
}

_REQUIRED_FIELDS: list[str] = ["pmi", "cpi", "close"]


# ─── Configuration ────────────────────────────────────────────────────────────


class MacroRotationConfig(StrategyConfig):
    """Configuration for Macro 4-Regime Rotation strategy.

    Extends StrategyConfig with PMI/CPI thresholds and fund code mapping.
    """

    pmi_threshold: float = Field(
        default=50.0,
        gt=0,
        description="PMI threshold: above=expansion, below=contraction",
    )
    cpi_threshold: float = Field(
        default=3.0,
        gt=0,
        description="CPI threshold: above=inflation, below=stable",
    )
    bond_fund: str = Field(default="511010", description="Bond ETF for recession regime")
    stock_fund: str = Field(default="510300", description="Stock ETF for recovery regime")
    commodity_fund: str = Field(
        default="159980", description="Commodity ETF for overheating regime",
    )
    cash_fund: str = Field(
        default="511880", description="Cash/money-market ETF for stagflation",
    )


# ─── Strategy ─────────────────────────────────────────────────────────────────


class MacroRotation(BaseStrategy):
    """Macro 4-Regime Rotation strategy.

    Uses PMI and CPI data to classify the current macro regime, then rotates
    the portfolio into the regime-optimal asset class:
      - Recession → bonds (defensive)
      - Recovery → stocks (growth)
      - Overheating → commodities (inflation hedge)
      - Stagflation → cash (capital preservation)
    """

    config: MacroRotationConfig = Field(
        default_factory=lambda: MacroRotationConfig(name="macro_rotation"),
        description="Macro rotation strategy configuration",
    )

    def generate_signals(
        self,
        dt: date,  # noqa: ARG002
        market_data: dict[str, pl.DataFrame],
        portfolio: object = None,  # noqa: ARG002
    ) -> list[dict[str, float | int | str]]:
        """Generate trading signals based on current macro regime.

        Reads PMI and CPI from the 'macro' key in market_data, classifies
        the regime, and emits a BUY signal for the regime-optimal fund.
        All other configured funds get TRIM signals.

        Args:
            dt: Trading date.
            market_data: Must contain 'macro' key with pmi/cpi columns,
                         plus DataFrames for each configured fund.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts.

        """
        if "macro" not in market_data:
            return []

        macro_df = market_data["macro"]
        if macro_df.is_empty() or "pmi" not in macro_df.columns or "cpi" not in macro_df.columns:
            return []

        pmi = float(macro_df["pmi"][0])
        cpi = float(macro_df["cpi"][0])

        regime = self.classify_regime(pmi, cpi)
        target_attr = _REGIME_ASSET_MAP[regime]
        target_fund: str = getattr(self.config, target_attr)

        all_funds = [
            self.config.bond_fund,
            self.config.stock_fund,
            self.config.commodity_fund,
            self.config.cash_fund,
        ]

        signals: list[dict[str, float | int | str]] = []
        regime_label = _REGIME_LABELS[regime]
        max_weight = self.config.max_position_pct

        for fund_code in all_funds:
            if fund_code not in market_data or market_data[fund_code].is_empty():
                continue

            if fund_code == target_fund:
                signals.append({
                    "fund_code": fund_code,
                    "direction": SignalDirection.BUY.value,
                    "confidence": 0.85,
                    "target_weight": max_weight,
                    "reason": (
                        f"{regime_label} regime (PMI={pmi:.1f} CPI={cpi:.1f})"
                        f" → rotate to {target_attr.replace('_fund', '')}"
                    ),
                })
            else:
                signals.append({
                    "fund_code": fund_code,
                    "direction": SignalDirection.TRIM.value,
                    "confidence": 0.70,
                    "target_weight": 0.0,
                    "reason": f"Not optimal for {regime_label} regime",
                })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields."""
        return _REQUIRED_FIELDS

    def classify_regime(self, pmi: float, cpi: float) -> MacroRegime:
        """Classify macro regime from PMI and CPI values.

        Args:
            pmi: Purchasing Managers Index value.
            cpi: Consumer Price Index year-over-year change.

        Returns:
            The corresponding MacroRegime.

        """
        if pmi >= self.config.pmi_threshold:
            if cpi >= self.config.cpi_threshold:
                return MacroRegime.OVERHEATING
            return MacroRegime.RECOVERY
        if cpi >= self.config.cpi_threshold:
            return MacroRegime.STAGFLATION
        return MacroRegime.RECESSION
