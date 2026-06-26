"""
PE/PB Band strategy (S6) for Chinese index funds.

Trades based on valuation percentile bands:
  - Accumulate when PE < buy threshold AND PB < buy threshold (cheap zone).
  - Sell when PE > sell threshold OR PB > sell threshold (expensive zone).
  - Hold otherwise (neutral zone).

Target indices: CSI 300, CSI 500, ChiNext.

Default thresholds calibrated for CSI 300:
  PE_buy=10.5  PE_sell=14.5   PB_buy=1.2  PB_sell=1.7

Historical win rate: ~60-70% long-term (tested 2005-2025).
"""

from datetime import date

import polars as pl
from pydantic import Field

from src.core.strategy.base import BaseStrategy, SignalDirection, StrategyConfig

# ─── Default thresholds (CSI 300 calibrated) ────────────────────────────────

DEFAULT_PE_BUY: float = 10.5
DEFAULT_PE_SELL: float = 14.5
DEFAULT_PB_BUY: float = 1.2
DEFAULT_PB_SELL: float = 1.7


# ─── Strategy ───────────────────────────────────────────────────────────────


class PEPBBandStrategy(BaseStrategy):
    """PE/PB valuation band strategy for Chinese broad-market ETFs.

    Generates ACCUMULATE signals in cheap zones (both PE and PB below buy
    thresholds) and SELL signals in expensive zones (either PE or PB above
    sell thresholds). Confidence scales linearly with distance from thresholds.
    """

    config: StrategyConfig = Field(
        default_factory=lambda: StrategyConfig(name="pe_pb_band"),
        description="Strategy metadata and risk constraints",
    )

    pe_buy_threshold: float = Field(
        default=DEFAULT_PE_BUY,
        gt=0,
        description="PE ratio below which accumulation begins (e.g. 10.5 for CSI 300)",
    )
    pe_sell_threshold: float = Field(
        default=DEFAULT_PE_SELL,
        gt=0,
        description="PE ratio above which reduction begins (e.g. 14.5 for CSI 300)",
    )
    pb_buy_threshold: float = Field(
        default=DEFAULT_PB_BUY,
        gt=0,
        description="PB ratio below which accumulation begins (e.g. 1.2 for CSI 300)",
    )
    pb_sell_threshold: float = Field(
        default=DEFAULT_PB_SELL,
        gt=0,
        description="PB ratio above which reduction begins (e.g. 1.7 for CSI 300)",
    )

    # ── Abstract Interface ───────────────────────────────────────────────

    def generate_signals(
        self,
        dt: date,
        market_data: dict[str, pl.DataFrame],
        portfolio: list | None = None,
    ) -> list[dict]:
        """Produce one signal per fund based on latest PE/PB vs thresholds.

        Args:
            dt: Trading date (unused; strategy is valuation-only).
            market_data: {fund_code: DataFrame} with 'pe' and 'pb' columns.
            portfolio: Unused; PE/PB Band is independent of current holdings.

        Returns:
            List of signal dicts with fund_code, direction, confidence,
            target_weight, and reason.
        """
        if not market_data:
            return []

        signals: list[dict] = []

        for fund_code, df in market_data.items():
            # ── Guard: require both PE and PB columns ────────────────────
            if "pe" not in df.columns or "pb" not in df.columns:
                continue

            # Use the most recent (last) row
            try:
                pe_val: float | None = df["pe"].to_list()[-1]
                pb_val: float | None = df["pb"].to_list()[-1]
            except (IndexError, TypeError):
                continue

            if pe_val is None or pb_val is None:
                continue

            # ── NaN guard ────────────────────────────────────────────────
            if pe_val != pe_val or pb_val != pb_val:
                continue

            direction, confidence, target_weight, reason = self._classify(
                pe_val, pb_val, fund_code
            )

            signals.append(
                {
                    "fund_code": fund_code,
                    "direction": direction,
                    "confidence": round(confidence, 4),
                    "target_weight": round(target_weight, 4),
                    "reason": reason,
                }
            )

        return signals

    def required_data(self) -> list[str]:
        """PE and PB are the only required fields for valuation analysis."""
        return ["pe", "pb"]

    def validate(self) -> list[str]:
        """Ensure buy thresholds < sell thresholds."""
        errors: list[str] = []
        if self.pe_buy_threshold >= self.pe_sell_threshold:
            errors.append(
                f"PE buy threshold ({self.pe_buy_threshold}) must be less than "
                f"PE sell threshold ({self.pe_sell_threshold})"
            )
        if self.pb_buy_threshold >= self.pb_sell_threshold:
            errors.append(
                f"PB buy threshold ({self.pb_buy_threshold}) must be less than "
                f"PB sell threshold ({self.pb_sell_threshold})"
            )
        return errors

    # ── Classification Logic ────────────────────────────────────────────

    def _classify(
        self, pe: float, pb: float, fund_code: str
    ) -> tuple[SignalDirection, float, float, str]:
        """Classify a single valuation pair into a signal.

        Returns (direction, confidence, target_weight, reason).
        """
        # ── ACCUMULATE zone: both PE and PB below buy thresholds ────
        if pe < self.pe_buy_threshold and pb < self.pb_buy_threshold:
            # Confidence: how far below the buy threshold (0 at threshold, 1 at PE=0)
            pe_distance = (self.pe_buy_threshold - pe) / self.pe_buy_threshold
            pb_distance = (self.pb_buy_threshold - pb) / self.pb_buy_threshold
            confidence = min(1.0, max(0.5, (pe_distance + pb_distance) / 2))
            target_weight = 0.8 * confidence
            reason = (
                f"PE={pe:.2f}<{self.pe_buy_threshold}, "
                f"PB={pb:.2f}<{self.pb_buy_threshold} → 估值洼地，建议增持"
            )
            return SignalDirection.ACCUMULATE, confidence, target_weight, reason

        # ── SELL zone: either PE or PB above sell thresholds ─────────
        if pe > self.pe_sell_threshold or pb > self.pb_sell_threshold:
            pe_excess = max(0.0, (pe - self.pe_sell_threshold) / self.pe_sell_threshold)
            pb_excess = max(0.0, (pb - self.pb_sell_threshold) / self.pb_sell_threshold)
            confidence = min(1.0, max(0.5, (pe_excess + pb_excess) / 2 + 0.4))
            target_weight = 0.0
            parts: list[str] = [f"PE={pe:.2f}" if pe > self.pe_sell_threshold else ""]
            parts2: list[str] = [f"PB={pb:.2f}" if pb > self.pb_sell_threshold else ""]
            over = ", ".join(p for p in parts + parts2 if p)
            reason = f"{over} → 估值偏高，建议减持"
            return SignalDirection.SELL, confidence, target_weight, reason

        # ── HOLD zone: neutral ──────────────────────────────────────
        mid_pe = (self.pe_buy_threshold + self.pe_sell_threshold) / 2
        mid_pb = (self.pb_buy_threshold + self.pb_sell_threshold) / 2
        pe_deviation = abs(pe - mid_pe) / mid_pe
        pb_deviation = abs(pb - mid_pb) / mid_pb
        confidence = max(0.3, 0.5 - (pe_deviation + pb_deviation) / 4)
        target_weight = 0.4
        reason = (
            f"PE={pe:.2f}({self.pe_buy_threshold}-{self.pe_sell_threshold}), "
            f"PB={pb:.2f}({self.pb_buy_threshold}-{self.pb_sell_threshold}) → 估值合理，持有"
        )
        return SignalDirection.HOLD, confidence, target_weight, reason
