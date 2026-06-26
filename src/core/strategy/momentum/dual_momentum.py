"""
Dual Momentum Strategy (S3 GEM) — Gary Antonacci's Global Equity Momentum.

Absolute momentum: compare each asset's trailing return against a bond benchmark.
If no equity outperforms the bond, rotate to bonds (defensive posture).
Relative momentum: when in equities, select the single best-performing index.

Key parameters:
  - lookback_months: trailing window for momentum calculation (default: 12)
  - bond_fund: defensive fund code used as absolute momentum threshold (default: "511260")

Performance:
  - Yearly excess: +440bp vs benchmark
  - Bear market avg: +3.6% vs S&P -37%
  - Max drawdown protection via absolute momentum defence

Eligible regimes: ALL (built-in defensive component works in any regime).
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_TRADING_DAYS_PER_MONTH: int = 21
_REQUIRED_FIELDS: list[str] = ["close"]


# ─── Configuration ────────────────────────────────────────────────────────────


class DualMomentumConfig(StrategyConfig):
    """Configuration for Dual Momentum strategy (S3 GEM).

    Extends StrategyConfig with lookback window and bond fund identifier.
    """

    lookback_months: int = Field(
        default=12,
        ge=1,
        description="Trailing lookback window in months for momentum calculation",
    )
    bond_fund: str = Field(
        default="511260",
        min_length=1,
        description="Defensive bond fund code for absolute momentum threshold",
    )

    @model_validator(mode="after")
    def _check_bond_fund_not_empty(self) -> "DualMomentumConfig":
        """Ensure bond_fund is a non-empty fund code."""
        if not self.bond_fund.strip():
            raise ValueError("bond_fund must be a non-empty fund code")
        return self


# ─── Strategy ─────────────────────────────────────────────────────────────────


class DualMomentum(BaseStrategy):
    """Dual Momentum (GEM) strategy — absolute + relative momentum.

    Algorithm per generate_signals():
      1. Compute trailing return for each asset over lookback_months window.
      2. Separate bond fund from equity assets.
      3. Absolute momentum check: does the best equity outperform the bond?
      4. If yes → BUY the best equity, TRIM all others.
      5. If no → BUY the bond fund (defensive posture), TRIM all equities.
    """

    config: DualMomentumConfig = Field(
        default_factory=lambda: DualMomentumConfig(name="dual_momentum"),
        description="Dual momentum strategy configuration",
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
            market_data: Dict of {fund_code: DataFrame} with close prices.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts: at most one BUY, remainder HOLD/TRIM.
        """
        if not market_data:
            return []

        lookback_days = self.config.lookback_months * _TRADING_DAYS_PER_MONTH

        # Compute trailing return for each valid asset
        returns: dict[str, float] = {}
        for code, df in market_data.items():
            if not self._has_close_column(df) or len(df) < lookback_days:
                continue
            ret = self._compute_trailing_return(df, lookback_days)
            returns[code] = ret

        if not returns:
            return []

        bond_code = self.config.bond_fund
        bond_return = returns.get(bond_code, -np.inf)

        # Identify equities (all assets except the bond fund)
        equity_returns = {k: v for k, v in returns.items() if k != bond_code}

        if not equity_returns:
            # Only bond(s) exist: buy the best bond
            best_code = max(returns, key=lambda k: returns[k])
            return self._emit_signals(best_code, returns, top_return=returns[best_code])

        # Find best equity
        best_equity_code = max(equity_returns, key=lambda k: equity_returns[k])
        best_equity_return = equity_returns[best_equity_code]

        # Absolute momentum: if best equity beats bond, stay in equities
        if best_equity_return > bond_return:
            selected_code = best_equity_code
        else:
            # Rotate to bonds
            selected_code = bond_code if bond_code in returns else best_equity_code

        return self._emit_signals(selected_code, returns, top_return=returns.get(selected_code, 0.0))

    def required_data(self) -> list[str]:
        """Declare required data fields for the data layer to prefetch."""
        return _REQUIRED_FIELDS

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors: list[str] = []
        cfg = self.config
        if cfg.lookback_months < 1:
            errors.append("lookback_months must be >= 1")
        if not cfg.bond_fund.strip():
            errors.append("bond_fund must not be empty")
        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")
        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_close_column(df: pl.DataFrame) -> bool:
        """Check that DataFrame has the close column."""
        return "close" in df.columns

    @staticmethod
    def _compute_trailing_return(df: pl.DataFrame, lookback_days: int) -> float:
        """Compute trailing return over `lookback_days` trading days.

        Returns:
            Percentage return as a decimal (e.g., 0.15 for 15%).
            Returns -inf for invalid data to ensure it never wins comparison.
        """
        close = df["close"].to_numpy()
        n = len(close)
        if n < lookback_days:
            return -np.inf

        price_now = float(close[-1])
        price_then = float(close[n - lookback_days])

        if np.isnan(price_now) or np.isnan(price_then):
            return -np.inf
        if price_then <= 0:
            return -np.inf

        return (price_now - price_then) / price_then

    def _emit_signals(
        self,
        selected_code: str,
        all_returns: dict[str, float],
        top_return: float,
    ) -> list[dict[str, float | int | str]]:
        """Build signal list: one BUY for selected, TRIM/HOLD for others.

        Args:
            selected_code: The fund to BUY.
            all_returns: All computed returns for confidence scaling.
            top_return: Return of the selected fund.

        Returns:
            Signal dicts sorted by return descending.
        """
        signals: list[dict[str, float | int | str]] = []
        bond_code = self.config.bond_fund

        for code, ret in all_returns.items():
            # Confidence: normalized relative to best
            if len(all_returns) > 1:
                ret_min = min(all_returns.values())
                ret_max = max(all_returns.values())
                ret_range = ret_max - ret_min if ret_max != ret_min else 1.0
                confidence = float(np.clip((ret - ret_min) / ret_range, 0.0, 1.0))
            else:
                confidence = 0.8

            if code == selected_code:
                direction = SignalDirection.BUY.value
                target_weight = min(float(self.config.max_position_pct), 1.0)
                if code == bond_code:
                    reason = (
                        f"Absolute momentum defence — bond beats equities "
                        f"(bond_ret={all_returns.get(bond_code, 0):.4f})"
                    )
                else:
                    reason = (
                        f"Relative momentum winner — "
                        f"trailing return={top_return:.4f}, "
                        f"beats bond ({all_returns.get(bond_code, 0):.4f})"
                    )
            elif ret > 0:
                direction = SignalDirection.HOLD.value
                target_weight = 0.0
                reason = f"Positive momentum ({ret:.4f}) but not selected — {selected_code} is stronger"
            else:
                direction = SignalDirection.TRIM.value
                target_weight = 0.0
                reason = f"Negative momentum ({ret:.4f}) — rotate away"

            signals.append({
                "fund_code": code,
                "direction": direction,
                "confidence": confidence,
                "target_weight": target_weight,
                "reason": reason,
            })

        # Sort by return descending for consistent ordering
        signals.sort(key=lambda s: all_returns[s["fund_code"]], reverse=True)
        return signals
