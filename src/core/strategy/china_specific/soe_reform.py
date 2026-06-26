"""
SOE Reform Strategy (S15) — "中特估" composite scoring for China SOE stocks.

Strategy: Build a composite score from dividend yield, north-bound capital flow,
and institutional flow to identify undervalued SOE (State-Owned Enterprise) stocks.
Select top-N stocks for investment.

"中特估" (zhōng tè gū) = China Special Valuation — a policy framework emphasizing
the re-rating of SOE stocks based on their strategic value, dividend reliability,
and capital flow characteristics.

Key parameters:
  - top_n: number of stocks to select (default: 3)
  - dividend_weight: weight for dividend yield component (default: 0.4)
  - flow_weight: weight for combined flow component (default: 0.6)

Composite score formula:
  score = dividend_weight * dividend_yield_norm
        + flow_weight * (north_bound_flow + institutional_flow) / 2

Eligible regimes: ALL (designed for long-term value regardless of market state).

Required data: ["dividend_yield", "north_bound_flow", "institutional_flow"]
"""

from datetime import date

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

# ─── Constants ─────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS: list[str] = ["dividend_yield", "north_bound_flow", "institutional_flow"]
_DIVIDEND_YIELD_CAP: float = 0.15  # cap dividend yield at 15% for normalization


# ─── Configuration ────────────────────────────────────────────────────────────


class SoeReformConfig(StrategyConfig):
    """Configuration for SOE Reform strategy (S15).

    Extends StrategyConfig with SOE-specific scoring weights and selection count.
    """

    top_n: int = Field(
        default=3,
        ge=1,
        description="Number of top-ranked SOE stocks to select for BUY signals",
    )
    dividend_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for dividend yield component in composite score",
    )
    flow_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for combined flow component (north-bound + institutional) in composite score",
    )
    max_position_pct: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum position as fraction of portfolio per stock",
    )

    @model_validator(mode="after")
    def _check_weights_sum(self) -> "SoeReformConfig":
        """Ensure dividend_weight + flow_weight approximate 1.0."""
        total = self.dividend_weight + self.flow_weight
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"dividend_weight + flow_weight must sum to ~1.0, got {total}"
            )
        return self


# ─── Strategy ─────────────────────────────────────────────────────────────────


class SoeReform(BaseStrategy):
    """SOE Reform strategy — composite scoring of China SOE stocks.

    Algorithm per generate_signals():
      1. Filter stocks with all three required columns.
      2. For each stock, extract latest row values for dividend_yield,
         north_bound_flow, and institutional_flow.
      3. Normalize dividend yield to [0, 1] using _DIVIDEND_YIELD_CAP.
      4. Compute composite score:
           score = dividend_weight * div_norm + flow_weight * (nb_flow + inst_flow) / 2
      5. Rank stocks by composite score descending.
      6. Emit BUY for top_n stocks, HOLD for others with positive scores, TRIM for negative.
    """

    config: SoeReformConfig = Field(
        default_factory=lambda: SoeReformConfig(name="soe_reform"),
        description="SOE reform strategy configuration",
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
            market_data: Dict of {stock_code: DataFrame} with dividend_yield,
                         north_bound_flow, and institutional_flow columns.
            portfolio: Current portfolio (unused).

        Returns:
            List of signal dicts sorted by composite score descending.
        """
        if not market_data:
            return []

        # 1. Collect latest values for cross-sectional ranking
        fund_codes: list[str] = []
        fund_scores: list[dict[str, object]] = []
        all_nb: list[float] = []
        all_inst: list[float] = []

        for code, df in market_data.items():
            if not self._has_required_columns(df):
                continue
            row = df.row(-1, named=True)
            fund_codes.append(code)
            all_nb.append(float(row["north_bound_flow"]))
            all_inst.append(float(row["institutional_flow"]))
            fund_scores.append({"code": code, "df": df, "div_yield": float(row["dividend_yield"])})

        if not fund_codes:
            return []

        # 2. Cross-sectional percentile rank (v0.2.5 audit fix)
        nb_ranks = _percentile_rank(np.array(all_nb, dtype=np.float64))
        inst_ranks = _percentile_rank(np.array(all_inst, dtype=np.float64))

        # 3. Compute composite scores with cross-sectional ranks
        scores: list[tuple[str, float]] = []
        for i, fs in enumerate(fund_scores):
            composite = self._compute_composite_score(
                fs["df"],  # type: ignore[arg-type]
                dividend_yield=float(fs["div_yield"]),
                flow_rank=(nb_ranks[i] + inst_ranks[i]) / 2.0,
            )
            scores.append((str(fs["code"]), composite))

        if not scores:
            return []

        # 2. Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Emit signals
        signals: list[dict[str, float | int | str]] = []
        top_n = self.config.top_n
        base_weight = 1.0 / top_n

        for rank, (code, composite) in enumerate(scores):
            # Confidence: normalized score range
            if len(scores) > 1:
                score_min = scores[-1][1]
                score_max = scores[0][1]
                score_range = score_max - score_min if score_max != score_min else 1.0
                confidence = float(np.clip((composite - score_min) / score_range, 0.0, 1.0))
            else:
                confidence = 0.7

            if rank < top_n:
                direction = SignalDirection.BUY.value
                target_weight = base_weight
                reason = (
                    f"Rank #{rank + 1}/{len(scores)} — "
                    f"composite score={composite:.4f}, "
                    f"belongs in top {top_n}"
                )
            elif composite > 0:
                direction = SignalDirection.HOLD.value
                target_weight = 0.0
                reason = (
                    f"Rank #{rank + 1}/{len(scores)} — "
                    f"positive score ({composite:.4f}) but outside top {top_n}"
                )
            else:
                direction = SignalDirection.TRIM.value
                target_weight = 0.0
                reason = (
                    f"Rank #{rank + 1}/{len(scores)} — "
                    f"negative score ({composite:.4f})"
                )

            signals.append({
                "fund_code": code,
                "direction": direction,
                "confidence": confidence,
                "target_weight": target_weight,
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
        if cfg.top_n < 1:
            errors.append("top_n must be >= 1")
        if not (0.0 <= cfg.dividend_weight <= 1.0):
            errors.append("dividend_weight must be in [0, 1]")
        if not (0.0 <= cfg.flow_weight <= 1.0):
            errors.append("flow_weight must be in [0, 1]")
        total = cfg.dividend_weight + cfg.flow_weight
        if not (0.99 <= total <= 1.01):
            errors.append(
                f"dividend_weight + flow_weight must sum to ~1.0, got {total}"
            )
        if not (0.0 <= cfg.max_position_pct <= 1.0):
            errors.append("max_position_pct must be in [0, 1]")
        return errors

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _has_required_columns(df: pl.DataFrame) -> bool:
        """Check that DataFrame has all three required SOE scoring columns."""
        df_cols = set(df.columns)
        return all(col in df_cols for col in _REQUIRED_FIELDS)

    def _compute_composite_score(
        self, df: pl.DataFrame, dividend_yield: float, flow_rank: float
    ) -> float:
        """
        Compute weighted composite score from cross-sectional ranks.

        【v0.2.5 审计修复】彻底移除时间序列 Expanding Window Z-Score。
        现在由 generate_signals 在横截面上对所有标的进行 Percentile Rank 映射，
        _compute_composite_score 只接收已去量纲化的 flow_rank ∈ [0, 1]。
        消除了 ETF 生命周期错位的异方差灾难。
        """
        div_norm = min(dividend_yield, _DIVIDEND_YIELD_CAP) / _DIVIDEND_YIELD_CAP
        score = self.config.dividend_weight * div_norm + self.config.flow_weight * flow_rank
        return float(np.clip(score, 0.0, 1.0))


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank mapping → [0, 1]. Identical values share the same rank."""
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5, dtype=np.float64)
    # scipy-style: average rank for ties → maps to [0, 1]
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Average ranks for ties
    uniq, inv = np.unique(values, return_inverse=True)
    for u in range(len(uniq)):
        mask = inv == u
        ranks[mask] = np.mean(ranks[mask])
    return ranks / (n - 1)
