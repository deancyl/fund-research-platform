"""Grid Trading + Hurst Veto Strategy (S10) — 网格交易 + Hurst 一票否决.

Target: CSI 300 ETF (510300), range-bound (SIDEWAYS) markets.

Algorithm:
  1. Compute Hurst exponent via R/S analysis on close prices.
  2. Hurst >= veto_threshold (default 0.6) → VETO: grid blocked entirely.
  3. Hurst < 0.4 → full grid: BUY signals at each grid layer below current price.
  4. 0.4 <= Hurst < veto_threshold → scaled partial grid with reduced confidence.

Grid mechanics:
  - grid_spacing_pct: vertical distance between layers (default 5%).
  - grid_layers: number of buy levels below current price (default 5).
  - Each layer receives equal position allocation.
"""

from __future__ import annotations

from datetime import date  # noqa: TC003 — runtime parameter type

import numpy as np
import polars as pl  # noqa: TC002 — runtime DataFrame access

from src.core.strategy.base import (
    BaseStrategy,
    MarketRegime,
    SignalDirection,
    StrategyConfig,
)

# ── Constants ────────────────────────────────────────────────────────────────

_MIN_DATA_POINTS: int = 30
_HURST_FULL_GRID_THRESHOLD: float = 0.4
_CONFIDENCE_FLOOR: float = 0.05
_EPSILON: float = 1e-12
_MIN_RETURNS: int = 20
_MIN_WINDOW_EXPONENT: int = 3
_MIN_CHUNKS: int = 2
_MIN_RS_POINTS: int = 4
_MIN_WINDOW: int = 8
_RS_WINDOW_DIVISOR: int = 40
_MAX_WINDOWS: int = 15
_NEUTRAL_HURST: float = 0.5
_CONFIDENCE_LAYER_DECAY: float = 0.12


class GridHurstStrategy(BaseStrategy):
    """Grid trading strategy gated by Hurst exponent mean-reversion check.

    Only generates signals in SIDEWAYS regime. A high Hurst exponent
    (>= veto_threshold) indicates trending behavior — grid trading is
    inappropriate, and signals are suppressed (one-vote rejection).

    Attributes:
        grid_spacing_pct: Price distance between adjacent grid layers.
        grid_layers: Number of buy levels below the current price.
        hurst_veto_threshold: Hurst value above which all grid signals are vetoed.

    """

    grid_spacing_pct: float = 0.05
    grid_layers: int = 5
    hurst_veto_threshold: float = 0.6

    config: StrategyConfig = StrategyConfig(
        name="grid_hurst",
        version="1.0.0",
        eligible_regimes={MarketRegime.SIDEWAYS},
        max_position_pct=0.20,
        min_holding_days=1,
    )

    # ── Abstract Interface ───────────────────────────────────────────────

    def generate_signals(
        self,
        dt: date,  # noqa: ARG002 — reserved for future signal stamping
        market_data: dict[str, pl.DataFrame],
        portfolio: list | None = None,  # noqa: ARG002
    ) -> list[dict[str, float | int | str]]:
        """Generate grid-trading signals per fund, gated by Hurst exponent.

        For each fund in market_data:
          1. Extract close prices from the DataFrame.
          2. Compute R/S Hurst exponent.
          3. If Hurst >= veto_threshold → emit HOLD (veto).
          4. If Hurst < 0.4 → full grid: BUY at every layer.
          5. Otherwise → scaled partial grid.

        Returns:
            List of signal dicts with keys:
              fund_code, direction, confidence, target_weight, reason.

        """
        if not market_data:
            return []

        signals: list[dict[str, float | int | str]] = []

        for fund_code, df in market_data.items():
            close = df["close"].to_numpy()
            if len(close) < _MIN_DATA_POINTS:
                # Insufficient data for reliable Hurst estimation
                continue

            hurst = self._hurst_rs(close)

            if hurst >= self.hurst_veto_threshold:
                signals.append({
                    "fund_code": fund_code,
                    "direction": SignalDirection.HOLD,
                    "confidence": 0.0,
                    "target_weight": 0.0,
                    "reason": (
                        f"Hurst={hurst:.3f} ≥ veto={self.hurst_veto_threshold}"
                        " — trending, grid blocked"
                    ),
                })
                continue

            last_price = float(close[-1])

            if hurst < _HURST_FULL_GRID_THRESHOLD:
                # Full grid — strong mean reversion
                active_layers = self.grid_layers
                confidence_base = 1.0
            else:
                # Partial grid — moderate mean reversion
                veto_range = self.hurst_veto_threshold - _HURST_FULL_GRID_THRESHOLD
                active_layers = max(
                    1, int(self.grid_layers * (self.hurst_veto_threshold - hurst) / veto_range),
                )
                confidence_base = (self.hurst_veto_threshold - hurst) / veto_range

            weight_per_layer = 1.0 / active_layers

            for layer in range(active_layers):
                discount = self.grid_spacing_pct * (layer + 1)
                target_price = last_price * (1.0 - discount)
                confidence = confidence_base * (1.0 - layer * _CONFIDENCE_LAYER_DECAY)
                signals.append({
                    "fund_code": fund_code,
                    "direction": SignalDirection.BUY,
                    "confidence": round(max(confidence, _CONFIDENCE_FLOOR), 4),
                    "target_weight": round(weight_per_layer, 4),
                    "reason": (
                        f"Grid L{layer + 1}/{active_layers} "
                        f"@ {target_price:.3f} "
                        f"(Hurst={hurst:.3f})"
                    ),
                })

        return signals

    def required_data(self) -> list[str]:
        """Declare required data fields."""
        return ["close"]

    def validate(self) -> list[str]:
        """Validate strategy parameters."""
        errors: list[str] = []
        if self.grid_spacing_pct <= 0.0:
            errors.append(
                f"grid_spacing_pct must be > 0, got {self.grid_spacing_pct}",
            )
        if self.grid_layers < 1:
            errors.append(
                f"grid_layers must be >= 1, got {self.grid_layers}",
            )
        if not (0.0 < self.hurst_veto_threshold <= 1.0):
            errors.append(
                f"hurst_veto_threshold must be in (0, 1], got {self.hurst_veto_threshold}",
            )
        return errors

    # ── Hurst Exponent (R/S Analysis) ────────────────────────────────────

    @staticmethod
    def _hurst_rs(series: np.ndarray) -> float:
        """Estimate Hurst exponent via Rescaled Range (R/S) analysis.

        Algorithm:
          1. Compute log returns from the price series.
          2. For each window size w, partition the series into m = n//w chunks.
          3. For each chunk compute R = max(cumdev) - min(cumdev), S = std.
          4. Average R/S across chunks, fit log(R/S) = H * log(w) + C.

        Args:
            series: 1-D numpy array of close prices.

        Returns:
            Hurst exponent clamped to [0.0, 1.0].

        """
        # Use log returns for stationarity
        with np.errstate(divide="ignore", invalid="ignore"):
            log_prices = np.log(np.maximum(series, _EPSILON))
        returns = np.diff(log_prices)
        n_returns = len(returns)

        if n_returns < _MIN_RETURNS:
            return _NEUTRAL_HURST

        # Generate logarithmically spaced window sizes
        min_window = max(_MIN_WINDOW, n_returns // _RS_WINDOW_DIVISOR)
        max_exponent = int(np.log2(n_returns // 2))
        if max_exponent < _MIN_WINDOW_EXPONENT:
            return _NEUTRAL_HURST

        window_sizes = np.unique(
            np.logspace(
                np.log2(min_window),
                max_exponent,
                num=min(_MAX_WINDOWS, max_exponent),
                base=2,
            ).astype(int),
        )
        window_sizes = window_sizes[window_sizes >= min_window]

        rs_points: list[tuple[float, float]] = []

        for w in window_sizes:
            m = n_returns // w
            if m < _MIN_CHUNKS:
                continue
            chunk_rs: list[float] = []
            for i in range(m):
                chunk = returns[i * w : (i + 1) * w]
                chunk_mean = float(np.mean(chunk))
                cumdev = np.cumsum(chunk - chunk_mean)
                r_val = float(np.max(cumdev) - np.min(cumdev))
                s_val = float(np.std(chunk, ddof=1))
                if s_val > _EPSILON and r_val > _EPSILON:
                    chunk_rs.append(r_val / s_val)
            if len(chunk_rs) >= _MIN_CHUNKS:
                rs_points.append((np.log(float(w)), np.log(float(np.mean(chunk_rs)))))

        if len(rs_points) < _MIN_RS_POINTS:
            return _NEUTRAL_HURST

        xs = np.array([p[0] for p in rs_points], dtype=np.float64)
        ys = np.array([p[1] for p in rs_points], dtype=np.float64)

        # OLS slope: H = cov(x,y) / var(x)
        slope = np.polyfit(xs, ys, 1)[0]

        return float(np.clip(slope, 0.0, 1.0))
