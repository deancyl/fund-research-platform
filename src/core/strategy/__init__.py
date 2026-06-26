"""Layer 4: Strategy library.

20 strategies across 5 categories:
  momentum/         — Factor momentum, ETF multi-factor, Dual momentum, Sector rotation, 52-week high
  mean_reversion/   — PE/PB band, RSI+fixed income, Dividend timing, Bollinger+RSI, Grid trading
  factor_rotation/  — Macro 4-regime, Style rotation, ETF low-vol, Spring festival effect
  china_specific/   — SOE reform, North-bound flow, Limit-up probability
  portfolio/        — 3-layer defensive, Aggressive ETF, AI-enhanced

Plus:
  base.py       — BaseStrategy abstract class with Pydantic frozen config
  switching.py  — Market regime detection → strategy mapping → dynamic allocation
"""

from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

__all__ = [
    "BaseStrategy",
    "MarketRegime",
    "SignalDirection",
    "StrategyConfig",
]
