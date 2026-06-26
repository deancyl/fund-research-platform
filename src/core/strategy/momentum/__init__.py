"""Momentum strategy category (5 strategies).

Factor momentum, ETF multi-factor, Dual momentum, Sector rotation, 52-week high.
"""

from src.core.strategy.momentum.dual_momentum import DualMomentum
from src.core.strategy.momentum.etf_momentum import EtfMomentum
from src.core.strategy.momentum.factor_momentum import FactorMomentum
from src.core.strategy.momentum.fiftytwo_week_high import FiftyTwoWeekHigh
from src.core.strategy.momentum.sector_rotation import SectorRotation

__all__ = [
    "DualMomentum",
    "EtfMomentum",
    "FactorMomentum",
    "FiftyTwoWeekHigh",
    "SectorRotation",
]
