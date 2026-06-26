"""Momentum strategy category (5 strategies).

Factor momentum, ETF multi-factor, Dual momentum, Sector rotation, 52-week high.
"""

from src.core.strategy.momentum.etf_momentum import EtfMomentum
from src.core.strategy.momentum.factor_momentum import FactorMomentum

__all__ = ["EtfMomentum", "FactorMomentum"]
