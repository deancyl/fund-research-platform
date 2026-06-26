"""Factor rotation strategies: macro 4-regime, ETF low-vol, style rotation, spring festival."""

from src.core.strategy.factor_rotation.low_vol_rotation import LowVolRotation
from src.core.strategy.factor_rotation.macro_rotation import (
    MacroRegime,
    MacroRotation,
    MacroRotationConfig,
)
from src.core.strategy.factor_rotation.spring_festival import (
    SpringFestival,
    SpringFestivalConfig,
)

__all__: list[str] = [
    "LowVolRotation",
    "MacroRegime",
    "MacroRotation",
    "MacroRotationConfig",
    "SpringFestival",
    "SpringFestivalConfig",
]
