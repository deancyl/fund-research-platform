"""Factor rotation strategies: macro 4-regime, style rotation, ETF low-vol, spring festival."""

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
    "MacroRegime",
    "MacroRotation",
    "MacroRotationConfig",
    "SpringFestival",
    "SpringFestivalConfig",
]
