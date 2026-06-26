"""Single source of truth — reads pyproject.toml directly, no package reinstall needed."""

import tomllib
from pathlib import Path

_pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
with open(_pyproject, "rb") as _f:
    VERSION: str = tomllib.load(_f)["project"]["version"]
