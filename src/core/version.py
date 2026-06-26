"""Single source of truth — dynamic from pyproject.toml, fallback for dev."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    VERSION = _pkg_version("fund-research-platform")
except PackageNotFoundError:
    VERSION = "0.3.1"
