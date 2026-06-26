"""
Configuration loader — reads config.yaml via pydantic-settings.

All runtime parameters flow through this module. No hardcoded values in
any other module — they either accept params or read from AppConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─── Sub-models ─────────────────────────────────────────────────────────────


class DataConfig(BaseModel, frozen=True):
    primary: str = "akshare"
    fallback: str = "eastmoney"
    cache_dir: str = ".data"
    max_age_hours: int = 24


class LLMConfig(BaseModel, frozen=True):
    provider: str = "deepseek"
    api_base: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1)
    fallback: str = "ollama"


class SharpeAttenuation(BaseModel, frozen=True):
    beta_low: float = Field(default=0.90, ge=0, le=1)     # SR < 0.5
    beta_mid: float = Field(default=0.70, ge=0, le=1)     # 0.5-1.0
    beta_high: float = Field(default=0.50, ge=0, le=1)    # 1.0-1.5
    beta_extreme: float = Field(default=0.25, ge=0, le=1)  # > 2.0


class RiskConfig(BaseModel, frozen=True):
    max_single_position_pct: float = Field(default=0.20, ge=0, le=1)
    max_portfolio_drawdown_pct: float = Field(default=0.20, ge=0, le=1)
    max_strategy_drawdown_pct: float = Field(default=0.15, ge=0, le=1)
    sharpe_attenuation: SharpeAttenuation = Field(default_factory=SharpeAttenuation)


class RegimeConfig(BaseModel, frozen=True):
    hmm_states: int = Field(default=3, ge=2, le=10)
    hmm_features: list[str] = Field(default_factory=lambda: ["return_1d", "volatility_20d"])
    recheck_interval_days: int = 7


class TUIConfig(BaseModel, frozen=True):
    theme: str = "dark"
    refresh_interval_seconds: int = 5
    log_max_lines: int = 100
    colors: dict[str, str] = Field(default_factory=lambda: {"up": "red", "down": "green"})


class WebConfig(BaseModel, frozen=True):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


# ─── Root Config ────────────────────────────────────────────────────────────


class AppConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")

    data: DataConfig = Field(default_factory=DataConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    strategies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tui: TUIConfig = Field(default_factory=TUIConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @model_validator(mode="after")
    def _validate_risk_consistency(self) -> "AppConfig":
        """Risk thresholds must be consistent."""
        if self.risk.max_strategy_drawdown_pct > self.risk.max_portfolio_drawdown_pct:
            raise ValueError("Strategy drawdown limit must be <= portfolio limit")
        return self


# ─── Loader ──────────────────────────────────────────────────────────────────


_config: AppConfig | None = None
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML file. Cached after first load."""
    global _config
    if _config is not None and path is None:
        return _config

    target = Path(path) if path else _CONFIG_PATH
    raw: dict[str, Any] = {}
    if target.exists():
        with open(target, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # Map YAML keys to sub-models
    parsed: dict[str, Any] = {}
    if "data" in raw:
        parsed["data"] = DataConfig(**raw["data"])
    if "llm" in raw:
        parsed["llm"] = LLMConfig(**raw["llm"])
    if "risk" in raw:
        raw_risk = raw["risk"]
        if "sharpe_attenuation" in raw_risk:
            raw_risk["sharpe_attenuation"] = SharpeAttenuation(**raw_risk["sharpe_attenuation"])
        parsed["risk"] = RiskConfig(**raw_risk)
    if "regime" in raw:
        parsed["regime"] = RegimeConfig(**raw["regime"])
    if "strategies" in raw:
        parsed["strategies"] = raw["strategies"]
    if "tui" in raw:
        parsed["tui"] = TUIConfig(**raw["tui"])
    if "web" in raw:
        parsed["web"] = WebConfig(**raw["web"])

    _config = AppConfig(**parsed)
    return _config
