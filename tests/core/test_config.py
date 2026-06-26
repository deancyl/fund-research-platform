"""Tests for config loading and validation."""
import pytest
from pydantic import ValidationError

from src.core.config import AppConfig, load_config


class TestAppConfig:
    def test_loads_from_default_yaml(self) -> None:
        config = load_config()
        assert config.data.primary == "akshare"
        assert config.llm.provider == "deepseek"

    def test_risk_defaults(self) -> None:
        config = load_config()
        assert config.risk.max_single_position_pct == 0.20
        assert 0 < config.risk.sharpe_attenuation.beta_high < 1

    def test_strategies_parsed(self) -> None:
        config = load_config()
        assert "factor_momentum" in config.strategies
        s = config.strategies["factor_momentum"]
        assert s["top_n"] == 3

    def test_risk_validation_rejects_invalid_max_position(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(
                risk={"max_single_position_pct": 1.5, "max_portfolio_drawdown_pct": 0.2},
                data={"primary": "akshare"}, llm={"provider": "test"},
            )

    def test_pe_buy_threshold_type(self) -> None:
        config = load_config()
        s = config.strategies["pe_pb_band"]
        assert isinstance(s["pe_buy_threshold"], float)
        assert s["pe_buy_threshold"] == 10.5
