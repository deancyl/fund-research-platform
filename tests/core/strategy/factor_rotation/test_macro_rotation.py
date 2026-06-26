"""
Tests for Macro 4-Regime Rotation strategy (S11).

4 regimes driven by PMI + CPI thresholds:
  衰退(Recession): PMI < 50, CPI < 3.0 → bonds
  复苏(Recovery): PMI >= 50, CPI < 3.0 → stocks
  过热(Overheating): PMI >= 50, CPI >= 3.0 → commodities
  滞涨(Stagflation): PMI < 50, CPI >= 3.0 → cash

Annual excess: +16.1%, Info Ratio 1.78, Monthly win: 66%.
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.factor_rotation.macro_rotation import (
    MacroRegime,
    MacroRotation,
    MacroRotationConfig,
)


@pytest.fixture
def strategy() -> MacroRotation:
    config = MacroRotationConfig(
        name="macro_rotation",
        version="1.0.0",
        eligible_regimes={r for r in MarketRegime},
        max_position_pct=0.30,
        min_holding_days=1,
        pmi_threshold=50.0,
        cpi_threshold=3.0,
        bond_fund="511010",
        stock_fund="510300",
        commodity_fund="159980",
        cash_fund="511880",
    )
    return MacroRotation(config=config)


@pytest.fixture
def test_date() -> date:
    return date(2026, 6, 15)


def _make_macro_df(pmi: float, cpi: float) -> dict[str, pl.DataFrame]:
    return {
        "macro": pl.DataFrame([{"date": date(2026, 6, 15), "pmi": pmi, "cpi": cpi}]),
        "511010": pl.DataFrame([{"date": date(2026, 6, 15), "close": 102.5}]),
        "510300": pl.DataFrame([{"date": date(2026, 6, 15), "close": 4.20}]),
        "159980": pl.DataFrame([{"date": date(2026, 6, 15), "close": 1.85}]),
        "511880": pl.DataFrame([{"date": date(2026, 6, 15), "close": 100.05}]),
    }


@pytest.fixture
def recession_data() -> dict[str, pl.DataFrame]:
    return _make_macro_df(48.5, 1.5)


@pytest.fixture
def recovery_data() -> dict[str, pl.DataFrame]:
    return _make_macro_df(52.0, 2.0)


@pytest.fixture
def overheating_data() -> dict[str, pl.DataFrame]:
    return _make_macro_df(54.0, 4.5)


@pytest.fixture
def stagflation_data() -> dict[str, pl.DataFrame]:
    return _make_macro_df(47.0, 5.0)


class TestMacroRotationConfig:
    def test_default_config(self) -> None:
        config = MacroRotationConfig(name="macro_rotation")
        assert config.pmi_threshold == 50.0
        assert config.cpi_threshold == 3.0
        assert config.bond_fund == "511010"
        assert config.stock_fund == "510300"
        assert config.commodity_fund == "159980"
        assert config.cash_fund == "511880"
        assert config.max_position_pct == 0.20

    def test_rejects_invalid_pmi_threshold(self) -> None:
        with pytest.raises(ValueError):
            MacroRotationConfig(name="macro_rotation", pmi_threshold=-1.0)

    def test_rejects_invalid_cpi_threshold(self) -> None:
        with pytest.raises(ValueError):
            MacroRotationConfig(name="macro_rotation", cpi_threshold=-0.5)

    def test_inherits_strategy_config(self) -> None:
        config = MacroRotationConfig(name="macro_rotation")
        assert isinstance(config, StrategyConfig)


class TestMacroRegimeClassification:
    def test_recession(self, strategy: MacroRotation) -> None:
        assert strategy.classify_regime(48.5, 1.5) == MacroRegime.RECESSION

    def test_recovery(self, strategy: MacroRotation) -> None:
        assert strategy.classify_regime(52.0, 2.0) == MacroRegime.RECOVERY

    def test_overheating(self, strategy: MacroRotation) -> None:
        assert strategy.classify_regime(54.0, 4.5) == MacroRegime.OVERHEATING

    def test_stagflation(self, strategy: MacroRotation) -> None:
        assert strategy.classify_regime(47.0, 5.0) == MacroRegime.STAGFLATION

    def test_boundary_pmi_at_threshold(self) -> None:
        config = MacroRotationConfig(name="macro_rotation", pmi_threshold=50.0, cpi_threshold=3.0)
        s = MacroRotation(config=config)
        assert s.classify_regime(50.0, 2.0) == MacroRegime.RECOVERY

    def test_boundary_cpi_at_threshold(self) -> None:
        config = MacroRotationConfig(name="macro_rotation", pmi_threshold=50.0, cpi_threshold=3.0)
        s = MacroRotation(config=config)
        assert s.classify_regime(52.0, 3.0) == MacroRegime.OVERHEATING

    def test_custom_thresholds(self) -> None:
        config = MacroRotationConfig(name="macro_rotation", pmi_threshold=48.0, cpi_threshold=2.5)
        s = MacroRotation(config=config)
        assert s.classify_regime(49.0, 2.0) == MacroRegime.RECOVERY
        assert s.classify_regime(49.0, 3.0) == MacroRegime.OVERHEATING


class TestRegimeEligibility:
    def test_eligible_in_all_regimes(self, strategy: MacroRotation) -> None:
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True


class TestRequiredData:
    def test_returns_required_fields(self, strategy: MacroRotation) -> None:
        fields = strategy.required_data()
        assert "pmi" in fields
        assert "cpi" in fields


class TestGenerateSignalsHappyPath:
    def test_recession_bond_buy(self, strategy: MacroRotation, test_date: date, recession_data: dict[str, pl.DataFrame]) -> None:
        signals = strategy.generate_signals(test_date, recession_data)
        buy = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert len(buy) >= 1
        bond = [s for s in buy if s["fund_code"] == "511010"]
        assert len(bond) == 1
        assert bond[0]["target_weight"] > 0
        assert "衰退" in str(bond[0]["reason"])

    def test_recovery_stock_buy(self, strategy: MacroRotation, test_date: date, recovery_data: dict[str, pl.DataFrame]) -> None:
        signals = strategy.generate_signals(test_date, recovery_data)
        buy = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        stock = [s for s in buy if s["fund_code"] == "510300"]
        assert len(stock) == 1

    def test_overheating_commodity_buy(self, strategy: MacroRotation, test_date: date, overheating_data: dict[str, pl.DataFrame]) -> None:
        signals = strategy.generate_signals(test_date, overheating_data)
        buy = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        commodity = [s for s in buy if s["fund_code"] == "159980"]
        assert len(commodity) == 1

    def test_stagflation_cash_buy(self, strategy: MacroRotation, test_date: date, stagflation_data: dict[str, pl.DataFrame]) -> None:
        signals = strategy.generate_signals(test_date, stagflation_data)
        buy = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        cash = [s for s in buy if s["fund_code"] == "511880"]
        assert len(cash) == 1

    def test_signal_structure(self, strategy: MacroRotation, test_date: date, recovery_data: dict[str, pl.DataFrame]) -> None:
        signals = strategy.generate_signals(test_date, recovery_data)
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in {e.value for e in SignalDirection}
            assert isinstance(s["confidence"], float)
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], float)
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0


class TestGenerateSignalsEdgeCases:
    def test_empty_data(self, strategy: MacroRotation, test_date: date) -> None:
        signals = strategy.generate_signals(test_date, {})
        assert signals == []

    def test_missing_macro_key(self, strategy: MacroRotation, test_date: date) -> None:
        data = {"511010": pl.DataFrame([{"date": test_date, "close": 102.5}])}
        signals = strategy.generate_signals(test_date, data)
        assert signals == []

    def test_missing_pmi_column(self, strategy: MacroRotation, test_date: date) -> None:
        data = {"macro": pl.DataFrame([{"date": test_date, "cpi": 2.0}])}
        signals = strategy.generate_signals(test_date, data)
        assert signals == []


class TestStrategyIdentity:
    def test_name(self, strategy: MacroRotation) -> None:
        assert strategy.name == "macro_rotation"

    def test_validate_returns_empty(self, strategy: MacroRotation) -> None:
        assert strategy.validate() == []
