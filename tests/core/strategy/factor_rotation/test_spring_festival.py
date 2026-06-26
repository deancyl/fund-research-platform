"""
Tests for Spring Festival Effect strategy (S14).

Buy small-cap/ChiNext before Spring Festival, hold for N trading days.
Win rate: 80% (20yr data), Median return: +9.45%.

Key dates (lunar new year):
  2025-01-29, 2026-02-17, 2027-02-06, 2028-01-26
"""

from datetime import date

import polars as pl
import pytest

from src.core.strategy.base import MarketRegime, SignalDirection, StrategyConfig
from src.core.strategy.factor_rotation.spring_festival import (
    SpringFestival,
    SpringFestivalConfig,
)


@pytest.fixture
def strategy() -> SpringFestival:
    config = SpringFestivalConfig(
        name="spring_festival",
        version="1.0.0",
        eligible_regimes={r for r in MarketRegime},
        max_position_pct=0.25,
        min_holding_days=1,
        hold_days=20,
        pre_days=5,
        target_fund="159915",
    )
    return SpringFestival(config=config)


@pytest.fixture
def market_data() -> dict[str, pl.DataFrame]:
    return {"159915": pl.DataFrame([{"date": date(2026, 2, 10), "close": 2.35}])}


class TestSpringFestivalConfig:
    def test_default_config(self) -> None:
        config = SpringFestivalConfig(name="spring_festival")
        assert config.hold_days == 20
        assert config.pre_days == 5
        assert config.target_fund == "159915"
        assert config.max_position_pct == 0.20

    def test_rejects_invalid_hold_days(self) -> None:
        with pytest.raises(ValueError):
            SpringFestivalConfig(name="spring_festival", hold_days=0)

    def test_rejects_invalid_pre_days(self) -> None:
        with pytest.raises(ValueError):
            SpringFestivalConfig(name="spring_festival", pre_days=0)

    def test_inherits_strategy_config(self) -> None:
        config = SpringFestivalConfig(name="spring_festival")
        assert isinstance(config, StrategyConfig)


class TestLunarNewYearDetection:
    def test_2026_spring_festival_date(self, strategy: SpringFestival) -> None:
        result = strategy.get_spring_festival(2026)
        assert result == date(2026, 2, 17)

    def test_2025_spring_festival_date(self, strategy: SpringFestival) -> None:
        result = strategy.get_spring_festival(2025)
        assert result == date(2025, 1, 29)

    def test_2027_spring_festival_date(self, strategy: SpringFestival) -> None:
        result = strategy.get_spring_festival(2027)
        assert result == date(2027, 2, 6)

    def test_2028_spring_festival_date(self, strategy: SpringFestival) -> None:
        result = strategy.get_spring_festival(2028)
        assert result == date(2028, 1, 26)

    def test_returns_none_for_out_of_range(self, strategy: SpringFestival) -> None:
        assert strategy.get_spring_festival(1980) is None


class TestIsInPreFestivalWindow:
    def test_within_5_days_before_festival(self, strategy: SpringFestival) -> None:
        assert strategy.is_in_pre_festival_window(date(2026, 2, 12))
        assert strategy.is_in_pre_festival_window(date(2026, 2, 16))

    def test_exactly_5_days_before(self, strategy: SpringFestival) -> None:
        assert strategy.is_in_pre_festival_window(date(2026, 2, 12))  # 17-5=12

    def test_too_early(self, strategy: SpringFestival) -> None:
        assert not strategy.is_in_pre_festival_window(date(2026, 2, 11))

    def test_on_festival_day(self, strategy: SpringFestival) -> None:
        assert not strategy.is_in_pre_festival_window(date(2026, 2, 17))

    def test_after_festival(self, strategy: SpringFestival) -> None:
        assert not strategy.is_in_pre_festival_window(date(2026, 2, 18))


class TestRegimeEligibility:
    def test_eligible_in_all_regimes(self, strategy: SpringFestival) -> None:
        for regime in MarketRegime:
            assert strategy.is_eligible(regime) is True


class TestRequiredData:
    def test_returns_close(self, strategy: SpringFestival) -> None:
        fields = strategy.required_data()
        assert "close" in fields


class TestGenerateSignalsHappyPath:
    def test_buy_before_spring_festival(
        self, strategy: SpringFestival, market_data: dict[str, pl.DataFrame]
    ) -> None:
        dt = date(2026, 2, 13)  # 4 days before 2026-02-17
        signals = strategy.generate_signals(dt, market_data)
        buy = [s for s in signals if s["direction"] == SignalDirection.BUY.value]
        assert len(buy) == 1
        assert buy[0]["fund_code"] == "159915"
        assert buy[0]["target_weight"] > 0

    def test_no_signal_outside_window(self, strategy: SpringFestival) -> None:
        dt = date(2026, 3, 1)
        data = {"159915": pl.DataFrame([{"date": dt, "close": 2.35}])}
        signals = strategy.generate_signals(dt, data)
        assert signals == []

    def test_signal_structure(
        self, strategy: SpringFestival, market_data: dict[str, pl.DataFrame]
    ) -> None:
        dt = date(2026, 2, 14)
        signals = strategy.generate_signals(dt, market_data)
        for s in signals:
            assert isinstance(s["fund_code"], str)
            assert s["direction"] in {e.value for e in SignalDirection}
            assert isinstance(s["confidence"], float)
            assert 0.0 <= s["confidence"] <= 1.0
            assert isinstance(s["target_weight"], float)
            assert 0.0 <= s["target_weight"] <= 1.0
            assert isinstance(s["reason"], str)
            assert len(s["reason"]) > 0

    def test_confidence_reflects_proximity(self, strategy: SpringFestival) -> None:
        data = {"159915": pl.DataFrame([{"date": date(2026, 2, 15), "close": 2.35}])}
        close_signal = strategy.generate_signals(date(2026, 2, 16), data)
        far_signal = strategy.generate_signals(date(2026, 2, 12), data)
        assert close_signal[0]["confidence"] >= far_signal[0]["confidence"]


class TestGenerateSignalsEdgeCases:
    def test_empty_data(self, strategy: SpringFestival) -> None:
        dt = date(2026, 2, 13)
        signals = strategy.generate_signals(dt, {})
        assert signals == []

    def test_target_fund_not_in_data(self, strategy: SpringFestival) -> None:
        dt = date(2026, 2, 13)
        data = {"000001": pl.DataFrame([{"date": dt, "close": 1.0}])}
        signals = strategy.generate_signals(dt, data)
        assert signals == []

    def test_custom_target_fund(self) -> None:
        config = SpringFestivalConfig(
            name="spring_festival",
            target_fund="510050",
            hold_days=15,
            pre_days=3,
        )
        s = SpringFestival(config=config)
        dt = date(2026, 2, 14)
        data = {"510050": pl.DataFrame([{"date": dt, "close": 3.50}])}
        signals = s.generate_signals(dt, data)
        assert len(signals) == 1
        assert signals[0]["fund_code"] == "510050"


class TestStrategyIdentity:
    def test_name(self, strategy: SpringFestival) -> None:
        assert strategy.name == "spring_festival"

    def test_validate_returns_empty(self, strategy: SpringFestival) -> None:
        assert strategy.validate() == []
