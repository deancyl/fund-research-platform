"""
Tests for src/core/data/cache.py — dual-engine CacheManager (SQLite + DuckDB).
"""

import tempfile
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.core.data.cache import CacheManager


@pytest.fixture
def cache() -> CacheManager:
    """Create a CacheManager pointed at a temp directory."""
    with tempfile.TemporaryDirectory() as td:
        yield CacheManager(data_dir=Path(td))


def _nav_df() -> pl.DataFrame:
    return pl.DataFrame({
        "fund_code": ["005827", "005827"],
        "date": [date(2026, 6, 1), date(2026, 6, 2)],
        "nav": [1.50, 1.52],
        "accumulated_nav": [2.10, 2.12],
    })


def _index_df() -> pl.DataFrame:
    return pl.DataFrame({
        "index_code": ["000300", "000300"],
        "date": [date(2026, 6, 1), date(2026, 6, 2)],
        "open": [3900.0, 3920.0],
        "high": [3950.0, 3940.0],
        "low": [3880.0, 3900.0],
        "close": [3920.0, 3910.0],
        "volume": [1e9, 1.2e9],
    })


def _fund_info_df() -> pl.DataFrame:
    return pl.DataFrame({
        "fund_code": ["005827", "510300"],
        "fund_name": ["易方达蓝筹精选", "沪深300ETF"],
        "fund_type": ["混合型", "指数型"],
        "company": ["易方达基金", "华泰柏瑞"],
    })


class TestDuckDBNavStorage:
    """NAV data goes to DuckDB (time-series)."""

    def test_save_and_load_nav(self, cache: CacheManager) -> None:
        df = _nav_df()
        with cache:
            cache.save_nav("005827", df)
            loaded = cache.load_nav("005827", date(2026, 6, 1), date(2026, 6, 30))
        assert len(loaded) == 2
        assert loaded["nav"].to_list() == [1.50, 1.52]

    def test_nav_missing_columns_raises(self, cache: CacheManager) -> None:
        bad_df = pl.DataFrame({"wrong_col": [1.0]})
        with pytest.raises(ValueError):
            with cache:
                cache.save_nav("005827", bad_df)


class TestDuckDBIndexStorage:
    """Index data goes to DuckDB."""

    def test_save_and_load_index(self, cache: CacheManager) -> None:
        df = _index_df()
        with cache:
            cache.save_index_data("000300", df)
            loaded = cache.load_index_data("000300", date(2026, 6, 1), date(2026, 6, 30))
        assert len(loaded) == 2
        assert loaded["close"].to_list() == [3920.0, 3910.0]


class TestDuckDBBacktestStorage:
    """Backtest results go to DuckDB."""

    def test_save_backtest_result(self, cache: CacheManager) -> None:
        df = pl.DataFrame({
            "strategy_name": ["test_strat", "test_strat"],
            "date": [date(2026, 6, 1), date(2026, 6, 2)],
            "pnl": [100.0, -50.0],
            "nav": [1.01, 1.005],
        })
        with cache:
            cache.save_backtest_result("test_strat", df)


class TestSQLiteMetadata:
    """Fund metadata and watchlists go to SQLite."""

    def test_save_and_load_fund_info(self, cache: CacheManager) -> None:
        with cache:
            cache.save_fund_info(_fund_info_df())
            result = cache.load_fund_info("005827")
        assert len(result) == 1
        assert result["fund_name"][0] == "易方达蓝筹精选"

    def test_load_all_funds(self, cache: CacheManager) -> None:
        with cache:
            cache.save_fund_info(_fund_info_df())
            result = cache.load_fund_info()
        assert len(result) == 2

    def test_save_and_load_watchlist(self, cache: CacheManager) -> None:
        with cache:
            cache.save_watchlist("my_list", ["005827", "510300"])
            codes = cache.load_watchlist("my_list")
        assert codes == ["005827", "510300"]

    def test_watchlist_overwrite(self, cache: CacheManager) -> None:
        with cache:
            cache.save_watchlist("my_list", ["005827"])
            cache.save_watchlist("my_list", ["510300"])
            codes = cache.load_watchlist("my_list")
        assert codes == ["510300"]


class TestFreshnessCheck:
    """is_fresh returns whether cached data is recent."""

    def test_freshness_on_non_existent(self, cache: CacheManager) -> None:
        """Non-existent cache → not fresh."""
        with cache:
            assert not cache.is_fresh("fund_nav")

    def test_freshness_after_save(self, cache: CacheManager) -> None:
        """After saving, cache should be fresh."""
        with cache:
            cache.save_nav("005827", _nav_df())
            assert cache.is_fresh("fund_nav", max_age_hours=24)
