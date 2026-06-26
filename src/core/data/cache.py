"""
Data caching layer — dual-engine storage for fund research data.

Engine assignment (per SYSTEM-CONTRACT §5):
  - SQLite: static relational data (fund metadata, watchlist, portfolio lots).
  - DuckDB: time-series/high-frequency data (NAV history, backtest results, news).
  - Rule: any table expected to exceed 10,000 rows → DuckDB.

All operations are context-managed. Cache paths default to a local .data/ directory.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator

import polars as pl


# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR: Path = Path(os.environ.get("FUND_DATA_DIR", ".data"))
MAX_SQLITE_ROWS: int = 10_000


# ─── Cache Manager ──────────────────────────────────────────────────────────


class CacheManager:
    """
    Manages dual-engine storage: SQLite for static, DuckDB for time-series.

    Usage:
        with CacheManager() as cache:
            cache.save_nav("005827", nav_df)
            df = cache.load_nav("005827", start=date(2026,1,1), end=date(2026,6,1))
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sqlite_path = self._data_dir / "fund_meta.db"
        self._duckdb_path = self._data_dir / "fund_ts.duckdb"
        self._connected: bool = False

    def __enter__(self) -> "CacheManager":
        self._connected = True
        return self

    def __exit__(self, *args: object) -> None:
        self._connected = False

    # ── DuckDB: Time-Series (NAV, index, backtest results) ─────────────────

    def save_nav(self, fund_code: str, df: pl.DataFrame) -> None:
        """Save daily NAV data to DuckDB. Overwrites existing for the same date range."""
        _validate_required_columns(df, {"date", "nav"})
        with self._duckdb_conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS fund_nav ("
                "    fund_code VARCHAR, date DATE, nav DOUBLE, accumulated_nav DOUBLE"
                ")"
            )
            con.execute(
                "DELETE FROM fund_nav WHERE fund_code = ? AND date BETWEEN ? AND ?",
                [fund_code, df["date"].min(), df["date"].max()],
            )
            con.execute("INSERT INTO fund_nav SELECT * FROM df")

    def load_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        """Load NAV data from DuckDB for a fund and date range."""
        with self._duckdb_conn() as con:
            return con.execute(
                "SELECT date, nav, accumulated_nav FROM fund_nav "
                "WHERE fund_code = ? AND date BETWEEN ? AND ? ORDER BY date",
                [fund_code, start, end],
            ).pl()

    def save_index_data(self, index_code: str, df: pl.DataFrame) -> None:
        """Save index OHLCV data to DuckDB."""
        _validate_required_columns(df, {"date", "close"})
        with self._duckdb_conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS index_data ("
                "    index_code VARCHAR, date DATE, open DOUBLE, high DOUBLE, "
                "    low DOUBLE, close DOUBLE, volume DOUBLE"
                ")"
            )
            con.execute(
                "DELETE FROM index_data WHERE index_code = ? AND date BETWEEN ? AND ?",
                [index_code, df["date"].min(), df["date"].max()],
            )
            con.execute("INSERT INTO index_data SELECT * FROM df")

    def load_index_data(
        self, index_code: str, start: date, end: date
    ) -> pl.DataFrame:
        """Load index OHLCV data from DuckDB."""
        with self._duckdb_conn() as con:
            return con.execute(
                "SELECT date, open, high, low, close, volume FROM index_data "
                "WHERE index_code = ? AND date BETWEEN ? AND ? ORDER BY date",
                [index_code, start, end],
            ).pl()

    def save_backtest_result(self, strategy_name: str, df: pl.DataFrame) -> None:
        """Save backtest result time-series to DuckDB."""
        _validate_required_columns(df, {"date", "pnl"})
        with self._duckdb_conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS backtest_results ("
                "    strategy_name VARCHAR, date DATE, pnl DOUBLE, nav DOUBLE"
                ")"
            )
            con.execute(
                "DELETE FROM backtest_results WHERE strategy_name = ?",
                [strategy_name],
            )
            con.execute("INSERT INTO backtest_results SELECT * FROM df")

    # ── SQLite: Static metadata (fund info, watchlist) ────────────────────

    def save_fund_info(self, fund_list: pl.DataFrame) -> None:
        """Save fund basic info (code, name, type, company) to SQLite."""
        _validate_required_columns(fund_list, {"fund_code", "fund_name"})
        with self._sqlite_conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS fund_info ("
                "    fund_code TEXT PRIMARY KEY, fund_name TEXT, "
                "    fund_type TEXT, company TEXT"
                ")"
            )
            con.execute("DELETE FROM fund_info")
            rows = fund_list.select(["fund_code", "fund_name", "fund_type", "company"]).rows()
            con.executemany(
                "INSERT INTO fund_info (fund_code, fund_name, fund_type, company) "
                "VALUES (?, ?, ?, ?)",
                [tuple(r) for r in rows],
            )
            con.commit()

    def load_fund_info(self, fund_code: str = "") -> pl.DataFrame:
        """Load fund info from SQLite. Empty code → all funds."""
        with self._sqlite_conn() as con:
            if fund_code:
                rows = con.execute(
                    "SELECT fund_code, fund_name, fund_type, company "
                    "FROM fund_info WHERE fund_code = ?",
                    [fund_code],
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT fund_code, fund_name, fund_type, company FROM fund_info"
                ).fetchall()
            return pl.DataFrame(
                rows,
                schema=["fund_code", "fund_name", "fund_type", "company"],
                orient="row",
            )

    def save_watchlist(self, watchlist_name: str, codes: list[str]) -> None:
        """Save a watchlist (list of fund codes) to SQLite."""
        with self._sqlite_conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS watchlists ("
                "    name TEXT, fund_code TEXT, PRIMARY KEY (name, fund_code)"
                ")"
            )
            con.execute("DELETE FROM watchlists WHERE name = ?", [watchlist_name])
            for code in codes:
                con.execute(
                    "INSERT INTO watchlists (name, fund_code) VALUES (?, ?)",
                    [watchlist_name, code],
                )
            con.commit()

    def load_watchlist(self, watchlist_name: str) -> list[str]:
        """Load a watchlist from SQLite."""
        with self._sqlite_conn() as con:
            rows = con.execute(
                "SELECT fund_code FROM watchlists WHERE name = ? ORDER BY fund_code",
                [watchlist_name],
            ).fetchall()
            return [row[0] for row in rows]

    # ── Cache Helpers ─────────────────────────────────────────────────────

    def is_fresh(self, table: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is fresh enough.

        Returns True if the table file was modified within max_age_hours.
        """
        if table == "fund_nav":
            path = self._duckdb_path
        elif table == "fund_info":
            path = self._sqlite_path
        else:
            path = self._duckdb_path

        if not path.exists():
            return False

        age_seconds = (date.today() - date.fromtimestamp(path.stat().st_mtime)).total_seconds()
        return age_seconds < max_age_hours * 3600

    # ── Internal Connections ──────────────────────────────────────────────

    @contextmanager
    def _duckdb_conn(self) -> Iterator["duckdb.DuckDBPyConnection"]:  # noqa: F821
        import duckdb
        con = duckdb.connect(str(self._duckdb_path))
        try:
            yield con
        finally:
            con.close()

    @contextmanager
    def _sqlite_conn(self) -> Iterator["sqlite3.Connection"]:  # noqa: F821
        import sqlite3
        con = sqlite3.connect(str(self._sqlite_path))
        try:
            yield con
        finally:
            con.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _validate_required_columns(df: pl.DataFrame, required: set[str]) -> None:
    """Raise ValueError if required columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
