"""
Data acquisition layer — fetcher protocol and AKShare/Eastmoney implementations.

All fetchers return polars DataFrames. The DataFetcher protocol allows
swapping implementations for testing or fallback scenarios.
"""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl


class DataFetcher(ABC):
    """Protocol for fetching Chinese fund/index data."""

    @abstractmethod
    def fetch_fund_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        """
        Fetch daily NAV history for a fund.

        Returns: DataFrame with columns [date, nav, accumulated_nav, daily_return].
        """
        ...

    @abstractmethod
    def fetch_fund_list(self) -> pl.DataFrame:
        """
        Fetch the complete list of publicly offered funds.

        Returns: DataFrame with columns [fund_code, fund_name, fund_type, company].
        """
        ...

    @abstractmethod
    def fetch_index_daily(self, index_code: str, start: date, end: date) -> pl.DataFrame:
        """
        Fetch daily index data.

        Returns: DataFrame with columns [date, open, high, low, close, volume].
        """
        ...

    @abstractmethod
    def fetch_etf_realtime(self) -> pl.DataFrame:
        """
        Fetch real-time ETF quotes.

        Returns: DataFrame with columns [fund_code, name, price, change_pct, volume].
        """
        ...


class NoopFetcher(DataFetcher):
    """
    No-op fetcher for testing — returns empty DataFrames.

    Use when AKShare is not available and you need the system to start.
    """
    _EMPTY_NAV = pl.DataFrame(
        schema={"date": pl.Date, "nav": pl.Float64, "accumulated_nav": pl.Float64, "daily_return": pl.Float64}
    )
    _EMPTY_FUND_LIST = pl.DataFrame(
        schema={"fund_code": pl.Utf8, "fund_name": pl.Utf8, "fund_type": pl.Utf8, "company": pl.Utf8}
    )
    _EMPTY_INDEX = pl.DataFrame(
        schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
    )
    _EMPTY_ETF = pl.DataFrame(
        schema={"fund_code": pl.Utf8, "name": pl.Utf8, "price": pl.Float64, "change_pct": pl.Float64, "volume": pl.Float64}
    )

    def fetch_fund_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        return self._EMPTY_NAV

    def fetch_fund_list(self) -> pl.DataFrame:
        return self._EMPTY_FUND_LIST

    def fetch_index_daily(self, index_code: str, start: date, end: date) -> pl.DataFrame:
        return self._EMPTY_INDEX

    def fetch_etf_realtime(self) -> pl.DataFrame:
        return self._EMPTY_ETF


class AKShareFetcher(DataFetcher):
    """
    Primary fetcher using AKShare.

    AKShare is the most comprehensive free Chinese financial data library.
    It scrapes Eastmoney, Sina, CSIndex and others under the hood.

    Install: pip install akshare
    """

    def fetch_fund_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            return NoopFetcher().fetch_fund_nav(fund_code, start, end)

        try:
            df = ak.fund_open_fund_info_em(
                symbol=fund_code,
                indicator="单位净值走势",
                period="成立来",
            )
            pdf = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
            if "净值日期" in pdf.columns:
                pdf = pdf.rename({"净值日期": "date", "单位净值": "nav", "累计净值": "accumulated_nav"})
            pdf = pdf.filter(pl.col("date").is_between(start, end))
            return pdf.select(["date", "nav", "accumulated_nav"])
        except Exception:  # noqa: BROAD_EXCEPT_OK — external HTTP scraper, many failure modes
            return NoopFetcher().fetch_fund_nav(fund_code, start, end)

    def fetch_fund_list(self) -> pl.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            return NoopFetcher().fetch_fund_list()

        try:
            df = ak.fund_name_em()
            pdf = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
            if "基金代码" in pdf.columns:
                pdf = pdf.rename({"基金代码": "fund_code", "基金简称": "fund_name", "基金类型": "fund_type"})
                if "基金公司" not in pdf.columns:
                    pdf = pdf.with_columns(pl.lit("").alias("company"))
            return pdf.select(["fund_code", "fund_name", "fund_type", "company"])
        except Exception:  # noqa: BROAD_EXCEPT_OK — external HTTP scraper, many failure modes
            return NoopFetcher().fetch_fund_list()

    def fetch_index_daily(self, index_code: str, start: date, end: date) -> pl.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            return NoopFetcher().fetch_index_daily(index_code, start, end)

        try:
            df = ak.stock_zh_index_daily_em(symbol=f"sh{index_code}" if not index_code.startswith("sz") else index_code)
            pdf = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
            if "date" in pdf.columns:
                pdf = pdf.filter(pl.col("date").is_between(start, end))
            return pdf.select(["date", "open", "high", "low", "close", "volume"])
        except Exception:  # noqa: BROAD_EXCEPT_OK — external HTTP scraper, many failure modes
            return NoopFetcher().fetch_index_daily(index_code, start, end)

    def fetch_etf_realtime(self) -> pl.DataFrame:
        try:
            import akshare as ak
        except ImportError:
            return NoopFetcher().fetch_etf_realtime()

        try:
            df = ak.fund_etf_spot_em()
            pdf = pl.from_pandas(df) if hasattr(df, "to_pandas") else pl.DataFrame(df)
            return pdf.select(["代码", "名称", "最新价", "涨跌幅", "成交量"]).rename({
                "代码": "fund_code", "名称": "name", "最新价": "price",
                "涨跌幅": "change_pct", "成交量": "volume",
            })
        except Exception:  # noqa: BROAD_EXCEPT_OK — external HTTP scraper, many failure modes
            return NoopFetcher().fetch_etf_realtime()


class FallbackFetcher(DataFetcher):
    """
    Composite fetcher with fallback chain.

    Tries primary (AKShare) first; falls back to secondary on any failure.
    """

    def __init__(self, primary: DataFetcher, fallback: DataFetcher) -> None:
        self._primary = primary
        self._fallback = fallback

    def fetch_fund_nav(self, fund_code: str, start: date, end: date) -> pl.DataFrame:
        result = self._primary.fetch_fund_nav(fund_code, start, end)
        if result.is_empty():
            result = self._fallback.fetch_fund_nav(fund_code, start, end)
        return result

    def fetch_fund_list(self) -> pl.DataFrame:
        result = self._primary.fetch_fund_list()
        if result.is_empty():
            result = self._fallback.fetch_fund_list()
        return result

    def fetch_index_daily(self, index_code: str, start: date, end: date) -> pl.DataFrame:
        result = self._primary.fetch_index_daily(index_code, start, end)
        if result.is_empty():
            result = self._fallback.fetch_index_daily(index_code, start, end)
        return result

    def fetch_etf_realtime(self) -> pl.DataFrame:
        result = self._primary.fetch_etf_realtime()
        if result.is_empty():
            result = self._fallback.fetch_etf_realtime()
        return result
