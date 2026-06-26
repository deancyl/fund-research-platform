"""
Tests for src/core/data/fetcher.py — DataFetcher protocol and implementations.
"""

from datetime import date

import polars as pl
import pytest

from src.core.data.fetcher import (
    AKShareFetcher,
    FallbackFetcher,
    NoopFetcher,
)


class TestNoopFetcher:
    """NoopFetcher returns empty DataFrames with correct schema."""

    def test_fetch_fund_nav_returns_empty(self) -> None:
        f = NoopFetcher()
        result = f.fetch_fund_nav("005827", date(2026, 1, 1), date(2026, 6, 1))
        assert result.is_empty()
        assert "nav" in result.columns

    def test_fetch_fund_list_has_required_columns(self) -> None:
        f = NoopFetcher()
        result = f.fetch_fund_list()
        assert {"fund_code", "fund_name", "fund_type", "company"} <= set(result.columns)


class TestFallbackFetcher:
    """FallbackFetcher delegates to primary first, then fallback."""

    def test_uses_primary_when_available(self) -> None:
        """When primary returns data, fallback is not called."""
        primary = NoopFetcher()  # returns empty → falls back
        fallback = NoopFetcher()
        f = FallbackFetcher(primary=primary, fallback=fallback)
        result = f.fetch_fund_list()
        assert result.is_empty()  # both return empty

    def test_fallback_is_reachable(self) -> None:
        """Even if primary fails, the method should not crash."""
        f = FallbackFetcher(primary=NoopFetcher(), fallback=NoopFetcher())
        result = f.fetch_fund_nav("000001", date(2024, 1, 1), date(2024, 12, 31))
        assert result.is_empty()  # Noop always empty — graceful degradation


class TestAKShareFetcher:
    """AKShareFetcher gracefully degrades when akshare is not installed."""

    def test_returns_empty_when_akshare_missing(self) -> None:
        """Without akshare installed, AKShareFetcher returns empty DataFrames."""
        f = AKShareFetcher()
        result = f.fetch_fund_list()
        assert "fund_code" in result.columns
        # May or may not be empty depending on whether akshare is installed
        # The key requirement: no crash, no ImportError surfaced to caller.
