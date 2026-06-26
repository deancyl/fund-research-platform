"""Layer 1: Data acquisition and caching.

Data sources: AKShare (primary), Eastmoney direct API (fallback), Baostock (index supplement).
Storage: SQLite for static metadata, DuckDB for time-series, Parquet for bulk exchange.
"""

__all__: list[str] = []
