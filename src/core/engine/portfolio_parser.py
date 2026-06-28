"""
Universal portfolio parser — CSV/Excel to FundPosition Lots.
Supports Chinese column headers from broker/exchange exports.
"""

from datetime import date as _date

from src.core.data.schema import FundChannel, FundCategory, FundPosition, PositionLot

_COLUMN_MAP: dict[str, str] = {
    "基金代码": "code", "证券代码": "code", "代码": "code",
    "买入日期": "date", "交易日期": "date", "成本日期": "date",
    "持有份额": "shares", "实际份额": "shares", "份额": "shares",
    "成本金额": "cost", "买入金额": "cost", "金额": "cost",
    "当前净值": "nav",
    "基金名称": "name", "名称": "name",
}


def parse_portfolio(file_path: str) -> list[FundPosition]:
    """Parse a CSV or Excel file into FundPosition objects with Lot tracking."""
    import polars as pl

    if file_path.endswith(".csv"):
        df = pl.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pl.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Normalize column names
    df = df.rename({k: v for k, v in _COLUMN_MAP.items() if k in df.columns})

    positions: list[FundPosition] = []
    for code in df["code"].unique().to_list():
        group = df.filter(pl.col("code") == code)
        lots: list[PositionLot] = []
        nav = group["nav"].mean() if "nav" in group.columns else 0.0
        name = group["name"][0] if "name" in group.columns else str(code)

        for row in group.iter_rows(named=True):
            shares = float(row.get("shares", 0))
            cost = float(row.get("cost", 0))
            purchase_nav = cost / shares if shares > 0 else 0.0
            try:
                purchase_date = _date.fromisoformat(str(row["date"])[:10])
            except (KeyError, ValueError):
                purchase_date = _date.today()

            lots.append(PositionLot(
                purchase_date=purchase_date,
                shares=shares,
                purchase_nav=purchase_nav,
                cost_amount=cost,
            ))

        total_shares = sum(l.shares for l in lots)
        market_value = total_shares * nav if nav > 0 else sum(l.cost_amount for l in lots)

        positions.append(FundPosition(
            fund_code=str(code),
            fund_name=name,
            category=FundCategory.EQUITY,
            channel=FundChannel.OTC_OPEN_END,
            lots=lots,
            current_nav=nav if nav > 0 else 1.0,
            total_shares=total_shares,
            market_value=market_value,
            weight_pct=0.0,
        ))

    return positions
