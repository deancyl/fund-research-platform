"""fund-research-platform — Headless Core.

All modules in this package are pure computation — zero UI framework imports.
Inputs and outputs use only standard Python types, dicts, and polars DataFrames.

See src/core/api.py for the unified service-layer entry point used by both
the TUI adapter (src/tui/) and the Web adapter (src/web/).

Key exports:
  - schema: FundChannel, FundCategory, FundPosition, PositionLot, FundTradingProfile
  - engine: OrderCutoffValidator, RedemptionFeeCalculator, CashLockManager, SmartRebalancer
  - strategy: BaseStrategy, StrategyConfig, MarketRegime, SignalDirection
  - api: resolve_execution_date, calculate_redemption_fee, generate_rebalance_plan
"""

from src.core.api import (
    calculate_redeem_fee_for_lots,
    calculate_redemption_fee,
    check_holding_warning,
    create_cash_manager,
    generate_rebalance_plan,
    resolve_execution_date,
    validate_backtest_order,
)
from src.core.data.schema import FundCategory, FundChannel, FundPosition, FundTradingProfile, PositionLot
from src.core.engine.cash_lock import CashLockManager
from src.core.engine.order_cutoff import OrderCutoffValidator, TradingCalendar
from src.core.engine.rebalancer import SmartRebalancer
from src.core.engine.redemption_fee import HoldingWarning, RedemptionFeeCalculator
from src.core.strategy.base import BaseStrategy, MarketRegime, SignalDirection, StrategyConfig

__all__ = [
    # Schema
    "FundChannel",
    "FundCategory",
    "FundPosition",
    "FundTradingProfile",
    "PositionLot",
    # Engine
    "OrderCutoffValidator",
    "TradingCalendar",
    "RedemptionFeeCalculator",
    "HoldingWarning",
    "CashLockManager",
    "SmartRebalancer",
    # Strategy
    "BaseStrategy",
    "StrategyConfig",
    "MarketRegime",
    "SignalDirection",
    # API
    "resolve_execution_date",
    "validate_backtest_order",
    "calculate_redemption_fee",
    "calculate_redeem_fee_for_lots",
    "check_holding_warning",
    "create_cash_manager",
    "generate_rebalance_plan",
]
