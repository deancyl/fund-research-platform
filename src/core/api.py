"""
Unified service-layer API for the Headless Core.

This module is the SINGLE entry point for all adapters (TUI and Web).
It exposes only standard Python types and dataclass returns — never
Textual widgets, FastAPI Request/Response objects, or any UI imports.

Usage from TUI adapter:
    from src.core.api import validate_backtest_order, calculate_redemption_fee
    result = validate_backtest_order(signal_time, ...)

Usage from Web adapter:
    from src.core.api import generate_rebalance_plan
    plan = generate_rebalance_plan(portfolio_json, target_weights_json)
    # serialize plan to JSON response

Extending: add new functions here when new engine capabilities are built.
Do NOT import api.py from within core modules — it is the outermost layer.
"""

from datetime import date, datetime
from typing import Protocol

from src.core.data.schema import FundCategory, FundChannel, FundPosition, FundTradingProfile, PositionLot
from src.core.engine.cash_lock import CashLockManager
from src.core.engine.order_cutoff import OrderCutoffValidator, SimpleTradingCalendar, TradingCalendar
from src.core.engine.rebalancer import RebalancePlan, SmartRebalancer
from src.core.engine.redemption_fee import HoldingWarning, RedemptionFeeCalculator


# ─── Engine instances (lazy) ─────────────────────────────────────────────────

_cutoff_validator: OrderCutoffValidator | None = None
_fee_calculator: RedemptionFeeCalculator | None = None
_rebalancer: SmartRebalancer | None = None


def _get_cutoff_validator() -> OrderCutoffValidator:
    global _cutoff_validator
    if _cutoff_validator is None:
        _cutoff_validator = OrderCutoffValidator(calendar=SimpleTradingCalendar())
    return _cutoff_validator


def _get_fee_calculator() -> RedemptionFeeCalculator:
    global _fee_calculator
    if _fee_calculator is None:
        _fee_calculator = RedemptionFeeCalculator()
    return _fee_calculator


def _get_rebalancer() -> SmartRebalancer:
    global _rebalancer
    if _rebalancer is None:
        _rebalancer = SmartRebalancer()
    return _rebalancer


# ─── TradingCalendar — injectable custom calendars ──────────────────────────


def set_calendar(calendar: TradingCalendar) -> None:
    """Inject a custom trading calendar (e.g., with China holiday data)."""
    global _cutoff_validator
    _cutoff_validator = OrderCutoffValidator(calendar=calendar)


# ─── Order Cutoff API ───────────────────────────────────────────────────────


def resolve_execution_date(signal_datetime: datetime) -> date:
    """
    Determine the NAV date for a signal generated at signal_datetime.

    Returns signal_datetime.date() if before 15:00 on a trading day,
    otherwise the next trading day.
    """
    return _get_cutoff_validator().resolve_execution_date(signal_datetime)


def validate_backtest_order(
    signal_datetime: datetime,
    nav_publish_datetime: datetime,
    execution_date: date,
) -> bool:
    """Check for look-ahead bias. Returns True if order timeline is valid."""
    return _get_cutoff_validator().validate_backtest_order(
        signal_datetime, nav_publish_datetime, execution_date
    )


# ─── Redemption Fee API ─────────────────────────────────────────────────────


def calculate_redemption_fee(
    fund_code: str,
    channel: FundChannel,
    category: FundCategory,
    holding_days: int,
    redemption_amount: float,
    contract_schedule: dict | None = None,
) -> dict[str, float]:
    """
    Calculate redemption fee for a single holding.

    Returns: {"fee_amount_cny": ..., "rate": ...}
    """
    profile = FundTradingProfile(
        fund_code=fund_code,
        channel=channel,
        redemption_fee_schedule=contract_schedule or {},
    )
    fee, rate = _get_fee_calculator().calculate(
        fund_profile=profile,
        fund_category=category,
        holding_days=holding_days,
        redemption_amount=redemption_amount,
    )
    return {"fee_amount_cny": round(fee, 2), "rate": round(rate, 6)}


def calculate_redeem_fee_for_lots(
    lots: list[PositionLot],
    fund_code: str,
    channel: FundChannel,
    category: FundCategory,
    shares_to_redeem: float,
    current_nav: float,
    current_date: date,
) -> dict[str, float]:
    """Calculate FIFO fee for partial position redemption."""
    profile = FundTradingProfile(fund_code=fund_code, channel=channel)
    fee, rate = _get_fee_calculator().calculate_for_lots(
        fund_profile=profile,
        fund_category=category,
        lots=lots,
        shares_to_redeem=shares_to_redeem,
        current_nav=current_nav,
        current_date=current_date,
    )
    return {"fee_amount_cny": round(fee, 2), "rate": round(rate, 6)}


def check_holding_warning(
    holding_days: int, channel: FundChannel
) -> HoldingWarning:
    """Returns PASS, WARN, or REJECT based on holding period."""
    return _get_fee_calculator().enforce_min_hold(holding_days, channel)


# ─── Cash Lock API ──────────────────────────────────────────────────────────


def create_cash_manager(
    initial_cash: float,
    delay_override: dict[FundChannel, int] | None = None,
) -> CashLockManager:
    """Create a new CashLockManager with optional custom settlement delays."""
    return CashLockManager(
        initial_cash=initial_cash,
        delay_override=delay_override,
    )


# ─── Rebalancing API ────────────────────────────────────────────────────────


def generate_rebalance_plan(
    current_portfolio: list[FundPosition],
    target_weights: dict[str, float],
    current_date: date,
    total_portfolio_value: float,
    expected_alpha_pct: float = 0.03,
) -> RebalancePlan:
    """
    Generate a complete rebalancing plan.

    Args:
        current_portfolio: Current positions with Lot-level detail.
        target_weights: Desired allocation {fund_code: weight_pct}.
        current_date: Reference date for holding period calculation.
        total_portfolio_value: Total portfolio value in CNY.
        expected_alpha_pct: Expected annual alpha (fee penalty threshold).

    Returns:
        RebalancePlan with actions, timeline, friction cost, and AI note.
    """
    return _get_rebalancer().generate_rebalance_plan(
        current_portfolio=current_portfolio,
        target_weights=target_weights,
        current_date=current_date,
        total_portfolio_value=total_portfolio_value,
        expected_alpha_pct=expected_alpha_pct,
    )


# ─── Strategy Protocol (extensibility foundation) ───────────────────────────


class StrategyProtocol(Protocol):
    """
    Contract for all trading strategies in the platform.

    Every strategy must implement:
      - name: unique identifier
      - generate_signals(date): produce Buy/Sell/Hold signals for the given date
      - required_data(): declare what market data the strategy needs
    """

    name: str

    def generate_signals(self, dt: date) -> list[dict[str, float | str]]:
        """
        Generate trading signals for a given date.

        Returns: list of {fund_code, signal, confidence, reason}
        """
        ...

    def required_data(self) -> list[str]:
        """
        Declare required data fields (e.g., ['nav', 'volume', 'pe']).
        Used by the data layer to prefetch needed data.
        """
        ...
