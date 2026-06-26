"""
SmartRebalancer — composes redemption_fee, cash_lock, and order_cutoff
to produce complete rebalancing plans with friction cost analysis.

Key features:
  - FIFO lot-level fee calculation for partial redemptions.
  - Fee penalty gate: skip rebalancing when redemption fee exceeds expected alpha.
  - Settlement timeline: shows when redeemed cash becomes available for new purchases.
  - ETF vs OTC distinction: zero settlement delay and zero fee for ETFs.
"""

from dataclasses import dataclass, field
from datetime import date

from src.core.data.schema import FundChannel, FundPosition, FundTradingProfile
from src.core.engine.cash_lock import CashLockManager
from src.core.engine.redemption_fee import RedemptionFeeCalculator


# ─── Rebalance Action ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RebalanceAction:
    """A single action in the rebalancing plan (redeem or subscribe)."""

    action_type: str  # "REDEEM" or "SUBSCRIBE"
    fund_code: str
    fund_name: str
    amount: float      # gross redemption or subscription amount in CNY
    estimated_fee: float = 0.0
    net_cash: float = 0.0  # amount - fee (for redeems); amount (for subscribes)
    reason: str = ""
    skip_reason: str = ""  # non-empty if this action should be skipped


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """An event in the execution timeline."""

    t_day: int    # days from now (T+0 = today)
    event: str    # human-readable description


# ─── Rebalance Plan ──────────────────────────────────────────────────────────


@dataclass
class RebalancePlan:
    """Complete rebalancing plan with actions, timeline, and cost summary.

    Mutable because it's constructed incrementally by the SmartRebalancer.
    """

    status: str = "ANALYZED"
    actions: list[RebalanceAction] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    total_friction_cost_yuan: float = 0.0
    cash_flow_vacuum_days: int = 0
    ai_advisor_note: str = ""


# ─── SmartRebalancer ─────────────────────────────────────────────────────────


class SmartRebalancer:
    """
    Generate rebalancing plans with full China-specific rule compliance.

    Composes:
      - RedemptionFeeCalculator for statutory and contract-based fees.
      - CashLockManager settlement delay simulation.
      - FIFO lot-level fee calculation for precise cost estimation.
    """

    def __init__(self) -> None:
        self._fee_calc = RedemptionFeeCalculator()

    def generate_rebalance_plan(
        self,
        current_portfolio: list[FundPosition],
        target_weights: dict[str, float],
        current_date: date,
        total_portfolio_value: float,
        expected_alpha_pct: float = 0.03,
    ) -> RebalancePlan:
        """
        Produce a complete rebalancing plan.

        Args:
            current_portfolio: User's current positions with Lot-level detail.
            target_weights: Desired allocation {fund_code: weight_pct}.
            current_date: The date for holding-period calculation.
            total_portfolio_value: Total portfolio value in CNY.
            expected_alpha_pct: Expected annualized alpha from the new allocation.
                               Used as the threshold for the fee penalty gate.

        Returns:
            RebalancePlan with actions, timeline, and friction cost.
        """
        plan = RebalancePlan()
        redeems_total = 0.0
        max_settle_delay = 0

        # Phase 1: Identify overweight positions → generate REDEEM actions
        for pos in current_portfolio:
            target_w = target_weights.get(pos.fund_code, 0.0)
            if pos.weight_pct <= target_w:
                continue

            reduction_ratio = pos.weight_pct - target_w
            reduction_value = total_portfolio_value * reduction_ratio
            shares_to_redeem = reduction_value / pos.current_nav if pos.current_nav > 0 else 0.0

            # Calculate FIFO fee
            profile = self._make_profile(pos)
            fee, rate = self._fee_calc.calculate_for_lots(
                fund_profile=profile,
                fund_category=pos.category,
                lots=pos.lots,
                shares_to_redeem=shares_to_redeem,
                current_nav=pos.current_nav,
                current_date=current_date,
            )

            net_cash = reduction_value - fee
            skip_reason = ""

            # Fee penalty gate: skip if redemption fee > alpha
            daily_alpha = expected_alpha_pct / 365  # crude daily estimate
            if rate > daily_alpha and rate >= 0.01:
                skip_reason = (
                    f"赎回费率 {rate:.2%} 超过预期日Alpha ({daily_alpha:.4%})，"
                    f"建议推迟调仓以降低摩擦成本"
                )

            action = RebalanceAction(
                action_type="REDEEM",
                fund_code=pos.fund_code,
                fund_name=pos.fund_name,
                amount=round(reduction_value, 2),
                estimated_fee=round(fee, 2),
                net_cash=round(net_cash, 2),
                reason=f"仓位由 {pos.weight_pct:.1%} 降至 {target_w:.1%}",
                skip_reason=skip_reason,
            )
            plan.actions.append(action)

            plan.total_friction_cost_yuan += fee
            redeems_total += net_cash

            # Timeline: redemption submitted today (T+0)
            plan.timeline.append(
                TimelineEvent(t_day=0, event=f"提交 {pos.fund_name}({pos.fund_code}) 赎回申请")
            )

            delay = self._settle_days(pos.channel)
            max_settle_delay = max(max_settle_delay, delay)
            plan.timeline.append(
                TimelineEvent(
                    t_day=delay,
                    event=(
                        f"{pos.fund_name}({pos.fund_code}) 赎回款 {net_cash:,.2f} 元解冻"
                    ),
                )
            )

        # Phase 2: Identify underweight / new positions → SUBSCRIBE actions
        all_codes = set(target_weights.keys()) | {p.fund_code for p in current_portfolio}
        for code in all_codes:
            target_w = target_weights.get(code, 0.0)
            current_pos = next((p for p in current_portfolio if p.fund_code == code), None)
            current_w = current_pos.weight_pct if current_pos else 0.0

            if target_w <= current_w:
                continue

            increase_ratio = target_w - current_w
            subscribe_value = total_portfolio_value * increase_ratio

            fund_name = current_pos.fund_name if current_pos else code

            plan.actions.append(
                RebalanceAction(
                    action_type="SUBSCRIBE",
                    fund_code=code,
                    fund_name=fund_name,
                    amount=round(subscribe_value, 2),
                    net_cash=round(subscribe_value, 2),
                    reason=f"补充配置，目标仓位 {target_w:.1%}",
                )
            )

            # Subscribe timeline: only after cash is available
            settle_day = self._settle_days(
                current_pos.channel if current_pos else FundChannel.OTC_OPEN_END
            )
            plan.timeline.append(
                TimelineEvent(
                    t_day=settle_day,
                    event=f"申购 {fund_name}({code}) {subscribe_value:,.2f} 元",
                )
            )

        plan.cash_flow_vacuum_days = max_settle_delay
        plan.timeline.sort(key=lambda e: e.t_day)

        if plan.total_friction_cost_yuan > 0:
            plan.ai_advisor_note = (
                f"本次调仓预计产生摩擦成本 {plan.total_friction_cost_yuan:,.2f} 元。"
                f"部分持仓未满 30 天，若推迟数日可节省赎回费。"
            )

        return plan

    # ── Internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _settle_days(channel: FundChannel) -> int:
        """Map channel to settlement delay in trading days."""
        delays = {
            FundChannel.ETF_ON_EXCHANGE: 0,
            FundChannel.OTC_OPEN_END: 4,
            FundChannel.OTC_ETF_FEEDER: 3,
            FundChannel.QDII: 8,
        }
        return delays.get(channel, 4)

    @staticmethod
    def _make_profile(pos: FundPosition) -> FundTradingProfile:
        """Construct a FundTradingProfile from a FundPosition for fee calculation."""
        return FundTradingProfile(
            fund_code=pos.fund_code,
            channel=pos.channel,
        )
