"""
SmartRebalancer — composes redemption_fee, cash_lock, and order_cutoff
to produce complete rebalancing plans with friction cost analysis.

v0.1.1: Fixed 3 critical bugs per external audit:
  - Fee penalty gate now uses holding-period-amortized alpha (not daily alpha).
  - Config drift eliminated: binds to CashLockManager for settlement delays.
  - Timeline vacuum fixed: SUBSCRIBE t_day = max_settle_delay of all redeems.
"""

from dataclasses import dataclass, field
from datetime import date

from src.core.data.schema import FundChannel, FundPosition, FundTradingProfile
from src.core.engine.cash_lock import CashLockManager
from src.core.engine.redemption_fee import RedemptionFeeCalculator


@dataclass(frozen=True, slots=True)
class RebalanceAction:
    action_type: str
    fund_code: str
    fund_name: str
    amount: float
    estimated_fee: float = 0.0
    net_cash: float = 0.0
    reason: str = ""
    skip_reason: str = ""


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    t_day: int
    event: str


@dataclass
class RebalancePlan:
    status: str = "ANALYZED"
    actions: list[RebalanceAction] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    total_friction_cost_yuan: float = 0.0
    cash_flow_vacuum_days: int = 0
    ai_advisor_note: str = ""


class SmartRebalancer:
    """
    Generate rebalancing plans with full China-specific rule compliance.

    v0.1.1 fixes:
      - Binds to CashLockManager for settlement delays (single source of truth).
      - Fee gate uses holding-period-amortized alpha, not daily alpha.
      - SUBSCRIBE timeline uses max_settle_delay from redeems, not target channel.
    """

    def __init__(self, delay_override: dict[FundChannel, int] | None = None) -> None:
        self._fee_calc = RedemptionFeeCalculator()
        self._cash_ref = CashLockManager(initial_cash=0.0, delay_override=delay_override)

    def generate_rebalance_plan(
        self,
        current_portfolio: list[FundPosition],
        target_weights: dict[str, float],
        current_date: date,
        total_portfolio_value: float,
        expected_alpha_pct: float = 0.03,
        expected_holding_days: int = 90,
    ) -> RebalancePlan:
        """
        Produce a complete rebalancing plan.

        Args:
            current_portfolio: User's current positions with Lot-level detail.
            target_weights: Desired allocation {fund_code: weight_pct}.
            current_date: The date for holding-period calculation.
            total_portfolio_value: Total portfolio value in CNY.
            expected_alpha_pct: Expected annualized alpha (default 3%).
            expected_holding_days: Expected holding period — used to amortize
                                   one-time redemption fee against total alpha.
        """
        plan = RebalancePlan()
        max_settle_delay = 0
        executable_redeem_cash = 0.0  # Only non-skipped redeems release real cash

        # ── Phase 1: Overweight → REDEEM ────────────────────────────────
        for pos in current_portfolio:
            target_w = target_weights.get(pos.fund_code, 0.0)
            if pos.weight_pct <= target_w:
                continue

            reduction_ratio = pos.weight_pct - target_w
            reduction_value = total_portfolio_value * reduction_ratio
            shares_to_redeem = reduction_value / pos.current_nav if pos.current_nav > 0 else 0.0

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

            # 【审计修复】Fee penalty gate: amortize one-time fee over holding period
            holding_period_alpha = (expected_alpha_pct / 365.0) * expected_holding_days
            if rate > holding_period_alpha and rate >= 0.005:
                skip_reason = (
                    f"赎回费 {rate:.2%} 超出持仓周期预期Alpha ({holding_period_alpha:.2%})，建议推迟调仓"
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

            if not skip_reason:
                plan.total_friction_cost_yuan += fee
                executable_redeem_cash += net_cash  # Only non-skipped redeems release cash

            # 【审计修复】Use CashLockManager as single source of truth for delays
            delay = self._cash_ref.settlement_delays.get(pos.channel, 4)
            max_settle_delay = max(max_settle_delay, delay)

            plan.timeline.append(
                TimelineEvent(t_day=0, event=f"提交 {pos.fund_name}({pos.fund_code}) 赎回申请")
            )
            plan.timeline.append(
                TimelineEvent(
                    t_day=delay,
                    event=f"{pos.fund_name}({pos.fund_code}) 赎回款 {net_cash:,.2f} 元解冻 (交收延迟 {delay}天)",
                )
            )

        # ── Phase 2: Underweight / new → SUBSCRIBE ──────────────────────
        # Constrained by actual cash from non-skipped REDEEM actions (audit fix).
        # Exception: when NO redeems at all, assume fresh cash injection is available.
        has_redeems = any(a.action_type == "REDEEM" for a in plan.actions)
        deployable_cash = executable_redeem_cash if has_redeems else float("inf")
        all_codes = set(target_weights.keys()) | {p.fund_code for p in current_portfolio}
        for code in all_codes:
            target_w = target_weights.get(code, 0.0)
            current_pos = next((p for p in current_portfolio if p.fund_code == code), None)
            current_w = current_pos.weight_pct if current_pos else 0.0

            if target_w <= current_w:
                continue

            increase_ratio = target_w - current_w
            subscribe_value = total_portfolio_value * increase_ratio

            # 【审计修复】Constrained by actual cash from non-skipped redeems
            # When deployable_cash is inf (no redeems at all), use fresh cash injection.
            if deployable_cash == float("inf"):
                reason_suffix = ""
            elif deployable_cash <= 0:
                subscribe_value = 0.0
                reason_suffix = " (无可用于申购的赎回资金)"
            elif subscribe_value > deployable_cash:
                subscribe_value = deployable_cash
                reason_suffix = f" (受限于可用现金 {deployable_cash:,.2f})"
            else:
                reason_suffix = ""

            fund_name = current_pos.fund_name if current_pos else code

            if subscribe_value <= 0:
                continue  # No cash available → skip this SUBSCRIBE

            plan.actions.append(
                RebalanceAction(
                    action_type="SUBSCRIBE",
                    fund_code=code,
                    fund_name=fund_name,
                    amount=round(subscribe_value, 2),
                    net_cash=round(subscribe_value, 2),
                    reason=f"补充配置，目标仓位 {target_w:.1%}{reason_suffix}",
                )
            )

            # 【审计修复】SUBSCRIBE after ALL redeems' cash is unlocked (max_settle_delay)
            plan.timeline.append(
                TimelineEvent(
                    t_day=max_settle_delay,
                    event=f"申购 {fund_name}({code}) {subscribe_value:,.2f} 元 (资金已全部解冻)",
                )
            )
            deployable_cash = deployable_cash - subscribe_value if deployable_cash != float("inf") else deployable_cash

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
    def _make_profile(pos: FundPosition) -> FundTradingProfile:
        return FundTradingProfile(fund_code=pos.fund_code, channel=pos.channel)
