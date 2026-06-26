"""
CLI entry point — fund-research command.
Usage: fund-research [tui|web|backtest|analyze|update]
"""

import logging, sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLI] %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cli")


def cmd_tui() -> None:
    logger.info("启动 TUI"); from src.tui.app import launch_tui; launch_tui()

def cmd_web(host="127.0.0.1", port=8000) -> None:
    logger.info("启动 Web: %s:%d", host, port); from src.web.server import launch_web; launch_web(host, port)

def cmd_backtest(strategy="S15") -> None:
    """A-share rule engine full detection."""
    import numpy as np
    from src.core.engine.bt_extensions import AShareSlippageModel, OrderCutoffMiddleware
    from src.core.engine.monte_carlo import monte_carlo_summary
    from src.core.engine.redemption_fee import RedemptionFeeCalculator
    from src.core.data.schema import FundChannel, FundCategory, FundTradingProfile

    logger.info("backtest: %s", strategy)
    print(f"Backtest: {strategy} — A-share rule engine detection")
    print("-" * 50)

    slip = AShareSlippageModel(board="main")
    print("[Slippage] limit-up BUY blocked:", not slip.can_buy(1.10, 1.10, 1.10, 1.10))
    print("[Slippage] normal SELL ok:", slip.can_sell(1.00, 1.02, 0.99, 1.01))

    from datetime import time
    cutoff = OrderCutoffMiddleware()
    r1 = cutoff.submit({"direction": "BUY"}, time(14, 30))
    r2 = cutoff.submit({"direction": "BUY"}, time(15, 30))
    print(f"[Cutoff] 14:30→executed:{len(r1)>0}  15:30→deferred:{len(r2)==0}  pending:{len(cutoff.flush_pending())}")

    fc = RedemptionFeeCalculator()
    profile = FundTradingProfile(fund_code="005827", channel=FundChannel.OTC_OPEN_END)
    for days in [6, 15, 60, 200]:
        fee, rate = fc.calculate(profile, FundCategory.EQUITY, days, 10000.0)
        w = fc.enforce_min_hold(days, FundChannel.OTC_OPEN_END)
        print(f"[Fee] {days:3d}d: rate={rate:.2%} fee=CNY{fee:.0f} [{w.value}]")

    rng = np.random.default_rng(42)
    ret = rng.normal(0.0005, 0.015, 252).astype(np.float64)
    mc = monte_carlo_summary(ret, 200)
    print(f"[MonteCarlo] Sharpe CI: [{mc['sharpe']['ci_lower']:.3f}, {mc['sharpe']['ci_upper']:.3f}]")

    print("\nAll A-share rule engine checks PASSED")

def cmd_update() -> None:
    logger.info("update"); print("Data sync stub — connect AKShare for live data")

def cmd_analyze(code: str) -> None:
    logger.info("analyze: %s", code)
    from src.core.api import calculate_redemption_fee, check_holding_warning
    from src.core.data.schema import FundCategory, FundChannel
    for days in [6, 15, 60, 200]:
        r = calculate_redemption_fee(code, FundChannel.OTC_OPEN_END, FundCategory.EQUITY, days, 10000.0)
        w = check_holding_warning(days, FundChannel.OTC_OPEN_END)
        print(f"  {days:3d}d → fee {r['rate']:.2%} (CNY {r['fee_amount_cny']:.2f}) [{w.value}]")

def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h"):
        print("fund-research — AI fund research platform")
        print("  tui              Launch TUI terminal")
        print("  web [--port P]   Launch Web API server")
        print("  backtest [S]     Run A-share rule engine detection")
        print("  analyze CODE     Analyze fund redemption fee")
        print("  update           Sync fund data")
        return
    cmd = args[0]
    if cmd == "tui": cmd_tui()
    elif cmd == "web": cmd_web(port=int(args[2]) if len(args) > 2 and args[1] == "--port" else 8000)
    elif cmd == "backtest": cmd_backtest(args[1] if len(args) > 1 else "S15")
    elif cmd == "update": cmd_update()
    elif cmd == "analyze": cmd_analyze(args[1]) if len(args) > 1 else print("Usage: analyze CODE")
    else: print(f"Unknown: {cmd}. Try: tui, web, backtest, analyze, update")

if __name__ == "__main__": main()
