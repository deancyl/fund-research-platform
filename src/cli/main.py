"""
CLI entry point — fund-research command.

Usage:
    fund-research tui       Launch the Textual terminal interface
    fund-research web       Launch the FastAPI web server
    fund-research update    Sync fund data from AKShare
    fund-research analyze   Run analysis on a specific fund
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLI] %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cli")


def cmd_tui() -> None:
    """Launch the TUI terminal interface."""
    logger.info("启动 TUI 终端界面")
    from src.tui.app import launch_tui
    launch_tui()


def cmd_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the Web API server."""
    logger.info("启动 Web 服务: %s:%d", host, port)
    from src.web.server import launch_web
    launch_web(host=host, port=port)


def cmd_update() -> None:
    """Sync fund data from AKShare."""
    logger.info("开始数据同步...")
    print("数据同步功能开发中 — 连接 AKShare 以获取实时数据")
    logger.info("数据同步完成 (stub)")


def cmd_analyze(fund_code: str) -> None:
    """Run analysis on a specific fund."""
    logger.info("分析基金: %s", fund_code)
    from src.core.api import calculate_redemption_fee, check_holding_warning
    from src.core.data.schema import FundCategory, FundChannel

    for days in [6, 15, 60, 200]:
        result = calculate_redemption_fee(
            fund_code=fund_code, channel=FundChannel.OTC_OPEN_END,
            category=FundCategory.EQUITY, holding_days=days, redemption_amount=10000.0,
        )
        warning = check_holding_warning(days, FundChannel.OTC_OPEN_END)
        print(f"  持有 {days:3d}天 → 赎回费 {result['rate']:.2%} ({result['fee_amount_cny']:.2f} CNY) [{warning.value}]")
    logger.info("分析完成: %s", fund_code)


def main() -> None:
    """Main entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("fund-research — 中国基金/指数基金 AI 投研平台")
        print()
        print("用法:")
        print("  fund-research tui              启动 TUI 终端界面")
        print("  fund-research web [--port PORT] 启动 Web API 服务")
        print("  fund-research update            同步基金数据")
        print("  fund-research analyze CODE      分析指定基金")
        return

    cmd = args[0]

    if cmd == "tui":
        cmd_tui()
    elif cmd == "web":
        port = 8000
        if len(args) > 2 and args[1] == "--port":
            port = int(args[2])
        cmd_web(port=port)
    elif cmd == "update":
        cmd_update()
    elif cmd == "analyze":
        if len(args) < 2:
            print("错误: 请指定基金代码, 例如: fund-research analyze 005827")
            sys.exit(1)
        cmd_analyze(args[1])
    else:
        print(f"未知命令: {cmd}")
        print("可用命令: tui, web, update, analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()
