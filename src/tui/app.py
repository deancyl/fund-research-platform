"""
TUI adapter v0.5.0 — Professional terminal with summary dashboard.
Color-coded signals (red=up, green=down per CN convention).
"""

from datetime import date, datetime, timedelta
import logging

from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Label, Log, Static

from src.core.api import generate_rebalance_plan, get_fund_kline, get_fund_profile
from src.core.data.schema import FundCategory, FundChannel, FundPosition, PositionLot
from src.core.data.lookup import FundLookupEngine
from src.core.engine.ledger import LedgerEngine, TradeDirection
from src.core.version import VERSION

logger = logging.getLogger("tui")
logger.setLevel(logging.DEBUG)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s [TUI] %(levelname)s %(message)s", "%H:%M:%S"))
logger.addHandler(_h)


class FundResearchTUI(App):
    CSS = """
    Screen { background: #0d1117; color: #c9d1d9; }
    #summary-bar { height: 3; background: #161b22; border-bottom: solid #30363d; padding: 0 2; }
    #main-grid { grid-size: 2 2; grid-gutter: 1 2; height: 70%; padding: 1; }
    #watchlist { border: solid #30363d; background: #161b22; height: 100%; }
    #watchlist:focus { border: solid #58a6ff; }
    #chart { border: solid #30363d; background: #161b22; padding: 1; content-align: center middle; }
    #recommendations { border: solid #30363d; background: #161b22; height: 100%; }
    #recommendations:focus { border: solid #58a6ff; }
    #portfolio { border: solid #30363d; background: #161b22; padding: 1; }
    #status-bar { dock: bottom; height: 1; background: #21262d; color: #58a6ff; padding: 0 2; }
    #log-panel { border: solid #30363d; background: #0d1117; height: 6; dock: bottom; display: none; }
    #command-input { dock: bottom; width: 100%; border: none; background: #21262d; color: #58a6ff; }
    .up { color: #ef4444; } .down { color: #22c55e; }
    .buy { color: #ef4444; } .sell { color: #22c55e; } .hold { color: #f59e0b; }
    .summary-label { color: #8b949e; text-style: bold; }
    .summary-value { color: #f0f6fc; }
    """

    BINDINGS = [
        ("q", "quit", "退出"),
        ("r", "refresh_portfolio", "刷新"),
        ("s", "run_strategies", "策略"),
        ("l", "toggle_log", "日志"),
        ("/", "focus_input", "搜索"),
    ]

    TITLE = f"🏦 基金量化投研终端 v{VERSION}"
    _log_visible: bool = False
    _lookup: FundLookupEngine = FundLookupEngine()
    _ledger: LedgerEngine = LedgerEngine()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("", id="summary-bar")
        with Grid(id="main-grid"):
            yield DataTable(id="watchlist", cursor_type="row")
            yield Static("📊 输入代码或名称查看走势图 (完整K线请用Web端)", id="chart")
            yield DataTable(id="recommendations", cursor_type="row")
            yield Static("📋 基金画像 | 点击自选列表基金行查看详情", id="portfolio")
        yield Label(" 💡 就绪 — /搜索 r刷新 s策略 q退出", id="status-bar")
        yield Log(id="log-panel", max_lines=100)
        yield Input(placeholder="🔍 输入基金代码/名称/拼音 如 005827 或 易方达...", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_skeleton()

    async def on_ready(self) -> None:
        self._update_status("⏳ 加载持仓数据...")
        await self.action_refresh_portfolio()

    def _setup_skeleton(self) -> None:
        wl = self.query_one("#watchlist", DataTable)
        wl.clear(columns=True)
        wl.add_columns("代码", "名称", "净值", "涨跌%", "持仓占比")
        recs = self.query_one("#recommendations", DataTable)
        recs.clear(columns=True)
        recs.add_columns("动作", "代码", "名称", "金额", "规费", "理由")

    # ── Row click → K-line ────────────────────────────────────────────

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        table = event.data_table
        if table.id == "watchlist":
            row_key = event.row_key
            if row_key is not None:
                row = table.get_row(row_key)
                if row and len(row) >= 1:
                    fund_code = str(row[0])
                    self._update_status(f"📈 加载 {fund_code} K线...")
                    self.run_worker(self._render_kline(fund_code), thread=True)

    # ── Portfolio diagnosis ─────────────────────────────────────────────

    async def action_refresh_portfolio(self) -> None:
        self.run_worker(self._diagnose(), thread=True)

    async def _diagnose(self) -> None:
        lots = [
            PositionLot(purchase_date=date(2026, 6, 10), shares=8000.0, purchase_nav=1.92, cost_amount=15360.0),
            PositionLot(purchase_date=date(2026, 1, 15), shares=30000.0, purchase_nav=1.80, cost_amount=54000.0),
        ]
        portfolio = [
            FundPosition(fund_code="005827", fund_name="易方达蓝筹精选", category=FundCategory.EQUITY,
                         channel=FundChannel.OTC_OPEN_END, lots=lots, current_nav=1.85,
                         total_shares=38000.0, market_value=70300.0, weight_pct=0.65),
        ]
        target = {"005827": 0.20, "510300": 0.80}
        total_value = 108153.0
        plan = generate_rebalance_plan(portfolio, target, date(2026, 6, 26), total_value)
        self.call_from_thread(self._render_plan, portfolio, plan, total_value)

    def _render_plan(self, portfolio: list, plan, total_value: float) -> None:
        wl = self.query_one("#watchlist", DataTable)
        wl.clear(columns=False)
        wl.add_row("005827", "易方达蓝筹精选", "1.8500", "[-0.41%]", "[65.0%]")
        wl.add_row("510300", "沪深300ETF", "3.9200", "[+0.51%]", "[35.0%]")

        recs = self.query_one("#recommendations", DataTable)
        recs.clear(columns=False)
        for a in plan.actions:
            skip = a.skip_reason or "执行"
            recs.add_row(a.action_type, a.fund_code, a.fund_name,
                         f"¥{a.amount:,.0f}", f"¥{a.estimated_fee:,.0f}",
                         f"{a.reason} [{skip}]")

        self._update_summary(total_value, plan.total_friction_cost_yuan)
        self._update_status(f"✅ 就绪 | 市值 ¥{total_value:,.0f} | 摩擦 ¥{plan.total_friction_cost_yuan:,.0f}")

    def _update_summary(self, total_value: float, friction: float) -> None:
        bar = self.query_one("#summary-bar", Label)
        bar.update(
            f"  📊 总资产: ¥{total_value:,.0f}  │  ⚠️ 摩擦成本: ¥{friction:,.0f}  │  "
            f"📈 CSI300: 3,920 [+0.51%]  │  📉 创业板: 2,145 [-1.20%]  │  🔍 /搜索基金"
        )

    # ── Strategy run ────────────────────────────────────────────────────

    async def action_run_strategies(self) -> None:
        self._update_status("⏳ 策略运行中...")
        self.run_worker(self._run_strats(), thread=True)

    async def _run_strats(self) -> None:
        import time; time.sleep(0.5)
        self.call_from_thread(self._strat_done)

    def _strat_done(self) -> None:
        recs = self.query_one("#recommendations", DataTable)
        recs.clear(columns=False)
        recs.add_row("📈BUY", "159915", "创业板ETF", "¥35,000", "¥0", "动量因子排名#1")
        recs.add_row("⏸HOLD", "510300", "沪深300ETF", "¥0", "¥0", "PE=14.4 中性区间")
        recs.add_row("⏸HOLD", "005827", "易方达蓝筹", "¥0", "¥0", "30d持有费¥45.9 [REJECT]")
        self._update_status("✅ 策略完成 — 3信号")

    # ── Input: fuzzy search + K-line ────────────────────────────────────

    def action_focus_input(self) -> None:
        self.query_one("#command-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        q = event.value.strip()
        if len(q) < 1: return
        matches = self._lookup.search(q)
        if matches:
            preview = ", ".join([f"{m['code']} {m['name']}" for m in matches[:3]])
            self._update_status(f"🔎 {preview}")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        q = event.value.strip()
        event.input.clear()
        if q.lower() in ("q", "quit", "exit"): self.exit(); return

        # ── Command palette: :add / :rem ────────────────────────────
        if q.startswith(":add"):
            self._cmd_add(q); return
        if q.startswith(":rem"):
            self._cmd_rem(q); return

        # ── Fuzzy search ────────────────────────────────────────────
        matches = self._lookup.search(q)
        if matches:
            fc = matches[0]["code"]; nm = matches[0]["name"]
            self._update_status(f"📈 加载 {nm} ({fc}) K线...")
            self.run_worker(self._render_kline(fc), thread=True)
        elif q.isdigit() and len(q) >= 4:
            self.run_worker(self._render_kline(q), thread=True)
        else:
            self._update_status(f"🔍 未匹配: {q}")

    def _cmd_add(self, cmd: str) -> None:
        """Parse :add CODE buy AMOUNT [DATE]"""
        parts = cmd.split()
        if len(parts) < 4:
            self._update_status("❌ 用法: :add CODE buy 金额 [YYYY-MM-DD]"); return
        fund_code = parts[1]; date_str = parts[4] if len(parts) > 4 else str(date.today())
        try:
            amount = float(parts[3])
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=10)
            nav = 1.85  # stub — in production fetch from cache
            entry = self._ledger.record_subscribe(fund_code, dt, amount, nav)
            self._update_status(f"✅ 申购: {fund_code} ¥{amount:,.0f} → {entry.shares}份 [ID:{entry.tx_id}]")
            self._refresh_ledger_panel()
        except Exception as e:
            self._update_status(f"❌ 命令错误: {e}")

    def _cmd_rem(self, cmd: str) -> None:
        """Parse :rem CODE sell SHARES [DATE]"""
        parts = cmd.split()
        if len(parts) < 4:
            self._update_status("❌ 用法: :rem CODE sell 份额 [YYYY-MM-DD]"); return
        fund_code = parts[1]; date_str = parts[4] if len(parts) > 4 else str(date.today())
        try:
            shares = float(parts[3])
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=10)
            nav = 1.85  # stub
            entry = self._ledger.record_redeem(fund_code, dt, shares, nav)
            warning = ""
            if entry.fee_charged > 0:
                rate = entry.fee_charged / (shares * nav) * 100
                warning = f" ⚠️ 赎费¥{entry.fee_charged:,.2f} ({rate:.1f}%)"
            self._update_status(f"✅ 赎回: {fund_code} {shares}份 → ¥{entry.amount:,.0f}{warning}")
            self._refresh_ledger_panel()
        except Exception as e:
            self._update_status(f"❌ 命令错误: {e}")

    def _refresh_ledger_panel(self) -> None:
        """Update the ledger display in the chart/portfolio panel."""
        entries = self._ledger.get_ledger()
        lines = ["📒 调整流水账簿:"]
        for e in entries[-5:]:
            d = e.timestamp.strftime("%m-%d")
            sign = "+" if e.direction == TradeDirection.SUBSCRIBE else "-"
            amt = f"¥{e.amount:,.0f}" if e.amount else f"{e.shares}份"
            lines.append(f"  {d} {e.fund_code} {sign} {amt} [费¥{e.fee_charged:,.0f}]")
        self.query_one("#portfolio", Static).update("\n".join(lines))

    async def _render_kline(self, fund_code: str) -> None:
        """Readable ASCII chart with OHLC markers (full candlestick → use Web)."""
        import plotext as plt
        kline = get_fund_kline(fund_code, limit=30)
        profile = get_fund_profile(fund_code)
        plt.clf(); plt.theme("dark")
        dates = [b["date"][5:] for b in kline]
        closes = [b["close"] for b in kline]
        opens = [b["open"] for b in kline]
        highs = [b["high"] for b in kline]
        lows = [b["low"] for b in kline]
        plt.date_form("m-d")
        plt.plot(dates, closes, label="收盘价", color="cyan")
        plt.plot(dates, highs, label="最高", color="gray")
        plt.plot(dates, lows, label="最低", color="gray")
        for i, (o, c) in enumerate(zip(opens, closes)):
            marker = "▲" if c >= o else "▼"
            color = "red" if c >= o else "green"
            plt.scatter([dates[i]], [c], marker=marker, color=color)
        plt.title(f"{profile['fund_name']} ({fund_code}) ▸ 完整K线请用Web端")
        canvas = plt.build()
        self.call_from_thread(self._on_kline_done, canvas, profile)

    def _on_kline_done(self, canvas: str, profile: dict) -> None:
        self.query_one("#chart", Static).update(canvas)
        stocks = profile.get("top_ten_stocks", [])
        stocks_str = ", ".join([f"{s['name']}({s['pct']}%)" for s in stocks[:3]]) if stocks else "无数据"
        self.query_one("#portfolio", Static).update(
            f"🏦 {profile['fund_name']} ({profile['fund_code']})\n"
            f"👤 经理: {profile['manager']} | 成立: {profile['establishment_date']} | 规模: {profile['total_asset']}亿\n"
            f"📊 风格: [{profile['style_box']}]\n"
            f"📦 重仓: {stocks_str}"
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _update_status(self, text: str) -> None:
        self.query_one("#status-bar", Label).update(text)

    def action_toggle_log(self) -> None:
        self._log_visible = not self._log_visible
        lp = self.query_one("#log-panel", Log)
        lp.display = self._log_visible


def launch_tui() -> None:
    logger.info("launch_tui: v%s", VERSION)
    FundResearchTUI().run()
