"""
Dedicated K-line chart screen using textual-plotext for interactive OHLCV display.
With zoom/pan controls and volume subplot. Activated by "/" + fund code.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header
from textual_plotext import PlotextPlot

from src.core.api import get_fund_kline, get_fund_profile


class KLineScreen(Screen):
    """Full-screen interactive candlestick chart with volume subplot."""

    BINDINGS = [
        ("q", "dismiss", "返回主界面"),
        ("escape", "dismiss", "返回主界面"),
        ("left", "pan_left", "◀ 平移"),
        ("right", "pan_right", "▶ 平移"),
        ("up", "zoom_in", "＋ 放大"),
        ("down", "zoom_out", "－ 缩小"),
        ("home", "reset_view", "重置"),
    ]

    def __init__(self, fund_code: str) -> None:
        super().__init__()
        self.fund_code = fund_code
        self.all_bars: list[dict] = []
        self.view_size: int = 40
        self.start_idx: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield PlotextPlot()
        yield Footer()

    def on_mount(self) -> None:
        self.all_bars = get_fund_kline(self.fund_code, limit=200)
        profile = get_fund_profile(self.fund_code)
        self.title = f"📈 {profile['fund_name']} ({self.fund_code})"
        self.start_idx = max(0, len(self.all_bars) - self.view_size)
        self._render()

    def _render(self) -> None:
        """Render K-line + volume using plotext subplots."""
        end_idx = self.start_idx + self.view_size
        bars = self.all_bars[self.start_idx:end_idx]
        if not bars:
            return

        dates = [b["date"] for b in bars]
        opens = [b["open"] for b in bars]
        highs = [b["high"] for b in bars]
        lows = [b["low"] for b in bars]
        closes = [b["close"] for b in bars]
        volumes = [b["volume"] for b in bars]
        price_labels = [b["date"][5:] for b in bars]  # MM-DD for x-axis

        plot = self.query_one(PlotextPlot)
        plt = plot.plt
        plt.clear_figure()
        plt.subplots(2, 1)
        plt.subplot(1, 1).title(f"{self.title} ({self.view_size}根K线)")
        plt.date_form("")
        # Use numeric x-axis to avoid weekend gaps
        x_idx = list(range(len(dates)))
        plt.candlestick(x_idx, {"Open": opens, "High": highs, "Low": lows, "Close": closes})
        plt.ylabel("净值")
        plt.grid(True, True)
        # Label every 5th bar
        tick_pos = list(range(0, len(dates), max(1, len(dates) // 8)))
        tick_labels = [price_labels[i] for i in tick_pos]
        plt.xticks(tick_pos, tick_labels)

        plt.subplot(2, 1)
        plt.bar(x_idx, volumes, color="cyan")
        plt.ylabel("成交额")
        plt.xticks(tick_pos, tick_labels)
        plt.grid(True, True)

        plot.refresh()

    def action_pan_left(self) -> None:
        self.start_idx = max(0, self.start_idx - 5)
        self._render()

    def action_pan_right(self) -> None:
        max_idx = max(0, len(self.all_bars) - self.view_size)
        self.start_idx = min(max_idx, self.start_idx + 5)
        self._render()

    def action_zoom_in(self) -> None:
        if self.view_size > 10:
            self.view_size -= 5
            self.start_idx = max(0, min(self.start_idx + 3, len(self.all_bars) - self.view_size))
            self._render()

    def action_zoom_out(self) -> None:
        if self.view_size < len(self.all_bars):
            self.view_size += 5
            self.start_idx = max(0, self.start_idx - 2)
            self._render()

    def action_reset_view(self) -> None:
        self.view_size = 40
        self.start_idx = max(0, len(self.all_bars) - self.view_size)
        self._render()
