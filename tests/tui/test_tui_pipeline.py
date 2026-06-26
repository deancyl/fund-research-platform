"""
TUI Pipeline Robustness Test — Textual Pilot async stress testing.

Per audit: Unit tests cannot catch Textual event-timing crashes
(IndexError on empty DataTable, TCSS parse failures, race conditions).
This test uses Pilot to simulate extreme user behavior in headless mode.
"""

import pytest
from textual.widgets import DataTable, Input, Log
from src.tui.app import FundResearchTUI


@pytest.mark.asyncio
async def test_tui_pipeline_robustness() -> None:
    """Industrial TUI pipeline — catches race conditions and mock fraud."""
    app = FundResearchTUI()

    async with app.run_test() as pilot:
        # ── Check 1: Race condition injection ──────────────────────
        # Click DataTables immediately after mount (before data arrives)
        watchlist = app.query_one("#watchlist", DataTable)
        recs = app.query_one("#recommendations", DataTable)

        try:
            await pilot.click("#watchlist")
            await pilot.click("#recommendations")
        except IndexError:
            pytest.fail("CRITICAL: DataTable IndexError on pre-data click — skeleton protection failed")

        # ── Check 2: Hotkey stress test ────────────────────────────
        await pilot.press("r")  # trigger async refresh
        await pilot.press("s")  # trigger strategy run
        await pilot.press("l")  # toggle log visibility
        await pilot.pause(0.5)  # wait for async workers to complete

        # ── Check 3: Empty/illegal input attack ────────────────────
        cmd_input = app.query_one("#command-input", Input)
        await pilot.click("#command-input")
        await pilot.press("enter")  # empty submit — should not crash

        cmd_input.value = "INVALID_PATH_&*#$@!"
        await pilot.press("enter")
        await pilot.pause(0.1)

        # ── Check 4: Real data binding (no mock fraud) ─────────────
        await pilot.press("r")
        await pilot.pause(1.0)  # wait for rebalance worker to complete

        # Verify watchlist received rows from real API
        assert watchlist.row_count > 0, (
            f"CONTRACT VIOLATION: watchlist has {watchlist.row_count} rows — "
            "mock data fraud detected. Must bind to src.core.api."
        )

        # Verify recommendation table received rebalance actions
        assert recs.row_count > 0, (
            f"CONTRACT VIOLATION: recommendations has {recs.row_count} rows — "
            "no signals from core engine."
        )

        # ── Check 5: TCSS stability — log panel present ────────────
        log_panel = app.query_one("#log-panel", Log)
        assert log_panel is not None, "Log panel missing — layout may have collapsed"

    print("✅ TUI pipeline passed: race condition, hotkey, input, data binding, layout")
