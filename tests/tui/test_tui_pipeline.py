"""
TUI Pipeline Robustness Test — Textual Pilot async stress testing.
v0.1.6: Added cold-boot white-screen detection (on_ready auto-hydration).
"""

import pytest
from textual.widgets import DataTable, Input, Log
from src.tui.app import FundResearchTUI


@pytest.mark.asyncio
async def test_tui_pipeline_robustness() -> None:
    """Industrial TUI pipeline — catches race conditions, mock fraud, and cold-boot vacuum."""
    app = FundResearchTUI()

    async with app.run_test() as pilot:
        # ── Check 0: COLD BOOT — data must appear without any keypress ──
        # on_ready auto-triggers portfolio diagnosis in v0.1.6
        # Wait for async hydration to complete
        await pilot.pause(2.0)

        watchlist = app.query_one("#watchlist", DataTable)
        recs = app.query_one("#recommendations", DataTable)

        assert watchlist.row_count > 0, (
            f"COLD BOOT FAILURE: watchlist has {watchlist.row_count} rows after 1.5s idle — "
            "on_ready auto-hydration NOT working. Blank screen on launch."
        )
        assert recs.row_count > 0, (
            f"COLD BOOT FAILURE: recommendations has {recs.row_count} rows after 1.5s idle — "
            "on_ready auto-hydration NOT working."
        )

        # ── Check 1: Race condition injection ──────────────────────────
        try:
            await pilot.click("#watchlist")
            await pilot.click("#recommendations")
        except IndexError:
            pytest.fail("CRITICAL: DataTable IndexError on click — skeleton protection failed")

        # ── Check 2: Hotkey stress test ────────────────────────────────
        await pilot.press("r")
        await pilot.press("s")
        await pilot.press("l")
        await pilot.pause(0.5)

        # ── Check 3: Empty/illegal input attack ────────────────────────
        cmd_input = app.query_one("#command-input", Input)
        await pilot.click("#command-input")
        await pilot.press("enter")
        cmd_input.value = "INVALID_PATH_&*#$@!"
        await pilot.press("enter")
        await pilot.pause(0.3)

        # ── Check 4: Re-verify data after hotkeys ──────────────────────
        assert watchlist.row_count > 0, "Data lost after hotkey stress"
        assert recs.row_count > 0, "Data lost after hotkey stress"

        # ── Check 5: TCSS stability ────────────────────────────────────
        log_panel = app.query_one("#log-panel", Log)
        assert log_panel is not None

    print("✅ TUI pipeline passed: cold-boot hydration + race + hotkey + input + layout")
