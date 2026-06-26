#!/usr/bin/env python
"""
TUI CI Pipeline — runs Textual Pilot test, captures failures,
and prepares remediation context for AI agent self-healing.

Usage:
    python run_ai_tui_ci.py          # Run TUI pipeline test
    python run_ai_tui_ci.py --loop   # Run in loop until pass (AI fix mode)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REMEDIATION_FILE: Path = Path(".omo/remediation_context.txt")


def run_tui_test() -> tuple[bool, str]:
    """Run the TUI pipeline test. Returns (passed, output)."""
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/tui/test_tui_pipeline.py", "-vv", "--tb=long", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent,
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0, output


def write_remediation_context(output: str) -> None:
    """Package test failure output for AI agent consumption."""
    REMEDIATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    prompt = (
        "你的 TUI 流程化测试流水线失败了。Textual Pilot 在对 TUI 进行全真时序拷问时引发了崩溃。\n\n"
        "=== pytest 事故第一现场 ===\n"
        f"{output}\n\n"
        "请无条件执行以下自愈重构:\n"
        "1. 分析失败的组件 ID (如 #watchlist 或 #command-input)，如果是 IndexError，\n"
        "   必须使用'骨架屏保护范式'(skeleton columns on mount, clear(columns=False) only)。\n"
        "2. 严禁使用 Mock 数据应付测试，必须确保底层数据流从 src.core.api 回填。\n"
        "3. 重新输出修改后的 src/tui/app.py 源码，不准包含任何 TODO，\n"
        "   CI 流水线会自动重新跑测直到通过。\n"
    )
    REMEDIATION_FILE.write_text(prompt, encoding="utf-8")


def main() -> int:
    loop_mode = "--loop" in sys.argv

    while True:
        success, output = run_tui_test()

        if success:
            print("✅ TUI pipeline PASSED — 0 crash, real data bound.")
            return 0

        print("❌ TUI pipeline FAILED")
        print("-" * 60)
        # Print last 30 lines of output for quick diagnosis
        lines = output.strip().split("\n")
        for line in lines[-30:]:
            print(line)
        print("-" * 60)

        write_remediation_context(output)
        print(f"🤖 事故现场已导出至 {REMEDIATION_FILE}")
        print("   使用指令: opencode run --task fix_tui --input .omo/remediation_context.txt")

        if not loop_mode:
            return 1

        print("🔄 等待 AI 修复后重新跑测...")
        input("   修复完成后按 Enter 重新跑测...")


if __name__ == "__main__":
    sys.exit(main())
