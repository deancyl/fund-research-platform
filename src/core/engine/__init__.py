"""Layer 3: Backtesting and validation engine.

Includes the China-specific trading rules:
  - order_cutoff.py  — 15:00 subscription/redemption cutoff
  - redemption_fee.py — 2026 statutory redemption fee schedule + FIFO lot tracking
  - cash_lock.py      — Settlement delay state machine (CashLockManager)
  - capacity_gate.py  — Large subscription limits / mass redemption triggers
  - slippage.py       — A-share ETF slippage model (Level-2 order book)
  - validation.py     — CPCV + PBO + DSR overfitting detection

Dual-track architecture:
  - VectorBT: factor rough screening only (NO CashLockManager)
  - Backtrader: full event-driven with all China rules
"""

__all__: list[str] = []
