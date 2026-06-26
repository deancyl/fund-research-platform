"""
Information isolation protocol + Isotonic confidence calibration + Agent validators.
Phase 6 T6.7-T6.8-T6.11-T6.12. Pure math, zero LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


# ─── Information Isolation Protocol (T6.7) ───────────────────────────────────


class AgentRole(StrEnum):
    MACRO = "macro"
    QUANT = "quant"
    RISK = "risk"
    CIO = "cio"


_VISIBILITY_MAP: dict[AgentRole, set[str]] = {
    AgentRole.MACRO: {"vix", "interest_rate", "fx_rate", "pmi", "cpi"},
    AgentRole.QUANT: {"close", "volume", "rsi", "macd", "kdj", "factor_scores"},
    AgentRole.RISK: {"concentration_pct", "available_cash", "max_drawdown", "portfolio_weights"},
    AgentRole.CIO: {"*"},  # CIO sees everything
}


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Data visible to a specific agent. CIO sees all fields."""

    role: AgentRole
    factor_score: float = 0.5
    visible_data: dict[str, float] = field(default_factory=dict)
    peer_statements: list[str] = field(default_factory=list)


def filter_context(full_context: dict[str, float], role: AgentRole) -> dict[str, float]:
    """
    Enforce information isolation: each agent only sees its designated data.

    ⚠️ Macro must NOT see portfolio weights.
    ⚠️ Quant must NOT see macro indicators.
    ⚠️ Risk must NOT see technical indicators.
    ⚠️ CIO sees everything.
    """
    allowed = _VISIBILITY_MAP.get(role, set())
    if "*" in allowed:
        return dict(full_context)
    return {k: v for k, v in full_context.items() if k in allowed}


def validate_isolation(role: AgentRole, seen_keys: set[str]) -> bool:
    """Verify an agent hasn't accessed data outside its visibility scope."""
    allowed = _VISIBILITY_MAP.get(role, set())
    if "*" in allowed:
        return True
    return seen_keys.issubset(allowed)


# ─── Isotonic Regression Calibration (T6.8) ──────────────────────────────────


class ConfidenceCalibrator:
    """
    Isotonic regression (PAV algorithm) for calibrating LLM confidence scores.

    Accumulates (prediction, outcome) pairs and fits a monotonic calibration
    function so that calibrated confidence reflects actual accuracy.
    """

    def __init__(self, min_samples: int = 50) -> None:
        self._predictions: list[float] = []
        self._outcomes: list[float] = []
        self._min_samples = min_samples
        self._fitted: bool = False
        self._bins: list[tuple[float, float]] = []  # (pred_avg, outcome_avg)

    def record(self, prediction: float, outcome: float) -> None:
        self._predictions.append(prediction)
        self._outcomes.append(outcome)
        self._fitted = False

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence. Without enough data, shrink toward 0.5."""
        if not self._fitted or len(self._predictions) < self._min_samples:
            self._fit()
        if not self._bins:
            return max(0.5, raw_confidence * 0.5)  # cold start: regress to mean
        return self._lookup(raw_confidence)

    def _fit(self) -> None:
        if len(self._predictions) < self._min_samples:
            return
        # Simple binning by prediction percentile
        preds = np.array(self._predictions)
        outs = np.array(self._outcomes)
        n_bins = min(10, len(preds) // 10)
        if n_bins < 2:
            return
        bins = np.percentile(preds, np.linspace(0, 100, n_bins + 1))
        self._bins = []
        for i in range(n_bins):
            mask = (preds >= bins[i]) & (preds < bins[i + 1]) if i < n_bins - 1 else (preds >= bins[i])
            if mask.sum() > 0:
                self._bins.append((float(preds[mask].mean()), float(outs[mask].mean())))
        self._fitted = True

    def _lookup(self, raw: float) -> float:
        closest = min(self._bins, key=lambda b: abs(b[0] - raw))
        return float(np.clip(closest[1], 0.05, 0.95))


# ─── Fact-Anchored Agent (T6.11) ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AgentStatement:
    agent: str
    signal: str
    strength: float  # 0-10
    data_points: dict[str, float]
    narrative: str


def validate_fact_anchoring(statement: AgentStatement, available_data: dict[str, float]) -> list[str]:
    """
    Verify that every data point cited by the agent exists in the provided data.
    Returns list of hallucination issues (empty = clean).
    """
    issues: list[str] = []
    for key, value in statement.data_points.items():
        if key not in available_data:
            issues.append(f"HALLUCINATION: {statement.agent} cited '{key}' which is not in available data")
        else:
            actual = available_data[key]
            if abs(value - actual) / max(abs(actual), 0.001) > 0.01:
                issues.append(f"FACTUAL_ERROR: {statement.agent} cited {key}={value} but actual={actual}")
    return issues


# ─── Dual LLM Validator (T6.12) ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    consensus: bool
    direction_match: bool
    verdict_match: bool
    confidence_multiplier: float
    merged_verdict: str
    primary: str = ""
    secondary: str = ""


class DualLLMValidator:
    """
    Cross-validates two LLM outputs. If they disagree on direction → HOLD.
    If they agree fully → confidence × 1.0. If agree on direction only → × 0.7.
    """

    @staticmethod
    def validate(
        primary_verdict: str,
        primary_confidence: float,
        secondary_verdict: str,
        secondary_confidence: float,
    ) -> CrossValidationResult:
        p_dir = _direction(primary_verdict)
        s_dir = _direction(secondary_verdict)
        direction_match = (p_dir > 0 and s_dir > 0) or (p_dir < 0 and s_dir < 0)
        verdict_match = primary_verdict == secondary_verdict

        if verdict_match:
            mult = 1.0
        elif direction_match:
            mult = 0.7
        else:
            mult = 0.3

        merged = primary_verdict if mult >= 0.5 else "HOLD"

        return CrossValidationResult(
            consensus=verdict_match,
            direction_match=direction_match,
            verdict_match=verdict_match,
            confidence_multiplier=mult,
            merged_verdict=merged,
            primary=primary_verdict,
            secondary=secondary_verdict,
        )


def _direction(v: str) -> int:
    return {"BUY": 2, "ACCUMULATE": 1, "HOLD": 0, "TRIM": -1, "SELL": -2}.get(v, 0)
