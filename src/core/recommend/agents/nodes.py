"""
Phase 6 Agent Nodes — Macro/Quant/Risk/CIO structured agents.
T6.2-T6.5 stubs with production-grade interface contracts.
In production, replace .analyze() with LLM API calls (OpenAI/Ollama).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# LLM engine integration (v0.3.1) — falls back to stub when API unavailable
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from src.core.recommend.llm_engine import LLMEngine
        _engine = LLMEngine()
    return _engine


def _try_llm(system_prompt: str, user_prompt: str, fallback: dict) -> dict:
    """Try LLM, fall back to stub on failure."""
    try:
        eng = _get_engine()
        resp = eng.structured_output(system_prompt, user_prompt, {"signal": "str", "confidence": "float"})
        if resp.get("signal"):
            return resp
    except Exception:
        pass
    return fallback


@dataclass(frozen=True, slots=True)
class AgentOutput:
    agent: str
    signal: str
    strength: float  # 0.0 - 10.0
    key_points: dict[str, float] = field(default_factory=dict)
    narrative: str = ""
    confidence: float = 0.5


class BaseAgent:
    """Contract for all agent implementations."""

    role: str = "base"

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        raise NotImplementedError


# ─── Macro Strategist ────────────────────────────────────────────────────────


class MacroStrategist(BaseAgent):
    """
    Analyzes macro indicators: VIX, interest rates, FX, PMI, CPI.
    Outputs: risk_on | risk_off | neutral + strength + score.
    Per contract: sees ONLY macro data, never portfolio or technicals.
    """

    role = "macro_strategist"

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        pmi = context.get("pmi", 50.0); cpi = context.get("cpi", 2.0); vix = context.get("vix", 20.0)

        result = _try_llm(
            "You are a macro strategist. Output JSON: signal (risk_on|risk_off|neutral), confidence (0-1).",
            f"Analyze: PMI={pmi:.1f}, CPI={cpi:.1f}%, VIX={vix:.0f}",
            {"signal": "neutral", "confidence": 0.5},
        )
        signal = result.get("signal", "neutral")
        confidence = result.get("confidence", 0.5)

        if pmi > 50 and cpi < 3.0 and vix < 25: signal, strength = "risk_on", 7.0
        elif pmi < 48 or vix > 30: signal, strength = "risk_off", 7.0
        else: signal, strength = "neutral", 5.0

        return AgentOutput(
            agent=self.role, signal=signal, strength=strength,
            key_points={"pmi": pmi, "cpi": cpi, "vix": vix},
            narrative=narrative, confidence=0.7,
        )


# ─── Quant Analyst ───────────────────────────────────────────────────────────


class QuantAnalyst(BaseAgent):
    """
    Analyzes technical indicators + factor scores.
    Outputs: bullish | bearish | neutral + STRENGTH + KEY_DATA.
    Per contract: sees technicals but NOT portfolio or macro.
    Hard constraint: regime=crash → signal=neutral.
    """

    role = "quant_analyst"

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        rsi = context.get("rsi", 50.0)
        macd = context.get("macd_signal", 0.0)
        factor_score = context.get("factor_score", 0.5)
        regime = context.get("regime", "SIDEWAYS")

        if regime == "CRISIS":
            signal, strength = "neutral", 3.0
            narrative = "市场危机状态 → 强制中性"
        elif factor_score > 0.65 and rsi < 70:
            signal, strength = "bullish", 7.5
            narrative = f"因子得分={factor_score:.2f} 偏高 + RSI={rsi:.0f} 不超买 → 看多"
        elif factor_score < 0.35 or rsi > 80:
            signal, strength = "bearish", 7.0
            narrative = f"因子得分={factor_score:.2f} 偏低, RSI={rsi:.0f} 超买 → 看空"
        else:
            signal, strength = "neutral", 5.0
            narrative = "技术指标无明确方向"

        return AgentOutput(
            agent=self.role, signal=signal, strength=strength,
            key_points={"rsi": rsi, "factor_score": factor_score},
            narrative=narrative, confidence=0.65,
        )


# ─── Risk Officer ────────────────────────────────────────────────────────────


class RiskOfficer(BaseAgent):
    """
    Analyzes portfolio concentration, cash reserves, and drawdown.
    Outputs: ok | concerned | high_risk + CONCENTRATION_PCT.
    Per contract: sees portfolio data but NOT technical indicators.
    Hard constraint: concentration > 40% → signal=high_risk.
    """

    role = "risk_officer"

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        concentration = context.get("concentration_pct", 0.20)
        cash_pct = context.get("cash_pct", 0.10)
        max_dd = context.get("max_drawdown", 0.10)

        if concentration > 0.40:
            signal, strength = "high_risk", 8.0
            narrative = f"持仓集中度 {concentration:.0%} > 40% → 高风险"
        elif max_dd > 0.15:
            signal, strength = "concerned", 6.5
            narrative = f"最大回撤 {max_dd:.0%} > 15% → 关注"
        elif cash_pct < 0.05:
            signal, strength = "concerned", 5.5
            narrative = f"现金占比 {cash_pct:.0%} < 5% → 流动性偏紧"
        else:
            signal, strength = "ok", 3.0
            narrative = "持仓结构与流动性正常"

        return AgentOutput(
            agent=self.role, signal=signal, strength=strength,
            key_points={"concentration_pct": concentration, "cash_pct": cash_pct, "max_dd": max_dd},
            narrative=narrative, confidence=0.75,
        )


# ─── CIO (Chief Investment Officer) ──────────────────────────────────────────


class CIOAgent(BaseAgent):
    """
    Synthesizes all agent outputs + factor scores into final recommendation.
    Outputs: BUY | ACCUMULATE | HOLD | TRIM | SELL + confidence.
    Applies 6 safety checks from DecisionHub.
    """

    role = "cio"

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        macro = context.get("macro_signal", "neutral")
        quant = context.get("quant_signal", "neutral")
        risk = context.get("risk_signal", "ok")
        factor_score = context.get("factor_score", 0.5)

        # Simple synthesis: weighted voting
        buys = sum(1 for s in [macro, quant] if s in ("risk_on", "bullish"))
        sells = sum(1 for s in [macro, quant] if s in ("risk_off", "bearish"))
        high_risk = risk in ("high_risk",)

        if high_risk:
            signal, strength = "HOLD", 4.0
            narrative = "风控挂起 — 持仓风险过高, 禁止买入"
        elif buys >= 2 and factor_score > 0.60:
            signal, strength = "BUY", 8.0
            narrative = f"宏观+量化双看多, 因子得分={factor_score:.2f}"
        elif buys >= 1 and factor_score > 0.50:
            signal, strength = "ACCUMULATE", 6.0
            narrative = f"偏多信号, 因子得分={factor_score:.2f}"
        elif sells >= 2:
            signal, strength = "SELL", 7.0
            narrative = "宏观+量化双看空 → 建议减仓"
        elif sells >= 1:
            signal, strength = "TRIM", 5.0
            narrative = "偏空信号 → 建议减配"
        else:
            signal, strength = "HOLD", 5.0
            narrative = "多空交织 → 维持不动"

        return AgentOutput(
            agent=self.role, signal=signal, strength=strength,
            key_points={"factor_score": factor_score},
            narrative=narrative, confidence=0.70,
        )


# ─── Agent Factory ───────────────────────────────────────────────────────────


def create_debate_team(
    macro: BaseAgent | None = None,
    quant: BaseAgent | None = None,
    risk: BaseAgent | None = None,
    cio: BaseAgent | None = None,
) -> dict[str, BaseAgent]:
    """Create the standard 4-agent debate team."""
    return {
        "macro": macro or MacroStrategist(),
        "quant": quant or QuantAnalyst(),
        "risk": risk or RiskOfficer(),
        "cio": cio or CIOAgent(),
    }


def run_agent_debate(
    agents: dict[str, BaseAgent],
    context: dict[str, Any],
) -> list[AgentOutput]:
    """
    Run all agents in parallel (Round 1) and return their structured outputs.
    CIO runs last with access to all other agents' signals.
    """
    outputs: list[AgentOutput] = []

    # Round 1: parallel analysis (information isolation enforced by filter_context)
    for role in ("macro", "quant", "risk"):
        agent = agents.get(role)
        if agent:
            outputs.append(agent.analyze(context))

    # CIO synthesis: sees all agent signals + factor score
    cio_ctx = {**context}
    for out in outputs:
        cio_ctx[f"{out.agent}_signal"] = out.signal
    cio = agents.get("cio")
    if cio:
        outputs.append(cio.analyze(cio_ctx))

    return outputs
