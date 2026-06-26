"""
LLM Engine for agent physical ignition (Sprint 1).
Supports OpenAI-compatible APIs (DeepSeek, Ollama, Qwen).
Falls back gracefully to stub responses when no API key is configured.
"""

from __future__ import annotations

import json as _json
import os
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    content: str
    model: str = "unknown"
    tokens_used: int = 0
    error: str = ""


class LLMEngine:
    """
    OpenAI-compatible chat completion engine.

    Configure via environment variables:
      LLM_API_BASE  — API endpoint (default: http://localhost:11434/v1 for Ollama)
      LLM_API_KEY   — API key (default: "ollama" for local)
      LLM_MODEL     — Model name (default: "deepseek-r1:8b")
    """

    def __init__(self) -> None:
        self._base = os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
        self._key = os.getenv("LLM_API_KEY", "ollama")
        self._model = os.getenv("LLM_MODEL", "deepseek-r1:8b")

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> LLMResponse:
        """
        Send a chat completion request. Falls back to stub if API unavailable.

        Returns LLMResponse with content (stub if API down) and error field.
        """
        try:
            import requests
            resp = requests.post(
                f"{self._base}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=self._model,
                    tokens_used=data.get("usage", {}).get("total_tokens", 0),
                )
            return LLMResponse(content=self._stub_response(user_prompt), error=f"HTTP {resp.status_code}")
        except Exception as e:
            return LLMResponse(content=self._stub_response(user_prompt), error=str(e))

    def structured_output(
        self, system_prompt: str, user_prompt: str, output_schema: dict | None = None
    ) -> dict:
        """
        Request a structured JSON response from the LLM.
        Falls back to a stub dict on failure.
        """
        full_prompt = f"{user_prompt}\n\nOutput MUST be valid JSON matching this schema:\n{_json.dumps(output_schema or {}, indent=2)}"
        response = self.chat(system_prompt, full_prompt, temperature=0.0)
        try:
            # Extract JSON from response (may be wrapped in markdown)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return _json.loads(content.strip())
        except (_json.JSONDecodeError, IndexError):
            return {"signal": "neutral", "confidence": 0.5, "note": response.error or "parse_failed"}

    @staticmethod
    def _stub_response(prompt: str) -> str:
        """Stub response when LLM is unavailable."""
        if "macro" in prompt.lower():
            return '{"signal": "risk_on", "confidence": 0.65, "reason": "PMI>50, CPI<3% — stub"}'
        if "quant" in prompt.lower():
            return '{"signal": "bullish", "confidence": 0.60, "reason": "factor_score>0.6 — stub"}'
        if "risk" in prompt.lower():
            return '{"signal": "ok", "confidence": 0.75, "reason": "concentration<40% — stub"}'
        if "cio" in prompt.lower():
            return '{"signal": "HOLD", "confidence": 0.55, "reason": "mixed signals — stub"}'
        return '{"signal": "neutral", "confidence": 0.50, "reason": "stub"}'
