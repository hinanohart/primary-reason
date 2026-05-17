from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from primary_reason.adapters.base import LLMMessage


class MockAdapter:
    """Deterministic mock LLM adapter for testing.

    Behaviour:
      - If `responses` mapping is provided, return the value whose key is a substring of the prompt
        (first match wins by insertion order). Useful for golden-fixture tests.
      - Else if `response_fn` is provided, call it.
      - Else return a deterministic SHA-1 echo string.
      - `complete_json` defaults to a stub object {"primary_reasons": [...]} suitable for T1 tests.
    """

    name = "mock"

    def __init__(
        self,
        model: str = "mock-model",
        *,
        responses: Mapping[str, str] | None = None,
        json_responses: Mapping[str, dict[str, Any]] | None = None,
        response_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.model = model
        self.responses: dict[str, str] = dict(responses or {})
        self.json_responses: dict[str, dict[str, Any]] = dict(json_responses or {})
        self.response_fn = response_fn
        self.call_log: list[dict[str, Any]] = []

    def _echo(self, prompt: str) -> str:
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"MOCK[{h}]"

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        self.call_log.append({"op": "complete", "prompt": prompt, "system": system})
        for key, val in self.responses.items():
            if key in prompt:
                return val
        if self.response_fn is not None:
            return self.response_fn(prompt)
        return self._echo(prompt)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        self.call_log.append({"op": "chat", "messages": messages})
        prompt = "\n".join(m["content"] for m in messages)
        return self.complete(prompt, max_tokens=max_tokens, temperature=temperature)

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.call_log.append({"op": "complete_json", "prompt": prompt, "system": system})
        for key, val in self.json_responses.items():
            if key in prompt:
                return val
        # Default: extract reasons for any step text mentioned (very simple heuristic)
        text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, system=system)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {"primary_reasons": [], "raw": text}
