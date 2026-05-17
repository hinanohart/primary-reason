from __future__ import annotations

import json
import os
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from primary_reason.adapters.base import AdapterError, LLMMessage


class AnthropicAdapter:
    """Anthropic Messages API adapter.

    API key read from ANTHROPIC_API_KEY (R11: env var passthrough, value never written).
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7", *, api_key: str | None = None) -> None:
        self.model = model
        self._api_key_provided = api_key is not None
        self._explicit_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise AdapterError(
                "anthropic package not installed; install with 'pip install anthropic'"
            ) from e
        key = self._explicit_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AdapterError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        msgs: list[LLMMessage] = [{"role": "user", "content": prompt}]
        return self._call(msgs, system=system, max_tokens=max_tokens, temperature=temperature)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        system: str | None = None
        user_msgs: list[LLMMessage] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        return self._call(user_msgs, system=system, max_tokens=max_tokens, temperature=temperature)

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
        json_system = "Respond with valid JSON only. No markdown fences, no prose."
        if system:
            json_system = f"{system}\n\n{json_system}"
        if schema:
            prompt = f"{prompt}\n\nSchema:\n{json.dumps(schema, indent=2)}"
        text = self.complete(
            prompt, max_tokens=max_tokens, temperature=temperature, system=json_system
        )
        return _extract_json(text)


_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from possibly fenced or noisy text. Returns {} on failure (caller decides)."""
    text = text.strip()
    if not text:
        return {}
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"_value": parsed}
    except json.JSONDecodeError:
        # try to find first top-level {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"_raw": text}
