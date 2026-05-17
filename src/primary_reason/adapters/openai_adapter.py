from __future__ import annotations

import json
import os
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from primary_reason.adapters.anthropic_adapter import _extract_json
from primary_reason.adapters.base import AdapterError, LLMMessage


class OpenAIAdapter:
    """OpenAI Chat Completions adapter.

    API key read from OPENAI_API_KEY (R11: env var passthrough, value never written).
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None) -> None:
        self.model = model
        self._explicit_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise AdapterError(
                "openai package not installed; install with 'pip install openai'"
            ) from e
        key = self._explicit_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise AdapterError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=key)
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
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        msgs: list[LLMMessage] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self._call(msgs, max_tokens=max_tokens, temperature=temperature)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        return self._call(messages, max_tokens=max_tokens, temperature=temperature)

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
        full_prompt = prompt
        if schema:
            full_prompt = f"{prompt}\n\nSchema:\n{json.dumps(schema, indent=2)}"
        msgs: list[LLMMessage] = [
            {"role": "system", "content": json_system},
            {"role": "user", "content": full_prompt},
        ]
        text = self._call(
            msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return _extract_json(text)
