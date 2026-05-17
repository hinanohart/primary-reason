from __future__ import annotations

import json
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from primary_reason.adapters.anthropic_adapter import _extract_json
from primary_reason.adapters.base import AdapterError, LLMMessage


class OllamaAdapter:
    """Ollama local LLM adapter (lazy-import).

    No API key required; assumes ollama daemon at OLLAMA_HOST (default http://localhost:11434).
    """

    name = "ollama"

    def __init__(self, model: str = "llama3.2", *, host: str | None = None) -> None:
        self.model = model
        self.host = host
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import ollama
        except ImportError as e:
            raise AdapterError(
                "ollama package not installed; install with 'pip install primary-reason[ollama]'"
            ) from e
        if self.host:
            self._client = ollama.Client(host=self.host)
        else:
            self._client = ollama
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
        format_json: bool = False,
    ) -> str:
        client = self._get_client()
        options = {"temperature": temperature, "num_predict": max_tokens}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": options,
        }
        if format_json:
            kwargs["format"] = "json"
        resp = client.chat(**kwargs)
        if isinstance(resp, dict):
            return str(resp.get("message", {}).get("content", ""))
        return str(getattr(resp, "message", {}).get("content", ""))

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
        text = self._call(msgs, max_tokens=max_tokens, temperature=temperature, format_json=True)
        return _extract_json(text)
