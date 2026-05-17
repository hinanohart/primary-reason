from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


class LLMMessage(TypedDict):
    role: str
    content: str


@runtime_checkable
class LLMAdapter(Protocol):
    """Minimal LLM provider interface.

    All adapters implement:
      - complete(prompt, max_tokens, ...) -> str  (text completion convenience)
      - chat(messages, max_tokens, ...) -> str    (multi-turn)
      - complete_json(prompt, schema, ...) -> dict (structured output, schema-guided)
      - name -> str
    """

    name: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> str: ...

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str: ...

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class AdapterError(RuntimeError):
    """Raised when an adapter call fails after retries."""
