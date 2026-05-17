from __future__ import annotations

from primary_reason.adapters.base import LLMAdapter, LLMMessage
from primary_reason.adapters.mock import MockAdapter


def build_adapter(name: str, *, model: str) -> LLMAdapter:
    name = name.lower()
    if name == "mock":
        return MockAdapter(model=model)
    if name == "anthropic":
        from primary_reason.adapters.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(model=model)
    if name == "openai":
        from primary_reason.adapters.openai_adapter import OpenAIAdapter

        return OpenAIAdapter(model=model)
    if name == "ollama":
        from primary_reason.adapters.ollama_adapter import OllamaAdapter

        return OllamaAdapter(model=model)
    raise ValueError(f"Unknown adapter: {name!r}. Known: anthropic, openai, ollama, mock")


__all__ = ["LLMAdapter", "LLMMessage", "MockAdapter", "build_adapter"]
