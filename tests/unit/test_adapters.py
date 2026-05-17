from __future__ import annotations

import pytest

from primary_reason.adapters import build_adapter
from primary_reason.adapters.base import LLMAdapter
from primary_reason.adapters.mock import MockAdapter


def test_build_mock() -> None:
    a = build_adapter("mock", model="m")
    assert isinstance(a, MockAdapter)
    assert isinstance(a, LLMAdapter)


def test_build_unknown_raises() -> None:
    with pytest.raises(ValueError):
        build_adapter("unknown_provider", model="m")


def test_mock_responses_substring_match() -> None:
    a = MockAdapter(responses={"capital of France": "Paris"})
    assert a.complete("What is the capital of France?") == "Paris"


def test_mock_response_fn() -> None:
    a = MockAdapter(response_fn=lambda p: f"answered: {p[:5]}")
    assert a.complete("hello world").startswith("answered:")


def test_mock_deterministic_echo() -> None:
    a = MockAdapter()
    assert a.complete("abc") == a.complete("abc")


def test_mock_complete_json_with_mapping() -> None:
    a = MockAdapter(json_responses={"trigger": {"primary_reasons": [{"step_index": 0}]}})
    out = a.complete_json("contains trigger here")
    assert out["primary_reasons"][0]["step_index"] == 0


def test_mock_complete_json_fallback() -> None:
    a = MockAdapter()
    out = a.complete_json("anything")
    assert "primary_reasons" in out or "_raw" in out


def test_mock_chat() -> None:
    a = MockAdapter(responses={"hello": "hi"})
    out = a.chat([{"role": "user", "content": "hello there"}])
    assert out == "hi"


def test_anthropic_adapter_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from primary_reason.adapters.anthropic_adapter import AnthropicAdapter
    from primary_reason.adapters.base import AdapterError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = AnthropicAdapter(model="claude-opus-4-7")
    with pytest.raises(AdapterError):
        a.complete("hi")


def test_openai_adapter_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from primary_reason.adapters.base import AdapterError
    from primary_reason.adapters.openai_adapter import OpenAIAdapter

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = OpenAIAdapter(model="gpt-4o-mini")
    with pytest.raises(AdapterError):
        a.complete("hi")


def test_extract_json_fenced() -> None:
    from primary_reason.adapters.anthropic_adapter import _extract_json

    out = _extract_json('```json\n{"a": 1}\n```')
    assert out == {"a": 1}


def test_extract_json_noisy() -> None:
    from primary_reason.adapters.anthropic_adapter import _extract_json

    out = _extract_json('garble {"x": 2} trailing')
    assert out == {"x": 2}


def test_extract_json_invalid() -> None:
    from primary_reason.adapters.anthropic_adapter import _extract_json

    out = _extract_json("not json at all")
    assert "_raw" in out
