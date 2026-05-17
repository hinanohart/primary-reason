from __future__ import annotations

import sys

from primary_reason.plugins.langfuse import langfuse_trace


def test_langfuse_trace_no_op_when_uninstalled(monkeypatch) -> None:
    """The plugin should be a no-op (yield None) when langfuse is not importable."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langfuse" or name.startswith("langfuse."):
            raise ImportError("simulated missing langfuse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)

    with langfuse_trace("test.op", model="m") as trace:
        assert trace is None
