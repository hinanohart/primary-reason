from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any


@contextlib.contextmanager
def langfuse_trace(name: str, **metadata: Any) -> Iterator[Any]:
    """Optional Langfuse tracing context. No-op if langfuse is not installed.

    Usage:
        with langfuse_trace("primary_reason.verify", model="claude-opus-4-7") as trace:
            result = verifier.verify(prompt, cot)
            if trace is not None:
                trace.update(output=result.to_dict())
    """
    try:
        from langfuse import Langfuse
    except ImportError:
        yield None
        return
    client = Langfuse()
    trace = client.trace(name=name, metadata=metadata)
    try:
        yield trace
    finally:
        with contextlib.suppress(Exception):
            client.flush()
