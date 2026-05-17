from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from primary_reason.adapters.base import LLMAdapter
from primary_reason.core.types import CoTStep, PrimaryReason
from primary_reason.extractors.prompts import (
    PRIMARY_REASON_SCHEMA,
    PRIMARY_REASON_SYSTEM,
    build_primary_reason_prompt,
)

_MAX_RETRIES = 3


def extract_primary_reasons(
    steps: Sequence[CoTStep],
    prompt: str,
    adapter: LLMAdapter,
    *,
    max_retries: int = _MAX_RETRIES,
) -> list[PrimaryReason]:
    """Extract Davidson primary reasons (pro-attitude + belief) for each CoT step.

    Retries up to max_retries times on malformed JSON, returning whatever subset successfully parses.
    """
    if not steps:
        return []
    steps_text = "\n".join(f"{s.index}. {s.text}" for s in steps)
    user_prompt = build_primary_reason_prompt(prompt, steps_text)

    last_raw: dict[str, Any] = {}
    for _attempt in range(max_retries):
        raw = adapter.complete_json(
            user_prompt,
            schema=PRIMARY_REASON_SCHEMA,
            system=PRIMARY_REASON_SYSTEM,
            max_tokens=2048,
            temperature=0.0,
        )
        last_raw = raw
        items = raw.get("primary_reasons") if isinstance(raw, dict) else None
        if isinstance(items, list) and items:
            parsed = _parse_items(items, steps_count=len(steps))
            if parsed:
                return parsed

    return _fallback(steps, last_raw)


def _parse_items(items: list[Any], *, steps_count: int) -> list[PrimaryReason]:
    out: list[PrimaryReason] = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("step_index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= steps_count or idx in seen:
            continue
        pro = str(item.get("pro_attitude", "(none)"))
        belief = str(item.get("belief", "(none)"))
        causal = item.get("causal_role")
        conf_raw = item.get("confidence", 0.5)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        out.append(
            PrimaryReason(
                step_index=idx,
                pro_attitude=pro,
                belief=belief,
                causal_role=str(causal) if causal else None,
                confidence=conf,
                raw=item,
            )
        )
        seen.add(idx)
    return out


def _fallback(steps: Sequence[CoTStep], last_raw: dict[str, Any]) -> list[PrimaryReason]:
    """If JSON extraction completely failed, return low-confidence stubs so downstream still runs."""
    return [
        PrimaryReason(
            step_index=s.index,
            pro_attitude="(extraction_failed)",
            belief="(extraction_failed)",
            causal_role=None,
            confidence=0.0,
            raw={"_last_raw": last_raw},
        )
        for s in steps
    ]
