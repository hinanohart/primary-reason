from __future__ import annotations

from collections.abc import Sequence

from primary_reason.adapters.base import LLMAdapter
from primary_reason.core.types import CoTStep, InterventionStrategy
from primary_reason.extractors.prompts import NEGATE_SYSTEM, PARAPHRASE_SYSTEM

_DELETE_MARKER = "[step omitted]"


def corrupt_step(
    step: CoTStep,
    strategy: InterventionStrategy,
    *,
    adapter: LLMAdapter | None = None,
) -> str:
    """Produce a perturbed version of a single CoT step.

    - delete: replace text with a placeholder
    - paraphrase: ask the adapter to paraphrase
    - negate: ask the adapter to negate the propositional content
    """
    if strategy == "delete":
        return _DELETE_MARKER
    if strategy == "paraphrase":
        if adapter is None:
            return step.text
        out = adapter.complete(step.text, system=PARAPHRASE_SYSTEM, max_tokens=256).strip()
        return out or step.text
    if strategy == "negate":
        if adapter is None:
            return f"It is not the case that: {step.text}"
        out = adapter.complete(step.text, system=NEGATE_SYSTEM, max_tokens=256).strip()
        return out or f"It is not the case that: {step.text}"
    raise ValueError(f"Unknown intervention strategy: {strategy!r}")


def perturbed_cot(steps: Sequence[CoTStep], target_index: int, perturbed_text: str) -> str:
    """Reassemble the CoT with one step replaced by its perturbed version."""
    lines: list[str] = []
    for s in steps:
        text = perturbed_text if s.index == target_index else s.text
        lines.append(text)
    return "\n".join(lines)
