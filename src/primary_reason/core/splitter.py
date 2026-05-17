from __future__ import annotations

import re

from primary_reason.core.types import CoTStep

_NUMBERED = re.compile(r"(?:^|(?<=[\s.!?]))(?:\d+[.)]\s+|step\s*\d+\s*:?\s+)", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def split_cot(cot: str) -> list[CoTStep]:
    """Split a CoT trace into steps.

    Strategy: prefer explicit numbering (1./2./Step 1:), fall back to sentence boundaries.
    """
    cot = cot.strip()
    if not cot:
        return []

    if _NUMBERED.search(cot):
        parts = _NUMBERED.split(cot)
        chunks = [p.strip() for p in parts if p.strip()]
    else:
        chunks = [c.strip() for c in _SENTENCE_BOUNDARY.split(cot) if c.strip()]

    steps: list[CoTStep] = []
    for i, chunk in enumerate(chunks):
        role: str = "inference"
        if i == 0:
            role = "premise"
        elif i == len(chunks) - 1:
            role = "conclusion"
        steps.append(CoTStep(index=i, text=chunk, role=role))  # type: ignore[arg-type]
    return steps
