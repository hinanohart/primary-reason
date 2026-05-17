from __future__ import annotations

import re

from primary_reason.core.types import CoTStep

_NUMBERED = re.compile(
    # numbered list "1." / "2)" — line-anchored only (avoids breaking arithmetic CoTs)
    r"(?:^|\n)\s*\d+[.)]\s+"
    r"|"
    # "Step N:" label — colon required, allowed inline (the explicit ":" is contextually
    # distinctive and unlikely to occur inside math), but it still requires a preceding
    # boundary char so it doesn't fire inside a longer word.
    r"(?:^|(?<=[\s.!?]))step\s*\d+\s*:\s+",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


def split_cot(cot: str) -> list[CoTStep]:
    """Split a CoT trace into steps.

    Numbered-list split is anchored to line starts only, so inline numerics inside a step
    (e.g. "17 * 4. Step 1: ...") are not used as split boundaries — that previously
    broke arithmetic / math CoTs. The "Step N:" label form still splits inline because the
    explicit colon is contextually distinctive.
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
