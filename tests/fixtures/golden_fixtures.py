from __future__ import annotations

from primary_reason.core.types import CoTStep


def arithmetic_steps() -> list[CoTStep]:
    return [
        CoTStep(index=0, text="We need to compute 17 * 4.", role="premise"),
        CoTStep(index=1, text="17 * 4 = 68 since 17 * 2 = 34 and 34 * 2 = 68.", role="inference"),
        CoTStep(index=2, text="The answer is 68.", role="conclusion"),
    ]


def qa_steps() -> list[CoTStep]:
    return [
        CoTStep(index=0, text="The capital of France is asked.", role="premise"),
        CoTStep(index=1, text="Paris is the capital of France.", role="inference"),
        CoTStep(index=2, text="So the answer is Paris.", role="conclusion"),
    ]


def code_steps() -> list[CoTStep]:
    return [
        CoTStep(index=0, text="We want to reverse a string.", role="premise"),
        CoTStep(index=1, text="In Python, s[::-1] reverses a string.", role="inference"),
        CoTStep(index=2, text="So return s[::-1].", role="conclusion"),
    ]


def long_steps() -> list[CoTStep]:
    return [
        CoTStep(index=0, text="A train leaves at 9:00.", role="premise"),
        CoTStep(index=1, text="The trip takes 2 hours and 30 minutes.", role="inference"),
        CoTStep(index=2, text="Adding 2:30 to 9:00 gives 11:30.", role="inference"),
        CoTStep(index=3, text="There is a 15 minute stop.", role="inference"),
        CoTStep(index=4, text="So total arrival is 11:45.", role="conclusion"),
    ]


ALL_GOLDEN: list[tuple[str, list[CoTStep]]] = [
    ("arithmetic", arithmetic_steps()),
    ("qa", qa_steps()),
    ("code", code_steps()),
    ("long", long_steps()),
]
