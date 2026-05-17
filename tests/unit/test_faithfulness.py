from __future__ import annotations

from primary_reason.adapters.mock import MockAdapter
from primary_reason.core.types import CoTStep
from primary_reason.metrics.faithfulness import score_faithfulness


def _necessary_step_adapter() -> MockAdapter:
    """Adapter that returns '5' when CoT contains '2 + 3 = 5', else returns '?'.

    Simulates the case where one CoT step is causally necessary for the answer.
    """

    def fn(prompt: str) -> str:
        return "5" if "2 + 3 = 5" in prompt else "?"

    return MockAdapter(response_fn=fn)


def _ignored_step_adapter() -> MockAdapter:
    """Adapter that always returns '5' regardless of CoT — CoT is causally irrelevant."""
    return MockAdapter(response_fn=lambda p: "5")


def test_score_faithfulness_empty_steps() -> None:
    a = MockAdapter()
    fs = score_faithfulness(
        prompt="p",
        steps=[],
        adapter=a,
        strategies=("delete",),
        distance_metric="exact",
        original_answer="x",
    )
    assert fs.score == 0.0
    assert fs.per_step == {}


def test_score_faithfulness_necessary_step_high_score() -> None:
    steps = [
        CoTStep(index=0, text="Need to compute."),
        CoTStep(index=1, text="2 + 3 = 5 is the key step."),
        CoTStep(index=2, text="Therefore the answer."),
    ]
    a = _necessary_step_adapter()
    fs = score_faithfulness(
        prompt="what is 2+3?",
        steps=steps,
        adapter=a,
        strategies=("delete",),
        distance_metric="exact",
        original_answer="5",
    )
    # deleting the key step (index 1) should produce a different answer
    assert fs.per_step[1] == 1.0
    # other steps unchanged because adapter only triggers on key text
    assert fs.score > 0.0


def test_score_faithfulness_unfaithful_low_score() -> None:
    steps = [
        CoTStep(index=0, text="alpha"),
        CoTStep(index=1, text="beta"),
        CoTStep(index=2, text="gamma"),
    ]
    a = _ignored_step_adapter()
    fs = score_faithfulness(
        prompt="p",
        steps=steps,
        adapter=a,
        strategies=("delete",),
        distance_metric="exact",
        original_answer="5",
    )
    assert fs.score == 0.0


def test_score_faithfulness_multiple_strategies() -> None:
    steps = [CoTStep(index=0, text="step a"), CoTStep(index=1, text="step b")]
    a = MockAdapter(response_fn=lambda p: "constant")
    fs = score_faithfulness(
        prompt="p",
        steps=steps,
        adapter=a,
        strategies=("delete", "paraphrase"),
        distance_metric="exact",
        original_answer="constant",
    )
    # constant answer -> distance 0 always -> low faithfulness
    assert fs.score == 0.0
    # 2 steps x 2 strategies = 4 interventions
    assert len(fs.interventions) == 4
