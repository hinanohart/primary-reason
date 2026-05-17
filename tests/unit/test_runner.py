from __future__ import annotations

from primary_reason.core.types import CoTStep
from primary_reason.interventions.runner import iter_interventions


def test_iter_interventions_yields_step_strategy_pairs() -> None:
    steps = [CoTStep(index=i, text=f"step {i}") for i in range(2)]
    out = list(iter_interventions(steps, strategies=("delete",), adapter=None))
    assert len(out) == 2
    for step, strat, perturbed_text, perturbed_full in out:
        assert strat == "delete"
        assert "[step omitted]" in perturbed_text
        assert "[step omitted]" in perturbed_full


def test_iter_interventions_multiple_strategies() -> None:
    steps = [CoTStep(index=0, text="alpha"), CoTStep(index=1, text="beta")]
    out = list(iter_interventions(steps, strategies=("delete", "negate"), adapter=None))
    assert len(out) == 4
