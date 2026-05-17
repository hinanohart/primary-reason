"""Intervention orchestration: thin wrapper to apply a strategy to every step.

Currently a convenience layer over corruption.corrupt_step + corruption.perturbed_cot.
metrics/faithfulness.py uses corruption.* directly; this module exists so external callers
can drive interventions without depending on the metric layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from primary_reason.adapters.base import LLMAdapter
from primary_reason.core.types import CoTStep, InterventionStrategy
from primary_reason.interventions.corruption import corrupt_step, perturbed_cot


def iter_interventions(
    steps: Sequence[CoTStep],
    *,
    strategies: Sequence[InterventionStrategy],
    adapter: LLMAdapter | None = None,
) -> Iterator[tuple[CoTStep, InterventionStrategy, str, str]]:
    """Yield (step, strategy, perturbed_step_text, perturbed_full_cot) for every (step, strategy)."""
    for step in steps:
        for strat in strategies:
            text = corrupt_step(step, strat, adapter=adapter)
            yield step, strat, text, perturbed_cot(steps, step.index, text)
