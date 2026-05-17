from __future__ import annotations

import pytest

from primary_reason.adapters.mock import MockAdapter
from primary_reason.core.types import CoTStep
from primary_reason.interventions.corruption import corrupt_step, perturbed_cot


def test_corrupt_delete() -> None:
    s = CoTStep(index=0, text="A premise.")
    out = corrupt_step(s, "delete")
    assert "[step omitted]" in out


def test_corrupt_paraphrase_no_adapter_returns_original() -> None:
    s = CoTStep(index=0, text="A premise.")
    out = corrupt_step(s, "paraphrase", adapter=None)
    assert out == "A premise."


def test_corrupt_paraphrase_with_adapter() -> None:
    s = CoTStep(index=0, text="A premise.")
    a = MockAdapter(response_fn=lambda p: "Paraphrased premise")
    out = corrupt_step(s, "paraphrase", adapter=a)
    assert out == "Paraphrased premise"


def test_corrupt_negate_no_adapter() -> None:
    s = CoTStep(index=0, text="The sky is blue.")
    out = corrupt_step(s, "negate", adapter=None)
    assert out.startswith("It is not the case that")


def test_corrupt_unknown_strategy() -> None:
    s = CoTStep(index=0, text="x")
    with pytest.raises(ValueError):
        corrupt_step(s, "bogus")  # type: ignore[arg-type]


def test_perturbed_cot_replaces_target() -> None:
    steps = [CoTStep(index=i, text=f"step {i}") for i in range(3)]
    out = perturbed_cot(steps, 1, "REPLACED")
    assert "REPLACED" in out
    assert "step 0" in out
    assert "step 2" in out
    assert "step 1" not in out
