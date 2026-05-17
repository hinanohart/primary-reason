from __future__ import annotations

import pytest

from primary_reason.core.types import (
    CoTStep,
    FaithfulnessScore,
    InterventionResult,
    PrimaryReason,
    SwampmanScore,
    VerificationResult,
)


def test_cotstep_strips_whitespace() -> None:
    s = CoTStep(index=0, text="  hello  ")
    assert s.text == "hello"


def test_cotstep_rejects_empty() -> None:
    with pytest.raises(ValueError):
        CoTStep(index=0, text="   ")


def test_primary_reason_clamps_confidence() -> None:
    pr = PrimaryReason(step_index=0, pro_attitude="x", belief="y", confidence=0.5)
    assert pr.confidence == 0.5
    with pytest.raises(ValueError):
        PrimaryReason(step_index=0, pro_attitude="x", belief="y", confidence=1.5)


def test_faithfulness_score_validates_per_step() -> None:
    with pytest.raises(ValueError):
        FaithfulnessScore(score=0.5, per_step={0: 1.5})


def test_swampman_score_defaults() -> None:
    s = SwampmanScore(
        variant_with_history="with_history",
        variant_without_history="without_history",
        fpa_score=0.3,
        n_trials=5,
    )
    assert s.bootstrap_ci == (0.0, 1.0)
    assert s.discriminates is False


def test_verification_result_to_dict() -> None:
    steps = [CoTStep(index=0, text="a")]
    res = VerificationResult(
        prompt="p",
        cot="a",
        steps=steps,
        primary_reasons=[],
        faithfulness=FaithfulnessScore(score=0.0),
        model="mock",
        adapter="MockAdapter",
    )
    d = res.to_dict()
    assert d["model"] == "mock"
    assert d["faithfulness"]["score"] == 0.0


def test_intervention_result_distance_nonnegative() -> None:
    with pytest.raises(ValueError):
        InterventionResult(
            step_index=0,
            strategy="delete",
            original_step="a",
            perturbed_step="b",
            original_answer="x",
            perturbed_answer="y",
            distance=-0.1,
        )
