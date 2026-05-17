"""End-to-end integration via MockAdapter — analogue of FaithCoT-Bench mini 20-item smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from primary_reason import ReasonCauseVerifier
from primary_reason.adapters.mock import MockAdapter


def _build_payload(n_steps: int, prefix: str) -> dict:
    return {
        "primary_reasons": [
            {
                "step_index": i,
                "pro_attitude": f"{prefix}-attitude-{i}",
                "belief": f"{prefix}-belief-{i}",
                "confidence": 0.7,
            }
            for i in range(n_steps)
        ]
    }


@pytest.mark.parametrize(
    ("idx", "prompt", "cot", "answer", "expect_faithful"),
    [
        # 20 mini-fixtures: 10 faithful (necessary step), 10 unfaithful (irrelevant CoT)
        (i, f"task {i}", f"1. first.\n2. key step {i}.\n3. so answer.", str(i), i % 2 == 0)
        for i in range(20)
    ],
)
def test_integration_mini_bench(
    idx: int, prompt: str, cot: str, answer: str, expect_faithful: bool
) -> None:
    payload = _build_payload(n_steps=3, prefix=f"task{idx}")

    if expect_faithful:
        # answer depends on key step
        def fn(p: str) -> str:
            return answer if f"key step {idx}" in p else "DIFFERENT"
    else:

        def fn(p: str) -> str:
            return answer  # constant — CoT irrelevant

    a = MockAdapter(json_responses={"primary_reasons JSON": payload}, response_fn=fn)
    v = ReasonCauseVerifier(
        model="mock",
        adapter=a,
        intervention_strategies=("delete",),
        distance_metric="exact",
    )
    res = v.verify(prompt=prompt, cot=cot, answer=answer)

    assert len(res.steps) == 3
    assert len(res.primary_reasons) == 3
    if expect_faithful:
        assert res.faithfulness.score > 0.0
    else:
        assert res.faithfulness.score == 0.0


def test_integration_full_with_swampman(tmp_path: Path) -> None:
    payload = _build_payload(n_steps=2, prefix="full")
    a = MockAdapter(
        json_responses={"primary_reasons JSON": payload},
        response_fn=lambda p: "answer-X",
    )
    v = ReasonCauseVerifier(
        model="mock",
        adapter=a,
        intervention_strategies=("delete", "paraphrase"),
        distance_metric="lexical",
    )
    res = v.verify(
        prompt="run full",
        cot="step a.\nstep b.",
        answer="answer-X",
        run_swampman=True,
    )
    assert res.swampman_score is not None
    out_path = tmp_path / "out.json"
    out_path.write_text(json.dumps(res.to_dict()))
    assert json.loads(out_path.read_text())["model"] == "mock"
