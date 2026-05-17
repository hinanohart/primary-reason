from __future__ import annotations

import json

from primary_reason import ReasonCauseVerifier
from primary_reason.adapters.mock import MockAdapter


def test_verifier_repr_and_dict() -> None:
    a = MockAdapter()
    v = ReasonCauseVerifier(model="m", adapter=a, intervention_strategies=("delete",))
    assert "ReasonCauseVerifier" in repr(v)
    d = v.to_dict()
    assert d["model"] == "m"
    assert d["intervention_strategies"] == ["delete"]


def test_verifier_full_path_mock() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 0, "pro_attitude": "want sum", "belief": "2+3", "confidence": 0.9},
            {"step_index": 1, "pro_attitude": "report", "belief": "answer", "confidence": 0.8},
        ]
    }
    a = MockAdapter(
        responses={"Final answer": "5"},
        json_responses={"primary_reasons JSON": payload},
    )
    v = ReasonCauseVerifier(
        model="mock-1",
        adapter=a,
        intervention_strategies=("delete",),
        distance_metric="exact",
    )
    res = v.verify(prompt="what is 2+3?", cot="1. compute 2+3\n2. answer is 5", answer="5")
    assert len(res.steps) == 2
    assert len(res.primary_reasons) == 2
    assert res.faithfulness.score >= 0.0
    assert res.model == "mock-1"


def test_verifier_swampman_path() -> None:
    a = MockAdapter(response_fn=lambda p: "same")
    v = ReasonCauseVerifier(model="m", adapter=a, intervention_strategies=("delete",))
    score = v.swampman_score()
    assert score.fpa_score == 0.0
    assert score.n_trials > 0


def test_verifier_serializes_to_json() -> None:
    a = MockAdapter(
        responses={"Final answer": "ok"},
        json_responses={
            "primary_reasons JSON": {
                "primary_reasons": [
                    {"step_index": 0, "pro_attitude": "p", "belief": "b", "confidence": 0.5}
                ]
            }
        },
    )
    v = ReasonCauseVerifier(model="m", adapter=a, intervention_strategies=("delete",))
    res = v.verify(prompt="p", cot="single step.", answer="ok")
    s = json.dumps(res.to_dict())
    assert "primary_reasons" in s
