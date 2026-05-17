from __future__ import annotations

import json

from primary_reason.adapters.mock import MockAdapter
from primary_reason.core.types import CoTStep
from primary_reason.extractors.primary_reason import extract_primary_reasons


def test_extract_happy_path() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 0, "pro_attitude": "want truth", "belief": "2+3=5", "confidence": 0.9},
            {
                "step_index": 1,
                "pro_attitude": "deliver answer",
                "belief": "answer is 5",
                "confidence": 0.8,
            },
        ]
    }
    a = MockAdapter(json_responses={"primary_reasons JSON": payload})
    steps = [CoTStep(index=0, text="2+3=5"), CoTStep(index=1, text="answer is 5")]
    reasons = extract_primary_reasons(steps, "what is 2+3?", a)
    assert len(reasons) == 2
    assert reasons[0].pro_attitude == "want truth"
    assert reasons[1].confidence == 0.8


def test_extract_empty_steps_returns_empty() -> None:
    a = MockAdapter()
    assert extract_primary_reasons([], "p", a) == []


def test_extract_clamps_out_of_range_indices() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 99, "pro_attitude": "x", "belief": "y", "confidence": 0.5},
            {"step_index": 0, "pro_attitude": "ok", "belief": "ok", "confidence": 0.5},
        ]
    }
    a = MockAdapter(json_responses={"primary_reasons JSON": payload})
    steps = [CoTStep(index=0, text="a")]
    reasons = extract_primary_reasons(steps, "p", a)
    assert len(reasons) == 1
    assert reasons[0].step_index == 0


def test_extract_dedupes_indices() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 0, "pro_attitude": "first", "belief": "b1", "confidence": 0.5},
            {"step_index": 0, "pro_attitude": "duplicate", "belief": "b2", "confidence": 0.5},
        ]
    }
    a = MockAdapter(json_responses={"primary_reasons JSON": payload})
    steps = [CoTStep(index=0, text="a")]
    reasons = extract_primary_reasons(steps, "p", a)
    assert len(reasons) == 1
    assert reasons[0].pro_attitude == "first"


def test_extract_fallback_when_empty() -> None:
    a = MockAdapter(json_responses={"primary_reasons JSON": {"primary_reasons": []}})
    steps = [CoTStep(index=0, text="a"), CoTStep(index=1, text="b")]
    reasons = extract_primary_reasons(steps, "p", a, max_retries=2)
    assert len(reasons) == 2
    assert all(r.confidence == 0.0 for r in reasons)
    assert all("extraction_failed" in r.pro_attitude for r in reasons)


def test_extract_clamps_confidence() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 0, "pro_attitude": "p", "belief": "b", "confidence": 5.0},
        ]
    }
    a = MockAdapter(json_responses={"primary_reasons JSON": payload})
    steps = [CoTStep(index=0, text="a")]
    reasons = extract_primary_reasons(steps, "p", a)
    assert reasons[0].confidence == 1.0


def test_extract_handles_malformed_confidence() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 0, "pro_attitude": "p", "belief": "b", "confidence": "high"},
        ]
    }
    a = MockAdapter(json_responses={"primary_reasons JSON": payload})
    steps = [CoTStep(index=0, text="a")]
    reasons = extract_primary_reasons(steps, "p", a)
    assert reasons[0].confidence == 0.5


def test_extract_with_json_string_raw_response() -> None:
    payload = {
        "primary_reasons": [
            {"step_index": 0, "pro_attitude": "x", "belief": "y", "confidence": 0.7},
        ]
    }
    a = MockAdapter(responses={"primary_reasons JSON now": json.dumps(payload)})
    steps = [CoTStep(index=0, text="a")]
    reasons = extract_primary_reasons(steps, "p", a)
    assert reasons[0].pro_attitude == "x"


def test_extract_golden_fixtures_schema_100() -> None:
    """All 4 golden fixtures (arithmetic / qa / code / long) → schema valid extraction."""
    from tests.fixtures.golden_fixtures import ALL_GOLDEN

    n_steps_total = 0
    n_extracted = 0
    for name, steps in ALL_GOLDEN:
        payload = {
            "primary_reasons": [
                {
                    "step_index": s.index,
                    "pro_attitude": f"goal for {name} step {s.index}",
                    "belief": f"belief for {name} step {s.index}",
                    "confidence": 0.7,
                }
                for s in steps
            ]
        }
        a = MockAdapter(json_responses={"primary_reasons JSON": payload})
        reasons = extract_primary_reasons(steps, f"prompt for {name}", a)
        n_steps_total += len(steps)
        n_extracted += len(reasons)

    assert n_extracted == n_steps_total
    assert n_steps_total >= 8
