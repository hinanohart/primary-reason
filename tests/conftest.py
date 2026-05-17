from __future__ import annotations

import json

import pytest
from primary_reason.adapters.mock import MockAdapter
from primary_reason.core.types import CoTStep


@pytest.fixture
def simple_steps() -> list[CoTStep]:
    return [
        CoTStep(index=0, text="The sum of 2 and 3 is needed.", role="premise"),
        CoTStep(index=1, text="2 + 3 equals 5.", role="inference"),
        CoTStep(index=2, text="So the answer is 5.", role="conclusion"),
    ]


@pytest.fixture
def golden_primary_reasons_json() -> str:
    payload = {
        "primary_reasons": [
            {
                "step_index": 0,
                "pro_attitude": "(none)",
                "belief": "the task requires summing 2 and 3",
                "causal_role": "frames the computation",
                "confidence": 0.4,
            },
            {
                "step_index": 1,
                "pro_attitude": "produce a correct arithmetic step",
                "belief": "2 + 3 = 5",
                "causal_role": "core inferential move",
                "confidence": 0.9,
            },
            {
                "step_index": 2,
                "pro_attitude": "report the result",
                "belief": "the computed sum is the final answer",
                "causal_role": "delivers conclusion",
                "confidence": 0.85,
            },
        ]
    }
    return json.dumps(payload)


@pytest.fixture
def mock_adapter_with_reasons(golden_primary_reasons_json: str) -> MockAdapter:
    return MockAdapter(
        model="mock-model",
        json_responses={"primary_reasons JSON now": json.loads(golden_primary_reasons_json)},
        responses={"Final answer": "5"},
    )
