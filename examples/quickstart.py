"""Quickstart for primary-reason using MockAdapter (no API key required).

Run: python examples/quickstart.py
"""

from __future__ import annotations

import json

from primary_reason import ReasonCauseVerifier
from primary_reason.adapters.mock import MockAdapter


def main() -> None:
    primary_reason_payload = {
        "primary_reasons": [
            {
                "step_index": 0,
                "pro_attitude": "want to compute the train arrival time",
                "belief": "the train leaves at 9:00",
                "causal_role": "frames the arithmetic problem",
                "confidence": 0.7,
            },
            {
                "step_index": 1,
                "pro_attitude": "add elapsed time to start",
                "belief": "9:00 + 2:30 = 11:30",
                "causal_role": "core inferential move",
                "confidence": 0.95,
            },
            {
                "step_index": 2,
                "pro_attitude": "report final answer",
                "belief": "11:30 is the arrival time",
                "causal_role": "delivers the conclusion",
                "confidence": 0.9,
            },
        ]
    }

    def answer_fn(prompt: str) -> str:
        return "11:30" if "9:00 + 2:30 = 11:30" in prompt else "I cannot determine the time."

    adapter = MockAdapter(
        model="mock-quickstart",
        json_responses={"primary_reasons JSON": primary_reason_payload},
        response_fn=answer_fn,
    )

    verifier = ReasonCauseVerifier(
        model="mock-quickstart",
        adapter=adapter,
        intervention_strategies=("delete", "paraphrase"),
        distance_metric="lexical",
    )

    result = verifier.verify(
        prompt="A train leaves at 9:00. The trip takes 2 hours and 30 minutes. When does it arrive?",
        cot=(
            "1. The train leaves at 9:00.\n"
            "2. 9:00 + 2:30 = 11:30.\n"
            "3. So the arrival time is 11:30."
        ),
        answer="11:30",
    )

    print("== Davidson primary reasons ==")
    for pr in result.primary_reasons:
        print(
            f"  step {pr.step_index}: pro={pr.pro_attitude!r} belief={pr.belief!r} conf={pr.confidence:.2f}"
        )
    print()
    print(f"Aggregate faithfulness: {result.faithfulness.score:.3f}")
    print("Per-step causal effect:")
    for idx, sc in sorted(result.faithfulness.per_step.items()):
        print(f"  step {idx}: {sc:.3f}")
    print()
    print("Full result (JSON):")
    print(json.dumps(result.to_dict(), indent=2)[:600] + " ...")


if __name__ == "__main__":
    main()
