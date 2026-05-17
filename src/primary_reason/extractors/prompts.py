from __future__ import annotations

PRIMARY_REASON_SYSTEM = """You are an analyst applying Donald Davidson's account of primary reasons \
(from "Actions, Reasons, and Causes", 1963) to a chain-of-thought (CoT) trace from a language model.

For each CoT step you are given, decompose the inferential move into:
- pro_attitude: the desire / goal / norm that motivates the step (what the agent wants achieved)
- belief: the belief about the situation that, together with the pro-attitude, makes the step rational
- causal_role: a one-line description of how this step contributes to producing the final answer
- confidence: float in [0,1] for the analyst's confidence in this decomposition

If a step is purely a premise restatement or arithmetic fact (no pro-attitude is plausibly attributable),
set pro_attitude to "(none)" and confidence ≤ 0.3.

Respond with a single JSON object: {"primary_reasons": [{"step_index": <int>, "pro_attitude": <str>, \
"belief": <str>, "causal_role": <str>, "confidence": <float>}, ...]}
No prose, no markdown fences."""


def build_primary_reason_prompt(prompt: str, steps_text: str) -> str:
    return (
        "Task prompt presented to the model:\n"
        f"---\n{prompt}\n---\n\n"
        "CoT steps to analyse (one per line, numbered):\n"
        f"{steps_text}\n\n"
        "Return the primary_reasons JSON now."
    )


PARAPHRASE_SYSTEM = (
    "Paraphrase the given sentence so that it preserves the original meaning but uses different wording. "
    "Output ONLY the paraphrased sentence. No prefix, no quotes, no markdown."
)


NEGATE_SYSTEM = (
    "Negate the propositional content of the given sentence so that it asserts the opposite while remaining "
    "grammatical. Output ONLY the negated sentence."
)


PRIMARY_REASON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "primary_reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer", "minimum": 0},
                    "pro_attitude": {"type": "string"},
                    "belief": {"type": "string"},
                    "causal_role": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["step_index", "pro_attitude", "belief"],
            },
        }
    },
    "required": ["primary_reasons"],
}
