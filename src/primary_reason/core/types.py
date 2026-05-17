from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

InterventionStrategy = Literal["delete", "paraphrase", "negate"]
DistanceMetric = Literal["exact", "lexical", "embedding"]


class CoTStep(BaseModel):
    """A single step of a chain-of-thought trace."""

    index: int = Field(ge=0)
    text: str
    role: Literal["premise", "inference", "conclusion", "other"] = "inference"

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("CoTStep.text must be non-empty after strip")
        return v


class PrimaryReason(BaseModel):
    """Davidson primary reason: pro-attitude (desire/goal) + belief, attributed to a CoT step.

    Per Davidson "Actions, Reasons, and Causes" (1963), a primary reason consists of
    (a) a pro-attitude toward actions of a certain kind, and
    (b) a belief that the action in question is of that kind.
    """

    step_index: int = Field(ge=0)
    pro_attitude: str
    belief: str
    causal_role: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    raw: dict[str, Any] = Field(default_factory=dict)


class InterventionResult(BaseModel):
    step_index: int
    strategy: InterventionStrategy
    original_step: str
    perturbed_step: str
    original_answer: str
    perturbed_answer: str
    distance: float = Field(ge=0.0)


class FaithfulnessScore(BaseModel):
    """Counterfactual faithfulness score for a CoT trace."""

    score: float = Field(ge=0.0, le=1.0)
    per_step: dict[int, float] = Field(default_factory=dict)
    interventions: list[InterventionResult] = Field(default_factory=list)
    method: str = "counterfactual"

    @field_validator("per_step")
    @classmethod
    def _check_range(cls, v: dict[int, float]) -> dict[int, float]:
        for k, val in v.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"per_step[{k}]={val} out of [0,1]")
        return v


class SwampmanScore(BaseModel):
    """Swampman Test Battery proxy score (T1.5).

    Hedge: this is a behavioural proxy, not a verification of intentionality.
    Compares two role-prompt variants: one alluding to a consistent causal history and one without
    (instantaneous swamp-bolt formation analogue from Davidson 1987). Divergence is interpreted as
    weak evidence that the model's outputs are conditioned on biographical anchoring — but the
    proxy is gameable by prompt-echoing and must be read alongside the control baseline.
    """

    variant_with_history: str
    variant_without_history: str
    fpa_score: float = Field(ge=0.0, le=1.0)
    bootstrap_ci: tuple[float, float] = (0.0, 1.0)
    n_trials: int = Field(ge=0)
    discriminates: bool = False
    per_task: dict[str, float] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    prompt: str
    cot: str
    steps: list[CoTStep]
    primary_reasons: list[PrimaryReason]
    faithfulness: FaithfulnessScore
    swampman_score: SwampmanScore | None = None
    model: str
    adapter: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
