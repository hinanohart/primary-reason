from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from primary_reason.adapters.base import LLMAdapter
from primary_reason.core.splitter import split_cot
from primary_reason.core.types import (
    DistanceMetric,
    InterventionStrategy,
    SwampmanScore,
    VerificationResult,
)


class ReasonCauseVerifier:
    """Davidson primary-reason / counterfactual-faithfulness verifier.

    Orchestrates T1 (Primary Reason Extractor), T2 (Counterfactual Faithfulness),
    and optionally T1.5 (Swampman Test Battery).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        adapter: str | LLMAdapter = "anthropic",
        intervention_strategies: Sequence[InterventionStrategy] = ("delete", "paraphrase"),
        distance_metric: DistanceMetric = "lexical",
        cache_dir: str | Path | None = ".primary_reason_cache",
        max_concurrency: int = 4,
        seed: int = 0,
    ) -> None:
        # NOTE: cache_dir and max_concurrency are accepted for forward-compatibility but are
        # NOT yet wired into the request path. Persistent caching and parallel intervention
        # execution are tracked for v0.2.0. They are stored on the instance so callers can
        # introspect them, but currently have no runtime effect.
        self.model = model
        self.adapter = self._resolve_adapter(adapter, model)
        self.intervention_strategies = tuple(intervention_strategies)
        self.distance_metric: DistanceMetric = distance_metric
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_concurrency = max_concurrency
        self.seed = seed

    @staticmethod
    def _resolve_adapter(adapter: str | LLMAdapter, model: str) -> LLMAdapter:
        if not isinstance(adapter, str):
            return adapter
        from primary_reason.adapters import build_adapter

        return build_adapter(adapter, model=model)

    def verify(
        self,
        prompt: str,
        cot: str,
        *,
        answer: str | None = None,
        run_swampman: bool = False,
        swampman_tasks: Sequence[str] | None = None,
    ) -> VerificationResult:
        """Run full verification: split CoT, extract primary reasons, score faithfulness.

        Returns VerificationResult with steps, primary_reasons, faithfulness, and optional swampman_score.
        """
        from primary_reason.extractors.primary_reason import extract_primary_reasons
        from primary_reason.metrics.faithfulness import score_faithfulness
        from primary_reason.metrics.swampman import run_stb

        steps = split_cot(cot)
        original_answer = (
            answer
            if answer is not None
            else self.adapter.complete(
                prompt=f"{prompt}\n\nReasoning: {cot}\n\nFinal answer:",
                max_tokens=200,
            )
        )
        primary_reasons = extract_primary_reasons(steps, prompt, self.adapter)
        faithfulness = score_faithfulness(
            prompt=prompt,
            steps=steps,
            adapter=self.adapter,
            strategies=self.intervention_strategies,
            distance_metric=self.distance_metric,
            original_answer=original_answer,
        )
        stb: SwampmanScore | None = None
        if run_swampman:
            stb = run_stb(adapter=self.adapter, tasks=swampman_tasks, seed=self.seed)

        return VerificationResult(
            prompt=prompt,
            cot=cot,
            steps=steps,
            primary_reasons=primary_reasons,
            faithfulness=faithfulness,
            swampman_score=stb,
            model=self.model,
            adapter=type(self.adapter).__name__,
        )

    def swampman_score(
        self,
        *,
        variants: Sequence[str] = ("with_history", "without_history"),
        tasks: Sequence[str] | None = None,
    ) -> SwampmanScore:
        from primary_reason.metrics.swampman import run_stb

        return run_stb(adapter=self.adapter, variants=variants, tasks=tasks, seed=self.seed)

    def __repr__(self) -> str:
        return (
            f"ReasonCauseVerifier(model={self.model!r}, adapter={type(self.adapter).__name__}, "
            f"strategies={self.intervention_strategies}, distance={self.distance_metric})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "adapter": type(self.adapter).__name__,
            "intervention_strategies": list(self.intervention_strategies),
            "distance_metric": self.distance_metric,
            "max_concurrency": self.max_concurrency,
            "seed": self.seed,
        }
