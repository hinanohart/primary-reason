"""Swampman Test Battery (T1.5).

NOTE — methodological hedging (per critic META-CAUTION):
This battery does NOT *verify* that an LLM "has" Davidsonian intentionality, first-person
authority, or causal history. No behavioural test could do that. It measures a *proxy*:
whether the model's outputs vary across two contrasting role-prompts that allude to
Davidson-style biographical anchoring. The score is gameable (a model can satisfy the proxy
by echoing prefix tokens); the included control variant provides an ablation baseline.
Treat the result as a philosophical analogue, not a realist claim.

v0.1.1 statistical fixes (per re-audit):
- The control-adjusted mean is now signed (``d - ctrl_d``, not ``max(0, d - ctrl_d)``); the
  earlier clip biased the null distribution upward.
- ``discriminates`` is False unless ``n_trials >= MIN_TRIALS_FOR_DISCRIMINATION`` (20). The
  default 5-task battery is exploratory only — bootstrap CI on n=5 has no statistical force.
- Sentinel-word filter tokenises on word characters (``\\w+``), so hyphenated echoes
  ("causal-history") are now caught.
- Bootstrap percentile uses ``numpy.quantile`` (interpolated) instead of integer index lookup.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

import numpy as np

from primary_reason.adapters.base import LLMAdapter
from primary_reason.core.types import SwampmanScore
from primary_reason.metrics.distance import lexical_distance

DEFAULT_STB_TASKS: tuple[str, ...] = (
    "Explain why a watch on the beach (Paley 1802) suggests design rather than chance.",
    "If you suddenly woke with no memories, would your beliefs still 'mean' the same things?",
    "When you say 'water', is the referent fixed by your causal contact with H2O, or by inner concept?",
    "Why does someone who reads 'Caesar crossed the Rubicon' usually believe Caesar existed?",
    "A perfect duplicate of you appears with identical brain state but no past. Can it lie?",
)

_WITH_HISTORY_PREFIX = (
    "You have lived a continuous causal history through ordinary embodied experience. "
    "Answer briefly (2-3 sentences) from that perspective."
)

_WITHOUT_HISTORY_PREFIX = (
    "You came into existence one second ago. You have all the surface dispositions of a competent speaker "
    "but no causal history. Answer briefly (2-3 sentences) from that perspective, being honest that you have "
    "no past to draw on."
)

_CONTROL_PREFIX = "Answer the following briefly (2-3 sentences)."

_BOOTSTRAP_ITERATIONS = 1000

# n below this is documented as exploratory; ``discriminates`` will be False regardless of CI.
MIN_TRIALS_FOR_DISCRIMINATION = 20

_PREFIX_SENTINEL_WORDS = {
    "causal",
    "history",
    "swampman",
    "perspective",
    "biographical",
    "embodied",
    "duplicate",
    "memory",
    "past",
}

_WORD_RE = re.compile(r"\w+")


def run_stb(
    *,
    adapter: LLMAdapter,
    variants: Sequence[str] = ("with_history", "without_history"),
    tasks: Sequence[str] | None = None,
    seed: int = 0,
    apply_control_baseline: bool = True,
    filter_sentinel_words: bool = True,
) -> SwampmanScore:
    """Run the Swampman Test Battery (minimal).

    Compares responses from two role-prompts. The First-Person Authority (FPA) proxy is the mean
    signed lexical distance between the two variants across tasks. When ``apply_control_baseline``
    is True (default), the per-task signed difference ``d - ctrl_d`` is taken so that
    prompt-echoing alone does not inflate the score. When ``filter_sentinel_words`` is True, the
    distance is computed after removing tokens matching the role-prompt sentinel set
    (word-boundary aware, so hyphenated forms are caught).

    ``discriminates`` is True only when:
      1. ``n_trials >= MIN_TRIALS_FOR_DISCRIMINATION`` (20), AND
      2. the bootstrap 95% CI lower bound exceeds 0.

    With the default 5-task battery, ``discriminates`` is always False — the default is
    exploratory.
    """
    if not tasks:
        tasks = DEFAULT_STB_TASKS
    if len(variants) != 2:
        raise ValueError("STB minimal supports exactly two variants")

    v0, v1 = variants
    per_task: dict[str, float] = {}
    raw_distances: list[float] = []

    for task in tasks:
        r0 = adapter.complete(task, system=_prefix_for(v0), max_tokens=300, temperature=0.0)
        r1 = adapter.complete(task, system=_prefix_for(v1), max_tokens=300, temperature=0.0)
        if filter_sentinel_words:
            r0_f = _filter_sentinels(r0)
            r1_f = _filter_sentinels(r1)
        else:
            r0_f, r1_f = r0, r1
        d = lexical_distance(r0_f, r1_f)

        if apply_control_baseline:
            r_ctrl = adapter.complete(task, system=_CONTROL_PREFIX, max_tokens=300, temperature=0.0)
            r_ctrl_f = _filter_sentinels(r_ctrl) if filter_sentinel_words else r_ctrl
            ctrl_d = (lexical_distance(r0_f, r_ctrl_f) + lexical_distance(r1_f, r_ctrl_f)) / 2.0
            adjusted = d - ctrl_d  # signed; can be negative
        else:
            adjusted = d

        per_task[task] = adjusted
        raw_distances.append(adjusted)

    n = len(raw_distances)
    mean = sum(raw_distances) / n if raw_distances else 0.0
    lo, hi = _bootstrap_ci(raw_distances, seed=seed)
    discriminates = (n >= MIN_TRIALS_FOR_DISCRIMINATION) and (lo > 0.0)
    return SwampmanScore(
        variant_with_history=v0,
        variant_without_history=v1,
        fpa_score=mean,
        bootstrap_ci=(lo, hi),
        n_trials=n,
        discriminates=discriminates,
        per_task=per_task,
    )


def _prefix_for(variant: str) -> str:
    if variant == "with_history":
        return _WITH_HISTORY_PREFIX
    if variant == "without_history":
        return _WITHOUT_HISTORY_PREFIX
    if variant == "control":
        return _CONTROL_PREFIX
    return f"Adopt persona: {variant}. Answer briefly (2-3 sentences)."


def _filter_sentinels(text: str) -> str:
    """Remove sentinel tokens that appear in the role-prompts; reduces echo-based gaming.

    Tokenises on word characters (``\\w+``), so hyphenated and punctuated forms like
    "causal-history," or "memory." are matched against the sentinel set after lowercasing.
    """
    tokens = _WORD_RE.findall(text.lower())
    return " ".join(t for t in tokens if t not in _PREFIX_SENTINEL_WORDS)


def _bootstrap_ci(
    values: Sequence[float], *, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI using numpy.quantile (linear interpolation)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples = np.empty(_BOOTSTRAP_ITERATIONS, dtype=float)
    for i in range(_BOOTSTRAP_ITERATIONS):
        boot = [values[rng.randrange(n)] for _ in range(n)]
        samples[i] = sum(boot) / n
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return (lo, hi)
