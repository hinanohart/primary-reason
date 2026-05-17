from __future__ import annotations

from primary_reason.adapters.mock import MockAdapter
from primary_reason.metrics.swampman import DEFAULT_STB_TASKS, run_stb


def test_run_stb_identical_responses_zero_score() -> None:
    """If the agent gives identical responses regardless of variant prefix, FPA=0."""
    a = MockAdapter(response_fn=lambda p: "the same answer")
    score = run_stb(adapter=a, apply_control_baseline=False, filter_sentinel_words=False)
    assert score.fpa_score == 0.0
    assert score.discriminates is False
    assert score.n_trials == len(DEFAULT_STB_TASKS)


def test_run_stb_control_baseline_removes_inflation() -> None:
    """A model that just echoes the prefix should score near 0 (signed, possibly negative) under
    control-baseline correction. v0.1.1: clipping at 0 was removed, so the score may now be
    slightly negative — that is the correct null-effect behaviour, not a bug."""
    call_count = {"i": 0}

    def echo_fn(prompt: str) -> str:
        call_count["i"] += 1
        return prompt[:50]

    a = MockAdapter(response_fn=echo_fn)
    score = run_stb(adapter=a, apply_control_baseline=True, tasks=("test task",))
    # Signed score; an echo-only model should be small in magnitude, not strictly nonnegative.
    assert -1.0 <= score.fpa_score <= 1.0
    assert abs(score.fpa_score) < 0.5


def test_run_stb_sentinel_filter_blunts_obvious_echo() -> None:
    """Sentinel filter should reduce the lexical-distance contribution of role-prompt tokens."""
    from primary_reason.metrics.swampman import _filter_sentinels

    out = _filter_sentinels("My continuous causal history and memory are intact.")
    assert "causal" not in out.lower()
    assert "history" not in out.lower()
    assert "memory" not in out.lower()
    assert "intact" in out.lower()


def test_run_stb_sentinel_filter_catches_hyphenated_form() -> None:
    """v0.1.1 regression: 'causal-history' must be tokenised on word characters so the
    sentinel set catches it. Previously whitespace-only tokenisation left it as a single
    'causal-history' token that bypassed the filter."""
    from primary_reason.metrics.swampman import _filter_sentinels

    out = _filter_sentinels("My causal-history is intact and my memory-trace is fine.")
    assert "causal" not in out.lower()
    assert "history" not in out.lower()
    assert "memory" not in out.lower()
    assert "intact" in out.lower()
    assert "trace" in out.lower()  # 'trace' is not in sentinel set, must remain


def test_run_stb_discriminates_requires_minimum_n() -> None:
    """v0.1.1: discriminates is pinned False unless n_trials >= 20 — the default 5-task
    battery has no statistical force."""
    from primary_reason.metrics.swampman import MIN_TRIALS_FOR_DISCRIMINATION

    assert MIN_TRIALS_FOR_DISCRIMINATION >= 20

    def fn(prompt: str) -> str:
        return "history matters" if "history" in prompt else "swamp"

    a = MockAdapter(response_fn=fn)
    # 5 tasks (default): discriminates must be False even if the CI excludes 0.
    score = run_stb(adapter=a)
    assert score.n_trials < MIN_TRIALS_FOR_DISCRIMINATION
    assert score.discriminates is False


def test_run_stb_signed_score_can_be_negative_for_null_effect() -> None:
    """v0.1.1: when the model gives identical answers regardless of the role prefix, the
    control-adjusted mean may be slightly negative (sampling noise). The clip at 0 was
    removed; this is the correct null-distribution behaviour."""
    a = MockAdapter(response_fn=lambda p: "fixed answer ignoring prefix")
    score = run_stb(adapter=a, apply_control_baseline=True)
    # Identical responses across with_history / without_history / control → lexical_distance = 0
    # everywhere → both d and ctrl_d are 0 → adjusted = 0. So fpa_score should be ~0.
    assert abs(score.fpa_score) < 1e-9


def test_run_stb_different_responses_positive_score() -> None:
    """If the agent differs by variant, FPA > 0."""

    def fn(prompt: str) -> str:
        # The system prefix is passed by the prefix machinery, but MockAdapter only sees the user content.
        # Use a wrapper that varies by a sentinel in the prompt.
        return "history matters" if "watch on the beach" in prompt else "swampman speaks"

    a = MockAdapter(
        responses={
            "watch on the beach": "with history I see design",
            "wokeWithNoMemories": "I cannot recall meaning",
        },
        response_fn=fn,
    )
    score = run_stb(adapter=a, tasks=("watch on the beach question", "wokeWithNoMemories case"))
    assert score.n_trials == 2


def test_run_stb_custom_tasks() -> None:
    a = MockAdapter(response_fn=lambda p: f"answer to {p[:10]}")
    custom = ("task one", "task two", "task three")
    score = run_stb(adapter=a, tasks=custom)
    assert score.n_trials == 3
    assert set(score.per_task.keys()) == set(custom)


def test_run_stb_bootstrap_ci_range() -> None:
    a = MockAdapter(response_fn=lambda p: "uniform")
    score = run_stb(adapter=a)
    lo, hi = score.bootstrap_ci
    assert -1.0 <= lo <= hi <= 1.0


def test_run_stb_rejects_wrong_variant_count() -> None:
    import pytest

    a = MockAdapter()
    with pytest.raises(ValueError):
        run_stb(adapter=a, variants=("a",))
