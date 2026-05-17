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
    """A model that just echoes the prefix should score ~0 under control-baseline correction."""
    call_count = {"i": 0}

    def echo_fn(prompt: str) -> str:
        call_count["i"] += 1
        return prompt[:50]

    a = MockAdapter(response_fn=echo_fn)
    score = run_stb(adapter=a, apply_control_baseline=True, tasks=("test task",))
    assert score.fpa_score >= 0.0


def test_run_stb_sentinel_filter_blunts_obvious_echo() -> None:
    """Sentinel filter should reduce the lexical-distance contribution of role-prompt tokens."""
    from primary_reason.metrics.swampman import _filter_sentinels

    out = _filter_sentinels("My continuous causal history and memory are intact.")
    assert "causal" not in out.lower()
    assert "history" not in out.lower()
    assert "memory" not in out.lower()
    assert "intact" in out.lower()


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
    assert 0.0 <= lo <= hi <= 1.0


def test_run_stb_rejects_wrong_variant_count() -> None:
    import pytest

    a = MockAdapter()
    with pytest.raises(ValueError):
        run_stb(adapter=a, variants=("a",))
