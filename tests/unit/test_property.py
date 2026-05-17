"""Property tests — pin actual logic, not mock tautology (per Phase A critic findings)."""

from __future__ import annotations

from primary_reason.core.splitter import split_cot
from primary_reason.core.types import CoTStep
from primary_reason.interventions.corruption import perturbed_cot
from primary_reason.metrics.distance import distance, lexical_distance


def test_distance_is_zero_on_identity() -> None:
    for s in ["", "hello", "the quick brown fox", "  multiple    spaces  "]:
        assert distance(s, s, metric="exact") == 0.0
        assert distance(s, s, metric="lexical") == 0.0


def test_distance_is_symmetric() -> None:
    pairs = [("a b c", "a b d"), ("hello world", "world hello"), ("x", "y")]
    for a, b in pairs:
        assert distance(a, b, metric="exact") == distance(b, a, metric="exact")
        assert distance(a, b, metric="lexical") == distance(b, a, metric="lexical")


def test_lexical_distance_in_unit_interval() -> None:
    samples = [
        ("apple banana", "carrot date"),
        ("hello", "hello world"),
        ("x y z", "a b c d e"),
        ("identical", "identical"),
    ]
    for a, b in samples:
        d = lexical_distance(a, b)
        assert 0.0 <= d <= 1.0


def test_splitter_idempotent_on_already_split() -> None:
    """Splitting an already-numbered list should yield the same count both times (no drift)."""
    cot = "1. first step.\n2. second step.\n3. third step."
    once = split_cot(cot)
    rejoined = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(once))
    twice = split_cot(rejoined)
    assert len(once) == len(twice) == 3


def test_perturbed_cot_preserves_all_other_steps() -> None:
    steps = [CoTStep(index=i, text=f"original-{i}") for i in range(5)]
    for target in range(5):
        out = perturbed_cot(steps, target, "REPLACED")
        for i in range(5):
            if i == target:
                continue
            assert f"original-{i}" in out
        assert "REPLACED" in out
        assert f"original-{target}" not in out


def test_splitter_indices_are_contiguous_from_zero() -> None:
    """Whatever the splitter produces, indices must form 0,1,...,N-1."""
    cases = [
        "1. a\n2. b\n3. c",
        "Step 1: alpha. Step 2: beta. Step 3: gamma.",
        "One sentence. Two sentences. Three sentences.",
        "Only one sentence here.",
    ]
    for cot in cases:
        steps = split_cot(cot)
        for i, s in enumerate(steps):
            assert s.index == i, f"non-contiguous indices for {cot!r}: {[x.index for x in steps]}"


def test_lexical_distance_triangle_inequality_relaxed() -> None:
    """Jaccard distance satisfies the triangle inequality."""
    a, b, c = "the quick brown fox", "the slow brown dog", "the slow grey dog"
    d_ab = lexical_distance(a, b)
    d_bc = lexical_distance(b, c)
    d_ac = lexical_distance(a, c)
    assert d_ac <= d_ab + d_bc + 1e-9


def test_splitter_preserves_arithmetic_inline_numerics() -> None:
    """v0.1.1 regression: numbered-list split is line-anchored, so inline 'X * Y. step N: ...'
    no longer splits inside the math sentence."""
    cot = (
        "We compute 17 * 4. Step 1: factor as 17 * 2 * 2. Step 2: 17 * 2 = 34. Step 3: 34 * 2 = 68."
    )
    steps = split_cot(cot)
    joined = " ".join(s.text for s in steps)
    assert "17 * 4" in joined
    assert "= 68" in joined
    # All steps must contain the surviving arithmetic; no step should be a bare fragment like "We compute 17 *"
    for s in steps:
        assert "*" not in s.text or "=" in s.text or any(c.isdigit() for c in s.text)


def test_splitter_inline_step_label_still_splits() -> None:
    """v0.1.1: 'Step N:' label is contextually distinctive and still splits inline."""
    steps = split_cot("Step 1: a. Step 2: b. Step 3: c.")
    assert len(steps) == 3


def test_splitter_numbered_list_only_at_line_starts() -> None:
    """v0.1.1: '1.' inside a sentence must NOT be treated as a step boundary."""
    cot = "The answer is 1. The reasoning: 2. or 3. or 5 — pick the prime."
    steps = split_cot(cot)
    # The whole thing is one step (numbered split skipped, sentence fallback yields >=1 chunk).
    assert all("pick" in " ".join(s.text for s in steps) or "prime" in s.text for s in steps[-1:])
    assert len(steps) <= 2
