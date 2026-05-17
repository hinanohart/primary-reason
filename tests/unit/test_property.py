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
