from __future__ import annotations

import pytest

from primary_reason.metrics.distance import (
    distance,
    exact_distance,
    lexical_distance,
    normalize,
)


def test_normalize() -> None:
    assert normalize("  Hello World.  ") == "hello world"


def test_exact_distance() -> None:
    assert exact_distance("yes", "yes") == 0.0
    assert exact_distance("yes", "no") == 1.0
    assert exact_distance("YES.", "yes") == 0.0


def test_lexical_distance_identical() -> None:
    assert lexical_distance("the cat sat", "the cat sat") == 0.0


def test_lexical_distance_disjoint() -> None:
    assert lexical_distance("apple banana", "carrot date") == 1.0


def test_lexical_distance_partial() -> None:
    d = lexical_distance("the cat sat", "the dog sat")
    assert 0.0 < d < 1.0


def test_distance_dispatch_unknown() -> None:
    with pytest.raises(ValueError):
        distance("a", "b", metric="bogus")  # type: ignore[arg-type]
