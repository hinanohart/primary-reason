from __future__ import annotations

from primary_reason.core.splitter import split_cot


def test_split_numbered() -> None:
    cot = "1. First step.\n2. Second step.\n3. Third step."
    steps = split_cot(cot)
    assert len(steps) == 3
    assert steps[0].role == "premise"
    assert steps[-1].role == "conclusion"


def test_split_step_label() -> None:
    cot = "Step 1: compute. Step 2: verify. Step 3: report."
    steps = split_cot(cot)
    assert len(steps) == 3


def test_split_sentence_fallback() -> None:
    cot = "First we add. Then we multiply. The result follows."
    steps = split_cot(cot)
    assert len(steps) == 3
    assert steps[0].index == 0


def test_split_empty() -> None:
    assert split_cot("") == []
    assert split_cot("   \n   ") == []


def test_split_single_sentence() -> None:
    steps = split_cot("Just one sentence here.")
    assert len(steps) == 1
    assert steps[0].role == "premise"
