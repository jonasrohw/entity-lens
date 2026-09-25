from __future__ import annotations

import random
from collections import Counter

import pytest

from entity_lens.data import Sample
from entity_lens.splits import full_evaluation_split, shuffled

SAMPLES = [Sample(subject=f"s{i}", object=f"o{i % 3}") for i in range(20)] + [Sample(subject="s0", object="o0")]


def test_shuffle_chain_matches_sequential_seeds() -> None:
    expected = list(SAMPLES)
    random.Random(0).shuffle(expected)
    random.Random(1000).shuffle(expected)
    assert shuffled(SAMPLES, [0, 1000]) == expected


def test_full_split_keeps_every_row_and_duplicates() -> None:
    split = full_evaluation_split(SAMPLES, [17], 8, 8)
    assert Counter(split.evaluation) == Counter(SAMPLES)
    assert list(split.demonstrations) == list(split.evaluation[:8])
    assert list(split.references) == list(split.evaluation[8:16])


def test_split_is_deterministic() -> None:
    assert full_evaluation_split(SAMPLES, [0, 1000], 8, 8) == full_evaluation_split(SAMPLES, [0, 1000], 8, 8)
    assert full_evaluation_split(SAMPLES, [17], 8, 8) != full_evaluation_split(SAMPLES, [0, 1000], 8, 8)


def test_split_requires_enough_rows() -> None:
    with pytest.raises(ValueError, match="cannot supply"):
        full_evaluation_split(SAMPLES[:5], [0], 4, 4)
