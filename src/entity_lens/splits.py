from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from entity_lens.data import Sample


class Split(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    demonstrations: tuple[Sample, ...]
    references: tuple[Sample, ...]
    evaluation: tuple[Sample, ...]


def shuffled(samples: Sequence[Sample], seeds: Sequence[int]) -> list[Sample]:
    pool = list(samples)
    for seed in seeds:
        random.Random(seed).shuffle(pool)
    return pool


def full_evaluation_split(
    samples: Sequence[Sample], seeds: Sequence[int], n_demonstrations: int, n_references: int
) -> Split:
    if len(samples) < n_demonstrations + n_references:
        message = f"{len(samples)} samples cannot supply the demonstrations and references"
        raise ValueError(message)
    pool = shuffled(samples, seeds)
    split = Split(
        demonstrations=tuple(pool[:n_demonstrations]),
        references=tuple(pool[n_demonstrations : n_demonstrations + n_references]),
        evaluation=tuple(pool),
    )
    if Counter(split.evaluation) != Counter(samples):
        message = "The evaluation set must contain every source row exactly once"
        raise AssertionError(message)
    return split
