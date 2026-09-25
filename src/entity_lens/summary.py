from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from pydantic import BaseModel, ConfigDict

from entity_lens.data import Relation
from entity_lens.records import Readout, RelationMeasurement
from entity_lens.statistics import Correlation, median_log_rank, spearman


class RelationSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    relation: str
    domain: str
    n_rows: int
    cardinality: float
    faithfulness: float
    best_layer: int
    best_beta: float
    excluded: bool
    scores: dict[Readout, float]
    literal_cel_score: float
    anchored_scores: dict[float, float]
    answer_prior: float


def summarize(measurement: RelationMeasurement, relation: Relation) -> RelationSummary:
    lens = measurement.lens
    betas = lens.rows[0].anchored_ranks.keys()
    return RelationSummary(
        model=measurement.model,
        relation=measurement.relation,
        domain=relation.domain,
        n_rows=len(lens.rows),
        cardinality=relation.cardinality,
        faithfulness=measurement.lre.faithfulness,
        best_layer=measurement.lre.best_layer,
        best_beta=measurement.lre.best_beta,
        excluded=bool(measurement.year_audit and measurement.year_audit.excluded),
        scores={readout: lens.score(readout) for readout in Readout},
        literal_cel_score=-median_log_rank([row.literal_ranks[Readout.CEL] for row in lens.rows]),
        anchored_scores={beta: -median_log_rank([row.anchored_ranks[beta] for row in lens.rows]) for beta in betas},
        answer_prior=float(np.mean([row.answer_prior for row in lens.rows])),
    )


class FactCorrelation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relation: str
    correlation: Correlation
    target_disagreements: int


def fact_correlation(measurement: RelationMeasurement) -> FactCorrelation:
    pending: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    lre_targets: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for row in measurement.lre.rows:
        pending[row.subject, row.object].append(row.rank)
        lre_targets[row.subject, row.object].append(row.target_id)
    lens_ranks, lre_ranks, disagreements = [], [], 0
    for lens_row in measurement.lens.rows:
        key = (lens_row.subject, lens_row.object)
        lens_ranks.append(lens_row.ranks[Readout.CEL])
        lre_ranks.append(pending[key].popleft())
        disagreements += lens_row.target_id != lre_targets[key].popleft()
    if any(pending.values()):
        message = f"{measurement.relation}: LRE and lens rows are not the same multiset"
        raise AssertionError(message)
    return FactCorrelation(
        relation=measurement.relation,
        correlation=spearman(lens_ranks, lre_ranks),
        target_disagreements=disagreements,
    )
