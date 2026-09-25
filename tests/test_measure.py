from __future__ import annotations

from collections import Counter

import pytest
import torch

from entity_lens import data, measure
from entity_lens.config import Project
from entity_lens.modeling import LanguageModel
from entity_lens.records import Readout, RelationMeasurement
from entity_lens.splits import full_evaluation_split


@pytest.fixture
def measured(project: Project, language_model: LanguageModel) -> dict[str, RelationMeasurement]:
    prior = measure.answer_prior(language_model, project.protocol.answer_prior.contexts)
    names = ("country_capital_city", "name_gender")
    relations = [data.load_relation(project.data / "relations", n, project.protocol.dataset) for n in names]
    return {r.key: measure.measure_relation(language_model, r, project.protocol, prior) for r in relations}


def test_answer_prior_is_log_distribution(project: Project, language_model: LanguageModel) -> None:
    prior = measure.answer_prior(language_model, project.protocol.answer_prior.contexts)
    assert prior.shape == (len(language_model.tokenizer),)
    assert bool((prior <= 0).all())
    assert float(prior.exp().sum()) <= 1.0 + 1e-5


def test_measurement_covers_every_row(project: Project, measured: dict[str, RelationMeasurement]) -> None:
    for name, record in measured.items():
        relation = data.load_relation(project.data / "relations", name, project.protocol.dataset)
        rows = Counter((s.subject, s.object) for s in relation.samples)
        assert Counter((r.subject, r.object) for r in record.lre.rows) == rows
        assert Counter((r.subject, r.object) for r in record.lens.rows) == rows
        assert len(record.lre.grid) == len(project.protocol.lre.betas) * 3
        best = max(record.lre.grid, key=lambda cell: cell.faithfulness)
        assert record.lre.faithfulness == best.faithfulness
        assert record.lre.faithfulness == sum(r.hit for r in record.lre.rows) / len(record.lre.rows)


def test_self_references_are_excluded(project: Project, measured: dict[str, RelationMeasurement]) -> None:
    record = measured["country_capital_city"]
    references = {r.subject for r in record.lens.split.references}
    for row in record.lens.rows:
        assert row.n_references == project.protocol.split.n_references - (row.subject in references)


def test_gender_alternatives_never_worsen_ranks(measured: dict[str, RelationMeasurement]) -> None:
    for row in measured["name_gender"].lens.rows:
        for readout in Readout:
            assert row.ranks[readout] <= row.literal_ranks[readout]
    for row in measured["country_capital_city"].lens.rows:
        assert row.ranks == row.literal_ranks


def test_grid_predictions_reproduce_best_cell(measured: dict[str, RelationMeasurement]) -> None:
    record = measured["country_capital_city"].lre
    index = next(i for i, c in enumerate(record.grid) if (c.layer, c.beta) == (record.best_layer, record.best_beta))
    for row in record.rows:
        assert row.grid_predictions[index] == row.prediction_id
        assert row.hit == (row.prediction_id == row.target_id)


def test_lens_readouts_follow_definitions(project: Project, language_model: LanguageModel) -> None:
    relation = data.load_relation(project.data / "relations", "country_capital_city", project.protocol.dataset)
    split = full_evaluation_split(relation.samples, [17], 2, 3)
    layer = language_model.spec.lens_layer
    readouts = measure.lens_readouts(language_model, split, "{} is known for", layer)
    row = split.evaluation[-1]
    cache = {
        s.subject: language_model.cache_prompt(f"{s.subject} is known for", s.subject, [layer])
        for s in split.evaluation
    }
    intervention = language_model.intervention()
    contrastive, noncontrastive, centers, anchors, count = 0, 0, 0, 0, 0
    with language_model.hooked(layer, intervention):
        for reference in split.references:
            if reference.subject == row.subject:
                continue
            ref = cache[reference.subject]
            target = cache[row.subject].states[layer]
            products = language_model.jvps(intervention, ref, layer, torch.stack([target - ref.states[layer], target]))
            contrastive = contrastive + products[0]
            noncontrastive = noncontrastive + products[1]
            centers = centers + ref.states[layer]
            anchors = anchors + ref.final
            count += 1
    index = len(split.evaluation) - 1
    torch.testing.assert_close(readouts.hidden[Readout.CEL][index], contrastive / count, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(readouts.hidden[Readout.NEL][index], noncontrastive / count, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(readouts.hidden[Readout.LOGIT_LENS][index], cache[row.subject].states[layer])
    expected = cache[row.subject].states[layer] - centers / count
    torch.testing.assert_close(readouts.hidden[Readout.DIFFERENTIAL_LOGIT_LENS][index], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(readouts.anchor[index], anchors / count, rtol=1e-5, atol=1e-6)


def test_ranks_of_counts_strictly_larger_logits() -> None:
    logits = torch.tensor([[0.1, 0.5, 0.5, 0.2]])
    assert measure.ranks_of(logits, torch.tensor([1])).tolist() == [0]
    assert measure.ranks_of(logits, torch.tensor([3])).tolist() == [2]


def test_trailing_override(project: Project, language_model: LanguageModel) -> None:
    prior = measure.answer_prior(language_model, project.protocol.answer_prior.contexts)
    relation = data.load_relation(project.data / "relations", "country_capital_city", project.protocol.dataset)
    record = measure.measure_relation(language_model, relation, project.protocol, prior, trailing=True)
    assert record.lre.trailing_space
