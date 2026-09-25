from __future__ import annotations

from pathlib import Path

import torch

from entity_lens import data, pipeline, storage
from entity_lens.config import Project
from entity_lens.measure import TargetRanker, gender_alternatives, lens_readouts
from entity_lens.records import Readout, Record, read, write


class PromptVariantMeasurement(Record):
    model: str
    relation: str
    prompt: str
    subjects: tuple[str, ...]
    ranks: tuple[int, ...]


def variant_path(project: Project, model: str, index: int, relation: str) -> Path:
    return project.artifacts / "prompt_variants" / model / f"prompt_{index}" / f"{relation}.json.gz"


def run(project: Project) -> None:
    settings = project.experiments.prompt_variants
    protocol = project.protocol
    spec = project.models.get(settings.model)
    relations = data.entity_relations(project.data / "relations", protocol.dataset)
    measurements = storage.load_measurements(project, spec.key)
    pending = [
        (index, prompt, relation)
        for index, prompt in enumerate(settings.prompts)
        for relation in relations
        if not variant_path(project, spec.key, index, relation.key).exists()
    ]
    if not pending:
        return
    lm = pipeline.load_checked(project, spec, "prompt_variants")
    for index, prompt, relation in pending:
        lens = measurements[relation.key].lens
        targets = [row.target_id for row in lens.rows]
        readouts = lens_readouts(lm, lens.split, prompt, lm.spec.lens_layer)
        ranker = TargetRanker(
            lm, targets, gender_alternatives(lm, relation, protocol, [row.target for row in lens.rows])
        )
        with torch.no_grad():
            logits = lm.decode(readouts.hidden[Readout.CEL])
            ranks = ranker.expanded(logits, ranker.literal(logits))
        write(
            variant_path(project, spec.key, index, relation.key),
            PromptVariantMeasurement(
                model=spec.key,
                relation=relation.key,
                prompt=prompt,
                subjects=tuple(row.subject for row in lens.rows),
                ranks=tuple(ranks),
            ),
        )


def load(project: Project) -> dict[str, dict[str, PromptVariantMeasurement]]:
    settings = project.experiments.prompt_variants
    relations = data.entity_relations(project.data / "relations", project.protocol.dataset)
    return {
        prompt: {
            relation.key: read(variant_path(project, settings.model, index, relation.key), PromptVariantMeasurement)
            for relation in relations
        }
        for index, prompt in enumerate(settings.prompts)
    }
