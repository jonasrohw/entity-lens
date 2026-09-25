from __future__ import annotations

from pathlib import Path

from entity_lens import data
from entity_lens.config import Project
from entity_lens.records import RelationMeasurement, read


def measurement_path(project: Project, model: str, relation: str) -> Path:
    return project.artifacts / "measurements" / model / f"{relation}.json.gz"


def worker_path(project: Project, model: str, stage: str) -> Path:
    return project.artifacts / "workers" / f"{model}_{stage}.json"


def numeric_object_path(project: Project, model: str, relation: str) -> Path:
    return project.artifacts / "numeric_objects" / model / f"{relation}.json.gz"


def load_measurements(project: Project, key: str) -> dict[str, RelationMeasurement]:
    relations = data.entity_relations(project.data / "relations", project.protocol.dataset)
    return {r.key: read(measurement_path(project, key, r.key), RelationMeasurement) for r in relations}
