from __future__ import annotations

import pytest

from entity_lens import data, modeling, pipeline, storage
from entity_lens.config import Project
from entity_lens.modeling import LanguageModel
from entity_lens.records import RelationMeasurement, WorkerRecord, read


def test_evaluate_model_writes_every_relation(evaluated: Project) -> None:
    relations = data.entity_relations(evaluated.data / "relations", evaluated.protocol.dataset)
    measurements = storage.load_measurements(evaluated, "olmo3_7b")
    assert set(measurements) == {r.key for r in relations}
    worker = read(storage.worker_path(evaluated, "olmo3_7b", "evaluation"), WorkerRecord)
    assert worker.model == "tiny"
    assert worker.device == "cpu"
    assert "measure.py" in worker.source_sha256


def test_evaluate_model_is_resumable(evaluated: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> LanguageModel:
        raise AssertionError

    monkeypatch.setattr(modeling, "load", fail)
    pipeline.evaluate_model(evaluated, "olmo3_7b")


def test_numeric_objects_run(project: Project, loaded: LanguageModel) -> None:
    protocol = project.protocol
    years = protocol.years.model_copy(update={"relations": frozenset({"country_capital_city"})})
    project.__dict__["protocol"] = protocol.model_copy(update={"years": years})
    pipeline.evaluate_without_trailing_space(project, "olmo3_7b")
    record = read(storage.numeric_object_path(project, "olmo3_7b", "country_capital_city"), RelationMeasurement)
    assert not record.lre.trailing_space
    assert loaded.spec.key == record.model
