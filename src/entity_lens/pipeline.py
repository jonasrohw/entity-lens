from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import torch

from entity_lens import checks, data, modeling
from entity_lens.config import ModelSpec, Project
from entity_lens.measure import answer_prior, measure_relation
from entity_lens.records import RuntimeChecks, WorkerRecord, write
from entity_lens.storage import measurement_path, numeric_object_path, worker_path

PACKAGE_ROOT = Path(__file__).resolve().parent


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
    }


def worker_record(lm: modeling.LanguageModel, runtime_checks: RuntimeChecks) -> WorkerRecord:
    return WorkerRecord(
        model=lm.spec.key,
        hf_id=lm.spec.hf_id,
        revision=lm.spec.revision,
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        device=torch.cuda.get_device_name(lm.output_device)
        if lm.output_device.type == "cuda"
        else str(lm.output_device),
        checks=runtime_checks,
        source_sha256=source_hashes(),
    )


def load_checked(project: Project, spec: ModelSpec, stage: str) -> modeling.LanguageModel:
    lm = modeling.load(spec, project.protocol.runtime)
    write(worker_path(project, spec.key, stage), worker_record(lm, checks.prepare(lm, project.protocol.checks)))
    return lm


def evaluate_model(project: Project, key: str) -> None:
    protocol = project.protocol
    spec = project.models.get(key)
    relations = data.entity_relations(project.data / "relations", protocol.dataset)
    pending = [r for r in relations if not measurement_path(project, key, r.key).exists()]
    if not pending:
        return
    lm = load_checked(project, spec, "evaluation")
    prior = answer_prior(lm, protocol.answer_prior.contexts)
    for relation in sorted(pending, key=lambda r: -len(r.samples)):
        record = measure_relation(lm, relation, protocol, prior)
        write(measurement_path(project, key, relation.key), record)
        print(f"{key} {relation.key} faithfulness={record.lre.faithfulness:.4f} seconds={record.seconds:.0f}")


def evaluate_without_trailing_space(project: Project, key: str) -> None:
    protocol = project.protocol
    names = sorted(protocol.years.relations)
    pending = [name for name in names if not numeric_object_path(project, key, name).exists()]
    if not pending:
        return
    lm = load_checked(project, project.models.get(key), "numeric_objects")
    prior = answer_prior(lm, protocol.answer_prior.contexts)
    for name in pending:
        relation = data.load_relation(project.data / "relations", name, protocol.dataset)
        write(numeric_object_path(project, key, name), measure_relation(lm, relation, protocol, prior, trailing=False))
