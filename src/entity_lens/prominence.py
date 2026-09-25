from __future__ import annotations

import re
from pathlib import Path

from entity_lens import data, modeling, storage
from entity_lens.config import Project, ProminenceConfig, RuntimeConfig
from entity_lens.records import Readout, Record, read, write


class Generations(Record):
    model: str
    prompts: tuple[str, ...]
    continuations: dict[str, str]


class RelationProminence(Record):
    relation: str
    prominence: float
    prominence_rows: int
    lens_hit_rate: float
    lens_rows: int


class ProminenceResult(Record):
    relations: tuple[RelationProminence, ...]


def distinctive_words(obj: str, subject: str, settings: ProminenceConfig) -> set[str]:
    subject_words = {word.lower() for word in re.findall(settings.word_pattern, subject)}
    return {
        word.lower()
        for word in re.findall(settings.word_pattern, obj)
        if len(word) >= settings.min_word_length
        and word.lower() not in settings.stop_words
        and word.lower() not in subject_words
    }


def mentions(continuation: str, words: set[str]) -> bool:
    return any(word in continuation for word in words)


def lens_hit(tokens: tuple[str, ...], words: set[str], settings: ProminenceConfig) -> bool:
    return any(
        len(token) >= settings.min_lens_token_length and any(w.startswith(token) or token.startswith(w) for w in words)
        for token in (t.strip().lower() for t in tokens)
    )


def subjects(project: Project, settings: ProminenceConfig) -> list[str]:
    names: dict[str, None] = {}
    for key in settings.relations:
        relation = data.load_relation(project.data / "relations", key, project.protocol.dataset)
        for sample in relation.samples:
            if distinctive_words(sample.object, sample.subject, settings):
                names.setdefault(sample.subject)
    return list(names)


def shard_path(project: Project, shard: int) -> Path:
    return project.artifacts / "prominence" / f"generations_{shard}.json.gz"


def generate(project: Project, shard: int) -> None:
    settings = project.experiments.prominence
    path = shard_path(project, shard)
    if path.exists():
        return
    spec = project.models.get(settings.model)
    runtime = RuntimeConfig.model_validate(
        project.protocol.runtime.model_dump() | {"dtype": settings.dtype, "attention": settings.attention}
    )
    lm = modeling.load(spec, runtime)
    names = subjects(project, settings)[shard :: settings.shards]
    continuations = {}
    for name in names:
        text = ""
        for prompt in settings.prompts:
            text += " " + lm.generate(prompt.format(name), settings.max_new_tokens).lower()
        continuations[name] = text
    write(path, Generations(model=spec.key, prompts=settings.prompts, continuations=continuations))


def evaluate(project: Project) -> ProminenceResult:
    settings = project.experiments.prominence
    continuations: dict[str, str] = {}
    for shard in range(settings.shards):
        continuations.update(read(shard_path(project, shard), Generations).continuations)
    measurements = storage.load_measurements(project, settings.model)
    results = []
    for key in settings.relations:
        relation = data.load_relation(project.data / "relations", key, project.protocol.dataset)
        generated = [
            mentions(continuations[s.subject], words)
            for s in relation.samples
            if (words := distinctive_words(s.object, s.subject, settings))
        ]
        lens = [
            lens_hit(row.top_tokens[Readout.CEL], words, settings)
            for row in measurements[key].lens.rows
            if (words := distinctive_words(row.object, row.subject, settings))
        ]
        results.append(
            RelationProminence(
                relation=key,
                prominence=sum(generated) / len(generated),
                prominence_rows=len(generated),
                lens_hit_rate=sum(lens) / len(lens),
                lens_rows=len(lens),
            )
        )
    return ProminenceResult(relations=tuple(results))
