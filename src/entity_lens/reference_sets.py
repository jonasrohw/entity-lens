from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import torch
from torch import Tensor

from entity_lens import pipeline, storage
from entity_lens.config import Project, ReferenceSetConfig
from entity_lens.measure import ranks_of
from entity_lens.modeling import HiddenStateIntervention, LanguageModel, PromptCache
from entity_lens.records import LreRow, Record, write

LENSES = ("cel", "nel")


class GroupScores(Record):
    n: int
    median_log_rank: dict[str, float]
    median_answer_rank: dict[str, float]


class ArmResult(Record):
    references: tuple[str, ...]
    subtracted_top: tuple[str, ...]
    groups: dict[str, GroupScores]
    answer_ranks: dict[str, dict[str, int]]
    top_tokens: dict[str, dict[str, tuple[str, ...]]]


class ReferenceSetResult(Record):
    n_rows: int
    n_answer_gold: int
    n_answer_target: int
    arms: dict[str, ArmResult]
    monotone_subjects: int
    example: dict[str, int]
    example_top: tuple[str, ...]
    n_answer_test: int


def design(
    rows: Sequence[LreRow], cross: Sequence[Sequence[LreRow]], settings: ReferenceSetConfig
) -> dict[str, list[str]]:
    generator = random.Random(settings.seed)
    answer = [row.subject for row in rows if row.object == settings.answer]
    other = [row.subject for row in rows if row.object != settings.answer]
    answer_refs = sorted(generator.sample(answer, settings.n_references))
    other_refs = sorted(generator.sample(other, settings.n_references))
    half = settings.n_references // 2
    cross_refs = [subject for group in cross for subject in generator.sample([r.subject for r in group], half)]
    return {
        "other": other_refs,
        "cross": cross_refs,
        "mixed": answer_refs[:half] + other_refs[:half],
        "answer": answer_refs,
    }


def median_rank(ranks: Sequence[int]) -> float:
    return float(np.median(np.asarray(ranks)))


def run(project: Project) -> None:
    settings = project.experiments.reference_sets
    spec = project.models.get(settings.model)
    measurements = storage.load_measurements(project, spec.key)
    start = project.protocol.split.n_demonstrations + project.protocol.split.n_references

    def subset(name: str) -> tuple[LreRow, ...]:
        return measurements[name].lre.rows[start : start + settings.subset_size]

    rows = subset(settings.relation)
    arms = design(rows, [subset(name) for name in settings.cross_relations], settings)
    excluded = set(arms["answer"]) | set(arms["other"])
    groups = {
        "answer": [row for row in rows if row.object == settings.answer and row.subject not in excluded],
        "other": [row for row in rows if row.object != settings.answer and row.subject not in excluded],
    }
    lm = pipeline.load_checked(project, spec, "reference_sets")
    answer_token = lm.tokenizer.encode(" " + settings.answer, add_special_tokens=False)[0]
    layer = lm.spec.lens_layer
    prompt = project.protocol.prompts.lens_prompt(lm.spec)
    names = list(
        dict.fromkeys(
            [*[r.subject for group in groups.values() for r in group], *[s for a in arms.values() for s in a]]
        )
    )
    cache = {name: lm.cache_prompt(prompt.format(name), name, [layer]) for name in names}
    intervention = lm.intervention()
    results = {}
    with lm.hooked(layer, intervention):
        for arm, references in arms.items():
            results[arm] = evaluate_arm(lm, cache, layer, intervention, references, groups, answer_token, settings)
    answer_test = [row.subject for row in groups["answer"]]
    order = ("other", "mixed", "answer")
    monotone = sum(
        all(results[a].answer_ranks["cel"][s] < results[b].answer_ranks["cel"][s] for a, b in pairwise(order))
        for s in answer_test
    )
    write(
        project.artifacts / "reference_sets" / f"{spec.key}.json",
        ReferenceSetResult(
            n_rows=len(rows),
            n_answer_gold=sum(row.object == settings.answer for row in rows),
            n_answer_target=sum(row.object == settings.answer and row.target == settings.answer for row in rows),
            arms=results,
            monotone_subjects=monotone,
            example={
                f"{arm}.{lens}": ranks[settings.example_subject]
                for arm, result in results.items()
                for lens, ranks in result.answer_ranks.items()
                if settings.example_subject in ranks
            },
            example_top=results["answer"].top_tokens["cel"].get(settings.example_subject, ()),
            n_answer_test=len(answer_test),
        ),
    )


def evaluate_arm(
    lm: LanguageModel,
    cache: dict[str, PromptCache],
    layer: int,
    intervention: HiddenStateIntervention,
    references: Sequence[str],
    groups: dict[str, list[LreRow]],
    answer_token: int,
    settings: ReferenceSetConfig,
) -> ArmResult:
    tests = [row for group in groups.values() for row in group]
    raw = torch.zeros((len(tests), cache[tests[0].subject].final.numel()), device=lm.output_device)
    subtracted = torch.zeros(raw.shape[1], device=lm.output_device)
    for reference in references:
        directions = torch.stack([cache[row.subject].states[layer] for row in tests] + [cache[reference].states[layer]])
        products = lm.jvps(intervention, cache[reference], layer, directions).to(lm.output_device)
        raw += products[:-1]
        subtracted += products[-1]
    raw /= len(references)
    subtracted /= len(references)
    hidden: dict[str, Tensor] = {"cel": raw - subtracted[None], "nel": raw}
    targets = torch.tensor([row.target_id for row in tests], device=lm.output_device)
    answers = torch.full_like(targets, answer_token)
    target_ranks, answer_ranks, top_tokens = {}, {}, {}
    with torch.no_grad():
        for lens, values in hidden.items():
            logits = lm.decode(values)
            target_ranks[lens] = ranks_of(logits, targets).tolist()
            answer_ranks[lens] = ranks_of(logits, answers).tolist()
            top_tokens[lens] = logits.topk(settings.top_k).indices.tolist()
        subtracted_top = lm.decode(subtracted[None]).topk(settings.top_k).indices[0].tolist()
    scores = {}
    offset = 0
    for name, group in groups.items():
        span = range(offset, offset + len(group))
        scores[name] = GroupScores(
            n=len(group),
            median_log_rank={
                lens: float(np.median(np.log10(np.asarray([target_ranks[lens][i] for i in span]) + 1)))
                for lens in LENSES
            },
            median_answer_rank={lens: median_rank([answer_ranks[lens][i] for i in span]) for lens in LENSES},
        )
        offset += len(group)
    answer_group = groups["answer"]
    return ArmResult(
        references=tuple(references),
        subtracted_top=tuple(lm.token(t) for t in subtracted_top),
        groups=scores,
        answer_ranks={
            lens: {row.subject: answer_ranks[lens][i] for i, row in enumerate(answer_group)} for lens in LENSES
        },
        top_tokens={
            lens: {row.subject: tuple(lm.token(t) for t in top_tokens[lens][i]) for i, row in enumerate(answer_group)}
            for lens in LENSES
        },
    )
