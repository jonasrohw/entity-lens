from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from entity_lens import data, modeling, prompt_variants, storage
from entity_lens.config import ExperimentsConfig, Project
from entity_lens.data import Relation
from entity_lens.prominence import ProminenceResult
from entity_lens.records import Readout, Record, RelationMeasurement, read
from entity_lens.reference_sets import ReferenceSetResult
from entity_lens.statistics import (
    Correlation,
    Interval,
    held_out_selection_rate,
    leave_one_group_out,
    median_log_rank,
    ols_residuals,
    paired_difference_interval,
    partial_spearman,
    spearman,
    summarize_range,
)
from entity_lens.summary import FactCorrelation, RelationSummary, fact_correlation, summarize
from entity_lens.swap import SwapResult
from entity_lens.tokens import first_object_token, trailing_space

CORPUS_FEATURES = (
    "subject",
    "object",
    "joint",
    "mean_log_joint",
    "log_mean_joint",
    "forward",
    "reverse",
    "direction",
    "micro_forward",
    "micro_reverse",
)


class ModelResult(Record):
    model: str
    label: str
    newline: bool
    lens_layer: int
    relations: tuple[RelationSummary, ...]
    primary_relations: tuple[str, ...]
    scored_rows: int
    correlations: dict[Readout, Correlation]
    all_relations: Correlation
    attribute_rate: Correlation | None
    empty_attribute_rows: int | None
    cel_minus_nel: Interval


class Controls(Record):
    cardinality_faithfulness: Correlation
    cardinality_lens: Correlation
    partial_cardinality: Correlation
    prior_faithfulness: Correlation
    prior_lens: Correlation
    partial_prior: Correlation
    partial_both: Correlation
    cardinality_range: tuple[float, float]
    domain_counts: dict[str, int]
    leave_one_domain_out: dict[str, Correlation]
    leave_one_domain_out_range: tuple[float, float, float]
    residuals: dict[str, float]
    adjusted_residuals: dict[str, float]
    cardinality_residual: Correlation
    rows_per_relation: tuple[int, int]
    relations_below_ten_rows: int
    lre_best_layers: dict[int, int]


class FactLevel(Record):
    correlations: tuple[FactCorrelation, ...]
    defined_range: tuple[float, float, float]
    target_disagreements: int


class Collapse(Record):
    rates: dict[str, float]
    distinct_objects: dict[str, int]
    colliding_objects: dict[str, int]
    above_threshold: tuple[str, ...]
    weak_prefix_objects: int
    filtered_faithfulness: dict[str, float]
    filtered_rows: dict[str, int]
    weak_token_targets: int
    weak_token_agreements: int
    weak_token_answers: int
    weak_token_rows: int


class NumericObjects(Record):
    example_tokens: tuple[str, ...]
    corrected: dict[str, tuple[float, float]]
    uncorrected: dict[str, tuple[float, float]]
    uncorrected_correlation: Correlation


class Anchored(Record):
    in_sample: dict[float, Correlation]
    held_out_win_rate: float | None
    train_relations: int
    test_relations: int


class PromptVariant(Record):
    prompt: str
    correlation: Correlation


class PaperResults(Record):
    models: tuple[ModelResult, ...]
    controls: Controls
    facts: FactLevel
    corpus: dict[str, Correlation] | None
    corpus_unobserved_rows: int | None
    prompt_variants: tuple[PromptVariant, ...] | None
    prominence: ProminenceResult | None
    prominence_correlation: Correlation | None
    reference_sets: ReferenceSetResult | None
    swap: SwapResult | None
    collapse: Collapse
    numeric_objects: NumericObjects | None
    anchored: Anchored
    competition: dict[str, tuple[str, ...]]
    n_dataset_relations: int
    n_entity_relations: int


def eligible(summaries: list[RelationSummary]) -> list[RelationSummary]:
    return [s for s in summaries if not s.excluded]


def correlate(summaries: list[RelationSummary], readout: Readout) -> Correlation:
    return spearman([s.scores[readout] for s in summaries], [s.faithfulness for s in summaries])


def load_attribute_means(project: Project, model: str) -> tuple[dict[str, float], dict[str, int]] | None:
    path = project.artifacts / "attribute_rates" / f"{model}.json"
    if not path.exists():
        return None
    from entity_lens import wikipedia

    record = read(path, wikipedia.AttributeRates)
    return (
        {name: relation.mean_rate for name, relation in record.relations.items()},
        {name: relation.empty_rows for name, relation in record.relations.items()},
    )


def model_result(
    project: Project, key: str, relations: dict[str, Relation], settings: ExperimentsConfig
) -> tuple[ModelResult, dict[str, RelationMeasurement]]:
    spec = project.models.get(key)
    measurements = storage.load_measurements(project, key)
    summaries = [summarize(measurements[name], relations[name]) for name in sorted(measurements)]
    primary = eligible(summaries)
    attributes = load_attribute_means(project, key)
    attribute_correlation, empty = None, None
    if attributes is not None:
        means, empties = attributes
        attribute_correlation = spearman([means[s.relation] for s in primary], [s.faithfulness for s in primary])
        empty = sum(empties[s.relation] for s in primary)
    return (
        ModelResult(
            model=key,
            label=spec.label,
            newline=spec.newline_prefix,
            lens_layer=spec.lens_layer,
            relations=tuple(summaries),
            primary_relations=tuple(s.relation for s in primary),
            scored_rows=sum(s.n_rows for s in primary),
            correlations={readout: correlate(primary, readout) for readout in Readout},
            all_relations=correlate(summaries, Readout.CEL),
            attribute_rate=attribute_correlation,
            empty_attribute_rows=empty,
            cel_minus_nel=paired_difference_interval(
                [s.scores[Readout.CEL] for s in primary],
                [s.scores[Readout.NEL] for s in primary],
                [s.faithfulness for s in primary],
                seed=settings.bootstrap.seed,
                resamples=settings.bootstrap.resamples,
                level=settings.bootstrap.level,
            ),
        ),
        measurements,
    )


def controls(summaries: list[RelationSummary], settings: ExperimentsConfig, years: frozenset[str]) -> Controls:
    lens = [s.scores[Readout.CEL] for s in summaries]
    faith = [s.faithfulness for s in summaries]
    cardinality = [s.cardinality for s in summaries]
    prior = [s.answer_prior for s in summaries]
    domains = [s.domain for s in summaries]
    loto = leave_one_group_out(lens, faith, domains)
    residuals = ols_residuals(lens, faith)
    adjusted, cardinality_residual = adjusted_residuals(summaries, settings, years)
    return Controls(
        cardinality_faithfulness=spearman(cardinality, faith),
        cardinality_lens=spearman(cardinality, lens),
        partial_cardinality=partial_spearman(lens, faith, [cardinality]),
        prior_faithfulness=spearman(prior, faith),
        prior_lens=spearman(prior, lens),
        partial_prior=partial_spearman(lens, faith, [prior]),
        partial_both=partial_spearman(lens, faith, [cardinality, prior]),
        cardinality_range=(min(cardinality), max(cardinality)),
        domain_counts=dict(Counter(domains)),
        leave_one_domain_out=loto,
        leave_one_domain_out_range=summarize_range(loto),
        residuals={s.relation: float(r) for s, r in zip(summaries, residuals, strict=True)},
        adjusted_residuals=adjusted,
        cardinality_residual=cardinality_residual,
        rows_per_relation=(min(s.n_rows for s in summaries), max(s.n_rows for s in summaries)),
        relations_below_ten_rows=sum(s.n_rows < 10 for s in summaries),
        lre_best_layers=dict(sorted(Counter(s.best_layer for s in summaries).items())),
    )


def adjusted_residuals(
    summaries: list[RelationSummary], settings: ExperimentsConfig, years: frozenset[str]
) -> tuple[dict[str, float], Correlation]:
    config = settings.residuals
    kept = [s for s in summaries if not (config.exclude_year_relations and s.relation in years)]
    lens = [s.literal_cel_score if s.relation in config.literal_scoring else s.scores[Readout.CEL] for s in kept]
    residuals = ols_residuals(lens, [s.faithfulness for s in kept])
    adjusted = {s.relation: float(r) for s, r in zip(kept, residuals, strict=True)}
    return adjusted, spearman([s.cardinality for s in kept], list(residuals))


def fact_level(measurements: dict[str, RelationMeasurement]) -> FactLevel:
    correlations = tuple(fact_correlation(measurements[name]) for name in sorted(measurements))
    defined = {c.relation: c.correlation for c in correlations}
    return FactLevel(
        correlations=correlations,
        defined_range=summarize_range(defined),
        target_disagreements=sum(c.target_disagreements for c in correlations),
    )


def corpus_correlations(
    project: Project, relations: dict[str, Relation], summaries: list[RelationSummary]
) -> tuple[dict[str, Correlation], int] | None:
    path = project.artifacts / "corpus" / "facts.csv"
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        facts = {(r["relation"], r["subject"], r["object"]): r for r in csv.DictReader(handle)}
    features: dict[str, list[float]] = {name: [] for name in CORPUS_FEATURES}
    unobserved = 0
    for summary in summaries:
        rows = [facts[summary.relation, s.subject, s.object] for s in relations[summary.relation].samples]
        cs, co, cso = (
            np.array([float(r[key]) for r in rows]) for key in ("subject_count", "object_count", "cooccurrence")
        )
        unobserved += int(((cs == 0) | (co == 0)).sum())
        forward = np.divide(cso, cs, out=np.zeros_like(cso), where=cs > 0)
        reverse = np.divide(cso, co, out=np.zeros_like(cso), where=co > 0)
        values = {
            "subject": cs.mean(),
            "object": co.mean(),
            "joint": cso.mean(),
            "mean_log_joint": np.log1p(cso).mean(),
            "log_mean_joint": np.log1p(cso.mean()),
            "forward": forward.mean(),
            "reverse": reverse.mean(),
            "direction": (forward - reverse).mean(),
            "micro_forward": cso.sum() / cs.sum(),
            "micro_reverse": cso.sum() / co.sum(),
        }
        for name, value in values.items():
            features[name].append(float(value))
    faith = [s.faithfulness for s in summaries]
    return {name: spearman(values, faith) for name, values in features.items()}, unobserved


def identifying(relation: Relation, tokenizer: PreTrainedTokenizerBase, *, trailing: bool) -> tuple[set[str], int]:
    objects = sorted({s.object for s in relation.samples})
    first = {obj: first_object_token(tokenizer, obj, trailing_space=trailing) for obj in objects}
    counts = Counter(first.values())
    keep = {obj for obj, token in first.items() if counts[token] == 1}
    return keep, len(objects) - len(counts)


def collapse(
    project: Project,
    all_relations: list[Relation],
    measurements: dict[str, RelationMeasurement],
    settings: ExperimentsConfig,
) -> Collapse:
    spec = project.models.get(settings.primary.model)
    tokenizer = modeling.load_tokenizer(spec)
    protocol = project.protocol
    rates, distinct, colliding, keeps = {}, {}, {}, {}
    for relation in all_relations:
        trailing = trailing_space(tokenizer, relation, spec, protocol.years)
        keep, lost = identifying(relation, tokenizer, trailing=trailing)
        objects = {s.object for s in relation.samples}
        distinct[relation.key] = len(objects)
        colliding[relation.key] = len(objects) - len(keep)
        rates[relation.key] = lost / len(objects)
        keeps[relation.key] = keep
    filtered_faith, filtered_rows = {}, {}
    weak_targets = weak_agreements = 0
    for name in settings.collapse.filtered_relations:
        lre = measurements[name].lre
        rows = [row for row in lre.rows if row.object in keeps[name]]
        cells = len(lre.grid)
        scores = [sum(row.grid_predictions[c] == row.target_id for row in rows) / len(rows) for c in range(cells)]
        best = int(np.argmax(scores))
        filtered_faith[name] = scores[best]
        filtered_rows[name] = len(rows)
        if name == settings.collapse.weak_token_relation:
            weak = [row for row in rows if row.target == settings.collapse.weak_token]
            weak_targets = len(weak)
            weak_agreements = sum(row.grid_predictions[best] == row.target_id for row in weak)
    weak_prefix = sum(
        obj.split()[0] == settings.collapse.weak_token
        for obj in {
            s.object for s in next(r for r in all_relations if r.key == settings.collapse.weak_token_relation).samples
        }
    )
    return Collapse(
        rates=rates,
        distinct_objects=distinct,
        colliding_objects=colliding,
        above_threshold=tuple(sorted(name for name, rate in rates.items() if rate > settings.collapse.threshold)),
        weak_prefix_objects=weak_prefix,
        filtered_faithfulness=filtered_faith,
        filtered_rows=filtered_rows,
        weak_token_targets=weak_targets,
        weak_token_agreements=weak_agreements,
        weak_token_answers=sum(
            row.target == settings.collapse.weak_token
            for row in measurements[settings.collapse.weak_token_relation].lre.rows
        ),
        weak_token_rows=len(measurements[settings.collapse.weak_token_relation].lre.rows),
    )


def numeric_objects(
    project: Project,
    summaries: list[RelationSummary],
    measurements: dict[str, RelationMeasurement],
    settings: ExperimentsConfig,
) -> NumericObjects | None:
    names = sorted(project.protocol.years.relations)
    paths = {name: storage.numeric_object_path(project, settings.primary.model, name) for name in names}
    if not all(path.exists() for path in paths.values()):
        return None
    spec = project.models.get(settings.primary.model)
    tokenizer = modeling.load_tokenizer(spec)
    ids = tokenizer.encode(" " + settings.numeric_objects.example_object, add_special_tokens=False)
    uncorrected = {name: read(path, RelationMeasurement) for name, path in paths.items()}
    replaced = {
        s.relation: (
            (uncorrected[s.relation].lens.score(Readout.CEL), uncorrected[s.relation].lre.faithfulness)
            if s.relation in uncorrected
            else (s.scores[Readout.CEL], s.faithfulness)
        )
        for s in summaries
    }
    return NumericObjects(
        example_tokens=tuple(tokenizer.convert_ids_to_tokens(ids)),
        corrected={
            name: (measurements[name].lre.faithfulness, measurements[name].lens.score(Readout.CEL)) for name in names
        },
        uncorrected={name: (m.lre.faithfulness, m.lens.score(Readout.CEL)) for name, m in uncorrected.items()},
        uncorrected_correlation=spearman([v[0] for v in replaced.values()], [v[1] for v in replaced.values()]),
    )


def anchored(summaries: list[RelationSummary], settings: ExperimentsConfig) -> Anchored:
    faith = [s.faithfulness for s in summaries]
    betas = summaries[0].anchored_scores.keys()
    candidates = {beta: [s.anchored_scores[beta] for s in summaries] for beta in betas}
    return Anchored(
        in_sample={beta: spearman(values, faith) for beta, values in candidates.items()},
        held_out_win_rate=held_out_selection_rate(
            candidates,
            [s.scores[Readout.CEL] for s in summaries],
            faith,
            train_size=settings.anchored.train_relations,
            splits=settings.anchored.splits,
            seed=settings.anchored.seed,
            min_distinct=settings.anchored.min_distinct,
        ),
        train_relations=settings.anchored.train_relations,
        test_relations=len(summaries) - settings.anchored.train_relations,
    )


def prompt_variant_results(project: Project, primary: set[str]) -> tuple[PromptVariant, ...] | None:
    settings = project.experiments.prompt_variants
    try:
        measured = prompt_variants.load(project)
    except FileNotFoundError:
        return None
    main = {s.relation: s for s in model_summaries(project, settings.model)}
    results = []
    for prompt, relations in measured.items():
        names = [name for name in sorted(relations) if name in primary]
        scores = [-median_log_rank(relations[name].ranks) for name in names]
        results.append(
            PromptVariant(prompt=prompt, correlation=spearman(scores, [main[n].faithfulness for n in names]))
        )
    return tuple(results)


def model_summaries(project: Project, key: str) -> list[RelationSummary]:
    relations = {r.key: r for r in data.entity_relations(project.data / "relations", project.protocol.dataset)}
    measurements = storage.load_measurements(project, key)
    return [summarize(measurements[name], relations[name]) for name in sorted(measurements)]


def optional[T: BaseModel](path: Path, kind: type[T]) -> T | None:
    return read(path, kind) if path.exists() else None


def build(project: Project) -> PaperResults:
    settings = project.experiments
    protocol = project.protocol
    dataset = [
        data.load_relation(project.data / "relations", name, protocol.dataset)
        for name in data.all_relation_names(protocol.dataset)
    ]
    relations = {r.key: r for r in data.entity_relations(project.data / "relations", protocol.dataset)}
    results, primary_measurements = [], {}
    for spec in project.models.models:
        result, measurements = model_result(project, spec.key, relations, settings)
        results.append(result)
        if spec.key == settings.primary.model:
            primary_measurements = measurements
    primary = next(r for r in results if r.model == settings.primary.model)
    summaries = list(primary.relations)
    year_relations = project.protocol.years.relations
    corpus = corpus_correlations(project, relations, summaries)
    qwen = next(r for r in results if r.model == project.experiments.prompt_variants.model)
    prominence_path = project.artifacts / "prominence" / "prominence.json"
    prominence = optional(prominence_path, ProminenceResult)
    prominence_rho = (
        spearman([r.prominence for r in prominence.relations], [r.lens_hit_rate for r in prominence.relations])
        if prominence
        else None
    )
    return PaperResults(
        models=tuple(results),
        controls=controls(summaries, settings, year_relations),
        facts=fact_level(primary_measurements),
        corpus=corpus[0] if corpus else None,
        corpus_unobserved_rows=corpus[1] if corpus else None,
        prompt_variants=prompt_variant_results(project, set(qwen.primary_relations)),
        prominence=prominence,
        prominence_correlation=prominence_rho,
        reference_sets=optional(
            project.artifacts / "reference_sets" / f"{settings.primary.model}.json", ReferenceSetResult
        ),
        swap=optional(project.artifacts / "swap" / "swap.json", SwapResult),
        collapse=collapse(project, dataset, primary_measurements, settings),
        numeric_objects=numeric_objects(project, summaries, primary_measurements, settings),
        anchored=anchored(summaries, settings),
        competition={
            f"{relation}/{subject}": next(
                row.top_tokens[Readout.CEL]
                for row in primary_measurements[relation].lens.rows
                if row.subject == subject
            )
            for relation, subject in settings.competition.examples
        },
        n_dataset_relations=len(dataset),
        n_entity_relations=len(relations),
    )
