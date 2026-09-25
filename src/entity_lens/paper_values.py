from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from entity_lens.config import Project
from entity_lens.records import Readout
from entity_lens.report import PaperResults
from entity_lens.statistics import Correlation
from entity_lens.swap import SwapResult

Value = float | int | str


class PrintedValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    printed: Value
    digits: int | None = None
    subset: bool = False
    location: str


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    location: str
    printed: Value
    reproduced: Value | None
    status: str


def _correlation(values: dict[str, Value], name: str, correlation: Correlation | None) -> None:
    if correlation is None or correlation.rho is None or correlation.p is None:
        return
    values[f"{name}.rho"] = correlation.rho
    values[f"{name}.p"] = correlation.p
    values[f"{name}.n"] = correlation.n


def _model_values(values: dict[str, Value], results: PaperResults, primary: str) -> None:
    for model in results.models:
        for readout, correlation in model.correlations.items():
            _correlation(values, f"{model.model}.{readout.value}", correlation)
        _correlation(values, f"{model.model}.attribute_rate", model.attribute_rate)
        values[f"{model.model}.relations"] = len(model.primary_relations)
        values[f"{model.model}.scored_rows"] = model.scored_rows
        values[f"{model.model}.cel_minus_nel.low"] = model.cel_minus_nel.low
        values[f"{model.model}.cel_minus_nel.high"] = model.cel_minus_nel.high
        if model.empty_attribute_rows is not None:
            values[f"{model.model}.empty_attribute_rows"] = model.empty_attribute_rows
        if model.model == primary:
            for summary in model.relations:
                values[f"relation.{summary.relation}.cardinality"] = summary.cardinality
                values[f"relation.{summary.relation}.faithfulness"] = summary.faithfulness
                values[f"relation.{summary.relation}.faithfulness_swap_text"] = summary.faithfulness
                values[f"relation.{summary.relation}.lens"] = summary.scores[Readout.CEL]
                values[f"relation.{summary.relation}.rows"] = summary.n_rows
    values["models.nel_above_cel"] = sum(
        (m.correlations[Readout.NEL].rho or 0.0) > (m.correlations[Readout.CEL].rho or 0.0) for m in results.models
    )
    values["models.intervals_include_zero"] = sum(
        m.cel_minus_nel.low <= 0 <= m.cel_minus_nel.high for m in results.models
    )


def _control_values(values: dict[str, Value], results: PaperResults) -> None:
    controls = results.controls
    for name in (
        "cardinality_faithfulness",
        "cardinality_lens",
        "partial_cardinality",
        "prior_faithfulness",
        "prior_lens",
        "partial_prior",
        "partial_both",
    ):
        _correlation(values, f"controls.{name}", getattr(controls, name))
    values["controls.cardinality_min"], values["controls.cardinality_max"] = controls.cardinality_range
    low, high, median = controls.leave_one_domain_out_range
    values.update({"domains.loo_min": low, "domains.loo_max": high, "domains.loo_median": median})
    for domain, count in controls.domain_counts.items():
        values[f"domains.{domain}"] = count
    for relation, residual in controls.residuals.items():
        values[f"residual.{relation}"] = residual
    for relation, residual in controls.adjusted_residuals.items():
        values[f"residual.adjusted.{relation}"] = residual
    _correlation(values, "residual.cardinality", controls.cardinality_residual)
    ranked = sorted(controls.adjusted_residuals, key=controls.adjusted_residuals.__getitem__)
    for index, relation in enumerate(ranked[:3]):
        values[f"residual.most_negative.{index + 1}"] = relation
    for index, relation in enumerate(reversed(ranked[-3:])):
        values[f"residual.most_positive.{index + 1}"] = relation
    values["rows.min"], values["rows.max"] = controls.rows_per_relation
    values["rows.relations_below_ten"] = controls.relations_below_ten_rows
    best_layers = controls.lre_best_layers
    modal = max(best_layers, key=best_layers.__getitem__)
    values["layers.modal"] = modal
    values["layers.modal_count"] = best_layers[modal]


def _fact_values(values: dict[str, Value], results: PaperResults) -> None:
    low, high, median = results.facts.defined_range
    values.update({"facts.min": low, "facts.max": high, "facts.median": median})
    values["facts.target_disagreements"] = results.facts.target_disagreements
    for fact in results.facts.correlations:
        if fact.correlation.rho is not None:
            values[f"facts.{fact.relation}"] = fact.correlation.rho


def _corpus_values(values: dict[str, Value], results: PaperResults) -> None:
    if results.corpus is not None:
        for name, correlation in results.corpus.items():
            _correlation(values, f"corpus.{name}", correlation)
    if results.prompt_variants is not None:
        for index, variant in enumerate(results.prompt_variants):
            _correlation(values, f"prompt.{index}", variant.correlation)


def _prominence_values(values: dict[str, Value], results: PaperResults) -> None:
    if results.prominence is not None and results.prominence_correlation is not None:
        _correlation(values, "prominence", results.prominence_correlation)
        for row in results.prominence.relations:
            values[f"prominence.{row.relation}.rate"] = row.prominence
            values[f"prominence.{row.relation}.rows"] = row.prominence_rows
            values[f"prominence.{row.relation}.lens"] = row.lens_hit_rate
            values[f"prominence.{row.relation}.lens_rows"] = row.lens_rows


def _reference_values(values: dict[str, Value], results: PaperResults) -> None:
    if results.reference_sets is not None:
        reference = results.reference_sets
        values.update(
            {
                "refsets.rows": reference.n_rows,
                "refsets.answer_gold": reference.n_answer_gold,
                "refsets.answer_target": reference.n_answer_target,
                "refsets.answer_test": reference.n_answer_test,
                "refsets.monotone": reference.monotone_subjects,
                "refsets.other_test": reference.arms["answer"].groups["other"].n,
            }
        )
        for key, rank in reference.example.items():
            values[f"refsets.example.{key}"] = rank
        values["refsets.example.top"] = ", ".join(reference.example_top[:6])
        for arm, result in reference.arms.items():
            for group, scores in result.groups.items():
                for lens in ("cel", "nel"):
                    values[f"refsets.{arm}.{group}.{lens}.log_rank"] = scores.median_log_rank[lens]
                    values[f"refsets.{arm}.{group}.{lens}.answer_rank"] = scores.median_answer_rank[lens]
            values[f"refsets.{arm}.subtracted_top"] = ", ".join(result.subtracted_top)


def _swap_arm_values(values: dict[str, Value], swap: SwapResult) -> None:
    for arm in swap.arms:
        for name, hits in arm.hits.items():
            values[f"swap.{arm.relation}.{name}"] = hits
            values[f"swap.{arm.relation}.{name}.given_own"] = arm.conditional_on_own[name]
        values[f"swap.{arm.relation}.norm_ratio"] = arm.norm_ratio
        if arm.benchmark_own_hits is not None and arm.added_own_hits is not None:
            values[f"swap.{arm.relation}.benchmark_own"] = arm.benchmark_own_hits
            values[f"swap.{arm.relation}.added_own"] = arm.added_own_hits


def _swap_anchor_values(values: dict[str, Value], swap: SwapResult) -> None:
    for summary in swap.anchors:
        for arm, mean in summary.mean.items():
            values[f"swap.anchor.{summary.relation}.{arm}.mean"] = mean
            values[f"swap.anchor.{summary.relation}.{arm}.min"] = summary.minimum[arm]
            values[f"swap.anchor.{summary.relation}.{arm}.max"] = summary.maximum[arm]
    for relation, shares in swap.variance_shares.items():
        for source, share in shares.items():
            if share is not None:
                values[f"swap.variance.{relation}.{source}"] = share


def _swap_values(values: dict[str, Value], results: PaperResults) -> None:
    swap = results.swap
    if swap is None:
        return
    values["swap.countries"] = len(swap.countries)
    values["swap.athletes"] = len(swap.athletes)
    values["swap.candidates"] = len(swap.candidates)
    values["swap.added_passed"] = sum(row.passed and not row.benchmark for row in swap.filter)
    values["swap.shared_father_mother"] = swap.shared_subjects
    if swap.composition_spearman is not None:
        values["swap.composition_spearman"] = swap.composition_spearman
    _swap_arm_values(values, swap)
    _swap_anchor_values(values, swap)
    for beta, hits in swap.currency_own_hits.items():
        values[f"swap.currency_own.beta_{beta:g}"] = hits
    gaps = [summary.mean["own"] - summary.mean["foreign"] for summary in swap.anchors]
    values["swap.anchor.max_own_foreign_gap"] = max(gaps)
    for pool, size in swap.anchor_pool_sizes.items():
        values[f"swap.pool.{pool}"] = size


def _collapse_values(values: dict[str, Value], results: PaperResults) -> None:
    collapse = results.collapse
    values["collapse.above_threshold"] = len(collapse.above_threshold)
    values["collapse.relations"] = len(collapse.rates)
    for relation, rate in collapse.rates.items():
        values[f"collapse.{relation}"] = rate
        values[f"collapse.{relation}.objects"] = collapse.distinct_objects[relation]
    values["collapse.weak_prefix_objects"] = collapse.weak_prefix_objects
    values["collapse.weak_token_targets"] = collapse.weak_token_targets
    values["collapse.weak_token_agreements"] = collapse.weak_token_agreements
    values["collapse.weak_token_answers"] = collapse.weak_token_answers
    values["collapse.weak_token_rows"] = collapse.weak_token_rows
    for relation, faithfulness in collapse.filtered_faithfulness.items():
        values[f"collapse.filtered.{relation}"] = faithfulness
        values[f"collapse.filtered.{relation}.rows"] = collapse.filtered_rows[relation]


def _numeric_values(values: dict[str, Value], results: PaperResults) -> None:
    if results.numeric_objects is not None:
        numeric = results.numeric_objects
        values["numeric.tokens"] = " ".join(numeric.example_tokens)
        for relation, (faithfulness, lens) in numeric.corrected.items():
            values[f"numeric.{relation}.faithfulness"] = faithfulness
            values[f"numeric.{relation}.lens"] = lens
        for relation, (faithfulness, lens) in numeric.uncorrected.items():
            values[f"numeric.{relation}.uncorrected_faithfulness"] = faithfulness
            values[f"numeric.{relation}.uncorrected_lens"] = lens
        _correlation(values, "numeric.uncorrected", numeric.uncorrected_correlation)


def _misc_values(values: dict[str, Value], results: PaperResults) -> None:
    if results.anchored.held_out_win_rate is not None:
        values["anchored.held_out_win_rate"] = results.anchored.held_out_win_rate
    values["anchored.train"] = results.anchored.train_relations
    values["anchored.test"] = results.anchored.test_relations
    for key, tokens in results.competition.items():
        values[f"competition.{key}"] = ", ".join(tokens)
        values[f"competition.{key}.first"] = tokens[0]
    values["dataset.relations"] = results.n_dataset_relations
    values["dataset.entity_relations"] = results.n_entity_relations
    values["swap.currency_n"] = results.swap.currency_n if results.swap is not None else 0
    values["dataset.excluded_relations"] = results.n_dataset_relations - results.n_entity_relations


def flatten(results: PaperResults, primary: str) -> dict[str, Value]:
    values: dict[str, Value] = {}
    _model_values(values, results, primary)
    _control_values(values, results)
    _fact_values(values, results)
    _corpus_values(values, results)
    _prominence_values(values, results)
    _reference_values(values, results)
    _swap_values(values, results)
    _collapse_values(values, results)
    _numeric_values(values, results)
    _misc_values(values, results)
    return values


def compare(value: Value | None, printed: PrintedValue) -> str:
    if value is None:
        return "missing"
    if printed.subset:
        return "match" if set(str(printed.printed).split(", ")) <= set(str(value).split(", ")) else "differs"
    if isinstance(printed.printed, str) or isinstance(value, str):
        return "match" if str(value) == str(printed.printed) else "differs"
    if printed.digits is None:
        return "match" if float(value) == float(printed.printed) else "differs"
    return (
        "match" if round(float(value), printed.digits) == round(float(printed.printed), printed.digits) else "differs"
    )


def comparisons(project: Project, results: PaperResults) -> list[Comparison]:
    values = flatten(results, project.experiments.primary.model)
    expected = {name: PrintedValue.model_validate(entry) for name, entry in project.load("paper_values").items()}
    return [
        Comparison(
            name=name,
            location=entry.location,
            printed=entry.printed,
            reproduced=values.get(name),
            status=compare(values.get(name), entry),
        )
        for name, entry in expected.items()
    ]


def markdown(rows: list[Comparison]) -> str:
    def show(value: Value | None) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", "\\n")

    matched = sum(row.status == "match" for row in rows)
    lines = [
        f"{matched} of {len(rows)} printed values match the reproduction.",
        "",
        "| Value | Location | Paper | Reproduced | Status |",
        "|---|---|---:|---:|---|",
    ]
    lines += [f"| `{r.name}` | {r.location} | {show(r.printed)} | {show(r.reproduced)} | {r.status} |" for r in rows]
    return "\n".join(lines) + "\n"


def write(project: Project, results: PaperResults, directory: Path) -> tuple[Path, Path]:
    rows = comparisons(project, results)
    directory.mkdir(parents=True, exist_ok=True)
    values_path = directory / "values.json"
    values = flatten(results, project.experiments.primary.model)
    values_path.write_text(json.dumps(values, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    report_path = directory / "comparison.md"
    report_path.write_text(markdown(rows))
    return values_path, report_path
