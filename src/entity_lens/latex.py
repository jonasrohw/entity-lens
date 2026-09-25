from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from entity_lens.config import Project
from entity_lens.records import Readout
from entity_lens.report import PaperResults
from entity_lens.statistics import Correlation

READOUT_ORDER = (Readout.CEL, Readout.NEL, Readout.LOGIT_LENS, Readout.DIFFERENTIAL_LOGIT_LENS)


class TableSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    caption: str
    columns: str
    header: str
    arms: tuple[str, ...] = ()
    labels: dict[str, str] = {}


def escape(text: str) -> str:
    return text.replace("_", r"\_")


def signed(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    return rf"$\phantom{{-}}{text}$" if value >= 0 else f"${text}$"


def p_value(value: float) -> str:
    if value >= 0.001:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def table(spec: TableSpec, rows: Sequence[str]) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{" + spec.caption + "}",
            r"\label{" + spec.label + "}",
            r"\begin{tabular}{" + spec.columns + "}",
            r"\toprule",
            spec.header + r" \\",
            r"\midrule",
            *[row + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def rho(correlation: Correlation | None, digits: int = 4) -> str:
    if correlation is None or correlation.rho is None:
        return "--"
    return signed(correlation.rho, digits)


def relations_table(results: PaperResults, spec: TableSpec, primary: str) -> str:
    model = next(m for m in results.models if m.model == primary)
    ordered = sorted(model.relations, key=lambda s: -s.faithfulness)
    rows = [
        f"{escape(s.relation)} & {s.cardinality:.1f} & {s.faithfulness:.2f} & ${s.scores[Readout.CEL]:.2f}$"
        for s in ordered
    ]
    return table(spec, rows)


def prominence_table(results: PaperResults, spec: TableSpec) -> str | None:
    if results.prominence is None:
        return None
    ordered = sorted(results.prominence.relations, key=lambda r: -r.prominence)
    rows = [
        f"{escape(r.relation)} & {r.prominence:.2f} & {r.prominence_rows} & {r.lens_hit_rate:.2f} & {r.lens_rows}"
        for r in ordered
    ]
    return table(spec, rows)


def reference_table(results: PaperResults, spec: TableSpec) -> str | None:
    if results.reference_sets is None:
        return None
    rows = []
    for arm in spec.arms:
        label = spec.labels[arm]
        group = results.reference_sets.arms[arm].groups["answer"]
        rows.append(
            f"{label} & {group.median_log_rank['cel']:.2f} & {group.median_log_rank['nel']:.2f} & "
            f"{group.median_answer_rank['cel']:.0f} & {group.median_answer_rank['nel']:.0f}"
        )
    return table(spec, rows)


def swap_table(results: PaperResults, spec: TableSpec) -> str | None:
    if results.swap is None:
        return None
    rows = [
        rf"\textsc{{{escape(a.relation)}}} & {a.n} & "
        + " & ".join(str(a.hits[arm]) if arm in a.hits else "--" for arm in spec.arms)
        for a in results.swap.arms
    ]
    return table(spec, rows)


def swap_anchor_table(results: PaperResults, spec: TableSpec) -> str | None:
    if results.swap is None:
        return None
    header = spec.header + "".join(f" & {escape(s.relation)}" for s in results.swap.anchors)
    rows = [
        arm + "".join(f" & {s.mean[arm]:.1f} ({s.minimum[arm]}--{s.maximum[arm]})" for s in results.swap.anchors)
        for arm in spec.arms
    ]
    return table(spec.model_copy(update={"header": header}), rows)


def readout_table(results: PaperResults, spec: TableSpec) -> str:
    rows = []
    for model in results.models:
        cells = [rho(model.correlations[readout]) for readout in READOUT_ORDER] + [rho(model.attribute_rate)]
        rows.append(model.label + ("$^{*}$" if model.newline else "") + " & " + " & ".join(cells))
    return table(spec, rows)


def corpus_table(results: PaperResults, spec: TableSpec) -> str | None:
    if results.corpus is None:
        return None
    rows = [
        f"{spec.labels[name]} & {rho(c)} & {p_value(c.p) if c.p is not None else '--'}"
        for name, c in results.corpus.items()
    ]
    return table(spec, rows)


def prompt_table(results: PaperResults, spec: TableSpec) -> str | None:
    if results.prompt_variants is None:
        return None
    rows = [
        r"\texttt{"
        + escape(v.prompt).replace("\n", r"\textbackslash n").replace("{}", r"\{s\}")
        + "} & "
        + rho(v.correlation)
        for v in results.prompt_variants
    ]
    return table(spec, rows)


def write_all(project: Project, results: PaperResults, directory: Path) -> list[Path]:
    specs = {name: TableSpec.model_validate(value) for name, value in project.load("tables").items()}
    primary = project.experiments.primary.model
    outputs = {
        "table_relations": relations_table(results, specs["relations"], primary),
        "table_prominence": prominence_table(results, specs["prominence"]),
        "table_refsets": reference_table(results, specs["reference_sets"]),
        "table_swap": swap_table(results, specs["swap"]),
        "table_swapfull": swap_table(results, specs["swap_full"]),
        "table_swapanchor": swap_anchor_table(results, specs["swap_anchor"]),
        "readout_comparison": readout_table(results, specs["readouts"]),
        "olmo_frequencies": corpus_table(results, specs["corpus"]),
        "prompt_variations": prompt_table(results, specs["prompt_variants"]),
    }
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in outputs.items():
        if text is not None:
            path = directory / f"{name}.tex"
            path.write_text(text)
            written.append(path)
    return written
