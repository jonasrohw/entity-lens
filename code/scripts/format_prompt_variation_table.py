"""Validate saved prompt experiments and typeset their two-panel appendix table."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from format_shared_baseline_tables import MODELS, OUT, ROOT, coefficient
from scipy.stats import spearmanr

CODE = ROOT / "code"
QWEN = CODE / "results/qwen_prompt_sensitivity"
NEWLINE = CODE / "results/all_model_newline"


def main():
    hashes = {}

    def read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return json.loads(path.read_text())

    def csv_read(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        with path.open() as stream:
            return list(csv.DictReader(stream))

    prompt_csv = csv_read(QWEN / "correlations.csv")
    newline_csv = csv_read(NEWLINE / "correlations.csv")
    metadata = read(QWEN / "worker_0.json")
    prompts = metadata["prompts"]
    assert len(prompts) == 7
    results = []

    def score(lens):
        ranks = [r["rank_diff"] for r in lens["rows"]]
        value = -float(np.median(np.log10(np.array(ranks) + 1)))
        assert abs(value + lens["agg"]["diff"]["median_lograng"]) < 1e-12
        return value

    def identities(lens):
        return [
            [r[k] for k in ("subject", "model_top1_id", "n_references")]
            for r in lens["rows"]
        ]

    newline_records = {}
    for model, _ in MODELS:
        for protocol in (
            ["main", "shared_split"] if model == "qwen3_8b_base" else ["shared_split"]
        ):
            paths = sorted((NEWLINE / model / protocol).glob("*.json"))
            assert len(paths) == (28 if protocol == "main" else 25)
            values = [read(p) for p in paths]
            newline_records[model, protocol] = {
                p.stem: v for p, v in zip(paths, values, strict=True)
            }
            for value in values:
                worker = read(NEWLINE / model / f"worker_{value['worker']}.json")
                assert worker["dtype"] == "float32"
                assert worker["lens_layer"] == value["lens"]["layer"]
                assert (
                    worker["prompt"]
                    == value["lens"]["neutral_prompt"]
                    == "\n{} is known for"
                )
                baseline = value["baseline_lens"]
                if "model_top1_id" in baseline["rows"][0]:
                    assert identities(value["lens"]) == identities(baseline)
                else:
                    # The legacy OLMo/GPT-J files retain decoded targets only.
                    # Match them and independently verify current IDs against
                    # the retained ordinary-forward target audit.
                    keys = ("subject", "model_top1")
                    assert [[r[k] for k in keys] for r in value["lens"]["rows"]] == [
                        [r[k] for k in keys] for r in baseline["rows"]
                    ]
                    audit = read(
                        CODE
                        / "results/cross_model_audit/raw/targets"
                        / model
                        / protocol
                        / f"{value['relation']}.json"
                    )
                    assert audit["subjects"] == [
                        r["subject"] for r in value["lens"]["rows"]
                    ]
                    assert audit["targets"] == [
                        r["model_top1_id"] for r in value["lens"]["rows"]
                    ]
                    assert audit["split_fingerprint"] == value["split_fingerprint"]
                for relative, expected in value["baseline_sha256"].items():
                    path = CODE / relative
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
                    hashes[str(path.relative_to(ROOT))] = expected
            y = [v["faithfulness"]["best_faith"] for v in values]
            for version, key in [("original", "baseline_lens"), ("newline", "lens")]:
                rho = float(spearmanr([score(v[key]) for v in values], y).statistic)
                saved = [
                    r
                    for r in newline_csv
                    if r["model"] == model
                    and r["protocol"] == protocol
                    and r["version"] == version
                    and r["kind"] == "diff"
                    and r["cohort"] == "all"
                ]
                assert (
                    len(saved) == 1
                    and abs(rho - float(saved[0]["spearman_rho"])) < 1e-12
                )
                results.append(
                    dict(
                        model=model,
                        protocol=protocol,
                        prompt=version,
                        template="{} is known for"
                        if version == "original"
                        else "\n{} is known for",
                        n=len(values),
                        rho=rho,
                        source="all_model_newline",
                    )
                )

    for protocol in ("main", "shared_split"):
        paired = newline_records["qwen3_8b_base", protocol]
        for prompt, template in prompts.items():
            paths = sorted((QWEN / protocol / prompt).glob("*.json"))
            assert {p.stem for p in paths} == set(paired)
            x, y = [], []
            for path in paths:
                value, reference = read(path), paired[path.stem]
                worker = read(QWEN / f"worker_{value['worker']}.json")
                current_worker = read(
                    NEWLINE / "qwen3_8b_base" / f"worker_{reference['worker']}.json"
                )
                assert (
                    worker["revision"]
                    == metadata["revision"]
                    == current_worker["revision"]
                )
                assert (
                    worker["layer"]
                    == value["lens"]["layer"]
                    == reference["lens"]["layer"]
                    == 19
                )
                assert worker["dtype"] == "float32"
                assert value["lens"]["neutral_prompt"] == template
                assert value["faithfulness"] == reference["faithfulness"]["best_faith"]
                assert value["split_fingerprint"] == reference["split_fingerprint"]
                assert identities(value["lens"]) == identities(reference["lens"])
                assert value["data_sha256"] == reference["data_sha256"]
                data_path = CODE / "data/relations" / path.name
                assert (
                    hashlib.sha256(data_path.read_bytes()).hexdigest()
                    == value["data_sha256"]
                )
                hashes[str(data_path.relative_to(ROOT))] = value["data_sha256"]
                assert (
                    hashlib.sha256(
                        (CODE / value["retained_source"]).read_bytes()
                    ).hexdigest()
                    == value["retained_sha256"]
                )
                # The historical launcher later gained the newline variant;
                # validate the unchanged measurement modules, not that launcher.
                for relative, expected in worker["source_sha256"].items():
                    if relative.startswith("jlens/"):
                        source = CODE / relative
                        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                            # Validate the exact historical measurement source;
                            # this does not certify equivalence to today's code.
                            archive = read(
                                CODE
                                / "results/shared_baselines/provenance/executed_sources.json"
                            )[relative]
                            assert archive["executed_sha256"] == expected
                            source = CODE / archive["snapshot"]
                        assert (
                            hashlib.sha256(source.read_bytes()).hexdigest() == expected
                        )
                        hashes[str(source.relative_to(ROOT))] = expected
                x.append(score(value["lens"]))
                y.append(value["faithfulness"])
            rho = float(spearmanr(x, y).statistic)
            saved = [
                r
                for r in prompt_csv
                if r["protocol"] == protocol
                and r["prompt"] == prompt
                and r["cohort"] == "all"
            ]
            assert len(saved) == 1 and abs(rho - float(saved[0]["rho"])) < 1e-12
            results.append(
                dict(
                    model="qwen3_8b_base",
                    protocol=protocol,
                    prompt=prompt,
                    template=template,
                    n=len(paths),
                    rho=rho,
                    source="qwen_prompt_sensitivity",
                )
            )

    def lookup(model, protocol, prompt):
        return next(
            r["rho"]
            for r in results
            if (r["model"], r["protocol"], r["prompt"]) == (model, protocol, prompt)
        )

    tex = [
        r"% Generated by code/scripts/format_prompt_variation_table.py.",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Neutral-prompt variations.} Spearman $\rho$ between differential-Jacobian scores and retained LRE faithfulness. Panel A compares exact templates for Qwen3 8B Base; panel B isolates the newline-prefix comparison across all seven models on the shared split.}",
        r"\label{tab:shared-prompt-variations}",
        r"\begingroup",
        r"\small",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule",
        r"\multicolumn{3}{@{}l}{\textbf{A. Qwen3 8B Base: prompt wording and prefix}} \\",
        r"\addlinespace[3pt]",
        r"Neutral prompt & Shared ($n=25$) & Legacy ($n=28$) \\",
        r"\midrule",
    ]
    for prompt, template in list(prompts.items()) + [("newline", "\n{} is known for")]:
        if prompt == "newline":
            tex.append(r"\addlinespace[4pt]")
        label = template.replace("\n", r"\textbackslash n").replace("{}", r"\{s\}")
        tex.append(
            r"\texttt{"
            + label
            + "} & "
            + coefficient(lookup("qwen3_8b_base", "shared_split", prompt))
            + " & "
            + coefficient(lookup("qwen3_8b_base", "main", prompt))
            + r" \\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\par\vspace{10pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}",
        r"\toprule",
        r"\multicolumn{4}{@{}l}{\textbf{B. All models: newline prefix, shared split ($n=25$)}} \\",
        r"\addlinespace[3pt]",
        r"Model & Original & Newline & $\Delta\rho$ \\",
        r"\midrule",
    ]
    for model, label in MODELS:
        original, newline = [
            lookup(model, "shared_split", p) for p in ("original", "newline")
        ]
        label = label.replace(r"$^{\dagger}$", "")
        tex.append(
            label
            + " & "
            + coefficient(original)
            + " & "
            + coefficient(newline)
            + f" & ${newline - original:+.3f}$"
            + r" \\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\par\vspace{5pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"$s$ denotes the subject; \texttt{\textbackslash n} denotes one leading newline. Panel B compares \texttt{\{s\} is known for} with \texttt{\textbackslash n\{s\} is known for}. The newline is a position control measured after the seven-template Qwen comparison; no additional newline is implied for the other templates.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    output = OUT / "prompt_variations.tex"
    output.write_text("\n".join(tex))
    with (OUT / "prompt_variations.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (OUT / "prompt_variations_manifest.json").write_text(
        json.dumps(
            dict(
                displayed_prompt_variants=8,
                displayed_models=7,
                recomputed_correlations=len(results),
                model_revision=metadata["revision"],
                qwen_layer=19,
                source_sha256=hashes,
                generator_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                validation="Subject/target/reference-count identities, splits, fixed outcomes, model revisions, input data and recomputed rank aggregates checked. The historical Qwen launcher hash is retained in worker metadata; its later newline addition does not alter these experiments.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"Validated {len(results)} correlations; generated eight-prompt and seven-model panels."
    )


if __name__ == "__main__":
    main()
