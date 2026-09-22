"""Report complete, paired neutral-prompt correlations without loading a model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_rho(x, y, indices):
    a = rankdata(x[indices], axis=1)
    b = rankdata(y[indices], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.divide(
        (a * b).sum(1), denom, out=np.full(len(a), np.nan), where=denom > 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "results/qwen_prompt_sensitivity"
    )
    args = parser.parse_args()
    root = args.root
    metadata = json.loads((root / "worker_0.json").read_text())
    workers = {
        p.stem: json.loads(p.read_text()) for p in sorted(root.glob("worker_*.json"))
    }
    prompts = metadata["prompts"]
    retained = ROOT / "results/five_model_sweep/qwen3_8b_base"
    rows, relation_rows, audit = [], [], {}
    for protocol in ("main", "shared_split"):
        originals = {
            p.stem: json.loads(p.read_text())
            for p in (retained / protocol).glob("*.json")
        }
        names = sorted(k for k, v in originals.items() if "agg" in v.get("lens", {}))
        records = {}
        for prompt in prompts:
            files = {p.stem: p for p in (root / protocol / prompt).glob("*.json")}
            if set(files) != set(names):
                raise RuntimeError(
                    f"Incomplete {protocol}/{prompt}: {len(files)}/{len(names)}"
                )
            records[prompt] = [json.loads(files[n].read_text()) for n in names]
        baseline = records["known_for"]
        for prompt, values in records.items():
            for name, value, base in zip(names, values, baseline):
                assert value["faithfulness"] == base["faithfulness"]
                assert value["split_fingerprint"] == base["split_fingerprint"]
                assert value["retained_sha256"] == base["retained_sha256"]
                assert value["data_sha256"] == base["data_sha256"]
                worker = workers[f"worker_{value['worker']}"]
                assert worker["prompts"] == prompts
                assert worker["revision"] == metadata["revision"]
                assert worker["layer"] == metadata["layer"]
                assert worker["source_sha256"] == metadata["source_sha256"]
                relation_rows.append(
                    dict(
                        protocol=protocol,
                        prompt=prompt,
                        relation=name,
                        score=-value["lens"]["agg"]["diff"]["median_lograng"],
                        faithfulness=value["faithfulness"],
                        n_lens=value["lens"]["n"],
                        n_lre=value["n_lre"],
                    )
                )
        differences = [d for v in baseline for d in v["baseline_rank_differences"]]
        audit[protocol] = dict(
            n_relations=len(names),
            n_subject_rows=len(differences),
            ranks_changed=sum(d != 0 for d in differences),
            max_absolute_rank_difference=max(map(abs, differences)),
            retained_rho=float(
                spearmanr(
                    [
                        -originals[n]["lens"]["agg"]["diff"]["median_lograng"]
                        for n in names
                    ],
                    [originals[n]["faithfulness"]["best_faith"] for n in names],
                ).statistic
            ),
        )
        for cohort in ("all", "at_least_4_lre_tests"):
            selected = [
                i for i, b in enumerate(baseline) if cohort == "all" or b["n_lre"] >= 4
            ]
            y = np.array([baseline[i]["faithfulness"] for i in selected])
            base_x = np.array(
                [
                    -baseline[i]["lens"]["agg"]["diff"]["median_lograng"]
                    for i in selected
                ]
            )
            indices = np.random.default_rng(20260921).integers(
                0, len(y), size=(10000, len(y))
            )
            base_boot = bootstrap_rho(base_x, y, indices)
            base_rho = float(spearmanr(base_x, y).statistic)
            for prompt, values in records.items():
                x = np.array(
                    [
                        -values[i]["lens"]["agg"]["diff"]["median_lograng"]
                        for i in selected
                    ]
                )
                rho, p = spearmanr(x, y)
                r, pp = pearsonr(x, y)
                boot = bootstrap_rho(x, y, indices)
                delta = boot - base_boot
                low, high = np.nanpercentile(boot, [2.5, 97.5])
                dl, dh = np.nanpercentile(delta, [2.5, 97.5])
                rows.append(
                    dict(
                        protocol=protocol,
                        cohort=cohort,
                        prompt=prompt,
                        template=prompts[prompt],
                        n_relations=len(y),
                        rho=float(rho),
                        p=float(p),
                        pearson_r=float(r),
                        pearson_p=float(pp),
                        delta_rho=float(rho - base_rho),
                        rho_ci_low=float(low),
                        rho_ci_high=float(high),
                        delta_ci_low=float(dl),
                        delta_ci_high=float(dh),
                        n_valid_bootstraps=int(np.isfinite(delta).sum()),
                        score_rho_vs_baseline=float(spearmanr(x, base_x).statistic),
                    )
                )
    for name, values in [("correlations", rows), ("relation_scores", relation_rows)]:
        with (root / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (root / "summary.json").write_text(
        json.dumps(
            dict(
                metadata=metadata,
                workers=workers,
                bootstrap=dict(
                    draws=10000, seed=20260921, unit="relation", paired=True
                ),
                baseline_reproduction=audit,
                correlations=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    main_rows = [r for r in rows if r["protocol"] == "main" and r["cohort"] == "all"]
    main_base = next(r for r in main_rows if r["prompt"] == "known_for")
    main_high = max(main_rows, key=lambda r: r["rho"])
    main_low = min(main_rows, key=lambda r: r["rho"])
    shared_rows = [
        r for r in rows if r["protocol"] == "shared_split" and r["cohort"] == "all"
    ]
    shared_base = next(r for r in shared_rows if r["prompt"] == "known_for")
    shared_high = max(shared_rows, key=lambda r: r["rho"])
    lines = [
        "# Qwen3-8B-Base neutral-prompt sensitivity",
        "",
        f"The measured correlation depends on the neutral prompt. Across all 28 main-protocol "
        f"relations, the original gives Spearman ρ={main_base['rho']:.4f}; the tested prompts "
        f"range from {main_low['rho']:.4f} to {main_high['rho']:.4f}. "
        f"The highest observed value is for `{main_high['template']}`. "
        f"Under shared_split (25 relations), the original gives ρ={shared_base['rho']:.4f}, "
        f"rising to {shared_high['rho']:.4f} for `{shared_high['template']}`. "
        "This is an exploratory sensitivity result, not a newly validated default.",
        "",
        "Six alternatives and the original prompt were fixed before running this comparison. "
        "All variants use the same checkpoint revision, float32/eager attention, layer 19, "
        "subject/reference splits, eight references, relation-prompt target tokens, and retained "
        "LRE faithfulness. Only the neutral lens prompt changes; no layer or beta is retuned. "
        "The original prompt is freshly rerun as a numerical control.",
        "",
        "Score: negative median log10(differential target rank + 1), correlated across relations "
        "with retained LRE model-top-1 faithfulness. The main protocol uses all 28 relations; "
        "shared_split uses its existing eligible cohort. Prefix variants also change the subject "
        "representation; suffix-only variants change the continuation/Jacobian context. "
        "Bare subject removes the continuation entirely. None supplies relation-specific information.",
        "",
    ]
    for protocol in ("main", "shared_split"):
        lines += [
            f"## {protocol}",
            "",
            "| Neutral prompt | Relations | Spearman ρ | Δρ | Paired 95% CI for Δρ | Pearson r |",
            "| --- | ---: | ---: | ---: | --- | ---: |",
        ]
        for row in rows:
            if row["protocol"] == protocol and row["cohort"] == "all":
                lines.append(
                    f"| `{row['template']}` | {row['n_relations']} | {row['rho']:.4f} | "
                    f"{row['delta_rho']:+.4f} | [{row['delta_ci_low']:+.4f}, {row['delta_ci_high']:+.4f}] | "
                    f"{row['pearson_r']:.4f} |"
                )
        lines += ["", f"Baseline reproduction: `{json.dumps(audit[protocol])}`.", ""]
    lines += [
        "## Existing small-relation sensitivity",
        "",
        "This secondary analysis excludes relations with fewer than four LRE tests, "
        "using the rule from the earlier audit. It changes the cohort and does not replace "
        "the all-relation comparison above.",
        "",
        "| Prompt | Main ρ | Shared ρ |",
        "| --- | ---: | ---: |",
    ]
    for prompt in prompts:
        values = [
            next(
                r
                for r in rows
                if r["prompt"] == prompt
                and r["protocol"] == protocol
                and r["cohort"] == "at_least_4_lre_tests"
            )
            for protocol in ("main", "shared_split")
        ]
        lines.append(
            f"| `{prompts[prompt]}` | {values[0]['rho']:.4f} (n={values[0]['n_relations']}) | "
            f"{values[1]['rho']:.4f} (n={values[1]['n_relations']}) |"
        )
    lines += [
        "",
        "![Prompt sensitivity](correlations.png)",
        "",
        "## Interpretation and limits",
        "",
        "All three prefixed prompts improve the observed correlation in both protocols. "
        "The `Entity:` control keeps the words `is known for`, so the pattern suggests that "
        "subject context/position contributes to the issue. This experiment does not isolate "
        "positional effects from the prefix's semantics. Bare subject and `{} is` do not improve "
        "on the original in either protocol. The strong shared-split results and the more modest "
        "all-28 main results should be reported together.",
        "",
        "Intervals are paired percentile bootstraps over relations (10,000 draws, seed 20260921). "
        "They describe relation-sampling uncertainty, not new subject splits or LRE re-estimation. "
        "The prompts were specified before this run, but the investigation follows observed low "
        "Qwen correlation; selecting the best prompt remains exploratory. CSV p-values and "
        "intervals are not adjusted for multiple comparisons. Relations share some subjects "
        "and semantic domains, so the relation bootstrap is an approximate uncertainty estimate. "
        "The two protocols are sensitivity checks, not independent replications. "
        "Retained LRE scores still use in-sample layer/beta selection and the original first-token targets.",
        "",
        "## Reproduction",
        "",
        "From `code/`, run one or more workers on separate GPUs:",
        "",
        "```sh",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_qwen_prompt_sensitivity.py --worker 0",
        ".venv/bin/python scripts/report_qwen_prompt_sensitivity.py",
        "```",
        "",
        "Workers claim independent prompt/relation files and resume completed outputs. "
        "Each file retains subject ranks, target IDs, prompt, fixed faithfulness, split fingerprint, "
        "source/data hashes, and baseline rank differences. Worker files retain checkpoint, "
        "code hashes, runtime environment, and derivative checks. The report refuses incomplete cohorts.",
        "Main measurements use derivative batches of two on A100 GPUs. The larger shared cohorts "
        "use batches of eight on A100/A6000 GPUs after the initial workers are replaced; any "
        "already-complete files are retained. Each result identifies its worker and exact "
        "environment. Every worker repeats the scalar-product and finite-difference verification.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, sharex=True)
    all_rhos = [r["rho"] for r in rows if r["cohort"] == "all"]
    for ax, protocol in zip(axes, ("main", "shared_split")):
        values = [r for r in rows if r["protocol"] == protocol and r["cohort"] == "all"]
        x = np.array([r["rho"] for r in values])
        ax.barh(
            range(len(values)), x, color=["#636363"] + ["#3978a8"] * (len(values) - 1)
        )
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(range(len(values)), [r["template"] for r in values])
        ax.set_title(f"{protocol} (n={values[0]['n_relations']})")
        ax.set_xlabel("Spearman ρ with fixed LRE faithfulness")
        ax.set_xlim(min(-0.05, min(all_rhos) - 0.08), max(0.6, max(all_rhos) + 0.10))
        for i, value in enumerate(x):
            ax.text(value + 0.008, i, f"{value:.3f}", va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(root / "correlations.png", dpi=180)
    fig.savefig(root / "correlations.pdf")
    print(json.dumps(dict(baseline_reproduction=audit, correlations=rows), indent=2))


if __name__ == "__main__":
    main()
