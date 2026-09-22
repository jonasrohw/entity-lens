"""Validate all newline reruns and report Spearman/Pearson for every readout."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_cross_model_audit import LABELS, records
from report_qwen_prompt_sensitivity import bootstrap_rho
from scipy.stats import pearsonr, spearmanr

from jlens import affine_beta

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(lens):
    values = {k: -lens["agg"][k]["median_lograng"] for k in ("diff", "lre")}
    if "beta_sweep" in lens:
        grid = lens["beta_sweep"]
        values.update(
            {
                f"affine_beta_{beta}": -grid["agg"][str(beta)]["median_lograng"]
                for beta in grid["betas"]
            }
        )
        values.update(
            {
                f"affine_tuned_{metric}": affine_beta.tuned_score(grid, metric)[1]
                for metric in affine_beta.SELECTION_METRICS
            }
        )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/all_model_newline")
    parser.add_argument("--models", default=",".join(LABELS))
    args = parser.parse_args()
    root, models = args.root.resolve(), args.models.split(",")
    retained = records()
    audits, relation_rows, correlations, workers = {}, [], [], {}
    file_hashes = {}

    def checked_hash(path, expected):
        path = str(path)
        if path not in file_hashes:
            file_hashes[path] = sha(Path(path))
        assert file_hashes[path] == expected, path

    for name in models:
        workers[name] = {
            p.stem.removeprefix("worker_"): json.loads(p.read_text())
            for p in (root / name).glob("worker_*.json")
        }
        audits[name] = {}
        for protocol in ("main", "shared_split"):
            originals = retained[name][protocol]
            paths = {p.stem: p for p in (root / name / protocol).glob("*.json")}
            if set(paths) != set(originals):
                raise RuntimeError(
                    f"Incomplete {name}/{protocol}: {len(paths)}/{len(originals)}; missing {sorted(set(originals) - set(paths))}"
                )
            rows = []
            changes = []
            all_positions = []
            for relation in sorted(originals):
                path = paths[relation]
                current = json.loads(path.read_text())
                old = originals[relation]
                worker = workers[name][current["worker"]]
                if "revision" in old:
                    assert worker["revision"] == old["revision"]
                    assert worker["model_id"] == old["model_id"]
                else:
                    legacy_metadata = json.loads(
                        (
                            ROOT
                            / f"results/cross_model_audit/raw/targets/{name}/metadata.json"
                        ).read_text()
                    )
                    assert worker["revision"] == legacy_metadata["revision"]
                    assert worker["model_id"] == legacy_metadata["model"]
                assert (
                    worker["prompt"]
                    == current["lens"]["neutral_prompt"]
                    == "\n{} is known for"
                )
                assert (
                    worker["lens_layer"]
                    == current["lens"]["layer"]
                    == old["lens"]["layer"]
                )
                for filename, expected in worker["source_sha256"].items():
                    checked_hash(ROOT / filename, expected)
                for filename, expected in current["baseline_sha256"].items():
                    checked_hash(ROOT / filename, expected)
                checked_hash(
                    ROOT / "data/relations" / path.name, current["data_sha256"]
                )
                assert current["baseline_lens"] == old["lens"]
                assert current["faithfulness"] == old["faithfulness"]
                assert current["n_lre"] == old["n_test"]
                assert (
                    current["lens"]["split_fingerprint"]
                    == old["lens"]["split_fingerprint"]
                )
                keys = ["subject", "model_top1"]
                if "model_top1_id" in old["lens"]["rows"][0]:
                    keys += ["model_top1_id", "n_references"]
                assert [[r[k] for k in keys] for r in current["lens"]["rows"]] == [
                    [r[k] for k in keys] for r in old["lens"]["rows"]
                ]
                for kind in ("diff", "lre"):
                    ranks = [r[f"rank_{kind}"] for r in current["lens"]["rows"]]
                    assert (
                        abs(
                            np.median(np.log10(np.array(ranks) + 1))
                            - current["lens"]["agg"][kind]["median_lograng"]
                        )
                        < 1e-12
                    )
                for beta in current["lens"]["beta_sweep"]["betas"]:
                    ranks = [
                        r["rank_lre_beta"][str(beta)] for r in current["lens"]["rows"]
                    ]
                    assert (
                        abs(
                            np.median(np.log10(np.array(ranks) + 1))
                            - current["lens"]["beta_sweep"]["agg"][str(beta)][
                                "median_lograng"
                            ]
                        )
                        < 1e-12
                    )
                rows.append(current)
                all_positions.extend(current["positions"].values())
                changes.extend(
                    (relation, c["subject"])
                    for c in current["content_tokenization_changes"]
                )
                for version, lens in (
                    ("original", old["lens"]),
                    ("newline", current["lens"]),
                ):
                    for kind, score in scores(lens).items():
                        relation_rows.append(
                            dict(
                                model=name,
                                protocol=protocol,
                                relation=relation,
                                version=version,
                                kind=kind,
                                score=score,
                                faithfulness=current["faithfulness"]["best_faith"],
                                n_lens=lens["n"],
                                n_lre=current["n_lre"],
                                source=str(path.relative_to(ROOT)),
                                source_sha256=sha(path),
                            )
                        )
            assert min(all_positions) > 0
            audits[name][protocol] = dict(
                n_relations=len(rows),
                n_subject_rows=sum(r["lens"]["n"] for r in rows),
                minimum_subject_position=min(all_positions),
                tokenization_changed_relation_subjects=changes,
                n_checked_relation_subjects=len(all_positions),
            )
            for cohort in ("all", "at_least_4_lre_tests"):
                values = [
                    r
                    for r in relation_rows
                    if r["model"] == name
                    and r["protocol"] == protocol
                    and (cohort == "all" or r["n_lre"] >= 4)
                ]
                for version in ("original", "newline"):
                    kinds = sorted(
                        {r["kind"] for r in values if r["version"] == version}
                    )
                    for kind in kinds:
                        selected = [
                            r
                            for r in values
                            if r["version"] == version and r["kind"] == kind
                        ]
                        x = np.array([r["score"] for r in selected])
                        y = np.array([r["faithfulness"] for r in selected])
                        rho, p = spearmanr(x, y)
                        pearson, pp = pearsonr(x, y)
                        result = dict(
                            model=name,
                            protocol=protocol,
                            cohort=cohort,
                            version=version,
                            kind=kind,
                            n_relations=len(x),
                            spearman_rho=float(rho),
                            spearman_p=float(p),
                            pearson_r=float(pearson),
                            pearson_p=float(pp),
                            delta_rho=None,
                            delta_ci_low=None,
                            delta_ci_high=None,
                        )
                        if version == "newline" and kind in ("diff", "lre"):
                            previous = [
                                r
                                for r in values
                                if r["version"] == "original" and r["kind"] == kind
                            ]
                            assert [r["relation"] for r in previous] == [
                                r["relation"] for r in selected
                            ]
                            before = np.array([r["score"] for r in previous])
                            ids = np.random.default_rng(20260921).integers(
                                0, len(x), size=(10000, len(x))
                            )
                            delta = bootstrap_rho(x, y, ids) - bootstrap_rho(
                                before, y, ids
                            )
                            low, high = np.nanpercentile(delta, [2.5, 97.5])
                            result.update(
                                delta_rho=float(rho - spearmanr(before, y).statistic),
                                delta_ci_low=float(low),
                                delta_ci_high=float(high),
                            )
                        correlations.append(result)
    for filename, rows in (
        ("correlations.csv", correlations),
        ("relation_scores.csv", relation_rows),
    ):
        with (root / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(
        prompt="\n{} is known for",
        audits=audits,
        workers=workers,
        correlations=correlations,
        report_sha256=sha(Path(__file__)),
        bootstrap=dict(draws=10000, seed=20260921, unit="relation", paired=True),
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def lookup(name, protocol, kind="diff", version="newline", cohort="all"):
        return next(
            r
            for r in correlations
            if (r["model"], r["protocol"], r["kind"], r["version"], r["cohort"])
            == (name, protocol, kind, version, cohort)
        )

    lines = [
        "# All-model newline-prefix rerun",
        "",
        "All models receive the identical neutral text template `\\n{} is known for`, retaining "
        "their native tokenizer and special-token conventions. Model checkpoint, fixed lens layer, "
        "reference/test splits, and relation-prompt targets match the retained runs. Only the neutral "
        "lens prompt changes; LRE faithfulness and its selected layer/beta are retained. All measurements "
        "use float32/eager attention. No prompt, layer, or beta is selected to maximize the reported primary correlations.",
        "",
        "The score is −median(log10(rank+1)), correlated across relations with LRE model-top-1 "
        "faithfulness. This is not complete-answer accuracy. Tables show coefficient (two-sided p-value).",
        "",
    ]
    for statistic, label in (("spearman", "Spearman ρ"), ("pearson", "Pearson r")):
        key = "spearman_rho" if statistic == "spearman" else "pearson_r"
        lines += [
            f"**{label}, newline prompt, all eligible relations**",
            "",
            "| Model | Main differential | Main affine β=1 | Shared differential | Shared affine β=1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in models:
            cells = [
                lookup(name, protocol, kind)
                for protocol in ("main", "shared_split")
                for kind in ("diff", "lre")
            ]
            lines.append(
                "| "
                + LABELS[name]
                + " | "
                + " | ".join(f"{r[key]:.4f} ({r[statistic + '_p']:.4g})" for r in cells)
                + " |"
            )
        lines += [""]
    lines += [
        "**Differential Spearman before → after**",
        "",
        "| Model | Main, all relations | Shared, all relations | Main, ≥4 LRE tests | Shared, ≥4 LRE tests |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in models:
        cells = []
        for cohort in ("all", "at_least_4_lre_tests"):
            for protocol in ("main", "shared_split"):
                old = lookup(name, protocol, version="original", cohort=cohort)
                new = lookup(name, protocol, cohort=cohort)
                cells.append(
                    f"{old['spearman_rho']:.4f} → {new['spearman_rho']:.4f} (n={new['n_relations']})"
                )
        lines.append("| " + LABELS[name] + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![Before and after](comparison.png)",
        "",
        "**Coverage and tokenization**",
        "",
        "| Model | Main rows / relations | Shared rows / relations | Content retokenizations / checked relation-subject entries |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in models:
        a, b = audits[name]["main"], audits[name]["shared_split"]
        changes = sum(len(d["tokenization_changed_relation_subjects"]) for d in (a, b))
        checked = sum(d["n_checked_relation_subjects"] for d in (a, b))
        lines.append(
            f"| {LABELS[name]} | {a['n_subject_rows']} / {a['n_relations']} | {b['n_subject_rows']} / {b['n_relations']} | {changes} / {checked} |"
        )
    lines += [
        "",
        "Every selected subject position was checked against full-prompt character offsets and "
        "is after position 0. Content-token preservation is checked after removing automatic special tokens: "
        "new content IDs versus newline IDs concatenated with old content IDs. Any differences are retained "
        "per subject; identical text does not imply identical tokenization across model families.",
        "",
        "[Tokenizer examples](tokenizer_examples.json) show the exact IDs and pieces for Australia. "
        "For Mistral, `▁Australia` becomes `Austral` + `ia`; its newline effect includes subject "
        "retokenization. Gemma retains its BOS and adds the newline after it.",
        "",
        "The runner verifies reconstructed logits, no-op state restoration, scalar/batched JVPs, and "
        "finite differences. Qwen3.8 additionally verifies its convolution optimization and prefix cache. "
        "Reports require complete model/protocol cohorts, unchanged source/data hashes, matching targets "
        "and splits, and aggregates recomputed from individual ranks. Legacy OLMo/GPT-J records store "
        "decoded target strings rather than IDs; their target checks therefore compare decoded strings.",
        "",
        "`correlations.csv` includes unrounded Spearman/Pearson coefficients and p-values for both protocols "
        "and both cohorts, differential and affine β=1 readouts, all six fixed affine betas, and both "
        "in-sample beta-selection metrics. Original beta sweeps are included where retained in the source. "
        "The tuned rows are exploratory, not primary results. Paired 95% intervals for changes in differential "
        "and affine β=1 Spearman use 10,000 relation bootstrap draws (seed 20260921). "
        "`relation_scores.csv` stores all relation-level scores and file hashes; `summary.json` retains "
        "validation metadata and coverage.",
        "",
        "Main and shared-split differ in filtering, row selection, and sample sizes. Shared-split filters "
        "answers whose first token collides with another answer, reserves 8 demonstration and 8 reference "
        "rows, and scores up to 100 test rows. Main lens scores up to 20 rows with references drawn from "
        "those rows and self-comparisons excluded. Disjoint rows need not mean disjoint subjects. The ≥4 "
        "LRE-test sensitivity changes the relation cohort. Retained LRE layer/beta selection is in-sample; "
        "relation bootstrap intervals are approximate because relations share subjects and domains. "
        "These reruns follow the Qwen diagnostic and are not independent held-out validation. "
        "Reported p-values are unadjusted for multiple comparisons.",
        "",
        "Reproduce from `code/`, assigning suitable free GPU(s) to each worker:",
        "",
        "```sh",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_all_model_newline.py --name mistral7b --worker local_0 --batch-size 4",
        ".venv/bin/python scripts/report_all_model_newline.py",
        "```",
        "",
        "`workers.json` records the actual launch assignments. The 27B model needs multiple GPUs in float32. "
        "Workers use advisory claims and resume unfinished relations.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    yy = np.arange(len(models))
    for ax, protocol in zip(axes, ("main", "shared_split")):
        before = [
            lookup(n, protocol, version="original")["spearman_rho"] for n in models
        ]
        after = [lookup(n, protocol)["spearman_rho"] for n in models]
        ax.barh(yy - 0.17, before, height=0.32, label="Original", color="#9aa7b1")
        ax.barh(yy + 0.17, after, height=0.32, label="Newline", color="#356b9a")
        ax.set_yticks(yy, [LABELS[n] for n in models])
        ax.set_xlim(-1, 1)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(protocol)
        ax.set_xlabel("Differential lens / LRE Spearman ρ")
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    axes[1].legend()
    fig.suptitle(
        "Identical neutral prompt with one leading newline · all eligible relations"
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    print("\n".join(lines[:28]))


if __name__ == "__main__":
    main()
