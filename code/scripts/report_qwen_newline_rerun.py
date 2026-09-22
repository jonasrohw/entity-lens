"""Validate and report the complete Qwen newline-prefix rerun."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_qwen_prompt_sensitivity import bootstrap_rho
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT / "results/five_model_sweep/qwen3_8b_base"
PROMPT = "\n{} is known for"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "results/qwen_newline_rerun"
    )
    args = parser.parse_args()
    root = args.root
    workers = {
        int(p.stem.split("_")[-1]): json.loads(p.read_text())
        for p in root.glob("worker_*.json")
    }
    relation_rows, correlations, audits, country_rows = [], [], {}, []
    for protocol in ("main", "shared_split"):
        sources = {p.stem: p for p in (RETAINED / protocol).glob("*.json")}
        originals = {name: json.loads(p.read_text()) for name, p in sources.items()}
        names = sorted(n for n, d in originals.items() if "agg" in d.get("lens", {}))
        files = {
            p.stem: p for p in (root / protocol / "newline_known_for").glob("*.json")
        }
        if set(files) != set(names):
            raise RuntimeError(
                f"Incomplete {protocol}: {len(files)}/{len(names)}; "
                f"missing {sorted(set(names) - set(files))}"
            )
        records = []
        for name in names:
            path = files[name]
            current = json.loads(path.read_text())
            old = originals[name]
            worker = workers[current["worker"]]
            assert worker["prompts"] == {"newline_known_for": PROMPT}
            assert worker["revision"] == old["revision"]
            assert worker["layer"] == old["lens"]["layer"] == 19
            for filename, expected in worker["source_sha256"].items():
                assert sha(ROOT / filename) == expected, filename
            assert sha(sources[name]) == current["retained_sha256"]
            assert sha(ROOT / "data/relations" / path.name) == current["data_sha256"]
            assert current["faithfulness"] == old["faithfulness"]["best_faith"]
            assert current["n_lre"] == old["faithfulness"]["n_test"]
            assert current["lens"]["neutral_prompt"] == PROMPT
            assert (
                current["lens"]["split_fingerprint"] == old["lens"]["split_fingerprint"]
            )
            assert all(position > 0 for position in current["positions"].values())
            keys = ("subject", "model_top1_id", "n_references")
            assert [[r[k] for k in keys] for r in current["lens"]["rows"]] == [
                [r[k] for k in keys] for r in old["lens"]["rows"]
            ]
            for d in (current, old):
                for kind in ("diff", "lre"):
                    ranks = [r[f"rank_{kind}"] for r in d["lens"]["rows"]]
                    value = np.median(np.log10(np.array(ranks) + 1))
                    assert abs(value - d["lens"]["agg"][kind]["median_lograng"]) < 1e-12
            records.append(current)
            if protocol == "main" and name == "country_capital_city":
                for r, before in zip(current["lens"]["rows"], old["lens"]["rows"]):
                    country_rows.append(
                        dict(
                            subject=r["subject"],
                            target=r["model_top1"],
                            original_rank=before["rank_diff"],
                            new_rank=r["rank_diff"],
                            top_diff=r["top_diff"],
                        )
                    )
            for kind in ("diff", "lre"):
                relation_rows.append(
                    dict(
                        protocol=protocol,
                        relation=name,
                        kind=kind,
                        original_score=-old["lens"]["agg"][kind]["median_lograng"],
                        new_score=-current["lens"]["agg"][kind]["median_lograng"],
                        faithfulness=current["faithfulness"],
                        n_lens=current["lens"]["n"],
                        n_lre=current["n_lre"],
                        source=str(path.relative_to(ROOT)),
                        source_sha256=sha(path),
                        retained_sha256=current["retained_sha256"],
                    )
                )
        audits[protocol] = dict(
            n_relations=len(names),
            n_subject_rows=sum(d["lens"]["n"] for d in records),
            excluded_relations=sorted(set(originals) - set(names)),
            minimum_subject_position=min(min(d["positions"].values()) for d in records),
            completed_relations=names,
        )
        for kind in ("diff", "lre"):
            for cohort in ("all", "at_least_4_lre_tests"):
                selected = [
                    r
                    for r in relation_rows
                    if r["protocol"] == protocol
                    and r["kind"] == kind
                    and (cohort == "all" or r["n_lre"] >= 4)
                ]
                y = np.array([r["faithfulness"] for r in selected])
                before = np.array([r["original_score"] for r in selected])
                after = np.array([r["new_score"] for r in selected])
                indices = np.random.default_rng(20260921).integers(
                    0, len(y), size=(10000, len(y))
                )
                difference = bootstrap_rho(after, y, indices) - bootstrap_rho(
                    before, y, indices
                )
                ci = np.nanpercentile(difference, [2.5, 97.5]).tolist()
                for version, x in (("original", before), ("newline", after)):
                    rho, p = spearmanr(x, y)
                    r, pp = pearsonr(x, y)
                    correlations.append(
                        dict(
                            protocol=protocol,
                            kind=kind,
                            cohort=cohort,
                            version=version,
                            n_relations=len(y),
                            rho=float(rho),
                            p=float(p),
                            pearson_r=float(r),
                            pearson_p=float(pp),
                            delta_rho=float(
                                spearmanr(after, y).statistic
                                - spearmanr(before, y).statistic
                            ),
                            delta_rho_ci95=ci,
                        )
                    )
    summary = dict(
        prompt=PROMPT,
        workers=workers,
        audit=audits,
        correlations=correlations,
        country_predictions=country_rows,
        bootstrap=dict(draws=10000, seed=20260921, unit="relation", paired=True),
        report_sha256=sha(Path(__file__)),
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for filename, rows in (
        ("relation_scores.csv", relation_rows),
        ("correlations.csv", correlations),
    ):
        with (root / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Qwen3-8B-Base: complete newline-prefix rerun",
        "",
        "The neutral lens prompt is now `\\n{} is known for`. This rerun covers every "
        "previously eligible relation in both protocols, with the original checkpoint, "
        "state 19, float32/eager attention, splits, references, target tokens, and LRE "
        "faithfulness. No layer or beta was retuned. Only the neutral lens prompt changed.",
        "",
        "The runner verifies that prepending the newline preserves every original token ID "
        "after one added token, checks subject offsets, and requires identical subject/target/"
        "reference-count rows and split fingerprints. The report checks complete cohorts, "
        "source/data hashes, and recalculates rank aggregates from the individual rows.",
        "",
        "Score: negative median log10(target rank + 1). Correlations are across relations "
        "against retained LRE model-top-1 faithfulness, not gold-answer accuracy.",
        "",
        "| Protocol | Relations | Original Spearman ρ | Newline Spearman ρ | Newline p | Δρ, paired 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for protocol in ("main", "shared_split"):
        old, new = [
            r
            for r in correlations
            if r["protocol"] == protocol
            and r["kind"] == "diff"
            and r["cohort"] == "all"
        ]
        low, high = new["delta_rho_ci95"]
        lines.append(
            f"| {protocol} | {new['n_relations']} | {old['rho']:.4f} | {new['rho']:.4f} | "
            f"{new['p']:.4g} | {new['delta_rho']:+.4f} [{low:+.4f}, {high:+.4f}] |"
        )
    lines += [
        "",
        "![Original and corrected relation scores](comparison.png)",
        "",
        "The existing sensitivity excluding relations with fewer than four LRE tests:",
        "",
    ]
    for protocol in ("main", "shared_split"):
        old, new = [
            r
            for r in correlations
            if r["protocol"] == protocol
            and r["kind"] == "diff"
            and r["cohort"] == "at_least_4_lre_tests"
        ]
        lines.append(
            f"- {protocol}: ρ {old['rho']:.4f} → {new['rho']:.4f}, "
            f"p={new['p']:.4g}, {new['n_relations']} relations."
        )
    lines += [
        "",
        "Country capital predictions (zero-based ranks of the original target token):",
        "",
        "| Country | Target | Original rank | Newline rank | Newline top five |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in country_rows:
        top = ", ".join(f"`{t}`" for t in r["top_diff"])
        lines.append(
            f"| {r['subject']} | {r['target']} | {r['original_rank']} | {r['new_rank']} | {top} |"
        )
    lines += [
        "",
        "The newline removes the diagnosed first-position mismatch. These are revised "
        "measurements following that diagnosis, not independent held-out confirmation. "
        "The two protocols are not independent replications; retained LRE faithfulness "
        "still uses in-sample layer/beta selection. The relation bootstrap uses 10,000 "
        "paired draws (seed 20260921); relations share subjects and domains, so its "
        "intervals are approximate. Tiny relations and first-token target ambiguity remain.",
        "",
        "Raw per-subject outputs are in `main/newline_known_for/` and "
        "`shared_split/newline_known_for/`. `summary.json` includes validation metadata, "
        "counts, excluded relations, and both differential and affine correlations. "
        "`relation_scores.csv` and `correlations.csv` retain unrounded values.",
        "",
        "Reproduce from `code/` (four workers can share the output directory on separate GPUs):",
        "",
        "```sh",
        "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_qwen_prompt_sensitivity.py "
        "--root results/qwen_newline_rerun --prompts newline_known_for --batch-size 8 --worker 0",
        ".venv/bin/python scripts/report_qwen_newline_rerun.py",
        "```",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for col, protocol in enumerate(("main", "shared_split")):
        records = [
            r
            for r in relation_rows
            if r["protocol"] == protocol and r["kind"] == "diff"
        ]
        for row, version in enumerate(("original", "newline")):
            ax = axes[row, col]
            metric = next(
                r
                for r in correlations
                if r["protocol"] == protocol
                and r["kind"] == "diff"
                and r["cohort"] == "all"
                and r["version"] == version
            )
            x = [
                r["original_score" if version == "original" else "new_score"]
                for r in records
            ]
            y = [r["faithfulness"] for r in records]
            ax.scatter(
                x,
                y,
                c=[
                    "#c45e22" if r["relation"].startswith("country_") else "#356b9a"
                    for r in records
                ],
            )
            ax.set_title(f"{protocol} · {version} · ρ={metric['rho']:.3f}")
            ax.grid(alpha=0.2)
            ax.set_xlim(-5.5, 0.1)
            ax.set_ylim(-0.03, 1.05)
            if row == 1:
                ax.set_xlabel("Lens score: −median log₁₀(rank + 1)")
            if col == 0:
                ax.set_ylabel("Retained LRE faithfulness")
    fig.suptitle(
        "Qwen3-8B-Base: one newline before the subject\nCountry relations highlighted in orange"
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(audits, indent=2))
    print(
        json.dumps(
            [r for r in correlations if r["kind"] == "diff" and r["cohort"] == "all"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
