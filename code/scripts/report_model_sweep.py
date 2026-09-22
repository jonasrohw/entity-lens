"""Retain atomic measurements and publish only complete model correlations."""

import csv
import json
import shutil
from pathlib import Path

from scipy.stats import spearmanr

from jlens.model_sweep import summarize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "generated/five_model_sweep/results"
DEST = ROOT / "results/five_model_sweep"
LABELS = {
    "qwen3_8b_base": "Qwen3 8B Base",
    "mistral7b": "Mistral 7B v0.3",
    "pythia6_9b": "Pythia 6.9B",
    "gemma2_9b": "Gemma 2 9B",
    "qwen3_8_27b": "Qwen3.8 27B",
}


def sample_size_sensitivity(summary):
    results = {}
    for name in LABELS:
        if not summary.get(name, {}).get("main", {}).get("complete"):
            continue
        measurements = [
            json.loads(path.read_text())
            for path in (DEST / name / "main").glob("*.json")
        ]
        results[name] = {}
        for minimum in [1, 4, 10, 20, 50, 100]:
            cohort = [r for r in measurements if r["faithfulness"]["n_test"] >= minimum]
            rho, p = spearmanr(
                [-r["lens"]["agg"]["diff"]["median_lograng"] for r in cohort],
                [r["faithfulness"]["best_faith"] for r in cohort],
            )
            results[name][str(minimum)] = dict(
                n=len(cohort),
                rho=float(rho),
                p=float(p),
                excluded_relations=sorted(
                    r["relation"]
                    for r in measurements
                    if r["faithfulness"]["n_test"] < minimum
                ),
            )
    output = dict(
        note="Exploratory changed-cohort diagnostic with unchanged model/lens/LRE settings. Thresholds were examined after viewing results; these do not replace the primary all-28 correlations.",
        results=results,
    )
    (DEST / "sample_size_sensitivity.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return results


def plot_results(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    for column, (name, label) in enumerate(LABELS.items()):
        for row, protocol in enumerate(("main", "shared_split")):
            ax = axes[row, column]
            data = summary.get(name, {}).get(protocol, {})
            ax.set_title(label, fontsize=11)
            ax.set_xlim(-5.6, 0.1)
            ax.set_ylim(-0.04, 1.04)
            ax.grid(alpha=0.15)
            if not data.get("complete"):
                ax.text(
                    0.5,
                    0.5,
                    f"Pending ({data.get('n_finished', 0)}/28)",
                    ha="center",
                    transform=ax.transAxes,
                )
                continue
            measurements = [
                json.loads((DEST / name / protocol / f"{relation}.json").read_text())
                for relation in data["completed_relations"]
            ]
            for kind, color, label_kind in [
                ("diff", "#2878b5", "Differential"),
                ("lre", "#d57722", "Affine beta=1"),
            ]:
                ax.scatter(
                    [-d["lens"]["agg"][kind]["median_lograng"] for d in measurements],
                    [d["faithfulness"]["best_faith"] for d in measurements],
                    color=color,
                    s=18,
                    alpha=0.65,
                    label=label_kind,
                )
            metrics = data["correlations"]
            ax.text(
                0.04,
                0.04,
                f"n={data['n_relations']}   rho(diff)={metrics['diff']['rho']:.3f}\n"
                f"rho(affine)={metrics['lre']['rho']:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            )
    axes[0, 0].set_ylabel("Main protocol\nLRE faithfulness")
    axes[1, 0].set_ylabel("Shared-split protocol\nLRE faithfulness")
    for ax in axes[1]:
        ax.set_xlabel("− median log10(rank + 1)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(DEST / "correlations.png", dpi=160)
    figure.savefig(DEST / "correlations.pdf")
    plt.close(figure)


def main():
    summarize(SOURCE, verbose=False)
    provenance = DEST / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for pattern in ["dynamic_worker_*.json", "workers.json", "*.tokenizer_audit.json"]:
        for path in SOURCE.parent.glob(pattern):
            shutil.copy2(path, provenance / path.name)
    for source in SOURCE.rglob("*.json"):
        target = DEST / source.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    summary = json.loads((DEST / "summary.json").read_text())
    lines = [
        "# Five-model lens/faithfulness comparison",
        "",
        "Spearman correlations across relations between negative median log10(rank+1) "
        "and selected LRE faithfulness. Only complete protocol results are reported below.",
        "",
        "| Model | Protocol | Finished / 28 | Included n | Differential rho (p) | Affine beta=1 rho (p) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = []
    for name, label in LABELS.items():
        for protocol in ("main", "shared_split"):
            data = summary.get(name, {}).get(protocol, {})
            row = dict(
                model=name,
                protocol=protocol,
                n_finished=data.get("n_finished", 0),
                complete=data.get("complete", False),
                n_relations=data.get("n_relations") if data.get("complete") else None,
            )
            values = []
            for kind in ("diff", "lre"):
                metric = (
                    data.get("correlations", {}).get(kind, {})
                    if row["complete"]
                    else {}
                )
                row[kind + "_rho"] = metric.get("rho")
                row[kind + "_p"] = metric.get("p")
                values.append(
                    f"{metric['rho']:.3f} ({metric['p']:.3f})" if metric else "pending"
                )
            lines.append(
                f"| {label} | {protocol} | {row['n_finished']} | {row['n_relations'] if row['complete'] else 'pending'} | {' | '.join(values)} |"
            )
            rows.append(row)
    sensitivity = sample_size_sensitivity(summary)
    lines += [
        "",
        "## Sample-size sensitivity",
        "",
        "The three main relations `name_gender`, `presidents_birth_year`, and "
        "`presidents_election_year` each have only three LRE test subjects. "
        "Removing these changes the cohort, without changing any model, LRE, or lens settings:",
        "",
        "| Model | Main rho, all 28 | Main rho, 25 relations with at least four tests |",
        "| --- | --- | --- |",
    ]
    for name, values in sensitivity.items():
        lines.append(
            f"| {LABELS[name]} | {values['1']['rho']:.3f} | {values['4']['rho']:.3f} |"
        )
    lines += [
        "",
        "This is an exploratory diagnostic, not a replacement primary result. "
        "[All inspected thresholds](sample_size_sensitivity.json) and "
        "[the focused Mistral grid and fresh-split checks](../mistral_grid_search/README.md) "
        "separate cohort effects from hyperparameter tuning.",
        "",
        "The main and shared-split protocols have different subject/reference selection. "
        "Do not interpret their difference as a controlled split-only effect.",
        "The original dataset contains duplicate subjects: the shared split separates rows, "
        "which does not always separate subjects. The focused Mistral report includes "
        "a separate duplicate-cleaned confirmation; the table here retains the original input protocol.",
        "",
        "Qwen3.8-27B is a post-trained hybrid model; the other four are base language models. "
        "All receive raw completion prompts. Float32, 8 references, up to 100 test subjects, "
        "six LRE layers and six beta values are retained. Affine beta tuning in the full "
        "JSON is in-sample and exploratory.",
        "",
        "See [protocol and validation](../../docs/five-model-sweep.md), "
        "[model selection](models.json), and [checkpoint revisions](revisions.json). "
        "Per-relation JSON files retain outcomes, ranks, the full LRE grid, affine-beta "
        "sweep, split fingerprints, and pretrained numerical checks.",
        "",
        "![Relation-level correlations](correlations.png)",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(lines))
    with (DEST / "correlations.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(summary)
    print("Published", sum(row["complete"] for row in rows), "of 10 complete protocols")


if __name__ == "__main__":
    main()
