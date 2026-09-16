"""Plot lens ranks against binary LRE outcomes, one panel per relation.

Panels are ordered by AUC. Relations with no outcome variance remain visible
but are excluded from inferential summaries."""

import os

import matplotlib

from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lens_path = f"{ROOT}/results/shared_split/jacobian_lens.json"
FAITH = f"{ROOT}/results/shared_split"
output_path = f"{OUTPUT_DIR}/subject_outcomes.pdf"
json_path = f"{OUTPUT_DIR}/subject_outcomes.json"

INK, HIT, MISS, MUTED = "#1a1a1a", "#3b6ea5", "#c2603f", "#9a9a9a"


def load_outcome_panels():
    lens_results = read_json(lens_path)
    out = []
    for relation in lens_results:
        if relation.startswith("_"):
            continue
        split_fingerprint = f"{FAITH}/faithfulness_{relation}.json"
        if not os.path.exists(split_fingerprint):
            continue
        faithfulness = read_json(split_fingerprint)
        ps = faithfulness.get("per_subject")
        if not ps:
            continue
        if faithfulness.get("split_fingerprint") != lens_results[relation].get(
            "split_fingerprint"
        ):
            raise SystemExit(
                f"FATAL {relation}: split fingerprints differ; the two "
                f"measurements did not score the same subjects."
            )
        x, y = [], []
        for r in lens_results[relation]["rows"]:
            if r["subject"] not in ps:
                continue
            x.append(np.log10(r["rank_diff"] + 1))
            y.append(int(ps[r["subject"]]))
        if not x:
            continue
        x, y = np.array(x), np.array(y)
        auc = p = np.nan
        if 0 < y.sum() < len(y):
            u = stats.mannwhitneyu(-x[y == 1], -x[y == 0], alternative="two-sided")
            auc = u.statistic / (y.sum() * (len(y) - y.sum()))
            p = u.pvalue
        out.append(
            dict(
                rel=relation,
                x=x,
                y=y,
                auc=auc,
                p=p,
                faith=faithfulness["best_faith"],
                n=len(y),
            )
        )
    # measurable relations first, by AUC; degenerate ones last
    out.sort(
        key=lambda d: (np.isnan(d["auc"]), -(d["auc"] if d["auc"] == d["auc"] else 0))
    )
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    D = load_outcome_panels()
    ncol = 5
    nrow = int(np.ceil(len(D) / ncol))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 6.2,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(11, 2.05 * nrow), dpi=300, sharex=True
    )
    axes = np.atleast_2d(axes).ravel()
    rng = np.random.default_rng(0)

    for ax, d in zip(axes, D):
        degen = np.isnan(d["auc"])
        for val, colour, lab in ((0, MISS, "miss"), (1, HIT, "hit")):
            m = d["y"] == val
            if not m.any():
                continue
            jit = val + rng.uniform(-0.17, 0.17, m.sum())
            ax.scatter(
                d["x"][m],
                jit,
                s=7,
                facecolor=(MUTED if degen else colour),
                edgecolor="white",
                linewidth=0.25,
                alpha=0.85,
                zorder=3,
            )
            ax.plot(
                [np.median(d["x"][m])] * 2,
                [val - 0.30, val + 0.30],
                color=(MUTED if degen else colour),
                lw=1.4,
                zorder=4,
            )
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["miss", "hit"])
        ax.tick_params(length=2, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        sub = f"n={d['n']}   faith {d['faith']:.2f}   " + (
            "no variance to measure"
            if degen
            else f"AUC {d['auc']:.2f}  (p={d['p']:.3f})"
        )
        ax.text(
            0,
            1.20,
            d["rel"].replace("_", " "),
            transform=ax.transAxes,
            fontsize=6.6,
            color=INK,
            va="bottom",
        )
        ax.text(
            0,
            1.04,
            sub,
            transform=ax.transAxes,
            fontsize=5.4,
            color=(MUTED if degen else INK),
            va="bottom",
        )
    for ax in axes[len(D) :]:
        ax.set_visible(False)
    for ax in axes[max(0, len(D) - ncol) : len(D)]:
        ax.set_xlabel("lens $\\log_{10}$ rank of the model's answer", fontsize=6)

    fig.suptitle(
        "Subject-level separation within each relation: "
        "lens rank of the model's answer, split by whether the LRE "
        "reproduced it",
        fontsize=8,
        y=0.998,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.6)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"wrote {output_path}  ({len(D)} panels)")

    # Persist the table.  Without this the numbers exist only in the figure
    # and in stdout, which is how a result gets quoted from a chat log
    # instead of from a file.
    meas = [d for d in D if not np.isnan(d["auc"])]
    a = np.array([d["auc"] for d in meas])
    kept = np.array([d["auc"] for d in meas if 0.10 <= d["faith"] <= 0.90])
    drop = [d for d in meas if not (0.10 <= d["faith"] <= 0.90)]
    summary = {
        "n_relations_plotted": len(D),
        "n_measurable": len(meas),
        "n_degenerate": len(D) - len(meas),
        "median_auc_all": float(np.median(a)),
        "above_chance": int((a > 0.5).sum()),
        "wilcoxon_p": float(stats.wilcoxon(a - 0.5, alternative="greater").pvalue),
        "median_auc_under_paper_criteria": float(np.median(kept)),
        "n_under_paper_criteria": int(len(kept)),
        "median_auc_of_relations_the_criteria_exclude": float(
            np.median([d["auc"] for d in drop])
        )
        if drop
        else None,
        "excluded_by_criteria": {
            d["rel"]: {"auc": d["auc"], "faith": d["faith"]} for d in drop
        },
    }
    rows = [
        {
            "relation": d["rel"],
            "n": d["n"],
            "faith": d["faith"],
            "auc": None if np.isnan(d["auc"]) else float(d["auc"]),
            "p": None if np.isnan(d["p"]) else float(d["p"]),
            "hits": int(d["y"].sum()),
            "misses": int(len(d["y"]) - d["y"].sum()),
            "measurable": not np.isnan(d["auc"]),
        }
        for d in D
    ]
    write_json(json_path, {"summary": summary, "relations": rows})
    print(f"wrote {json_path}")

    print(f"\n{'relation':34s} {'n':>4s} {'faith':>6s} {'AUC':>6s} {'p':>7s}")
    for d in D:
        av = "  --  " if np.isnan(d["auc"]) else f"{d['auc']:6.3f}"
        pp = "   --  " if np.isnan(d["p"]) else f"{d['p']:7.3f}"
        print(f"{d['rel']:34s} {d['n']:4d} {d['faith']:6.2f} {av} {pp}")
    print(
        f"\nmedian AUC {summary['median_auc_all']:.3f} over "
        f"{summary['n_measurable']} measurable relations, "
        f"{summary['above_chance']} above chance, "
        f"Wilcoxon p={summary['wilcoxon_p']:.5f}"
    )
    print(
        f"the paper's criteria keep {summary['n_under_paper_criteria']} "
        f"(median {summary['median_auc_under_paper_criteria']:.3f}) and drop "
        f"{len(drop)} whose median is "
        f"{summary['median_auc_of_relations_the_criteria_exclude']:.3f}"
    )


if __name__ == "__main__":
    main()
