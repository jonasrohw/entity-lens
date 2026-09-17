"""Plot lens ranks against binary LRE outcomes, one panel per relation.

Panels are ordered by AUC. Relations with no outcome variance remain visible
but are excluded from inferential summaries."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from jlens import plotstyle as style
from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json

lens_path = f"{ROOT}/results/shared_split/jacobian_lens.json"
FAITH = f"{ROOT}/results/shared_split"
output_stem = f"{OUTPUT_DIR}/subject_outcomes"
json_path = f"{output_stem}.json"


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
    # Four columns at the text width, so the sheet needs no scaling on the
    # page and the point sizes in it are the point sizes of the body text.
    ncol = 4
    nrow = int(np.ceil(len(D) / ncol))
    style.use_paper_style()
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(style.FULL_WIDTH, 1.45 * nrow),
        layout="constrained",
        sharex=True,
    )
    axes = np.atleast_2d(axes).ravel()
    rng = np.random.default_rng(0)

    for ax, d in zip(axes, D):
        degen = np.isnan(d["auc"])
        # Hits first, so the shared legend reads hit before miss.
        for val, colour, lab in (
            (1, style.HIT, "LRE reproduced the answer"),
            (0, style.MISS, "LRE missed it"),
        ):
            m = d["y"] == val
            if not m.any():
                continue
            jit = val + rng.uniform(-0.17, 0.17, m.sum())
            ax.scatter(
                d["x"][m],
                jit,
                s=5,
                facecolor=(style.DEGENERATE if degen else colour),
                edgecolor=style.EDGE,
                linewidth=0.25,
                zorder=3,
                label=lab,
            )
            ax.plot(
                [np.median(d["x"][m])] * 2,
                [val - 0.30, val + 0.30],
                color=(style.DEGENERATE if degen else colour),
                lw=1.4,
                zorder=4,
                label="group median" if val == 1 else None,
            )
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["miss", "hit"])
        ax.tick_params(length=2, pad=1.5)
        style.deemphasise(ax, degen)
        # Three short lines, not one long one: at four panels to the text
        # width a single stat line runs into its neighbour.
        counts = rf"$n = {d['n']}$;  faith {d['faith']:.2f}"
        sub = (
            "no variance to measure"
            if degen
            else rf"AUC {d['auc']:.2f}  ($p = {d['p']:.3f}$)"
        )
        ax.set_title(
            "\n".join([d["rel"].replace("_", " "), counts, sub]),
            fontsize=style.TINY_PT - 0.5,
            color=(style.DEGENERATE if degen else style.INK),
            linespacing=1.3,
        )
    for ax in axes[len(D) :]:
        ax.set_visible(False)
    # Short label: the full "rank of the model's answer" does not fit the
    # column width four times over, and the caption carries the gloss.
    style.label_bottom_row(axes, ncol, len(D), "lens $\\log_{10}$ rank")

    handles, labels = style.collect_handles(axes[: len(D)])
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        fontsize=style.TINY_PT,
        columnspacing=1.4,
        handlelength=1.8,
    )
    output_path = style.save(fig, output_stem)
    plt.close(fig)
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
