"""The per-subject analogue of the paper's Figure 1, one panel per relation.

Figure 1 plots two continuous variables at the relation level: lens
agreement against LRE faithfulness.  This is the same plot with the
subject as the unit --- lens rank of the model's answer against the
**LRE's** rank of that same answer, both per subject, both continuous.

`panels_subject_statistics.py` gives the binary-outcome version (hit/miss),
which is what the within-relation AUC summarizes.  This one shows how
*close* the LRE came, which the binary version throws away: a relation
where the misses cluster just below rank 0 is a different object from one
where they are scattered over five orders of magnitude.

Ranks come from `jlens/measure_subject_ranks.py`, which recomputes the LRE
at the already-selected (layer, beta) and asserts that thresholding at
rank 0 reproduces the stored per-subject hits.
"""

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from jlens import plotstyle as style
from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json


def load_rank_panels(lens_path, rank_dir):
    lens_results = read_json(lens_path)
    out = []
    for path in sorted(glob.glob(f"{rank_dir}/subject_ranks_*.json")):
        rank_results = read_json(path)
        relation = rank_results["relation"]
        if relation not in lens_results:
            continue
        if rank_results["split_fingerprint"] != lens_results[relation].get(
            "split_fingerprint"
        ):
            raise SystemExit(f"FATAL {relation}: fingerprint mismatch.")
        lens = {r["subject"]: r["rank_diff"] for r in lens_results[relation]["rows"]}
        x, y = [], []
        for s, d in rank_results["ranks"].items():
            if s not in lens:
                continue
            x.append(np.log10(lens[s] + 1))
            y.append(np.log10(d["rank"] + 1))
        if len(x) < 3:
            continue
        x, y = np.array(x), np.array(y)
        # A relation at faithfulness 1.00 has every LRE rank at 0, so y is
        # constant and rho is undefined.  That is a fact about the relation,
        # not an error: it is plotted, greyed, and left out of the summary.
        if len(set(y.tolist())) < 2 or len(set(x.tolist())) < 2:
            rho = p = np.nan
        else:
            rho, p = stats.spearmanr(x, y)
        # Among the subjects the LRE MISSED, does the lens predict how far
        # off it was?  The all-points rho mixes that question with "did it
        # miss at all", in proportions set by the relation's faithfulness,
        # so the two are not comparable across relations.  y == 0 is a
        # censoring floor, not a value: once the LRE is right, rank 0 is
        # all we can observe.
        m = y > 0
        if m.sum() >= 3 and len(set(y[m].tolist())) > 1 and len(set(x[m].tolist())) > 1:
            rho_m, p_m = stats.spearmanr(x[m], y[m])
        else:
            rho_m = p_m = np.nan
        out.append(
            dict(
                rel=relation,
                x=x,
                y=y,
                rho=rho,
                p=p,
                rho_m=rho_m,
                p_m=p_m,
                faith=rank_results["best_faith"],
                n=len(x),
                n_miss=int(m.sum()),
            )
        )
    out.sort(
        key=lambda d: (np.isnan(d["rho"]), -(d["rho"] if d["rho"] == d["rho"] else 0))
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", default=None)
    parser.add_argument("--rank-dir", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lens_path = args.lens or f"{ROOT}/results/shared_split/jacobian_lens.json"
    rank_dir = args.rank_dir or f"{ROOT}/results/shared_split"
    output_stem = f"{OUTPUT_DIR}/subject_ranks{args.tag}"
    json_path = f"{output_stem}.json"

    D = load_rank_panels(lens_path, rank_dir)
    # Four columns at the text width: the sheet is then placed at
    # \linewidth without scaling, so the 7pt in it is 7pt on the page.
    ncol = 4
    nrow = int(np.ceil(len(D) / ncol))
    style.use_paper_style()
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(style.FULL_WIDTH, 1.55 * nrow),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes).ravel()

    for ax, d in zip(axes, D):
        degen = np.isnan(d["rho"])
        hit = d["y"] == 0  # LRE rank 0 == faithful
        ax.scatter(
            d["x"][hit],
            d["y"][hit],
            s=5,
            facecolor=(style.DEGENERATE if degen else style.HIT),
            edgecolor=style.EDGE,
            linewidth=0.25,
            zorder=3,
            label="LRE reproduced the answer (rank 0)",
        )
        ax.scatter(
            d["x"][~hit],
            d["y"][~hit],
            s=5,
            facecolor=(style.DEGENERATE if degen else style.MISS),
            edgecolor=style.EDGE,
            linewidth=0.25,
            zorder=4,
            label="LRE missed it",
        )
        if (~hit).sum() >= 3:
            xs = np.linspace(d["x"].min(), d["x"].max(), 30)
            # solid: all points -- but note this fit is pulled by the mass of
            # zeros, so it partly traces the censoring floor
            b, a_ = np.polyfit(d["x"], d["y"], 1)
            ax.plot(
                xs,
                a_ + b * xs,
                lw=0.8,
                color=style.FIT,
                zorder=1,
                label="fit, all subjects",
            )
            # dashed: misses only -- the graded claim, uncensored
            if not np.isnan(d["rho_m"]):
                bm, am = np.polyfit(d["x"][~hit], d["y"][~hit], 1)
                ax.plot(
                    xs,
                    am + bm * xs,
                    lw=0.8,
                    color=style.FIT_MISS,
                    zorder=2,
                    dashes=(3, 2),
                    label="fit, misses only",
                )
        ax.tick_params(length=2, pad=1.5)
        style.deemphasise(ax, degen)
        # Three short lines, not one long one: at four panels to the text
        # width a single stat line runs into its neighbour.
        counts = rf"$n = {d['n']}$;  faith {d['faith']:.2f}"
        if degen:
            sub = "every LRE rank 0"
        else:
            mm = (
                ""
                if np.isnan(d["rho_m"])
                else rf";  $\rho_{{\mathrm{{miss}}}} = {d['rho_m']:+.2f}$"
            )
            sub = rf"$\rho = {d['rho']:+.2f}${mm}"
        ax.set_title(
            "\n".join([d["rel"].replace("_", " "), counts, sub]),
            fontsize=style.TINY_PT - 0.5,
            color=(style.DEGENERATE if degen else style.INK),
            linespacing=1.3,
        )
    for ax in axes[len(D) :]:
        ax.set_visible(False)
    style.label_bottom_row(axes, ncol, len(D), "lens $\\log_{10}$ rank")
    for i in range(0, len(D), ncol):
        axes[i].set_ylabel("LRE $\\log_{10}$ rank", fontsize=style.TINY_PT)

    handles, labels = style.collect_handles(axes[: len(D)])
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        fontsize=style.TINY_PT,
        columnspacing=1.4,
        handlelength=1.8,
    )
    output_path = style.save(fig, output_stem)
    plt.close(fig)

    rhos = np.array([d["rho"] for d in D if not np.isnan(d["rho"])])
    summ = {
        "n_relations_plotted": len(D),
        "n_measurable": int(len(rhos)),
        "n_degenerate": int(len(D) - len(rhos)),
        "median_rho": float(np.median(rhos)),
        "positive": int((rhos > 0).sum()),
        "wilcoxon_p": float(stats.wilcoxon(rhos, alternative="greater").pvalue),
    }
    write_json(
        json_path,
        {
            "summary": summ,
            "relations": [
                {
                    "relation": d["rel"],
                    "n": d["n"],
                    "faith": d["faith"],
                    "rho": None if np.isnan(d["rho"]) else float(d["rho"]),
                    "p": None if np.isnan(d["p"]) else float(d["p"]),
                    "n_miss": d["n_miss"],
                    "rho_misses_only": None
                    if np.isnan(d["rho_m"])
                    else float(d["rho_m"]),
                    "p_misses_only": None if np.isnan(d["p_m"]) else float(d["p_m"]),
                    "measurable": not np.isnan(d["rho"]),
                }
                for d in D
            ],
        },
    )
    print(f"wrote {output_path}\nwrote {json_path}\n")
    print(
        f"{'relation':34s} {'n':>4s} {'faith':>6s} {'rho':>6s} {'p':>7s} "
        f"{'miss rho':>9s} {'n_miss':>7s}"
    )
    for d in D:
        r = "   --  " if np.isnan(d["rho"]) else f"{d['rho']:+6.2f}"
        pp = "   --  " if np.isnan(d["p"]) else f"{d['p']:7.3f}"
        rm = "      -- " if np.isnan(d["rho_m"]) else f"{d['rho_m']:+9.2f}"
        print(
            f"{d['rel']:34s} {d['n']:4d} {d['faith']:6.2f} {r} {pp} "
            f"{rm} {d['n_miss']:7d}"
        )
    print(
        f"\nmedian rho {summ['median_rho']:+.3f} over "
        f"{summ['n_measurable']} measurable relations, "
        f"{summ['positive']} positive, "
        f"Wilcoxon p={summ['wilcoxon_p']:.5f}   "
        f"({summ['n_degenerate']} degenerate, excluded)"
    )


if __name__ == "__main__":
    main()
