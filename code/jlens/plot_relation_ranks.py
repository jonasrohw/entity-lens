"""Figure 1 in the panels' coordinate system: rank against rank.

The paper's Figure 1 plots lens *agreement* ($-\\log_{10}$ median rank,
higher better) against LRE *faithfulness* (a proportion, higher better).
The per-relation panels plot lens $\\log_{10}$ rank against LRE
$\\log_{10}$ rank, both per subject, both lower-better.  The x axes are
therefore the same quantity with opposite sign, and the y axes are
different quantities altogether -- so the two figures cannot be read
against each other.

This draws the relation-level figure in the panels' system: median lens
log-rank against median LRE log-rank, one point per relation, lower-left
is better on both.  It is then literally the panels aggregated, and the
reader can move between the two without re-orienting.

Requires `jlens/measure_subject_ranks.py` to have run.
"""

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from jlens import plotstyle as style
from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens", default=None, help="lens JSON; default is the filtered run"
    )
    parser.add_argument(
        "--rank-dir", default=None, help="dir holding subject_ranks_*.json"
    )
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    parser.add_argument(
        "--stat",
        choices=["mean", "median"],
        default="mean",
        help="how to aggregate per-subject log ranks to the relation. "
        "median censors: any relation with faithfulness above 0.5 "
        "has a hit as its median subject, so 16 of 25 relations "
        "land on exactly 0 and the y axis collapses. mean does not.",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate = np.mean if args.stat == "mean" else np.median

    lens_path = args.lens or f"{ROOT}/results/shared_split/jacobian_lens.json"
    rank_dir = args.rank_dir or f"{ROOT}/results/shared_split"
    output_stem = f"{OUTPUT_DIR}/relation_ranks_{args.stat}{args.tag}"
    json_path = f"{output_stem}.json"

    lens_results = read_json(lens_path)
    rows = []
    for path in sorted(glob.glob(f"{rank_dir}/subject_ranks_*.json")):
        rank_results = read_json(path)
        relation = rank_results["relation"]
        if relation not in lens_results:
            continue
        if rank_results["split_fingerprint"] != lens_results[relation].get(
            "split_fingerprint"
        ):
            raise SystemExit(f"FATAL {relation}: fingerprint mismatch")
        lens = {r["subject"]: r["rank_diff"] for r in lens_results[relation]["rows"]}
        xs = [np.log10(lens[s] + 1) for s in rank_results["ranks"] if s in lens]
        ys = [
            np.log10(d["rank"] + 1)
            for s, d in rank_results["ranks"].items()
            if s in lens
        ]
        if len(xs) < 3:
            continue
        rows.append(
            dict(
                rel=relation,
                x=float(aggregate(xs)),
                y=float(aggregate(ys)),
                n=len(xs),
                faith=rank_results["best_faith"],
            )
        )

    x = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    rho, p = stats.spearmanr(x, y)

    style.use_paper_style()
    fig, ax = plt.subplots(figsize=(style.BODY_WIDTH, 2.85), layout="constrained")
    b, a_ = np.polyfit(x, y, 1)
    xs_ = np.linspace(x.min() - 0.15, x.max() + 0.15, 50)
    ax.plot(xs_, a_ + b * xs_, lw=1.0, color=style.FIT, zorder=1)
    ax.scatter(
        x,
        y,
        s=20,
        facecolor=style.INK,
        edgecolor=style.EDGE,
        linewidth=0.5,
        zorder=3,
    )

    # Offsets are in data units and the two aggregations have different y
    # ranges, so they are given as fractions of the axis span.
    LABEL = {
        "person_father": (-0.06, -0.10, "right"),
        # Leftmost point of the cloud: a right-aligned label would run off
        # the axes, so this one is set to the right of its point.
        "country_capital_city": (0.04, 0.09, "left"),
        "superhero_person": (0.04, 0.07, "left"),
        "person_occupation": (-0.05, 0.10, "right"),
    }
    xr = x.max() - x.min()
    yr = max(y.max() - y.min(), 1e-9)
    for r in rows:
        if r["rel"] in LABEL:
            fx, fy, ha = LABEL[r["rel"]]
            ax.annotate(
                r["rel"].replace("_", " "),
                (r["x"], r["y"]),
                (r["x"] + fx * xr, r["y"] + fy * yr),
                ha=ha,
                fontsize=style.TINY_PT,
                color=style.INK,
                arrowprops=dict(
                    arrowstyle="-", lw=0.4, color=style.MUTED, shrinkA=0, shrinkB=2
                ),
            )
    ax.margins(x=0.16, y=0.14)
    ax.set_xlabel(f"lens: {args.stat} $\\log_{{10}}$ rank of the model's answer")
    ax.set_ylabel(f"LRE: {args.stat} $\\log_{{10}}$ rank")
    ax.text(
        0.02,
        0.97,
        f"$\\rho = {rho:.3f}$",
        transform=ax.transAxes,
        fontsize=style.SCRIPT_PT,
        va="top",
        color=style.INK,
    )
    ax.grid(True, zorder=0)
    output_path = style.save(fig, output_stem, png=True)
    plt.close(fig)

    write_json(
        json_path,
        {
            "statistic": args.stat,
            "spearman_rho": float(rho),
            "p": float(p),
            "n_relations": len(rows),
            "note": f"both axes are {args.stat} log10 rank, lower is better",
            "relations": rows,
        },
    )
    print(f"wrote {output_path}\nwrote {json_path}")
    print(f"\nSpearman rho = {rho:+.3f}  p = {p:.2e}  n = {len(rows)} relations")
    print(f"\n{'relation':34s} {'lens':>9s} {'LRE':>8s} {'faith':>6s}")
    for r in sorted(rows, key=lambda r: r["y"]):
        print(f"{r['rel']:34s} {r['x']:9.2f} {r['y']:8.2f} {r['faith']:6.2f}")


if __name__ == "__main__":
    main()
