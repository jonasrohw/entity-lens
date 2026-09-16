"""Scatter for the paper: lens agreement vs LRE faithfulness, 28 relations.

Form: scatter -- two continuous variables whose relationship is the claim.
One series, so no legend box; the two repaired year relations carry a
distinct marker shape (not colour alone) because they are discussed in the
text.  Print medium: grayscale-safe via shape, recessive grid, serif to
match the body text, selective direct labels rather than all 28.
"""

import matplotlib

from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d = read_json(f"{ROOT}/results/paper_figure_data.json")
    rels = d["rels"]
    y = np.array(d["faith"])
    lens = np.array(d["lens"])
    x = -lens  # agreement: higher is better
    YR = {"presidents_birth_year", "presidents_election_year"}
    # Two labels only.  More than this collides at 3.15in; the rest are in
    # Table 1.  Leader lines make the association unambiguous.
    LABEL = {
        "person_father": ("person_father /\nperson_mother", -3.15, 0.235, "left"),
        "country_capital_city": ("country_capital", -1.45, 0.34, "left"),
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.4,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    fig, ax = plt.subplots(figsize=(3.15, 2.5), dpi=400)
    INK = "#1a1a1a"
    ACCENT = "#3b6ea5"
    MUTED = "#8a8a8a"

    # trend line on ranks is what the statistic uses; show an OLS guide on the raw values
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 0.15, x.max() + 0.15, 50)
    ax.plot(xs, a + b * xs, lw=0.9, color=MUTED, zorder=1, alpha=0.9)

    isyr = np.array([r in YR for r in rels])
    ax.scatter(
        x[~isyr],
        y[~isyr],
        s=17,
        facecolor=ACCENT,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        alpha=0.95,
    )
    ax.scatter(
        x[isyr],
        y[isyr],
        s=26,
        marker="^",
        facecolor="white",
        edgecolor=ACCENT,
        linewidth=0.9,
        zorder=3,
        label="year objects (rescored)",
    )
    ax.legend(
        loc="lower right",
        fontsize=5.8,
        frameon=False,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.2,
    )

    for i, r in enumerate(rels):
        if r in LABEL:
            txt, tx, ty, ha = LABEL[r]
            ax.annotate(
                txt,
                (x[i], y[i]),
                (tx, ty),
                fontsize=5.5,
                color=INK,
                ha=ha,
                va="center",
                zorder=4,
                linespacing=1.15,
                arrowprops=dict(
                    arrowstyle="-", lw=0.45, color=MUTED, shrinkA=1.5, shrinkB=3.5
                ),
            )

    rho = stats.spearmanr(x, y)
    ax.text(
        0.035,
        0.965,
        f"$\\rho$ = {rho.statistic:.2f}   $p$ = {rho.pvalue:.0e}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        color=INK,
    )
    ax.set_xlabel(
        "lens agreement  ($-\\log_{10}$ rank of the model's top-1)", fontsize=7
    )
    ax.set_ylabel("LRE faithfulness", fontsize=7)
    ax.grid(True, lw=0.35, color="#d8d8d8", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b0b0b0")
    ax.tick_params(colors=INK, length=2.5, pad=1.5)
    ax.set_ylim(-0.04, 1.08)
    fig.tight_layout(pad=0.25)
    fig.savefig(f"{OUTPUT_DIR}/relation_agreement.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/relation_agreement.png", bbox_inches="tight")
    print(f"wrote {OUTPUT_DIR}/relation_agreement.{{pdf,png}}")
    print(
        f"rho={rho.statistic:+.3f} p={rho.pvalue:.2e}  x range {x.min():.2f}..{x.max():.2f}"
    )


if __name__ == "__main__":
    main()
