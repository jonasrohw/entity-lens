"""Scatter for the paper: lens agreement vs LRE faithfulness, 28 relations.

Form: scatter -- two continuous variables whose relationship is the claim.
One series, so no legend box; the two repaired year relations carry a
distinct marker shape *and* fill (not colour alone) because they are
discussed in the text.  Print medium: greyscale-safe via shape, recessive
grid, the body text's own face at the body text's sizes (see
`jlens/plotstyle.py`), selective direct labels rather than all 28.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from jlens import plotstyle as style
from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json

# lens.tex places this at 0.8\linewidth; drawing it at that size is what
# keeps 9pt in the figure equal to 9pt on the page.
FIGSIZE = (style.BODY_WIDTH, 2.85)


def sci(value, digits=1):
    """`1.4e-04` as maths the body text would set the same way."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d = read_json(f"{ROOT}/results/paper_figure_data.json")
    rels = d["rels"]
    y = np.array(d["faith"])
    lens = np.array(d["lens"])
    x = -lens  # agreement: higher is better
    YR = {"presidents_birth_year", "presidents_election_year"}
    # Two labels only.  More than this collides; the rest are in Table 1.
    # Leader lines make the association unambiguous.
    LABEL = {
        "person_father": ("person_father /\nperson_mother", -3.15, 0.235, "left"),
        "country_capital_city": ("country_capital", -1.45, 0.34, "left"),
    }
    style.use_paper_style()
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

    # trend line on ranks is what the statistic uses; show an OLS guide on
    # the raw values
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 0.15, x.max() + 0.15, 50)
    ax.plot(xs, a + b * xs, lw=1.0, color=style.FIT, zorder=1)

    isyr = np.array([r in YR for r in rels])
    ax.scatter(
        x[~isyr],
        y[~isyr],
        s=20,
        facecolor=style.HIT,
        edgecolor=style.EDGE,
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        x[isyr],
        y[isyr],
        s=30,
        marker="^",
        facecolor=style.ACCENT,
        edgecolor=style.INK,
        linewidth=0.7,
        zorder=4,
        label="year objects (rescored)",
    )
    ax.legend(loc="lower right", fontsize=style.TINY_PT)

    for i, r in enumerate(rels):
        if r in LABEL:
            txt, tx, ty, ha = LABEL[r]
            ax.annotate(
                txt,
                (x[i], y[i]),
                (tx, ty),
                fontsize=style.TINY_PT,
                color=style.INK,
                ha=ha,
                va="center",
                zorder=5,
                linespacing=1.15,
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.45,
                    color=style.MUTED,
                    shrinkA=1.5,
                    shrinkB=3.5,
                ),
            )

    rho = stats.spearmanr(x, y)
    ax.text(
        0.035,
        0.965,
        rf"$\rho = {rho.statistic:.2f}$   $p = {sci(rho.pvalue)}$",
        transform=ax.transAxes,
        fontsize=style.SCRIPT_PT,
        va="top",
        color=style.INK,
    )
    ax.set_xlabel("lens agreement  ($-\\log_{10}$ rank of the model's top-1)")
    ax.set_ylabel("LRE faithfulness")
    ax.grid(True, zorder=0)
    ax.set_ylim(-0.04, 1.08)
    style.save(fig, f"{OUTPUT_DIR}/relation_agreement", png=True)
    plt.close(fig)
    print(f"wrote {OUTPUT_DIR}/relation_agreement.{{pdf,png}}")
    print(
        f"rho={rho.statistic:+.3f} p={rho.pvalue:.2e}  "
        f"x range {x.min():.2f}..{x.max():.2f}"
    )


if __name__ == "__main__":
    main()
