"""Plot the product relation and three outlier relations from matched subject ranks.

One panel is one relation: the lens's log rank of the model's answer against
the LRE's log rank of the same answer, one point per subject.  Points on the
y = 0 floor are the subjects the LRE reproduced -- a censoring floor, not a
value -- so the panel carries two fits: one over all points, which is partly
tracing that floor, and one over the misses alone, which is not.

Colour and type come from `jlens/plotstyle.py`; the two series differ in hue
and the two fits in dash pattern, so the panels survive greyscale printing.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from jlens import plotstyle as style
from jlens.config import OUTPUT_DIR
from jlens.paper_statistics import fact_points

RELATIONS = (
    "superhero_archnemesis",
    "person_occupation",
    "person_university",
)

XLABEL = r"lens $\log_{10}(\mathrm{rank}+1)$"
YLABEL = r"LRE $\log_{10}(\mathrm{rank}+1)$"

HIT_LABEL = "LRE reproduced the answer (rank 0)"
MISS_LABEL = "LRE missed it"
FIT_LABEL = "fit, all subjects"
FIT_MISS_LABEL = "fit, misses only"


def draw_relation_panel(ax, relation, *, title_size=None, label_size=None, labels=True):
    """Draw one relation and return the statistics the caption quotes.

    `labels=False` leaves the axis labels off, for a small-multiples figure
    that carries one shared pair instead of repeating them per panel.
    """
    points = fact_points(relation, "shared_split/jacobian_lens.json", "shared_split")
    x, y = points.T
    miss = y > 0

    # The censoring floor the hits pile up on, drawn once and recessively so
    # the eye reads it as an axis feature rather than as data.
    ax.axhline(0, color=style.GRID, lw=0.6, zorder=0)

    ax.scatter(
        x[~miss],
        y[~miss],
        s=13,
        color=style.HIT,
        edgecolor=style.EDGE,
        linewidth=0.3,
        zorder=3,
        label=HIT_LABEL,
    )
    ax.scatter(
        x[miss],
        y[miss],
        s=13,
        color=style.MISS,
        edgecolor=style.EDGE,
        linewidth=0.3,
        zorder=4,
        label=MISS_LABEL,
    )

    xs = np.linspace(x.min(), x.max(), 50)
    all_fit = np.polyfit(x, y, 1)
    ax.plot(
        xs,
        np.polyval(all_fit, xs),
        color=style.FIT,
        linewidth=1.0,
        zorder=1,
        label=FIT_LABEL,
    )
    if miss.sum() >= 3:
        miss_fit = np.polyfit(x[miss], y[miss], 1)
        ax.plot(
            xs,
            np.polyval(miss_fit, xs),
            color=style.FIT_MISS,
            linewidth=1.0,
            dashes=(3.5, 2.5),
            zorder=2,
            label=FIT_MISS_LABEL,
        )

    stat = dict(
        relation=relation,
        n=len(x),
        faith=1 - miss.mean(),
        rho=stats.spearmanr(x, y).statistic,
        n_miss=int(miss.sum()),
        rho_miss=stats.spearmanr(x[miss], y[miss]).statistic,
    )
    ax.set_title(panel_title(stat), fontsize=title_size or style.SMALL_PT)
    if labels:
        ax.set_xlabel(XLABEL, fontsize=label_size or style.SMALL_PT)
        ax.set_ylabel(YLABEL, fontsize=label_size or style.SMALL_PT)
    return stat


def panel_title(stat, lines=3):
    """Relation name plus the numbers the text quotes, in `lines` lines."""
    name = stat["relation"].replace("_", " ")
    # One maths group per clause: mixing text and maths around "=" makes
    # mathtext add relation spacing on one side only, which reads as a typo.
    counts = rf"$n = {stat['n']}$;  $\mathrm{{faith}} = {stat['faith']:.2f}$"
    rhos = (
        rf"$\rho = {stat['rho']:+.2f}$;  "
        rf"$\rho_{{\mathrm{{miss}}}} = {stat['rho_miss']:+.2f}$"
    )
    misses = rf"$n_{{\mathrm{{miss}}}} = {stat['n_miss']}$"
    if lines >= 4:
        return "\n".join([name, counts, rhos, misses])
    return "\n".join([name, counts, f"{rhos}; {misses}"])


def shared_axis_labels(axes, size=None):
    """One x label and one y label for a row of panels that share both scales.

    Put on the middle and leftmost axes rather than on the figure: a figure
    level label and an ``outside lower center`` legend both claim the bottom
    strip, and constrained layout lets them overlap.
    """
    size = size or style.SCRIPT_PT
    axes[len(axes) // 2].set_xlabel(XLABEL, fontsize=size)
    axes[0].set_ylabel(YLABEL, fontsize=size)


def add_series_legend(fig, ncol=4):
    """One shared legend for the whole figure, below the panels."""
    handles, labels = style.collect_handles(fig.axes)
    return fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=ncol,
        fontsize=style.TINY_PT,
        columnspacing=1.4,
        handlelength=1.8,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style.use_paper_style()

    fig, ax = plt.subplots(figsize=(style.BODY_WIDTH, 3.0), layout="constrained")
    draw_relation_panel(ax, "product_by_company")
    add_series_legend(fig, ncol=2)
    style.save(fig, OUTPUT_DIR / "product_fact_ranks", png=True)
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(style.FULL_WIDTH, 2.95),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for ax, relation in zip(axes, RELATIONS):
        draw_relation_panel(ax, relation, title_size=style.SCRIPT_PT, labels=False)
    shared_axis_labels(axes)
    add_series_legend(fig)
    style.save(fig, OUTPUT_DIR / "outlier_fact_ranks", png=True)
    plt.close(fig)
    print(
        f"wrote {OUTPUT_DIR / 'product_fact_ranks.pdf'} and "
        f"{OUTPUT_DIR / 'outlier_fact_ranks.pdf'}"
    )


if __name__ == "__main__":
    main()
