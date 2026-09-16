"""Plot the product relation and three outlier relations from matched subject ranks."""

import matplotlib

from jlens.config import OUTPUT_DIR
from jlens.paper_statistics import fact_points

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BLUE = "#3b6ea5"
ORANGE = "#c2603f"
GREY = "#9a9a9a"
RELATIONS = (
    "superhero_archnemesis",
    "person_occupation",
    "person_university",
)


def draw_relation_panel(ax, relation):
    points = fact_points(relation, "shared_split/jacobian_lens.json", "shared_split")
    x, y = points.T
    miss = y > 0
    ax.scatter(x[~miss], y[~miss], s=15, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.scatter(x[miss], y[miss], s=15, color=ORANGE, edgecolor="white", linewidth=0.3)
    xs = np.linspace(x.min(), x.max(), 50)
    all_fit = np.polyfit(x, y, 1)
    ax.plot(xs, np.polyval(all_fit, xs), color=GREY, linewidth=1)
    if miss.sum() >= 3:
        miss_fit = np.polyfit(x[miss], y[miss], 1)
        ax.plot(
            xs,
            np.polyval(miss_fit, xs),
            color=ORANGE,
            linewidth=1,
            linestyle=(0, (4, 3)),
        )
    rho = stats.spearmanr(x, y).statistic
    miss_rho = stats.spearmanr(x[miss], y[miss]).statistic
    ax.set_title(
        relation.replace("_", " ")
        + f"\nn={len(x)}, faith={1 - miss.mean():.2f}, rho={rho:+.2f}"
        + f"\nmisses: n={miss.sum()}, rho={miss_rho:+.2f}",
        fontsize=9,
    )
    ax.set_xlabel("lens log10 rank", fontsize=9)
    ax.set_ylabel("LRE log10 rank", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "serif"})

    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=180)
    draw_relation_panel(ax, "product_by_company")
    fig.tight_layout()
    fig.savefig(out / "product_fact_ranks.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), dpi=180)
    for ax, relation in zip(axes, RELATIONS):
        draw_relation_panel(ax, relation)
    fig.tight_layout()
    fig.savefig(out / "outlier_fact_ranks.png")
    plt.close(fig)
    print(
        f"wrote {out / 'product_fact_ranks.png'} and {out / 'outlier_fact_ranks.png'}"
    )


if __name__ == "__main__":
    main()
