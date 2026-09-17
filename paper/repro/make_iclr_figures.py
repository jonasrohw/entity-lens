"""Render the retained measurements for the ICLR single-column layout.

Each figure is drawn at the width `lens.tex` places it at -- `\\linewidth`
(5.5in) or `0.8\\linewidth` (4.4in) -- and saved without a tight bounding box,
so LaTeX scales it by exactly 1.0 and a 9pt label in a figure is a 9pt label on
the page.  Style, colour and PDF settings are in `jlens/plotstyle.py`.

Run with the locked code environment: uv run python ../paper/repro/make_iclr_figures.py
"""

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER.parent / "code"))


def main():
    from jlens import plot_relation_agreement, plotstyle as style
    from jlens.plot_fact_ranks import (
        RELATIONS,
        add_series_legend,
        draw_relation_panel,
        panel_title,
        shared_axis_labels,
    )

    style.use_paper_style()
    import matplotlib.pyplot as plt

    output = PAPER / "build" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    # Figure 1: lens agreement against LRE faithfulness, 28 relations.
    plot_relation_agreement.OUTPUT_DIR = output
    plot_relation_agreement.main()
    (PAPER / "fig_iclr_agreement.pdf").write_bytes(
        (output / "relation_agreement.pdf").read_bytes()
    )
    plt.close("all")

    # Figure 2: the rank-rank panel for product_by_company.
    fig, ax = plt.subplots(figsize=(style.BODY_WIDTH, 3.0), layout="constrained")
    draw_relation_panel(ax, "product_by_company")
    add_series_legend(fig, ncol=2)
    style.save(fig, PAPER / "fig_iclr_product")
    plt.close(fig)

    # Figure 3: the three worst-correlating relations, at full text width.
    # Small multiples, so the panels share both scales: the reader is meant to
    # compare slopes across them, which per-panel limits would misrepresent.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(style.FULL_WIDTH, 2.95),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for ax, relation in zip(axes, RELATIONS):
        stat = draw_relation_panel(
            ax, relation, title_size=style.SCRIPT_PT, labels=False
        )
        # Four lines: at 1.8in per panel the three-line form runs past the axes.
        ax.set_title(panel_title(stat, lines=4), fontsize=style.SCRIPT_PT)
    shared_axis_labels(axes)
    add_series_legend(fig)
    style.save(fig, PAPER / "fig_iclr_outliers")
    plt.close(fig)
    print("Wrote the three fig_iclr_*.pdf files from retained measurements.")


if __name__ == "__main__":
    main()
