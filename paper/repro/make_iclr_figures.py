"""Render the retained measurements for the ICLR single-column layout.

Run with the locked code environment: uv run python ../paper/repro/make_iclr_figures.py
"""

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER.parent / "code"))


def main():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt
    from jlens import plot_relation_agreement
    from jlens.plot_fact_ranks import RELATIONS, draw_relation_panel

    output = PAPER / "build" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_relation_agreement.OUTPUT_DIR = output
    plot_relation_agreement.main()
    (PAPER / "fig_iclr_agreement.pdf").write_bytes(
        (output / "relation_agreement.pdf").read_bytes()
    )
    plt.close("all")

    fig, ax = plt.subplots(figsize=(4.4, 3.0), layout="constrained")
    draw_relation_panel(ax, "product_by_company")
    ax.set_xlabel(r"lens $\log_{10}(\mathrm{rank}+1)$", fontsize=9)
    ax.set_ylabel(r"LRE $\log_{10}(\mathrm{rank}+1)$", fontsize=9)
    fig.savefig(PAPER / "fig_iclr_product.pdf", metadata={"Author": ""})
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.55), layout="constrained")
    for ax, relation in zip(axes, RELATIONS):
        draw_relation_panel(ax, relation)
        title, overall, misses = ax.get_title().splitlines()
        count, faith, rho = overall.split(", ")
        miss_count, miss_rho = misses.removeprefix("misses: ").split(", ")
        ax.set_title(
            f"{title}\n{count}; {faith}\n"
            rf"$\rho={rho.split('=')[1]}$; $\rho_{{\rm miss}}={miss_rho.split('=')[1]}$"
            "\n" + rf"$n_{{\rm miss}}={miss_count.split('=')[1]}$",
            fontsize=7.5,
        )
        ax.set_xlabel(r"lens $\log_{10}(\mathrm{rank}+1)$", fontsize=8)
        ax.set_ylabel(r"LRE $\log_{10}(\mathrm{rank}+1)$", fontsize=8)
    fig.savefig(
        PAPER / "fig_iclr_outliers.pdf", bbox_inches="tight", metadata={"Author": ""}
    )
    plt.close(fig)
    print("Wrote the three fig_iclr_*.pdf files from retained measurements.")


if __name__ == "__main__":
    main()
