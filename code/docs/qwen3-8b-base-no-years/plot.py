"""Rebuild the Qwen3 8B Base Figure 1 from the accompanying frozen values."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from jlens import plotstyle as style
import matplotlib.pyplot as plt


def main():
    with (HERE / "relation_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    names = [r["relation"] for r in rows]
    assert len(names) == len(set(names)) == 26
    assert "name_gender" in names
    assert not {"presidents_birth_year", "presidents_election_year"} & set(names)
    assert all(
        (r["model"], r["protocol"], r["version"], r["kind"])
        == ("qwen3_8b_base", "main", "newline", "diff")
        for r in rows
    )
    x = np.array([float(r["score"]) for r in rows])
    y = np.array([float(r["faithfulness"]) for r in rows])
    rho, p = spearmanr(x, y)
    pearson, pearson_p = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    summary = dict(
        model="Qwen/Qwen3-8B-Base",
        protocol="main",
        neutral_prompt="\n{} is known for",
        lens_layer=19,
        excluded_relations=["presidents_birth_year", "presidents_election_year"],
        n_relations=len(rows),
        n_lens=sum(int(r["n_lens"]) for r in rows),
        spearman_rho=float(rho),
        spearman_p=float(p),
        pearson_r=float(pearson),
        pearson_p=float(pearson_p),
        ols_slope=float(slope),
        ols_intercept=float(intercept),
    )
    (HERE / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")

    style.use_paper_style()
    fig, ax = plt.subplots(figsize=(style.BODY_WIDTH, 3.25), layout="constrained")
    xs = np.linspace(x.min() - 0.12, x.max() + 0.12, 100)
    ax.plot(xs, intercept + slope * xs, color=style.FIT, lw=1, zorder=1)
    ax.scatter(x, y, s=20, facecolor=style.HIT, edgecolor=style.EDGE,
               linewidth=0.5, zorder=3)
    labels = {
        "name_gender": ("name_gender", (-3.95, 1.025)),
        "company_ceo": ("company_ceo", (-3.75, 0.06)),
        "country_capital_city": ("country_capital", (-1.60, 0.46)),
    }
    for name, (label, position) in labels.items():
        i = names.index(name)
        ax.annotate(label, (x[i], y[i]), position, fontsize=style.TINY_PT,
                    va="center", color=style.INK,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=style.MUTED,
                                    shrinkA=1.5, shrinkB=3.5))
    ax.set_title("Qwen3 8B Base · newline prompt\n26 relations · only year relations excluded")
    ax.text(0.97, 0.04, rf"$\rho = {rho:.4f}$   $p = {p:.4f}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.SCRIPT_PT)
    ax.set_xlabel("Representation score " + r"($-\mathrm{median}\,\log_{10}(\mathrm{rank}+1)$)")
    ax.set_ylabel("LRE faithfulness")
    ax.set_ylim(-0.04, 1.10)
    ax.grid(True, zorder=0)
    style.save(fig, HERE / "relation_agreement", png=True)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
