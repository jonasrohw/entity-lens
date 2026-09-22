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
    fig, ax = plt.subplots(figsize=(7.8, 5.4), layout="constrained")
    xs = np.linspace(x.min() - 0.12, x.max() + 0.12, 100)
    ax.plot(xs, intercept + slope * xs, color=style.FIT, lw=1, zorder=1)
    ax.scatter(x, y, s=32, facecolor=style.HIT, edgecolor=style.EDGE,
               linewidth=0.5, zorder=3)
    # Descriptive labeling rule, not a statistical outlier test. Explicit
    # positions keep the frozen figure readable without extra dependencies.
    outliers = {names[i] for i in range(len(names))
                if abs(y[i] - (intercept + slope * x[i])) >= 0.20}
    labels = {
        "name_gender": (-4.80, 1.07),
        "landmark_on_continent": (-3.85, 1.035),
        "name_religion": (-2.15, 1.075),
        "person_native_language": (-4.95, 0.885),
        "person_plays_instrument": (-2.95, 0.91),
        "person_plays_position_in_sport": (-4.95, 0.69),
        "person_father": (-4.95, 0.17),
        "person_mother": (-3.80, 0.14),
        "company_ceo": (-3.80, 0.025),
        "superhero_archnemesis": (-3.55, 0.225),
        "pokemon_evolutions": (-3.55, 0.325),
        "superhero_person": (-2.60, 0.16),
        "country_capital_city": (-1.80, 0.49),
    }
    assert set(labels) == outliers
    for name, position in labels.items():
        i = names.index(name)
        ax.annotate(name, (x[i], y[i]), position, fontsize=style.BODY_PT,
                    va="center", color=style.INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.96),
                    arrowprops=dict(arrowstyle="-", lw=0.65, color=style.MUTED,
                                    shrinkA=2.5, shrinkB=4.5))
    ax.set_title("Qwen3 8B Base · newline prompt\n26 relations · only year relations excluded")
    ax.text(0.97, 0.04, rf"$\rho = {rho:.4f}$   $p = {p:.4f}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=style.BODY_PT)
    ax.set_xlabel("Representation score " + r"($-\mathrm{median}\,\log_{10}(\mathrm{rank}+1)$)",
                  fontsize=style.BODY_PT)
    ax.set_ylabel("LRE faithfulness", fontsize=style.BODY_PT)
    ax.set_xlim(-5.05, -0.35)
    ax.set_ylim(-0.04, 1.13)
    ax.grid(True, zorder=0)
    style.save(fig, HERE / "relation_agreement", png=True)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
