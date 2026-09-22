"""Export the domain-level results and paired-difference confidence intervals."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/anchor_swap_gemini"


def main():
    r = json.loads((OUT / "summary.json").read_text())
    names = list(r["domains"])
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    colors = ["#235789", "#d1495b", "#2a9d8f", "#7b7b7b"]
    conditions = ["own", "foreign", "neutral", "anchor"]
    labels = ["Own subject term", "Other-relation term", "Neutral lens", "Anchor alone"]
    for j, (condition, label, color) in enumerate(zip(conditions, labels, colors)):
        vals = []
        lo = []
        hi = []
        for name in names:
            s = r["domains"][name]["subsets"]["primary/all"]["rates"][condition]
            vals.append(s["mean"] * 100)
            lo.append((s["mean"] - s["ci95"][0]) * 100)
            hi.append((s["ci95"][1] - s["mean"]) * 100)
        axes[0].errorbar(
            vals,
            np.arange(len(names)) + (j - 1.5) * 0.14,
            xerr=[lo, hi],
            fmt="o",
            color=color,
            label=label,
            markersize=5,
            capsize=2,
        )
    axes[0].set_yticks(
        range(len(names)),
        [f"{n.capitalize()} (n={r['domains'][n]['n_test']})" for n in names],
    )
    axes[0].set_xlabel("First-token faithfulness (%)")
    axes[0].set_xlim(-2, 102)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    axes[0].set_title(
        "Held-out subjects; both relation directions averaged", fontsize=11
    )
    names2 = names + ["macro"]
    vals = []
    lo = []
    hi = []
    for name in names2:
        s = (
            r["macro"]["primary/all"]
            if name == "macro"
            else r["domains"][name]["subsets"]["primary/all"]
        )
        d = s["contrasts"]["foreign_minus_own"]
        vals.append(d["mean"] * 100)
        lo.append((d["mean"] - d["ci95"][0]) * 100)
        hi.append((d["ci95"][1] - d["mean"]) * 100)
    axes[1].axvspan(
        -10, 10, color="#dddddd", alpha=0.55, label="±10 pp equivalence margin"
    )
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].errorbar(
        vals, range(len(names2)), xerr=[lo, hi], fmt="o", color="#d1495b", capsize=3
    )
    axes[1].set_yticks(range(len(names2)), [n.capitalize() for n in names2])
    axes[1].set_xlabel("Foreign minus own (percentage points)")
    axes[1].set_title("Paired subject-bootstrap 95% intervals", fontsize=11)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9
    )
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle(
        "Does the anchor preserve the answer when the subject term is swapped?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"anchor_swap_results.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
