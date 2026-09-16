"""Rank relations by how far they sit from the trend, in both figures.

Figure 1 plots lens agreement ($-\\log_{10}$ median rank, higher better)
against LRE faithfulness (higher better).  Figure 2 plots mean lens
log-rank against mean LRE log-rank (both lower better).  **The y axes run
in opposite directions**, so a raw residual means opposite things in the
two plots.  Everything here is therefore reported on one convention:

    positive residual = the LRE does BETTER than its lens score predicts
    negative residual = the LRE does WORSE than its lens score predicts

which for Figure 2 requires negating the raw residual.

Residuals are from an OLS fit on the plotted values, matching the
convention already used in the paper's residual discussion.  Because the
reported statistic is Spearman, a fit on ranks is also computed; the two
orderings are reported side by side so a conclusion that depends on which
was used is visible as such.
"""

import argparse
import glob

import numpy as np
from scipy import stats

from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json


def regression_residuals(x, y):
    """OLS residuals on the values, and on the ranks."""
    b, a = np.polyfit(x, y, 1)
    raw = y - (a + b * x)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    b2, a2 = np.polyfit(rx, ry, 1)
    return raw, ry - (a2 + b2 * rx)


def agreement_residuals():
    d = read_json(f"{ROOT}/results/paper_figure_data.json")
    rels = d["rels"]
    x = -np.array(d["lens"])  # agreement, higher better
    y = np.array(d["faith"])  # higher better
    raw, rnk = regression_residuals(x, y)
    return rels, x, y, raw, rnk  # sign already aligned


def rank_residuals(lens_path, rank_dir):
    lens_results = read_json(lens_path)
    rels, xs, ys = [], [], []
    for p in sorted(glob.glob(f"{rank_dir}/subject_ranks_*.json")):
        rank_results = read_json(p)
        relation = rank_results["relation"]
        if relation not in lens_results:
            continue
        if not rank_results.get("split_fingerprint") or rank_results[
            "split_fingerprint"
        ] != lens_results[relation].get("split_fingerprint"):
            raise ValueError(f"{relation}: split fingerprints differ or are missing")
        lens = {r["subject"]: r["rank_diff"] for r in lens_results[relation]["rows"]}
        a = [np.log10(lens[s] + 1) for s in rank_results["ranks"] if s in lens]
        b = [
            np.log10(v["rank"] + 1)
            for s, v in rank_results["ranks"].items()
            if s in lens
        ]
        if len(a) < 3:
            continue
        rels.append(relation)
        xs.append(np.mean(a))
        ys.append(np.mean(b))
    x, y = np.array(xs), np.array(ys)
    raw, rnk = regression_residuals(x, y)
    return rels, x, y, -raw, -rnk  # negate: lower LRE rank is better


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", default=None)
    parser.add_argument("--rank-dir", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lens_path = args.lens or f"{ROOT}/results/shared_split/jacobian_lens.json"
    rank_dir = args.rank_dir or f"{ROOT}/results/shared_split"
    json_path = f"{OUTPUT_DIR}/relation_outliers{args.tag}.json"
    markdown_path = f"{OUTPUT_DIR}/relation_outliers{args.tag}.md"
    out = {}
    for name, (rels, x, y, raw, rnk) in (
        ("figure1", agreement_residuals()),
        ("figure2", rank_residuals(lens_path, rank_dir)),
    ):
        order = np.argsort(-np.abs(raw))
        out[name] = {
            "n_relations": len(rels),
            "sign_convention": "positive = LRE does better than its lens "
            "score predicts",
            "spearman_x_y": float(
                stats.spearmanr(x, y).statistic * (1 if name == "figure1" else -1)
            ),
            "ranking": [
                {
                    "rank": int(i + 1),
                    "relation": rels[j],
                    "residual": float(raw[j]),
                    "abs_residual": float(abs(raw[j])),
                    "rank_space_residual": float(rnk[j]),
                    "x": float(x[j]),
                    "y": float(y[j]),
                }
                for i, j in enumerate(order)
            ],
        }
    # do the two figures agree about who the outliers are?
    r1 = {e["relation"]: e["residual"] for e in out["figure1"]["ranking"]}
    r2 = {e["relation"]: e["residual"] for e in out["figure2"]["ranking"]}
    both = sorted(set(r1) & set(r2))
    a = np.array([r1[k] for k in both])
    b = np.array([r2[k] for k in both])
    agree = {
        "n_shared": len(both),
        "spearman_residuals": float(stats.spearmanr(a, b).statistic),
        "spearman_abs_residuals": float(
            stats.spearmanr(np.abs(a), np.abs(b)).statistic
        ),
    }
    out["agreement"] = agree
    write_json(json_path, out)

    lens_results = [
        "<!-- generated by jlens/rank_outliers.py; do not hand-edit -->",
        "",
        "# Relations ranked by distance from the trend",
        "",
        "Positive residual: the LRE does **better** than its lens score",
        "predicts. Negative: **worse**. Figure 2's raw residual is negated to",
        "match Figure 1, whose y axis runs the other way.",
        "",
        f"Figures agree on the residuals at Spearman "
        f"**{agree['spearman_residuals']:+.3f}** over {agree['n_shared']} shared "
        f"relations ({agree['spearman_abs_residuals']:+.3f} on absolute values).",
        "",
    ]
    for name, title in (
        ("figure1", "Figure 1 — lens agreement vs faithfulness"),
        ("figure2", "Figure 2 — mean lens log-rank vs mean LRE log-rank"),
    ):
        d = out[name]
        lens_results += [
            f"## {title}",
            "",
            f"{d['n_relations']} relations.",
            "",
            "| # | relation | residual | rank-space resid | x | y |",
            "|---|---|---|---|---|---|",
        ]
        for e in d["ranking"]:
            lens_results.append(
                f"| {e['rank']} | `{e['relation']}` | {e['residual']:+.3f} | "
                f"{e['rank_space_residual']:+.1f} | {e['x']:.2f} | {e['y']:.2f} |"
            )
        lens_results.append("")
    open(markdown_path, "w").write("\n".join(lens_results))

    print(f"wrote {json_path}\nwrote {markdown_path}\n")
    for name in ("figure1", "figure2"):
        d = out[name]
        print(f"{name}: {d['n_relations']} relations, top 6 by |residual|")
        for e in d["ranking"][:6]:
            print(f"   {e['relation']:32s} {e['residual']:+.3f}")
        print()
    print(
        f"agreement between the two rankings: "
        f"rho={agree['spearman_residuals']:+.3f} on signed, "
        f"{agree['spearman_abs_residuals']:+.3f} on absolute "
        f"(n={agree['n_shared']})"
    )


if __name__ == "__main__":
    main()
