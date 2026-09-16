"""Supplementary domain-stratified correlations on the retained 28 relations.

This repository contains the named-entity subset of the upstream expansion.
Both domain groups therefore cover the same retained inputs."""

import glob

import numpy as np
from scipy import stats

from jlens.config import ROOT
from jlens.figure_data import faithfulness_scores
from jlens.io import read_json

ENTITY_DOMAINS = {
    "person",
    "country",
    "name",
    "company",
    "landmark",
    "superhero",
    "city",
    "pokemon",
    "product",
    "star",
}  # primary, pre-registered
NOUN_DOMAINS = {"fruit", "food", "object", "substance"}  # secondary


def relation_domain(r):
    for f in glob.glob(f"{ROOT}/data/relations/{r}.json"):
        return read_json(f)["properties"].get("domain_name", "")
    return ""


def object_cardinality(r):
    for f in glob.glob(f"{ROOT}/data/relations/{r}.json"):
        S = read_json(f)["samples"]
        return len(S) / len(set(x["object"] for x in S))
    return float("nan")


rk = stats.rankdata


def partial_correlation(a, b, ctrl):
    A, B = rk(a), rk(b)
    C = np.c_[np.ones(len(A)), *[rk(c) for c in ctrl]]
    ra = A - C @ np.linalg.lstsq(C, A, rcond=None)[0]
    rb = B - C @ np.linalg.lstsq(C, B, rcond=None)[0]
    return stats.pearsonr(ra, rb)


def report_domain_group(sub, label, faithfulness_results, lens_results):
    if len(sub) < 5:
        print(f"\n=== {label}: only {len(sub)}, skipped")
        return
    y = np.array([faithfulness_results[r] for r in sub])
    x = np.array([-lens_results[r]["agg"]["diff"]["median_lograng"] for r in sub])
    K = np.array([object_cardinality(r) for r in sub])
    s = stats.spearmanr(x, y)
    rng = np.random.default_rng(17)
    null = np.mean(
        [
            abs(stats.spearmanr(rng.permutation(x), y).statistic) >= abs(s.statistic)
            for _ in range(10000)
        ]
    )
    print(f"\n=== {label}   n={len(sub)} ===")
    print(
        f"  Spearman(lens, faith)   {s.statistic:+.3f}  p={s.pvalue:.4f}   perm-null {null:.4f}"
    )
    pr = partial_correlation(x, y, [K])
    print(f"  partial | k             {pr.statistic:+.3f}  p={pr.pvalue:.4f}")
    print(
        f"  faith range {y.min():.2f}-{y.max():.2f}; at ceiling(1.0): {(y >= 0.999).sum()}/{len(y)}"
    )
    ts = sorted(set(relation_domain(r) for r in sub))
    lo = []
    for t in ts:
        k = [i for i, r in enumerate(sub) if relation_domain(r) != t]
        if len(k) > 4:
            lo.append(stats.spearmanr(x[k], y[k]).statistic)
    if lo:
        print(
            f"  leave-one-type-out      min {min(lo):+.3f} max {max(lo):+.3f} median {np.median(lo):+.3f}"
        )


def main():
    faithfulness_results = faithfulness_scores(ROOT)
    lens_results = read_json(f"{ROOT}/results/jacobian_lens.json")
    rels = sorted(
        [r for r in lens_results if not r.startswith("_") and r in faithfulness_results]
    )

    print(
        f"faithfulness (uniform): {len(faithfulness_results)} | lens: {len(rels)} usable"
    )
    print(f"\n{'relation':34s} {'dom':11s} {'faith':>6} {'lens':>6} {'k':>7}")
    for r in sorted(rels, key=lambda r: faithfulness_results[r]):
        print(
            f"  {r:32s} {relation_domain(r):11s} {faithfulness_results[r]:>6.2f} "
            f"{lens_results[r]['agg']['diff']['median_lograng']:>6.2f} {object_cardinality(r):>7.1f}"
        )
    report_domain_group(
        [r for r in rels if relation_domain(r) in ENTITY_DOMAINS],
        "PRIMARY (named-entity subjects)",
        faithfulness_results,
        lens_results,
    )
    report_domain_group(
        [r for r in rels if relation_domain(r) in ENTITY_DOMAINS | NOUN_DOMAINS],
        "SECONDARY (+ concrete nouns)",
        faithfulness_results,
        lens_results,
    )


if __name__ == "__main__":
    main()
