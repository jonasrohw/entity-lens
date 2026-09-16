"""Test whether lens ranks separate LRE hits from misses within each relation.

Check shared split fingerprints, compute per-relation AUC, combine with a
Wilcoxon signed-rank test, and permute outcomes within each relation."""

import argparse
import glob
import sys

import numpy as np
from scipy import stats

from jlens.config import ROOT
from jlens.io import read_json


def load_subject_outcomes(lens_path, faith_dir, min_n, lo, hi):
    lens_results = read_json(lens_path)
    out, skipped = {}, []
    for relation, lens_result in lens_results.items():
        if relation.startswith("_"):
            continue
        split_fingerprint = f"{faith_dir}/faithfulness_{relation}.json"
        if not glob.glob(split_fingerprint):
            skipped.append((relation, "no faithfulness run"))
            continue
        faithfulness = read_json(split_fingerprint)
        if faithfulness.get("split_fingerprint") != lens_result.get(
            "split_fingerprint"
        ):
            sys.exit(
                f"FATAL {relation}: split fingerprints differ "
                f"({lens_result.get('split_fingerprint')} vs {faithfulness.get('split_fingerprint')}). "
                f"The two measurements did not score the same subjects."
            )
        ps = faithfulness.get("per_subject") or {}
        rows = [
            (r["subject"], -np.log10(r["rank_diff"] + 1), ps[r["subject"]])
            for r in lens_result["rows"]
            if r["subject"] in ps
        ]
        if len(rows) < min_n:
            skipped.append((relation, f"n={len(rows)} < {min_n}"))
            continue
        y = np.array([r[2] for r in rows])
        if y.sum() == 0 or y.sum() == len(y):
            skipped.append(
                (relation, f"no within-relation variance ({y.sum()}/{len(y)})")
            )
            continue
        f = faithfulness["best_faith"]
        if not (lo <= f <= hi):
            skipped.append((relation, f"faith {f:.2f} outside [{lo},{hi}]"))
            continue
        out[relation] = (np.array([r[1] for r in rows]), y, f)
    return out, skipped


def outcome_auc(x, y):
    u = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
    return u.statistic / (y.sum() * (len(y) - y.sum())), u.pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens", default=f"{ROOT}/results/shared_split/jacobian_lens.json"
    )
    parser.add_argument("--faith-dir", default=f"{ROOT}/results/shared_split")
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--faith-lo", type=float, default=0.10)
    parser.add_argument("--faith-hi", type=float, default=0.90)
    args = parser.parse_args()

    relation_results, skipped = load_subject_outcomes(
        args.lens, args.faith_dir, args.min_n, args.faith_lo, args.faith_hi
    )
    print(f"usable relations: {len(relation_results)}   skipped: {len(skipped)}")
    for relation, why in skipped:
        print(f"    - {relation:32s} {why}")
    if not relation_results:
        sys.exit("nothing to analyse yet")

    print(f"\n{'relation':32s} {'n':>4} {'hits':>5} {'AUC':>6} {'p':>8}")
    aucs, rows_total = [], 0
    for relation, (x, y, f) in sorted(relation_results.items()):
        A, p = outcome_auc(x, y)
        aucs.append(A)
        rows_total += len(y)
        print(f"  {relation:30s} {len(y):>4} {int(y.sum()):>5} {A:>6.3f} {p:>8.3f}")
    aucs = np.array(aucs)

    print(f"\ntotal subject-level rows: {rows_total}")
    print(
        f"median AUC {np.median(aucs):.3f}   mean {aucs.mean():.3f}   "
        f"above 0.5: {(aucs > 0.5).sum()}/{len(aucs)}"
    )
    w = stats.wilcoxon(aucs - 0.5, alternative="greater")
    print(f"Wilcoxon signed-rank vs 0.5: statistic={w.statistic:.1f}  p={w.pvalue:.4f}")

    # shuffle control: permute outcomes within each relation
    rng = np.random.default_rng(17)
    null = []
    for _ in range(2000):
        null.append(
            np.median(
                [
                    outcome_auc(x, rng.permutation(y))[0]
                    for x, y, _ in relation_results.values()
                ]
            )
        )
    null = np.array(null)
    print(
        f"shuffle control: null median AUC {null.mean():.3f} "
        f"(should be ~0.500); observed >= null in "
        f"{np.mean(null >= np.median(aucs)):.4f} of 2000"
    )


if __name__ == "__main__":
    main()
