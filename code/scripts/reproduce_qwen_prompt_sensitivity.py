"""Verify the compact prompt-sensitivity snapshot using only NumPy and SciPy."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


def bootstrap_rho(x, y, indices):
    a = rankdata(x[indices], axis=1)
    b = rankdata(y[indices], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.divide(
        (a * b).sum(1),
        denominator,
        out=np.full(len(a), np.nan),
        where=denominator > 0,
    )


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/qwen_prompt_sensitivity",
    )
    args = parser.parse_args()
    points = read_csv(args.root / "relation_scores.csv")
    expected = read_csv(args.root / "correlations.csv")
    assert len(points) == 371
    assert len(expected) == 28
    grouped = {}
    for row in points:
        group = grouped.setdefault((row["protocol"], row["prompt"]), {})
        assert row["relation"] not in group, "Duplicate relation point"
        group[row["relation"]] = row
    assert len(grouped) == 14
    for row in expected:
        protocol, prompt, cohort = row["protocol"], row["prompt"], row["cohort"]
        base = grouped[protocol, "known_for"]
        values = grouped[protocol, prompt]
        assert set(values) == set(base)
        for name in base:
            for key in ("faithfulness", "n_lens", "n_lre"):
                assert values[name][key] == base[name][key]
        assert cohort in {"all", "at_least_4_lre_tests"}
        names = sorted(
            name for name in base if cohort == "all" or int(base[name]["n_lre"]) >= 4
        )
        assert len(names) == int(row["n_relations"])
        x = np.array([float(values[name]["score"]) for name in names])
        bx = np.array([float(base[name]["score"]) for name in names])
        y = np.array([float(base[name]["faithfulness"]) for name in names])
        indices = np.random.default_rng(20260921).integers(
            0, len(names), size=(10000, len(names))
        )
        boot = bootstrap_rho(x, y, indices)
        delta = boot - bootstrap_rho(bx, y, indices)
        rho, p = spearmanr(x, y)
        r, pp = pearsonr(x, y)
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        dl, dh = np.nanpercentile(delta, [2.5, 97.5])
        actual = dict(
            rho=rho,
            p=p,
            pearson_r=r,
            pearson_p=pp,
            delta_rho=rho - spearmanr(bx, y).statistic,
            rho_ci_low=low,
            rho_ci_high=high,
            delta_ci_low=dl,
            delta_ci_high=dh,
            n_valid_bootstraps=np.isfinite(delta).sum(),
            score_rho_vs_baseline=spearmanr(x, bx).statistic,
        )
        for key, value in actual.items():
            np.testing.assert_allclose(
                value,
                float(row[key]),
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"{protocol}/{prompt}/{cohort}/{key}",
            )
        if cohort == "all":
            print(f"{protocol:12s} {prompt:20s} n={len(names):2d} rho={rho:.6f}")
    print(
        "Verified all 28 summaries, including p-values and paired bootstrap intervals."
    )


if __name__ == "__main__":
    main()
