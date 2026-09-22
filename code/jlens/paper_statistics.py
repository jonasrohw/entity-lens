"""Recompute the paper statistics from the retained CPU-readable result snapshots."""

import json
from collections import Counter

import numpy as np
from scipy import stats

from jlens import affine_beta
from jlens.config import ROOT, YEAR_RELATIONS


def load(path):
    return json.loads((ROOT / path).read_text())


def partial_spearman(x, y, *controls):
    """Pearson correlation of rank residuals after regressing out controls."""
    a, b = stats.rankdata(x), stats.rankdata(y)
    design = np.column_stack([np.ones(len(a)), *(stats.rankdata(c) for c in controls)])
    a -= design @ np.linalg.lstsq(design, a, rcond=None)[0]
    b -= design @ np.linalg.lstsq(design, b, rcond=None)[0]
    return stats.pearsonr(a, b)


def relation_data():
    return {
        path.stem: json.loads(path.read_text())
        for path in (ROOT / "data/relations").rglob("*.json")
    }


def affine_lens_scores(relations):
    """Read the zeroth-order-restored lens with the main protocol's year fixes."""
    lens = load("results/jacobian_lens.json")
    years = load("results/jacobian_lens_years.json")
    return np.array(
        [
            -(years if relation in YEAR_RELATIONS else lens)[relation]["agg"]["lre"][
                "median_lograng"
            ]
            for relation in relations
        ]
    )


def report_affine_baseline(scores, faithfulness, differential_rho, *, prefix=""):
    result = stats.spearmanr(scores, faithfulness)
    print(
        f"{prefix}Affine Jacobian-lens baseline (+zeroth-order term) "
        f"({len(scores)} relations): rho={result.statistic:.3f}, "
        f"p={result.pvalue:.2e}, "
        f"delta_rho={result.statistic - differential_rho:+.3f}"
    )


def beta_sweep_lens(directory):
    """The separately measured beta sweep, if the protocol's run is retained.

    The sweep needs one rescaled readout per beta, so it cannot be recovered
    from the fixed-beta files; without it the report is simply skipped.
    """
    path = ROOT / directory / "jacobian_lens_beta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def report_beta_baseline(lens, relations, faithfulness, differential_rho, *, prefix=""):
    """Report both selection rules; the sweep retains every beta's aggregate."""
    sweeps = [lens.get(relation, {}).get("beta_sweep") for relation in relations]
    if not any(sweeps):
        return
    if not all(sweeps):
        raise ValueError("affine beta sweep does not cover all scored relations")
    for metric in affine_beta.SELECTION_METRICS:
        selected = [affine_beta.tuned_score(sweep, metric) for sweep in sweeps]
        result = stats.spearmanr([score for _, score in selected], faithfulness)
        print(
            f"{prefix}Beta-tuned affine Jacobian-lens baseline "
            f"({len(relations)} relations; {metric}, in-sample): "
            f"rho={result.statistic:.3f}, p={result.pvalue:.2e}, "
            f"delta_rho={result.statistic - differential_rho:+.3f}"
        )
        counts = Counter(beta for beta, _ in selected)
        print(
            "  Selected beta counts: "
            + ", ".join(f"{beta:g}: {count}" for beta, count in sorted(counts.items()))
        )


def fact_points(relation, lens, rank_dir):
    """Join subject ranks only when both measurements use the same split."""
    lens_result = load(f"results/{lens}")[relation]
    rank_result = load(f"results/{rank_dir}/subject_ranks_{relation}.json")
    if not rank_result.get("split_fingerprint") or rank_result[
        "split_fingerprint"
    ] != lens_result.get("split_fingerprint"):
        raise ValueError(f"{relation}: split fingerprints differ or are missing")
    ranks = {row["subject"]: row["rank_diff"] for row in lens_result["rows"]}
    return np.array(
        [
            (np.log10(ranks[subject] + 1), np.log10(value["rank"] + 1))
            for subject, value in rank_result["ranks"].items()
            if subject in ranks
        ]
    ).reshape(-1, 2)


def main():
    figure = load("results/paper_figure_data.json")
    rels = figure["rels"]
    x = -np.array(figure["lens"])
    y = np.array(figure["faith"])
    data = relation_data()
    k = np.array(
        [
            len(data[relation]["samples"])
            / len({sample["object"] for sample in data[relation]["samples"]})
            for relation in rels
        ]
    )
    prior = load("results/answer_prior.json")
    frequency = np.array([prior[relation]["prior_modeltop1"] for relation in rels])
    rho = stats.spearmanr(x, y)
    rng = np.random.default_rng(17)
    null = sum(
        abs(stats.spearmanr(rng.permutation(x), y).statistic) >= abs(rho.statistic)
        for _ in range(10000)
    )
    domains = [data[relation]["properties"]["domain_name"] for relation in rels]
    leave_out = [
        stats.spearmanr(
            x[[i for i, domain in enumerate(domains) if domain != excluded]],
            y[[i for i, domain in enumerate(domains) if domain != excluded]],
        ).statistic
        for excluded in sorted(set(domains))
    ]

    print(f"Main ({len(rels)} relations): rho={rho.statistic:.3f}, p={rho.pvalue:.2e}")
    report_affine_baseline(affine_lens_scores(rels), y, rho.statistic)
    report_beta_baseline(beta_sweep_lens("results"), rels, y, rho.statistic)
    # The ablation baselines: the lens without its Jacobian, without its
    # difference, and without either. Each is reported only once measured.
    for label, filename in [
        ("Logit-lens baseline", "logit_lens.json"),
        ("Differential logit-lens baseline", "logit_lens_diff.json"),
        ("Raw Jacobian-lens baseline", "jacobian_lens_raw.json"),
    ]:
        path = ROOT / "results" / filename
        if not path.exists():
            continue
        baseline = json.loads(path.read_text())
        if set(baseline) != set(rels):
            raise ValueError(f"{filename} does not cover the same 28 relations")
        scores = np.array(
            [-baseline[relation]["agg"]["median_lograng"] for relation in rels]
        )
        result = stats.spearmanr(scores, y)
        print(
            f"{label} ({len(rels)} relations): "
            f"rho={result.statistic:.3f}, p={result.pvalue:.2e}"
        )
    print(f"Permutation: {null}/10000 exceed |rho|")
    print(
        f"Cardinality: k/faith rho={stats.spearmanr(k, y).statistic:.3f}; "
        f"k/lens rho={stats.spearmanr(k, x).statistic:.3f}; "
        f"partial rho={partial_spearman(x, y, k).statistic:.3f}"
    )
    print(
        f"Answer prior: prior/faith rho={stats.spearmanr(frequency, y).statistic:.3f}; "
        f"prior/lens rho={stats.spearmanr(frequency, x).statistic:.3f}; "
        f"partial rho={partial_spearman(x, y, frequency).statistic:.3f}; "
        f"partial with k={partial_spearman(x, y, k, frequency).statistic:.3f}"
    )
    print(
        f"Leave-one-domain-out: {min(leave_out):.3f}..{max(leave_out):.3f}, "
        f"median {np.median(leave_out):.3f}"
    )

    lens = load("results/shared_split/jacobian_lens.json")
    faith = {
        p.stem.removeprefix("faithfulness_"): json.loads(p.read_text())
        for p in (ROOT / "results/shared_split").glob("faithfulness_*.json")
    }
    alternative = [relation for relation in lens if relation in faith]
    alt = stats.spearmanr(
        [-lens[relation]["agg"]["diff"]["median_lograng"] for relation in alternative],
        [faith[relation]["best_faith"] for relation in alternative],
    )
    print(
        f"Shared split, identifying first tokens ({len(alternative)} relations): "
        f"rho={alt.statistic:.3f}, p={alt.pvalue:.2e}"
    )
    report_affine_baseline(
        [-lens[relation]["agg"]["lre"]["median_lograng"] for relation in alternative],
        [faith[relation]["best_faith"] for relation in alternative],
        alt.statistic,
        prefix="Shared split, ",
    )
    report_beta_baseline(
        beta_sweep_lens("results/shared_split"),
        alternative,
        [faith[relation]["best_faith"] for relation in alternative],
        alt.statistic,
        prefix="Shared split, ",
    )

    prominence = load("results/prominence_statistics.json")
    prom = stats.spearmanr(
        [row["prom"] for row in prominence],
        [row["gold_lens"] for row in prominence],
    )
    print(
        f"Prominence ({len(prominence)} relations): "
        f"rho={prom.statistic:.3f}, p={prom.pvalue:.3f}"
    )

    for relation in (
        "product_by_company",
        "superhero_archnemesis",
        "person_occupation",
        "person_university",
    ):
        points = fact_points(
            relation, "shared_split/jacobian_lens.json", "shared_split"
        )
        misses = points[points[:, 1] > 0]
        all_rho = stats.spearmanr(points[:, 0], points[:, 1]).statistic
        miss_rho = stats.spearmanr(misses[:, 0], misses[:, 1]).statistic
        print(
            f"{relation}: n={len(points)}, misses={len(misses)}, "
            f"rho={all_rho:.3f}, misses rho={miss_rho:.3f}"
        )


if __name__ == "__main__":
    main()
