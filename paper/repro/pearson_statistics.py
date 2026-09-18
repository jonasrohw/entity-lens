"""Print a separate Pearson report from the paper's retained measurements.

Run from code/: uv run python ../paper/repro/pearson_statistics.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from jlens.config import ROOT, YEAR_RELATIONS
from jlens.figure_data import build_figure_data
from jlens.paper_statistics import affine_lens_scores, fact_points, load, relation_data


def partial_pearson(x, y, *controls):
    """OLS residual correlation, with covariate-adjusted test degrees of freedom."""
    design = np.column_stack([np.ones(len(x)), *controls])
    a = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    b = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = stats.pearsonr(a, b).statistic
    df = len(x) - np.linalg.matrix_rank(design) - 1
    t = r * np.sqrt(df / (1 - r**2))
    return r, 2 * stats.t.sf(abs(t), df)


def row(label, x, y, *controls):
    result = partial_pearson(x, y, *controls) if controls else stats.pearsonr(x, y)
    print(f"| {label} | {len(x)} | {result[0]:+.3f} | {result[1]:.4g} |")


def table():
    print("\n| Analysis | n | Pearson r | Two-sided p |")
    print("| --- | ---: | ---: | ---: |")


def main():
    figure = build_figure_data()
    assert figure == load("results/paper_figure_data.json")
    rels = figure["rels"]
    x = -np.array(figure["lens"])
    y = np.array(figure["faith"])
    k = np.array(figure["k"])
    prior = np.array(figure["prior"])

    print("# Pearson results for the paper\n")
    print(
        "Computed from the retained measurements, with the same samples and "
        "numeric-object corrections as the manuscript. Relation-level lens scores "
        "are negative median log10(rank + 1); faithfulness is the original fraction. "
        "Cardinality k and answer-token prior use their existing numerical values, "
        "without an additional transformation. Each relation has equal weight. "
        "The manuscript and figures are unchanged."
    )
    print("\n## Relation-level results")
    table()
    row("Main: lens score vs faithfulness", x, y)
    row(
        "Affine Jacobian-lens baseline (+zeroth-order term) vs faithfulness",
        affine_lens_scores(rels),
        y,
    )
    for label, filename in [
        ("Logit-lens baseline vs faithfulness", "logit_lens.json"),
        ("Differential logit-lens baseline vs faithfulness", "logit_lens_diff.json"),
        ("Raw Jacobian-lens baseline vs faithfulness", "jacobian_lens_raw.json"),
    ]:
        if not (ROOT / "results" / filename).exists():
            continue
        baseline = load(f"results/{filename}")
        assert set(baseline) == set(rels)
        scores = np.array([-baseline[r]["agg"]["median_lograng"] for r in rels])
        row(label, scores, y)
    row("Cardinality k vs faithfulness", k, y)
    row("Cardinality k vs lens score", k, x)
    row("Lens vs faithfulness, controlling for k", x, y, k)
    row("Answer prior vs faithfulness", prior, y)
    row("Answer prior vs lens score", prior, x)
    row("Lens vs faithfulness, controlling for answer prior", x, y, prior)
    row("Lens vs faithfulness, controlling for k and answer prior", x, y, k, prior)
    print(
        "\nControlled rows are partial Pearson correlations: regress both variables "
        "on the unranked covariates with an intercept and correlate the residuals. "
        "Their p-values use a t test with n − c − 2 degrees of freedom for c "
        "independent covariates. All p-values are unadjusted for multiple comparisons."
    )

    rng = np.random.default_rng(17)
    observed = stats.pearsonr(x, y).statistic
    permutations = 10000
    exceed = sum(
        abs(stats.pearsonr(rng.permutation(x), y).statistic) >= abs(observed)
        for _ in range(permutations)
    )
    print(
        f"\nMain-result permutation test: {exceed}/{permutations} shuffles "
        f"({100 * exceed / permutations:.2f}%) matched or exceeded |r|; "
        f"Monte Carlo p = (b + 1)/(B + 1) = {(exceed + 1) / (permutations + 1):.4g}. "
        "The random seed is 17, matching the existing statistics script."
    )

    print("\n## Leave-one-domain-out analysis")
    domains = np.array([relation_data()[r]["properties"]["domain_name"] for r in rels])
    correlations = []
    table()
    for domain in sorted(set(domains)):
        keep = domains != domain
        row(f"Exclude {domain}", x[keep], y[keep])
        correlations.append(stats.pearsonr(x[keep], y[keep]).statistic)
    print(
        f"\nRange: {min(correlations):.3f}–{max(correlations):.3f}; "
        f"median: {np.median(correlations):.3f}."
    )

    print("\n## Alternative protocol and numeric-object check")
    lens = load("results/shared_split/jacobian_lens.json")
    alternative = [
        r
        for r in lens
        if (ROOT / f"results/shared_split/faithfulness_{r}.json").exists()
    ]
    table()
    row(
        "Shared split, identifying first tokens",
        np.array([-lens[r]["agg"]["diff"]["median_lograng"] for r in alternative]),
        np.array(
            [
                load(f"results/shared_split/faithfulness_{r}.json")["best_faith"]
                for r in alternative
            ]
        ),
    )
    original_lens = load("results/jacobian_lens.json")
    original_x, original_y = x.copy(), y.copy()
    for i, relation in enumerate(rels):
        if relation in YEAR_RELATIONS:
            original_x[i] = -original_lens[relation]["agg"]["diff"]["median_lograng"]
            original_y[i] = load(f"results/main/faithfulness_{relation}.json")[
                "best_faith"
            ]
    row(
        "Main with original, uncorrected numeric-object measurements",
        original_x,
        original_y,
    )

    print("\n## Object prominence")
    prominence = load("results/prominence_statistics.json")
    table()
    for label, rows in [
        ("Prominence vs gold object in lens top five", prominence),
        (
            "Prominence vs gold in top five, excluding star_constellation",
            [r for r in prominence if r["rel"] != "star_constellation"],
        ),
    ]:
        row(
            label,
            np.array([r["prom"] for r in rows]),
            np.array([r["gold_lens"] for r in rows]),
        )

    print("\n## Fact-level results")
    print(
        "\nPearson correlations use log10(rank + 1) on both axes, as plotted in "
        "the paper; these are not correlations of raw token ranks. Ranks are "
        "zero-based. Misses have LRE rank > 0. Subject joins require matching "
        "split fingerprints."
    )
    table()
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
        row(f"{relation}, all subjects", *points.T)
        row(f"{relation}, misses only", *misses.T)

    print("\n## Reproduction\n")
    print("From `code/`:\n")
    print(
        "```sh\nuv run python ../paper/repro/pearson_statistics.py > ../paper/pearson_results.md\n```"
    )
    print(
        "\nInputs: `code/results/paper_figure_data.json` (verified against the "
        "underlying measurements), available baseline snapshots, "
        "`code/results/shared_split/`, `code/results/prominence_statistics.json`, "
        "the original year measurements, and `code/data/relations/`. "
        "No model measurements were rerun."
    )


if __name__ == "__main__":
    main()
