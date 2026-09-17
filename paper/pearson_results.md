# Pearson results for the paper

Computed from the retained measurements, with the same samples and numeric-object corrections as the manuscript. Relation-level lens scores are negative median log10(rank + 1); faithfulness is the original fraction. Cardinality k and answer-token prior use their existing numerical values, without an additional transformation. Each relation has equal weight. The manuscript and figures are unchanged.

## Relation-level results

| Analysis | n | Pearson r | Two-sided p |
| --- | ---: | ---: | ---: |
| Main: lens score vs faithfulness | 28 | +0.666 | 0.0001104 |
| Logit-lens baseline vs faithfulness | 28 | +0.194 | 0.3223 |
| Cardinality k vs faithfulness | 28 | +0.354 | 0.06418 |
| Cardinality k vs lens score | 28 | +0.067 | 0.7336 |
| Lens vs faithfulness, controlling for k | 28 | +0.688 | 7.298e-05 |
| Answer prior vs faithfulness | 28 | -0.036 | 0.8567 |
| Answer prior vs lens score | 28 | +0.077 | 0.698 |
| Lens vs faithfulness, controlling for answer prior | 28 | +0.671 | 0.0001281 |
| Lens vs faithfulness, controlling for k and answer prior | 28 | +0.688 | 0.0001032 |

Controlled rows are partial Pearson correlations: regress both variables on the unranked covariates with an intercept and correlate the residuals. Their p-values use a t test with n − c − 2 degrees of freedom for c independent covariates. All p-values are unadjusted for multiple comparisons.

Main-result permutation test: 1/10000 shuffles (0.01%) matched or exceeded |r|; Monte Carlo p = (b + 1)/(B + 1) = 0.0002. The random seed is 17, matching the existing statistics script.

## Leave-one-domain-out analysis

| Analysis | n | Pearson r | Two-sided p |
| --- | ---: | ---: | ---: |
| Exclude city | 27 | +0.668 | 0.0001388 |
| Exclude company | 26 | +0.632 | 0.0005362 |
| Exclude country | 24 | +0.601 | 0.001915 |
| Exclude landmark | 26 | +0.677 | 0.0001461 |
| Exclude name | 25 | +0.665 | 0.0002844 |
| Exclude person | 17 | +0.755 | 0.0004607 |
| Exclude pokemon | 27 | +0.663 | 0.0001637 |
| Exclude product | 27 | +0.668 | 0.0001411 |
| Exclude star | 27 | +0.660 | 0.0001802 |
| Exclude superhero | 26 | +0.686 | 0.0001099 |

Range: 0.601–0.755; median: 0.667.

## Alternative protocol and numeric-object check

| Analysis | n | Pearson r | Two-sided p |
| --- | ---: | ---: | ---: |
| Shared split, identifying first tokens | 25 | +0.676 | 0.0002086 |
| Main with original, uncorrected numeric-object measurements | 28 | +0.518 | 0.004738 |

## Object prominence

| Analysis | n | Pearson r | Two-sided p |
| --- | ---: | ---: | ---: |
| Prominence vs gold object in lens top five | 14 | +0.772 | 0.001203 |
| Prominence vs gold in top five, excluding star_constellation | 13 | +0.858 | 0.000177 |

## Fact-level results

Pearson correlations use log10(rank + 1) on both axes, as plotted in the paper; these are not correlations of raw token ranks. Ranks are zero-based. Misses have LRE rank > 0. Subject joins require matching split fingerprints.

| Analysis | n | Pearson r | Two-sided p |
| --- | ---: | ---: | ---: |
| product_by_company, all subjects | 99 | +0.453 | 2.458e-06 |
| product_by_company, misses only | 32 | +0.531 | 0.001751 |
| superhero_archnemesis, all subjects | 55 | +0.117 | 0.3968 |
| superhero_archnemesis, misses only | 37 | +0.264 | 0.1144 |
| person_occupation, all subjects | 100 | -0.125 | 0.2165 |
| person_occupation, misses only | 34 | -0.306 | 0.07832 |
| person_university, all subjects | 29 | -0.243 | 0.2048 |
| person_university, misses only | 19 | -0.009 | 0.9718 |

## Reproduction

From `code/`:

```sh
uv run python ../paper/repro/pearson_statistics.py > ../paper/pearson_results.md
```

Inputs: `code/results/paper_figure_data.json` (verified against the underlying measurements), available baseline snapshots, `code/results/shared_split/`, `code/results/prominence_statistics.json`, the original year measurements, and `code/data/relations/`. No model measurements were rerun.
