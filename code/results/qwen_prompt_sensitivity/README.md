# Qwen3-8B-Base neutral-prompt sensitivity

The measured correlation depends on the neutral prompt. Across all 28 main-protocol relations, the original gives Spearman ρ=0.0604; the tested prompts range from -0.1180 to 0.2976. The highest observed value is for `Facts about {}:`. Under shared_split (25 relations), the original gives ρ=0.2680, rising to 0.7881 for `Facts about {}:`. This is an exploratory sensitivity result, not a newly validated default.

Six alternatives and the original prompt were fixed before running this comparison. All variants use the same checkpoint revision, float32/eager attention, layer 19, subject/reference splits, eight references, relation-prompt target tokens, and retained LRE faithfulness. Only the neutral lens prompt changes; no layer or beta is retuned. The original prompt is freshly rerun as a numerical control.

Score: negative median log10(differential target rank + 1), correlated across relations with retained LRE model-top-1 faithfulness. The main protocol uses all 28 relations; shared_split uses its existing eligible cohort. Prefix variants also change the subject representation; suffix-only variants change the continuation/Jacobian context. Bare subject removes the continuation entirely. None supplies relation-specific information.

## main

| Neutral prompt | Relations | Spearman ρ | Δρ | Paired 95% CI for Δρ | Pearson r |
| --- | ---: | ---: | ---: | --- | ---: |
| `{} is known for` | 28 | 0.0604 | +0.0000 | [+0.0000, +0.0000] | 0.1590 |
| `{}` | 28 | 0.0579 | -0.0025 | [-0.4372, +0.3571] | 0.1507 |
| `{} is` | 28 | -0.1180 | -0.1784 | [-0.5162, +0.1504] | 0.0104 |
| `{} can be described as` | 28 | 0.1787 | +0.1183 | [-0.0479, +0.3104] | 0.2256 |
| `Information about {}:` | 28 | 0.2948 | +0.2344 | [-0.0089, +0.4948] | 0.3690 |
| `Facts about {}:` | 28 | 0.2976 | +0.2372 | [+0.0020, +0.4765] | 0.4122 |
| `Entity: {} is known for` | 28 | 0.2023 | +0.1419 | [-0.0850, +0.4187] | 0.3138 |

Baseline reproduction: `{"n_relations": 28, "n_subject_rows": 513, "ranks_changed": 41, "max_absolute_rank_difference": 4, "retained_rho": 0.06039006393402829}`.

## shared_split

| Neutral prompt | Relations | Spearman ρ | Δρ | Paired 95% CI for Δρ | Pearson r |
| --- | ---: | ---: | ---: | --- | ---: |
| `{} is known for` | 25 | 0.2680 | +0.0000 | [+0.0000, +0.0000] | 0.3263 |
| `{}` | 25 | -0.0575 | -0.3254 | [-0.6583, +0.0008] | 0.0780 |
| `{} is` | 25 | 0.1662 | -0.1018 | [-0.4906, +0.3014] | 0.2484 |
| `{} can be described as` | 25 | 0.3721 | +0.1041 | [-0.2384, +0.4086] | 0.4570 |
| `Information about {}:` | 25 | 0.7334 | +0.4654 | [+0.0821, +0.8639] | 0.7775 |
| `Facts about {}:` | 25 | 0.7881 | +0.5201 | [+0.1503, +0.8927] | 0.8124 |
| `Entity: {} is known for` | 25 | 0.7276 | +0.4596 | [+0.1006, +0.8367] | 0.7413 |

Baseline reproduction: `{"n_relations": 25, "n_subject_rows": 1544, "ranks_changed": 75, "max_absolute_rank_difference": 2, "retained_rho": 0.2679784780356726}`.

## Existing small-relation sensitivity

This secondary analysis excludes relations with fewer than four LRE tests, using the rule from the earlier audit. It changes the cohort and does not replace the all-relation comparison above.

| Prompt | Main ρ | Shared ρ |
| --- | ---: | ---: |
| `{} is known for` | 0.4483 (n=25) | 0.2699 (n=24) |
| `{}` | 0.2805 (n=25) | -0.0645 (n=24) |
| `{} is` | 0.2259 (n=25) | 0.1609 (n=24) |
| `{} can be described as` | 0.5888 (n=25) | 0.3806 (n=24) |
| `Information about {}:` | 0.6762 (n=25) | 0.7595 (n=24) |
| `Facts about {}:` | 0.6801 (n=25) | 0.8132 (n=24) |
| `Entity: {} is known for` | 0.6404 (n=25) | 0.7626 (n=24) |

![Prompt sensitivity](correlations.png)

## Interpretation and limits

All three prefixed prompts improve the observed correlation in both protocols. The `Entity:` control keeps the words `is known for`, so the pattern suggests that subject context/position contributes to the issue. This experiment does not isolate positional effects from the prefix's semantics. Bare subject and `{} is` do not improve on the original in either protocol. The strong shared-split results and the more modest all-28 main results should be reported together.

Intervals are paired percentile bootstraps over relations (10,000 draws, seed 20260921). They describe relation-sampling uncertainty, not new subject splits or LRE re-estimation. The prompts were specified before this run, but the investigation follows observed low Qwen correlation; selecting the best prompt remains exploratory. CSV p-values and intervals are not adjusted for multiple comparisons. Relations share some subjects and semantic domains, so the relation bootstrap is an approximate uncertainty estimate. The two protocols are sensitivity checks, not independent replications. Retained LRE scores still use in-sample layer/beta selection and the original first-token targets.

## Reproduction

From `code/`, run one or more workers on separate GPUs:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_qwen_prompt_sensitivity.py --worker 0
.venv/bin/python scripts/report_qwen_prompt_sensitivity.py
```

Workers claim independent prompt/relation files and resume completed outputs. Each file retains subject ranks, target IDs, prompt, fixed faithfulness, split fingerprint, source/data hashes, and baseline rank differences. Worker files retain checkpoint, code hashes, runtime environment, and derivative checks. The report refuses incomplete cohorts.
Main measurements use derivative batches of two on A100 GPUs. The larger shared cohorts use batches of eight on A100/A6000 GPUs after the initial workers are replaced; any already-complete files are retained. Each result identifies its worker and exact environment. Every worker repeats the scalar-product and finite-difference verification.
