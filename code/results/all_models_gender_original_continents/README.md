# All models: gender alternatives, original continent references

Status: complete. Only complete model cohorts receive correlations.

| Model | Completed relations | Validated rows |
| --- | ---: | ---: |
| OLMo 3 7B | 28/28 | 9747/9747 |
| GPT-J 6B | 28/28 | 9747/9747 |
| Pythia 6.9B | 28/28 | 9747/9747 |
| Mistral 7B | 28/28 | 9747/9747 |
| Gemma 2 9B | 28/28 | 9747/9747 |
| Qwen3 8B Base | 28/28 | 9747/9747 |
| Qwen3.8 27B | 28/28 | 9747/9747 |

| Model | Cohort | Relations | Scored rows | Rho reselected | Rho original settings |
| --- | --- | ---: | ---: | ---: | ---: |
| OLMo 3 7B | primary | 28 | 9747 | 0.73545916 | 0.70462636 |
| OLMo 3 7B | all_28_years_included | 28 | 9747 | 0.73545916 | 0.70462636 |
| GPT-J 6B | primary | 28 | 9747 | 0.18828681 | 0.19978106 |
| GPT-J 6B | all_28_years_included | 28 | 9747 | 0.18828681 | 0.19978106 |
| Pythia 6.9B | primary | 28 | 9747 | 0.17460317 | 0.13441008 |
| Pythia 6.9B | all_28_years_included | 28 | 9747 | 0.17460317 | 0.13441008 |
| Mistral 7B | primary | 26 | 9709 | 0.65606838 | 0.59863248 |
| Mistral 7B | all_28_years_included | 28 | 9747 | 0.36868756 | 0.32270426 |
| Gemma 2 9B | primary | 26 | 9709 | 0.68888889 | 0.68615385 |
| Gemma 2 9B | all_28_years_included | 28 | 9747 | 0.45408513 | 0.43295019 |
| Qwen3 8B Base | primary | 26 | 9709 | 0.66472902 | 0.64899984 |
| Qwen3 8B Base | all_28_years_included | 28 | 9747 | 0.41467087 | 0.37689887 |
| Qwen3.8 27B | primary | 26 | 9709 | 0.55388313 | 0.52957265 |
| Qwen3.8 27B | all_28_years_included | 28 | 9747 | 0.30371087 | 0.27398385 |

All seven retained paper models use the original 28 relation datasets. Every original row is evaluated by both methods, including eight demonstration rows and eight reference rows per method. Duplicates retain their source weight. No row cap, correctness filter or object-token collision filter is applied. Subject-self differential-lens reference comparisons are excluded without excluding queries.

Gender lens scoring accepts the established gender-matched single-token alternatives. Exact model-top-1 LRE faithfulness is unchanged. Both LRE and lens use their original seeded landmark-continent reference sets, per the user’s explicit request to disable diverse-continent selection. Qwen models retain exactly one leading newline in the neutral prompt; other models use the original neutral prompt.

Primary correlations retain the preceding relation-level year policy: exclude the two affected year relations for models whose tokenizer emits standalone 1. Every year row is nevertheless evaluated and retained with a tokenizer audit. The separate all-28 sensitivity includes all year results and requires no additional inference.

The differential lens score is minus median(log10(zero-based target rank + 1)). Reselected LRE maximizes faithfulness over each model’s original six-layer/six-beta grid on the full evaluation set; original settings freeze the previously selected layer/beta. These are overlapping-data, in-sample evaluations, not independent held-out estimates.

OLMo and Qwen 8B are entirely reused from the completed full-row experiments, including their original-reference landmark measurements. The other five models’ older capped/differently split results do not satisfy this evaluation protocol and require new measurements. Model revisions, source hashes, full split rows, numerical checks, grid predictions and per-row lens ranks are retained.

Execution checks: batched derivatives and Jacobian prefix caching are checked against ordinary forwards and scalar derivatives before each worker starts. Optional prompt batching or shared-prefix reuse is accepted only after comparisons against complete scalar forwards; failed comparisons fall back to full forwards. Each accepted shared-prefix call checks every fourth query and the final query, including exact top-1 equality. Per-relation checks and checkpoint provenance are stored with the measurements. Synthetic end-to-end equivalence checks are retained in `validation/`.

The original overlapping LRE-grid comparisons, including any floating-point threshold differences, are listed in `summary.json`. Overlapping lens targets and reference counts are checked, and numerical rank differences are recorded separately in `original_lens_subset_checks.json`. These audit differences are never substituted into the newly measured full-row scores.

Independent correlation verification: `validation/independent_correlations.py` recalculates the row aggregates and Spearman coefficients using standard-library average tied ranks, without SciPy. Its checked values and measurement hashes are stored in `independent_correlation_check.json`.

Rebuild: `.venv/bin/python scripts/run_all_models_full_rows.py --report`.
