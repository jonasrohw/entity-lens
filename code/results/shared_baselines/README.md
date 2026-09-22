# Current shared-split baseline appendix

**Merge note, 22 September 2026:** These retained measurements predate the
collaborator correction that calls each architecture's actual final-normalization
module. In particular, Gemma2 readouts require a corrected rerun. The source
snapshot in `provenance/executed_sources.json` preserves the executed
implementation; validating its hashes does not validate the corrected code.
The tables were not rerun as part of the Git synchronization.

All seven models, exact neutral template `\n{} is known for`, pinned revisions, fp32/eager, 25 relations each. Every statistic below is recomputed from retained current rows. Spearman is primary; Pearson, unadjusted p-values, and exact sample counts are also retained.

[Paper appendix](../../../paper/appendix_shared_baselines.tex) · [Standalone PDF](../../../paper/shared_baselines.pdf) · [All correlations](correlations.csv) · [Relation measurements](relation_scores.csv) · [Subject measurements](subject_scores.csv) · [Source hashes and validation](summary.json)

## Lens and Wikipedia baselines

Spearman ρ across 25 relations; columns follow the same model order.

| Predictor | OLMo | GPT-J | Mistral | Gemma | Qwen 8B | Qwen 27B | Pythia |
|---|---:|---:|---:|---:|---:|---:|---:|
| Differential Jacobian | +0.683 | +0.573 | +0.606 | +0.554 | +0.729 | +0.600 | +0.479 |
| Raw Jacobian | +0.689 | +0.567 | +0.586 | +0.575 | +0.736 | +0.644 | +0.501 |
| Logit lens | +0.234 | +0.531 | +0.402 | +0.554 | +0.743 | +0.402 | -0.343 |
| Differential logit lens | +0.326 | +0.394 | +0.249 | +0.221 | +0.607 | +0.454 | +0.217 |
| Affine Jacobian (beta=1) | +0.562 | +0.430 | +0.407 | +0.472 | +0.300 | +0.308 | +0.308 |
| Geva attribute rate | -0.114 | +0.002 | -0.304 | -0.292 | +0.052 | -0.196 | +0.096 |
| Wikipedia token-frequency rank | +0.277 | +0.140 | +0.276 | +0.419 | +0.374 | +0.304 | +0.154 |
| Wikipedia TF/logit vocabulary agreement | +0.301 | -0.109 | -0.168 | -0.102 | -0.175 | +0.065 | +0.310 |

## Prior and cardinality controls

Spearman ρ across 25 relations; columns follow the same model order.

| Predictor | OLMo | GPT-J | Mistral | Gemma | Qwen 8B | Qwen 27B | Pythia |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model answer-token prior | -0.137 | +0.067 | -0.080 | +0.283 | +0.001 | +0.171 | -0.223 |
| Gold answer-token prior | +0.027 | +0.175 | +0.056 | +0.358 | +0.167 | +0.252 | +0.010 |
| Full-data samples/objects | +0.304 | +0.258 | +0.743 | +0.666 | +0.362 | +0.498 | +0.249 |
| Filtered samples/objects | +0.303 | +0.265 | +0.743 | +0.664 | +0.361 | +0.499 | +0.255 |
| Test samples/objects | +0.291 | +0.126 | +0.699 | +0.665 | +0.338 | +0.498 | +0.198 |
| Relation-prompt intercept-only agreement | +0.293 | +0.172 | +0.682 | +0.600 | +0.346 | +0.530 | +0.283 |
| Modal-target fraction (oracle) | +0.584 | +0.494 | +0.804 | +0.768 | +0.549 | +0.709 | +0.413 |
| Target entropy | -0.753 | -0.699 | -0.777 | -0.749 | -0.736 | -0.762 | -0.764 |
| Model gold-token accuracy | +0.683 | +0.781 | +0.335 | +0.370 | +0.549 | +0.447 | +0.775 |

## Corpus frequency and direction

Spearman ρ across 25 relations; columns follow the same model order.

| Predictor | OLMo | GPT-J | Mistral | Gemma | Qwen 8B | Qwen 27B | Pythia |
|---|---:|---:|---:|---:|---:|---:|---:|
| Subject document frequency | +0.290 | +0.336 | -0.166 | -0.204 | +0.166 | +0.061 | +0.378 |
| Object document frequency | +0.644 | +0.642 | +0.618 | +0.590 | +0.603 | +0.627 | +0.610 |
| Joint document frequency | +0.578 | +0.620 | +0.158 | +0.232 | +0.489 | +0.382 | +0.631 |
| Mean log(1+joint frequency) | +0.568 | +0.590 | +0.147 | +0.239 | +0.478 | +0.377 | +0.607 |
| Log(1+mean joint frequency) | +0.578 | +0.620 | +0.158 | +0.232 | +0.489 | +0.382 | +0.631 |
| P(object \| subject) | +0.386 | +0.418 | +0.566 | +0.635 | +0.411 | +0.418 | +0.305 |
| P(subject \| object) | -0.328 | -0.287 | -0.740 | -0.654 | -0.421 | -0.502 | -0.230 |
| Forward minus reverse conditional | +0.362 | +0.415 | +0.636 | +0.687 | +0.432 | +0.452 | +0.288 |
| Micro P(object \| subject) | +0.615 | +0.574 | +0.639 | +0.708 | +0.597 | +0.484 | +0.513 |
| Micro P(subject \| object) | +0.030 | +0.014 | -0.485 | -0.361 | -0.095 | -0.213 | +0.021 |

## Affine beta sweep

Spearman ρ across 25 relations; columns follow the same model order.

| Predictor | OLMo | GPT-J | Mistral | Gemma | Qwen 8B | Qwen 27B | Pythia |
|---|---:|---:|---:|---:|---:|---:|---:|
| Affine beta=0.5 | +0.367 | +0.160 | +0.382 | +0.498 | +0.157 | +0.289 | +0.103 |
| Affine beta=1 | +0.562 | +0.430 | +0.407 | +0.472 | +0.300 | +0.308 | +0.308 |
| Affine beta=1.5 | +0.635 | +0.473 | +0.425 | +0.528 | +0.396 | +0.326 | +0.391 |
| Affine beta=2 | +0.658 | +0.486 | +0.420 | +0.585 | +0.507 | +0.364 | +0.428 |
| Affine beta=3 | +0.728 | +0.463 | +0.399 | +0.647 | +0.600 | +0.393 | +0.510 |
| Affine beta=5 | +0.746 | +0.481 | +0.459 | +0.657 | +0.675 | +0.447 | +0.537 |
| Affine tuned top1 (in sample) | +0.364 | +0.175 | +0.383 | +0.498 | +0.210 | +0.279 | +0.161 |
| Affine tuned median-logrank (in sample) | +0.746 | +0.475 | +0.459 | +0.657 | +0.666 | +0.427 | +0.510 |

## Interpretation and scope

Pythia has 413 rows with undefined Geva rates because no top-50 prediction survives the filter. They are excluded from its attribute-rate means. The appendix and [Geva-matched measurements](geva_matched_relation_scores.csv) additionally restrict every readout and the LRE outcome to the same Geva-defined rows. The other models have no undefined attribute rates.

Raw Jacobian, affine Jacobian, and differential Jacobian are distinct readouts. Geva's attribute rate measures subject-attribute overlap, while Wikipedia TF is our separate answer-token ranking extension. OLMo document counts measure its published training multiset; for six other models they are a common external corpus proxy, not their training frequencies. Tuned affine beta and selected LRE settings use the scored data and are not held-out estimates. P-values are exploratory, two-sided and unadjusted.

The two-model prominence diagnostic was rerun in fp32 with newline-prefixed prose prompts, 90 generated tokens per prompt and up to 40 test rows per relation. It is kept separate from the 25-relation primary table. [Matched prominence rows](prominence.csv).

[Paired differences and 95% relation-bootstrap intervals](paired_gaps.csv), [partial/outcome controls](sensitivity.csv), and [all corpus stages and overlap sensitivities](corpus_relation_scores.csv) are included. Bootstrap intervals describe relation resampling, not uncertainty over models or training runs.

See [curation and protocol notes](../../docs/shared-baseline-appendix.md) for definitions, historical exclusions, and reproduction commands.
