# Current full-evaluation paper results

Complete: **True**.

Original 28 relations; every one of 9,747 rows/model measured, including demonstrations/references and duplicates. Gender alternatives enabled; original continent references. Primary year exclusions leave 26 relations and 9,709 scored rows in Mistral, Gemma and both Qwen models. LRE hyperparameters are selected on the full evaluation; frozen-parameter and all-28 sensitivities are supplied. These are in-sample results.

| Model | Relations | Spearman ρ | Leading newline |
|---|---:|---:|:---:|
| OLMo 3 7B | 28 | 0.735459 | No |
| GPT-J 6B | 28 | 0.434592 | Yes |
| Pythia 6.9B | 28 | 0.502738 | Yes |
| Mistral 7B | 26 | 0.656068 | No |
| Gemma 2 9B | 26 | 0.688889 | No |
| Qwen3 8B Base | 26 | 0.664729 | Yes |
| Qwen3.8 27B | 26 | 0.553883 | Yes |

The newline models are GPT-J, Pythia and both Qwen models; the appendix documents first-position norm diagnostics. Approved differential ranks and all LRE outcomes are reused. Missing raw Jacobian, logit-lens, differential-logit-lens and attribute baselines are measured on the same full evaluation, with gender alternatives applied consistently to target-rank readouts.

Attribute rate explicitly assigns zero overlap to rows whose filtered top-50 prediction set is empty. Corpus conditional features explicitly use zero for an unobserved conditioning marginal (207 dataset rows have at least one unobserved marginal). Neither convention drops a row. These extensions are documented in the paper.

The paper uses CEL (contrastive/differential lens) as the primary readout; NEL is a separate raw-Jacobian baseline. Earlier shared-split/capped results and incompatible small-cohort numerical sections are preserved in the dated paper archive and excluded from compilation.

Artifacts: [all correlations](correlations.csv), [statistics/provenance](paper_statistics.json), [PDF](../../../paper/lens.pdf), [paper guide](../../../paper/README.md), and `paper_release_audit.json` (created only after all measurements and PDF checks pass).
