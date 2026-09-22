# Current shared-split baseline appendix

**Merge note, 22 September 2026:** These retained measurements predate the
collaborator correction that calls each architecture's actual final-normalization
module. In particular, Gemma2 readouts require a corrected rerun. The source
snapshot in `provenance/executed_sources.json` preserves the executed
implementation; validating its hashes does not validate the corrected code.
The tables were not rerun as part of the Git synchronization.

The PDF contains three tables: five readouts across all seven models,
OLMo-only corpus-frequency results, and prompt variations. The prompt table
collects eight Qwen templates and the seven-model newline comparison;
legacy and shared cohorts are explicitly separated. Generate it with
`scripts/format_prompt_variation_table.py`, which rechecks the saved raw
measurements and recomputes the displayed correlations. Run
`scripts/format_shared_baseline_tables.py`
after regenerating the full numerical report to update these presentation
tables. All measurements and additional analyses listed below remain in the
CSV/JSON artifacts; the earlier expanded TeX is preserved in
`paper/archive/appendix_shared_baselines_expanded_2026-09-21.tex`.

This is the authoritative baseline comparison for the newline-prefixed shared
protocol. The [results and tables](../results/shared_baselines/README.md) cover
seven pinned models, 25 relations per model, and 10,772 test rows. The
[standalone appendix](../../paper/shared_baselines.pdf) is also included in
the manuscript as Appendix A. Spearman is primary; Pearson, both p-values,
and the number of observations are retained in
[correlations.csv](../results/shared_baselines/correlations.csv).

## What was found and what was refreshed

| Analysis | Historical source | Current disposition |
|---|---|---|
| Differential Jacobian and affine beta=1 | `results/<model>/results/jacobian_lens*.json`; `results/five_model_sweep/` | Reuse the completed, validated all-seven-model newline rerun in `results/all_model_newline/*/shared_split/`; exact targets, splits, data and source hashes checked again. |
| Raw/normal Jacobian, mean J_i s_e | `results/{olmo,gptj}/results/jacobian_lens_raw.json`; `docs/raw-jacobian-baseline.md` | Fresh fp32/eager runs for all seven models. The current table replaces the stale OLMo file made before the final-normalization correction. Raw Jacobian is not the affine baseline. |
| Ordinary logit lens | `results/{olmo,gptj}/results/logit_lens.json` | Fresh all-seven-model runs at the same subject position/layer/prompt as differential Jacobian. |
| Differential logit lens | `results/{olmo,gptj}/results/logit_lens_diff.json` and corresponding measurement module | Fresh all-seven-model runs; subtract the eligible reference mean before the actual final norm/readout. |
| Affine beta sweep | `docs/affine-beta-sweep.md`, `docs/fit-vs-differential-by-relation.md` | Recompute all shared/newline correlations for beta 0.5, 1, 1.5, 2, 3, 5 and both in-sample selection rules. Old extended-beta convergence plots remain historical mechanism diagnostics. |
| Geva/Gerstner attribute rate | `results/olmo/results/geva_baseline.json`, `docs/geva-baseline*.md` | Fresh all-seven-model top-50 readouts and scoring; Wikipedia BM25 reindexed for all 2,344 current distinct test subjects. Source filtering and stopwords retained. |
| Wikipedia token-frequency answer ranks | Same historical Geva artifact | Fresh tokenizer-specific counts and midranks for the same retrievals. This is our extension, not Geva's published metric. |
| TF–logit vocabulary correlation | Same historical Geva artifact | Fresh per-subject eligible-vocabulary Spearman, with relation-level aggregation and correlations. |
| Answer-token prior | `results/{olmo,gptj}/results/answer_prior.json` | Fresh fp32 model priors on the original ten context definitions for all seven models; score current model-target and gold IDs. |
| Cardinality | `jlens/paper_statistics.py` | Recompute full-data, tokenizer-filtered-pool, and test-only samples-per-object controls. |
| Intercept-only agreement, target concentration/entropy, gold-token accuracy | `results/cross_model_audit/raw/targets/*/shared_split/` | Reuse ordinary-forward measurements after checking pinned revision, ordered subject/target/gold IDs, split fingerprint and selected-LRE outcome. These relation prompts were not changed by the neutral newline. |
| Corpus marginals, joint counts, both conditionals, directional difference, micro conditionals | `generated/corpus_conditionals/{all_stages,dolma3_*}.facts.csv`; `docs/corpus-conditionals*.md` | Reuse verified complete document counts; recompute exact-current-test-row aggregation for all stages and each stage, with/without lexical overlap. OLMo training corpus; explicitly an external corpus proxy for six other models. |
| Free-generation object prominence and gold-in-lens top five | `results/{olmo,gptj}/results/prominence*.json` | Fresh OLMo/GPT-J generation on the original 14-relation scope, with newline-prefixed prose prompts and fp32; fix generation/lens cohort mismatch. Report all rows and corrected name-leakage subset separately. |
| Partial Spearman, leave-one-domain-out, permutation tests | `jlens/paper_statistics.py` | Recompute on current shared/newline relation scores for all seven models. |
| Intercept-adjusted and model-correct-outcome sensitivity | `results/cross_model_audit/analysis.json` | Recompute with current lens scores and matched target audit. Correct-only outcome retains original selected LRE and original full-test lens aggregate; it is a sensitivity, not a refitted experiment. |
| Within-relation LRE-rank figures | `jlens/matched_baseline.py`, `results/*/results/subject_ranks*/`, manuscript fact-level figures | Historical rank-outcome analysis; not a relation-level predictor baseline. Current subject rows and selected-LRE binary agreement are supplied in `subject_scores.csv`, but old LRE-rank figures are not relabelled as current. |
| Layer/grid confirmation and duplicate-cleaned Mistral measurements | `results/mistral_grid_search/`, `docs/mistral-grid-search.md` | Different data/selection protocol; catalogued, not pooled into the retained shared benchmark. |
| Activation interventions, rescaling, transplantation, mediation, readout decomposition | `docs/mechanistic-results.md`, `docs/paired-mechanistic-results.md`, `docs/paired-relation-feasibility.md` | Mechanistic experiments, not comparable observational baseline predictors. Their historical settings and causal outcomes are not silently substituted into this appendix. |

Historical files are preserved to keep provenance reproducible. Their old
numbers are not included as current measurements. The manuscript's existing
main-text figures have an explicit historical-configuration notice; this
appendix is independently readable and contains only the current comparison.

## Protocol details that matter

- The common neutral text is exactly `\n{} is known for`. Native BOS/special
  tokens remain. Within-model lens baselines share ordered test rows and
  relation-prompt model-top-one target IDs. Tokenizer filtering can change
  example membership across models; “shared” does not mean all models have
  the same example strings.
- Object filtering removes **all rows of different complete objects whose
  first answer token collides**. Multi-token subjects and objects remain
  eligible when this criterion is met. Eight demos, eight references and
  up to 100 tests are drawn with seed 0 after filtering. Repeated complete
  objects and repeated rows remain; same-subject reference directions are
  excluded at readout time.
- Neural target ranks count strictly greater logits and start at zero. The
  relation score is negative median `log10(rank+1)`, not log of median rank.
  Wikipedia frequency ranks instead use full-vocabulary average ties.
- The Geva metric filters *after* top-50 selection, retaining the source's
  asymmetric case handling. Empty retrieval is scored, not dropped. The
  common Wikipedia query vocabulary is an explicit adaptation; it is not a
  claim to reproduce the source models or source study's fact-selection rule.
  Pythia has 413 undefined rows with no surviving predictions; these are
  excluded from attribute-rate means. A separate sensitivity restricts every
  readout and the LRE outcome to the same defined rows, requiring at least
  two per relation. Other models have no undefined attribute rates.
- Corpus means use common support where both marginals are nonzero. Missing
  denominators are not encoded as zero probabilities. Repeated test rows keep
  their test weight when joined to a unique corpus fact. Literal-name counts
  are not disambiguated entity counts, sequence-window counts, or exact
  training exposure.
- LRE hyperparameters and tuned affine beta are selected on scored data.
  Correlations and p-values are exploratory. Bootstrap intervals resample
  relations jointly, not independent model training runs.
- Scalar cached logit readout and batched readout can move several almost-tied
  tail tokens. Every difference is recorded with its crossed logit gap and
  gated at `1e-5 * max(1, abs(target_logit))`; primary ranks remain the saved
  batched ranks. The Wikipedia attribute metric uses cached scalar top-50
  predictions. No target or subject mismatch is accepted.

## Reproduction

From `code/`, using the locked `.venv` environment:

```sh
# The current differential/affine runs are already complete.
.venv/bin/python scripts/report_all_model_newline.py

# One pinned model per GPU worker; 27B requires multiple GPUs in float32.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_shared_baselines.py \
  --name olmo3_7b --worker example --batch-size 4
# Repeat for gptj, mistral7b, gemma2_9b, qwen3_8b_base,
# qwen3_8_27b and pythia6_9b. Relation claims allow resumable workers.

.venv/bin/python scripts/retrieve_shared_wikipedia.py --workers 48
.venv/bin/python scripts/validate_shared_wikipedia.py
.venv/bin/python scripts/score_shared_wikipedia.py --name olmo3_7b
# Repeat the scoring command for all seven model names.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_shared_prominence.py \
  --name olmo3_7b --worker example_prominence
# Repeat for gptj, the other model in the original prominence analysis.

.venv/bin/python scripts/report_shared_baselines.py
cd ../paper
latexmk -pdf -interaction=nonstopmode -halt-on-error shared_baselines.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error lens.tex
```

The report validates model/data/source hashes, source split fingerprints,
ordered subjects, target IDs, gold IDs and all fresh aggregate ranks before
writing tables. Missing model/relation files are errors. `summary.json`
records source hashes and pinned model revisions; workers retain numerical
verification results. `wikipedia_validation.json` independently checks new
sparse token counts, BM25 scores and retrieval IDs/scores against BM25S on
10,000 real paragraphs and 20 queries. Saved corpus and model caches under
`generated/` are local reproducible artifacts, not included in Git.

The exact executed measurement drivers are archived under
`results/shared_baselines/provenance/executed_sources/`. The maintained
drivers were subsequently formatted and their imports cleaned. The archive
manifest records both hashes and verifies identical non-import syntax trees;
worker metadata retains the original execution hash. The report checks the
matching immutable source snapshot rather than rewriting execution history.

## Artifacts

- `results/shared_baselines/README.md`: curated Spearman tables.
- `correlations.csv`: complete Spearman/Pearson coefficients, p-values,
  sample counts, cohort, corpus stage, subset and outcome labels.
- `relation_scores.csv`, `subject_scores.csv`: full current measurements.
- `corpus_relation_scores.csv`: each corpus stage, overlap subset and coverage.
- `geva_matched_relation_scores.csv`: all readouts and the LRE outcome
  restricted to identical Geva-defined rows.
- `paired_gaps.csv`: paired differences and 95% bootstrap intervals.
- `sensitivity.csv`, `robustness.csv`, `prominence.csv`: auxiliary analyses.
- `summary.json`: provenance and audits; `launch_manifest.json`: job history.
- `paper/generated/shared_baselines/`: generated LaTeX tables.
- `paper/appendix_shared_baselines.tex`: definitions and interpretation.
- `paper/shared_baselines.pdf`: standalone appendix; `paper/lens.pdf`: paper
  including the appendix.
