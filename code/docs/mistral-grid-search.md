# Expanded Mistral search

The five-model comparison did **not** exhaust hyperparameters: it used six
relative-depth LRE layers, six beta values, and one fixed lens layer. Its LREs
were already recomputed from each model's own reference states and Jacobians;
no GPT-J/OLMo operator was reused. Mistral-7B-v0.3 is the focus because its
completed main-protocol differential correlation was the strongest of the four
completed smaller models: rho=0.3245, p=0.0920 across 28 relations. This choice
uses observed results and the subsequent search is exploratory.

The expanded grid is fixed in `generated/mistral_full_grid/config.json`:

| Factor | Treatment |
| --- | --- |
| LRE state | Embedding state 0 and every block output 1–32 |
| Lens state | Independently search the same 33 states |
| Beta | 0, 1/16, 1/8, 0.25–8 in increments of 0.25, 10, 15, 20, 30, 50, 100 |
| Reference averaging | First 4 versus all 8 references, holding demos/tests fixed; main lens excludes self and can therefore use 3/7 |
| Lens intercept | Differential versus affine; zero-beta affine is a prior-only control |
| Neutral context | Original `{} is known for` versus bare subject `{}` |
| Prompt-matched control | Readout of the actual LRE predictions; shares predictions with the outcome, so is not independent lens evidence |
| Subject/reference split | Both original main and identifying-token-filtered shared-split protocols |
| Rank aggregation | Median log-rank is primary; mean log-rank, top-1, top-10 and MRR are sensitivity analyses |
| Hyperparameter selection | Report all-relation exploratory maxima separately from preassigned held-out relations and subject calibration/test selection |
| Numerics | HF normalization/output head, float32, exact batched derivatives, pretrained prefix-cache checks |
| Degenerate states/ties | Retain final-layer and zero-beta controls; flag constant logits and distinguish optimistic rank-zero from actual argmax agreement |

Beta is continuous, so a finite grid alone is not exhaustive. For Mistral's
bias-free output head and multiplicative RMSNorm, token ordering for an affine
state is the ordering of `a + beta*b`: the RMS denominator is a common positive
factor. The target token's top-1 region is therefore an interval in positive
beta. Intersecting its pairwise token inequalities and maximizing overlap
across subjects yields a continuous-beta LRE optimum. Candidate optima must be
checked with the actual float32 HF readout; floating-point boundary effects
are not an architectural mismatch.

The derivative products and intercepts are retained alongside each relation's
JSON so beta can be refined without refitting/recomputing its Jacobians.

Remaining factors are not silently assumed irrelevant: demonstration order,
new reference draws, reference counts above eight, longer/multiple-token
answers, factual gold targets versus model-top-1 targets, relation-specific
prompt rewrites, alternative operator estimators, nonlinear fits, and model
training differences. These would constitute further experiments. Fresh-draw
confirmation used frozen selected configurations without tuning on its own
reported outcome. A high correlation found among thousands of configurations
is not, by itself, evidence of a generalizable improvement.

## Completed results and interpretation

[The retained report](../results/mistral_grid_search/README.md) contains the
full analysis, compressed scoring records, selected settings, and figures.
All 28 relations were processed under both protocols. Main correlations use
28 relations; the shared protocol has 25 eligible relations after filtering
and minimum-size checks.

| Configuration | Main seed 0 | Main fresh seed 1 | Shared seed 0 | Shared fresh seed 1 |
| --- | --- | --- | --- | --- |
| Original LRE and lens | 0.325 | 0.387 | 0.625 | 0.681 |
| Expanded LRE, original lens | 0.357 | 0.479 | 0.710 | 0.673 |
| Expanded LRE, its best exploratory lens | 0.578 | 0.018 | 0.761 | 0.585 |

These are Spearman correlations, not LRE accuracies. On seed 1, layer/beta
settings stay frozen and the operators are fitted again using new
demonstrations/references. Test subjects partly overlap, so this is a
same-dataset robustness check rather than external validation. The main and
shared protocols use different cohorts and cannot be compared as a controlled
single-factor experiment. The best exploratory lenses differ between protocols.

The separate main winner against the original LRE target reaches 0.631
(bare subject, affine lens state 10, lens beta 30, eight references), but falls
to 0.286 on seed 1, below the corresponding original lens's 0.387. Lens tuning
therefore did not establish a reproducible improvement. Expanding LRE fitting
helps the fixed main lens on both draws, but is not uniformly better under the
shared protocol.

The original six-state/six-beta grid achieves mean main LRE faithfulness
0.561. Expanding only beta raises it to 0.640; expanding only the states raises
it to 0.617; expanding both raises it to 0.666. Continuous positive beta then
raises it to 0.668. Better LRE agreement does not necessarily produce a larger
correlation across relations. Per-relation LRE settings maximize faithfulness;
the lens settings are selected separately by correlation.

Three main relations have only three LRE test subjects. Both year relations
also have perfect intercept-only first-token agreement. Without tuning, the
original method has rho 0.637 on the 25 relations with at least four tests.
This diagnoses sensitivity to the evaluation cohort, rather than replacing
the primary all-28 result. Prior-normalized outcomes and subject-calibrated
LRE settings are retained as explicitly separate sensitivity analyses.

A final input audit found duplicate subjects in nine relations, including
natural multi-answer subjects. For example, the original main superhero split
shares three subjects between references and tests; the original shared product
split shares one. Row-wise splitting therefore did not always separate subjects.
An additional seed-1 sensitivity collapses identical subject/object rows and
removes subjects with multiple gold objects before splitting, with hyperparameters
still frozen. Only 17 relation/protocol measurements require recomputation;
unchanged effective input pools reuse their existing seed-1 measurements.
Cleaning can change all shuffled partitions, so this is not an isolated causal
estimate of leakage bias. Its results and complete overlap audit are in the report.

With the original lens unchanged, cleaned seed-1 main correlation improves
from 0.280 using the frozen original LRE settings to 0.471 using frozen expanded
LRE settings (28 relations). Cleaned shared correlation improves from 0.665 to
0.761 (25 relations). The shared expanded-target lens winner instead reaches
0.572. These checks favor broader LRE fitting over the in-sample lens winner,
but do not establish performance on an independent dataset or across many seeds.

The pretrained regression audit checks 1,944 overlapping LRE cells with no
faithfulness differences and 4,054 individual ranks with 12 changes of one
position. Splits agree exactly. All 6,798 continuous-beta candidate optima
were verified using the actual HF float32 output normalization and head.

## Reproducing the analysis

From `code/`, the completed generated records can be analyzed without loading
the full model:

```sh
.venv/bin/python scripts/audit_mistral_grid.py
.venv/bin/python scripts/report_mistral_grid.py --force
.venv/bin/python scripts/report_mistral_confirmation.py
.venv/bin/python scripts/report_mistral_confirmation.py --root generated/mistral_shared_confirmation
.venv/bin/python scripts/report_mistral_confirmation.py --root generated/mistral_clean_main
.venv/bin/python scripts/report_mistral_confirmation.py --root generated/mistral_clean_shared_split
.venv/bin/python scripts/retain_mistral_grid.py
```

`scripts/run_mistral_grid.py` runs resident GPU workers with atomic per-relation
claims. The retained configuration and relation folds define the experiment.
`scripts/refine_mistral_beta.py` reuses saved derivative products and loads
only the final normalization and output head. The larger `.pt` products remain
in the generated directory; compressed JSON scoring records and their SHA-256
manifest are retained under `results/mistral_grid_search/`.
