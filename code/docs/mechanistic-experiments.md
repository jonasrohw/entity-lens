# Mechanistic experiments, 2026-09-19

Exploratory protocol recorded alongside the run; the exact splits and settings
are frozen in the saved manifest. These are
experiments on OLMo-3-1025-7B, float32/eager attention, hidden-state boundaries
5, 11, 17 and 23. They are separate from the retained paper runs.

All 28 available relations are considered. Identifying-first-token filtering,
ambiguous-subject exclusion, and a minimum of 20 samples determine inclusion.
Seed 1909 creates disjoint sets per relation: 8 demonstrations (4 for datasets
with fewer than 48 surviving subjects), 4 Jacobian
references, up to 24 direction/intercept training subjects, up to 8 validation
subjects and up to 24 test subjects. Exact splits are saved in manifest.json.
No test answer selects a direction, beta, or layer. Splits are within relation,
not globally disjoint across all relations. Four fixed neutral references
(Ada Lovelace, Isaac Newton, Tokyo, Pacific Ocean) are excluded from every split.

The user's primary comparison and the intervention experiments use the exact
reference-centred Taylor form: mean_i[f(s_i) + beta J_i(s-s_i)]. The neutral
and relation maps use the same reference identities, in their respective
contexts, and are compared at beta 1 as well as after validation selection.

To isolate operator effects in the shared-operator experiment, readouts use a common training-centred
parameterization f_bar_train + beta W (s - s_bar_train). The relation operator
W averages four relation-prompt Jacobians. The matched neutral operator averages
Jacobians at the same four subjects under '{} is known for'. The global neutral
operator averages Jacobians at the four fixed references, shared across every
relation. This is a calibrated affine readout, not the paper's exact reference-
centred Taylor intercept. Keep this distinction explicit in the report.

1. Transformation comparison: raw affine-vector cosine at beta 1; centred
   native-input cosine; same-input operator-response cosine and relative error;
   shuffled-subject controls; vocabulary predictions. No full-matrix equality
   is claimed from agreement on these tested directions.
2. Neutral subject transplantation: replace the subject's final-token state
   at the same layer; hold the relation operator and its validation beta fixed.
   Also inject that state into the full model. Identity patch is the baseline.
   Rotate neutral donors across held-out subjects as an additional control:
   retaining the answer with the correct donor is informative only if the
   subject's identity/content matters relative to an incorrect donor.
3. Attribute erasure: use training-only neutral-state/one-hot-label cross
   covariance, at most 8 singular directions. Only classes with at least two
   training examples contribute; at least two such classes are required.
   Otherwise the attribute-erasure experiment is marked non-identifiable for
   this relation rather than treating subject PCA as an attribute direction.
   Erase their projection relative
   to the relation training mean. Compare three random subspaces of equal rank,
   scaling each perturbation to the targeted perturbation's per-example norm.
   Evaluate LRE and full model. Add back removed information as a numerical
   restoration check, not independent causal evidence. Alternate-relation
   prompts receive the identical perturbation where the dataset provides
   another relation of the same domain; these attributes can be correlated.
4. Shared operator: replace W_relation by the fixed global W_neutral. Both
   have the same training relation intercept/centering and validation beta grid
   (including beta zero, an intercept-only baseline). Thus successful transfer
   does not establish that relation-specific information is unnecessary.
5. Layer tracing: measure readouts and interventions at all four boundaries.
   At each boundary's block, replace the subject-token post-normalization
   attention or MLP update by its training-subject mean. Measure full-model
   effects and apply the fixed LRE to the resulting subject state. Restore the
   clean subject state as a correctness control. Since these component
   interventions change only the subject position, this rescue is expected
   by construction, not an independent mediation result. Other token paths
   remain available during ablation, so a small effect can reflect redundancy.
   This is a sparse
   layer scan, not exhaustive circuit localization.

Beta is chosen by validation top-1 agreement, with validation target log
probability breaking ties, then smaller beta. Report layer 17 as the primary
fixed-layer result, and validation-selected layer results separately. Full-model
targets are its original relation-prompt top tokens, not ground truth. Gold
first-token correctness is retained for a separate model-correct analysis.
Confidence intervals use paired, relation-clustered bootstrap and are
descriptive for this one split/model. Save per-subject results and vector
artifacts; do not modify the manuscript to claim proof from exploratory results.

Run from code/ using the existing environment:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m jlens.mechanistic --prepare --out generated/mechanistic_20260919
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m jlens.mechanistic --shard 0 --nshards 4 --out generated/mechanistic_20260919
```

Launch other shards on separate available GPUs. Completed relations are skipped
on restart; partially completed relations are recomputed.

The final run uses rho2 via SSH, with `--two-gpu` and GPU pairs (0,1) and (2,3),
and rho1 with `--model-gpus 3` and GPU groups (0,1,2) and (3,4,5).
Layers 0–15 and the input embedding reside on the first GPU; layers 16–31,
final norm, and output embedding reside on the second in the two-GPU case.
Three-GPU runs divide the 32 blocks into consecutive groups of 11, 11 and 10,
with the input embedding on the first and output embedding on the last.
All parameters and
computations remain float32 on GPU; no quantization, mixed precision, or CPU
offloading. Small-model tests compare single-GPU versus this placement for
forward values and Jacobian-vector products. Interrupted local attempts are
excluded from the final report.

When additional GPUs became free, the final product and superhero-archnemesis
relations moved to rho1 GPU 6, and star-constellation and superhero-person to
rho1 GPU 7, with the whole float32 model on each GPU. Supervisors stop the
original workers before reaching reassigned relations; their nonzero termination
codes are expected after completing their assigned subset.

## Supplementary checks recorded during analysis

The original grid remains reported. To check whether shared-operator transfer
is artificially limited by its output scale, `mechanistic_rescale` also selects
positive gains from 0 through 10,000 on the same validation subjects, applying
the same expanded grid to all three operators. It recreates the training
intercept and relation centering offset, rescores cached products, and requires
zero top-token mismatches and at most 0.001 nat log-probability error against
every saved original-grid cell before accepting the wider-gain result.
This is a supplementary control, not a replacement of the frozen primary grid.

The CPU report also evaluates a nearest-training-class-centroid classifier
inside the attribute subspace. This is restricted to held-out values with at
least two training examples; coverage and the training-majority baseline are
reported. It diagnoses whether the chosen subspace generalizes and does not
select a subspace or intervention using test labels. Reference-centred
differential-vector cosine is reconstructed from the saved affine vectors and
reference means, alongside the original affine and training-centred comparisons.
The same reference-centred neutral differential vectors are decoded directly
for a layer-wise answer-rank curve, without selecting a gain on relation data.

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m jlens.mechanistic_rescale generated/mechanistic_20260919_rho2 --wait
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m jlens.mechanistic_rescale generated/mechanistic_20260919_rho2 --differential-only
.venv/bin/python -m jlens.mechanistic_report generated/mechanistic_20260919_rho2
```

The final reporting command rejects incomplete primary measurements or missing
wider-gain controls; `--allow-partial` is only for monitoring.
