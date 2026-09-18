# Verification: the final-norm fix raises the headline correlation to 0.771

**Date:** 2026-09-18. **Host:** `rho2`, 8 × NVIDIA RTX A6000 (49 GB), driver 615.71.09.
**Code:** commit `da4b6ce` plus one uncommitted change to
`jlens/measure_jacobian_lens_raw.py` (§5). **Environment:** Python 3.12.12,
torch 2.14.0+cu130, transformers 5.17.0, `uv sync --locked --extra gpu`.
**Model:** `allenai/Olmo-3-1025-7B`, float32, eager attention.

## 1. Summary

An independently re-measured OLMo pipeline reproduces the reported
**ρ = 0.771** exactly, and the reported logit-lens (0.231), differential
logit-lens (0.284), cardinality (0.362 / −0.033 / 0.840), permutation (0/10000)
and leave-one-domain-out (0.737…0.816, median 0.772) lines to every printed
digit. The increase over the retained paper value of 0.658 is real and is caused
by the final-norm double-application fix documented in
[`../jlens/BUG-final-norm-twice.md`](../jlens/BUG-final-norm-twice.md).

It decomposes as follows — the **y-axis (LRE faithfulness) contributes most of
the gain**, which is easy to miss because the bug is usually discussed as a lens
(x-axis) problem:

| lens `x` | faithfulness `y` | Spearman ρ | p |
| --- | --- | --- | --- |
| retained | retained | 0.6578 | 1.42e-04 |
| **fixed** | retained | 0.6775 | 7.47e-05 |
| retained | **fixed** | 0.7520 | 3.95e-06 |
| **fixed** | **fixed** | **0.7709** | **1.58e-06** |

One number does **not** reproduce: the raw Jacobian-lens baseline, reported as
0.745. That figure still carries the double-norm bug, because
`measure_jacobian_lens_raw.py` was written after commit `8650154` and never
received the fix. Corrected, it is **0.766** (§5).

## 2. The bug, verified directly on the real checkpoint

`HiddenStateIntervention.forward()` obtains "the state coming out of the last
transformer block" and downstream code applies the model's final RMSNorm to it
before unembedding. On this stack `hidden_states[-1]` is *already* normalized, so
the norm was applied twice. Checked on the actual 7B checkpoint
(`generated/raw_jacobian/check_norm.py`):

```
transformers hidden_states[-1] vs raw block output: max|diff| = 11.957
                                                    (‖raw‖ = 102.8, ‖hs‖ = 81.1)
hidden_states[-1] vs norm(raw block output):        max|diff| = 0.0
```

So `hidden_states[-1] == norm(raw)` to float32 exactness. This is not a
rounding-level issue: `model.model.norm.weight` ranges over **0.60 … 2.64**
(mean 1.35, sd 0.09) across its 4096 entries, so normalizing twice decodes
`w² ⊙ P·u` instead of `w ⊙ u` — a genuinely different direction, hence different
vocabulary ranks.

## 3. What was actually re-measured, and a correction to the record

The `rerun/` artifacts from 2026-09-17 are **pre-fix**, despite being dated after
the fix commit:

* All 30 `rerun/results/main/faithfulness_*.json` were written between 10:16 and
  11:58; the fix was committed at 11:49 and each sweep takes ~21 min, so every
  one of them imported unfixed code. Their `best_faith` values are identical to
  the retained snapshot.
* `rerun/results/jacobian_lens.json` (12:16) is bit-identical to the retained
  snapshot. Re-running the same command today with the same data and the current
  code does **not** reproduce it — see §4. The fix was therefore not active in
  that run either.
* Commit `659524a`'s message states that the regenerated legacy-split scores
  "match the retained snapshot bit-for-bit (`city_in_country` median_lograng
  0.47712125471966244)". That check used a single relation, and
  `city_in_country` happens to be one of the two relations the fix leaves
  unchanged. It should not be read as evidence that the pipeline was verified.

## 4. Re-measurement of the differential lens (x-axis)

```sh
# 8 shards, one per GPU; see generated/difflens_fixed/run_shards.sh
CUDA_VISIBLE_DEVICES=$g uv run --no-sync jlens --root code \
  measure-jacobian-lens --data-dir code/data/relations \
  --rels <shard> --legacy-split --out generated/difflens_fixed/shard_$g.json \
  --device cuda:0
# plus --trailing-space runs for the two year relations -> jacobian_lens_years.json
```

Subjects, prompt templates and model answer tokens are unchanged from the
retained snapshot; only the decoded ranks move, and they move in one direction.
Example, `person_occupation` (`rank_diff`, retained → fixed):

| subject | retained | fixed |
| --- | --- | --- |
| Jack Boucher | 4246 | 2205 |
| Ken Murray | 7476 | 3380 |
| Karl Hampton | 15754 | 6740 |
| Tom Terriss | 707 | 351 |
| Raoul Walsh | 9 | 9 |

Relation-level: the lens improves in 22/28 relations, is unchanged in 2, and the
mean `median_lograng` drops by 0.104. Largest moves: `person_native_language`
2.559 → 2.120, `person_occupation` 2.814 → 2.419, `presidents_election_year`
4.170 → 3.803.

## 5. Re-measurement of faithfulness (y-axis), and why it matters more

The same fix changes `base`, the LRE's own prediction, in
`measure_faithfulness.py`. All 30 legacy-split sweeps (28 relations + 2
trailing-space year variants) were re-run — 5.5 GPU-hours total, dynamic queue
across 8 GPUs, `generated/faith_fixed/run_faith.sh`, all rc=0.

`best_faith` changes in 15/28 relations (11 up, 4 down; mean +0.003, Spearman
between old and new 0.951). The mean barely moves but the *ordering* does, which
is what a rank correlation sees. Biggest single move: `name_gender` 0.667 →
0.333. This is why re-measuring `y` alone takes ρ from 0.658 to 0.752, while
re-measuring `x` alone only reaches 0.678.

## 6. The raw Jacobian-lens baseline

`measure_jacobian_lens_raw.py` (the "Jacobian without the difference" ablation,
`v_e = mean_i J_i s_e`) was added in `b9a80ea`, before the fix commit, and was
not among the four modules `8650154` patched. It still built
`HiddenStateIntervention(model)` with no `final_block`, so it differentiated the
post-norm state and then normalized again.

This is now fixed — `final_block = resolve_blocks(model)[-1]`, matching the
differential lens — alongside a `--rels` option so the relation loop can be
sharded across GPUs (the measurement is embarrassingly parallel; 8-way sharding
takes it from 8.5 min to ~6 min wall-clock, model load dominating).
`tests/test_baseline_measurements.py`, `tests/test_model.py`, `ruff check` and
`ruff format --check` all pass.

The baseline in §7's table was measured against the retained reference set,
which now lives at `results/olmo-old/`; the `results/olmo/` and `results/gptj/`
trees hold separate full reruns whose raw baselines are not comparable to it.

Consequence for the reported statistics block:

| Raw Jacobian baseline | ρ | p |
| --- | --- | --- |
| double-normed (as reported) | 0.745 | 5.45e-06 |
| **corrected** | **0.766** | 2.04e-06 |

Scientifically this sharpens an already awkward result: against the corrected
differential lens (0.771), the Jacobian **without** any reference subtraction
scores 0.766 — a gap of −0.005, with a paired bootstrap 95% CI over relations of
[−0.086, +0.077] (20000 resamples, seed 17). The two relation-level score vectors
correlate at ρ = 0.962. Whatever the lens is measuring, the differencing step is
not carrying it.

## 7. Independent reproduction of the full statistics block

`generated/fixed_pipeline/` is a self-contained `--root` holding the re-measured
main-protocol results. On CPU, with no model download:

```sh
cd code
uv run --no-sync jlens --root generated/fixed_pipeline statistics
```

Output, against the numbers under review:

| Line | Reported | This run |
| --- | --- | --- |
| Main (28 relations) | 0.771, p=1.58e-06 | **0.771, p=1.58e-06** |
| Logit-lens baseline | 0.231, p=2.37e-01 | **0.231, p=2.37e-01** |
| Differential logit-lens | 0.284, p=1.43e-01 | **0.284, p=1.43e-01** |
| Raw Jacobian-lens | 0.745, p=5.45e-06 | 0.766, p=2.04e-06 (§6) |
| Permutation | 0/10000 | **0/10000** |
| Cardinality k/faith, k/lens, partial | 0.362, −0.033, 0.840 | **0.362, −0.033, 0.840** |
| Answer prior partial | 0.771 | **0.771** |
| Leave-one-domain-out | 0.737…0.816, median 0.772 | **0.737…0.816, median 0.772** |

The `Answer prior: prior/faith` sub-value differs slightly (−0.019 reported,
−0.023 here) because this run reuses the retained `answer_prior.json` rather than
re-measuring it; those forward passes run in bfloat16 and are not affected by the
fix. The partial correlation, which is the quantity the paper uses, matches.

## 8. What this run does **not** verify

The shared-split protocol was not re-measured here. The last four lines of the
statistics block (`Shared split`, `Prominence`, and the per-relation fact-level
rows) come from retained copies, so this run prints the retained 0.644 / 0.614
where the report shows 0.624 / 0.596. The reported values are consistent with
those stages having been re-measured under the fix as well; confirming them
needs Phase B — 28 shared-split faithfulness sweeps, the shared-split lens, and
28 subject-rank runs, roughly 3 GPU-hours on 8 GPUs. Nothing in the main-protocol
verification depends on it.

Also unverified: whether the retained snapshots themselves came from the original
authors' code or from an earlier refactor state. All statements above are about
what the *current* code produces.

## 9. Artifacts

| Path | Contents |
| --- | --- |
| `generated/fixed_pipeline/` | Complete `--root` for the CPU check in §7 |
| `generated/difflens_fixed/` | Re-measured differential lens, shards, logs, runner |
| `generated/faith_fixed/` | Re-measured faithfulness, 30 runs, `status.tsv`, logs |
| `generated/raw_jacobian/` | Fixed raw-Jacobian baseline shards, logs, `check_norm.py` |
| `results/olmo-old/jacobian_lens_raw.json` | Corrected raw-Jacobian baseline for the retained snapshot |

`generated/` is gitignored; the runner scripts inside each directory contain the
exact commands and shard assignments used.
