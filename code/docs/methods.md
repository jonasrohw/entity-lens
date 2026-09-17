# Experimental protocols and result schema

## Measurements

The model is `allenai/Olmo-3-1025-7B`. The Jacobian lens uses hidden-state index
17 and the neutral prompt `"{} is known for"`. Index 17 refers to the output of
transformer block 16 in Transformers' hidden-state tuple.

The differential lens averages `J_i (s - s_i)` over reference subjects. The
affine form also adds each reference output `f(s_i)`.

Three baselines drop one or both of the lens's two ingredients, at the same
layer, neutral prompt, reference subjects, and target tokens:

| Readout | Jacobian | Difference | Command |
| --- | --- | --- | --- |
| `mean_i J_i (s - s_i)` | yes | yes | `measure-jacobian-lens` |
| `mean_i J_i s` | yes | no | `measure-jacobian-lens-raw` |
| `s - mean_i s_i` | no | yes | `measure-logit-lens-diff` |
| `s` | no | no | `measure-logit-lens` |

LRE faithfulness is the
fraction of subjects for which the LRE's top vocabulary token equals the full
model's top vocabulary token under the relation's in-context prompt. It does
not measure accuracy against a restricted object list.

The LRE sweep uses layers `5, 11, 14, 17, 20, 23` and beta values
`0.5, 1, 1.5, 2, 3, 5`. It averages each cell across trials before selecting a
maximum. The retained paper results use one trial. They should not be described
as reproducing the 24-trial experimental setting of Hernandez et al. (2024).

The refactor preserves the original normalization and decoding arithmetic.
Jacobian products use double backward with eager attention and float32 by
default. Prominence and answer-prior forward passes use bfloat16.

## Splits

| Protocol | Demonstrations | References | Test subjects |
| --- | --- | --- | --- |
| Legacy lens | First 8 after `Random(17)` shuffle | First 8 scored subjects, skipping self-comparisons | Next up to 20 subjects |
| Legacy LRE | First 8 after `Random(0)`, then `Random(1000 + trial)` shuffles | Next 8 | Next up to 100 |
| Shared | First 8 after `Random(seed)` shuffle | Next 8, disjoint from test subjects | Next up to 100 |

Legacy lens and LRE splits are independent. Their results support a
relation-level comparison; they do not establish a matched per-subject analysis.
Shared-split results record a fingerprint of ordered subject names in all three
partitions. Subject-level analyses reject mismatched fingerprints.

The identifying-first-token filter keeps an object only if its first token
uniquely identifies that object within the relation. Repeated occurrences of
the same object are allowed. The shared, filtered comparison contains 25 usable
relations; the smallest relations cannot supply the required split.

For the two presidential-year relations, a leading space can become the target
instead of a content token. The main analysis prefers the `_ts` LRE runs and
`jacobian_lens_years.json`, measured with a trailing space in the relation query.
The neutral lens prompt is unchanged.

## Statistics and controls

The main lens score is the negative median `log10(rank + 1)` of the model's
answer token. Reported correlations are Spearman correlations. Partial
correlations regress ranked variables on ranked controls with an intercept,
then correlate their residuals. Permutations use NumPy seed 17.

Object cardinality is the number of samples divided by the number of distinct
objects. The answer-prior control averages log-probabilities over ten neutral
contexts. The prominence control generates up to 90 tokens from each of two
prose prompts for at most 40 subjects, tests for distinctive object words, and
separately reports name-leakage corrections.

## Retained JSON schemas

JSON field names are intentionally unchanged from the source snapshots:

| Field | Meaning |
| --- | --- |
| `rank_diff`, `rank_lre`, `rank_logit`, `rank_logit_diff`, `rank_raw`, `rank` | Zero-based vocabulary rank; rank 0 is top-1 |
| `n_references` | Reference subjects averaged for a row, after skipping self-comparisons |
| `median_lograng` | Historical spelling for median `log10(rank + 1)` |
| `best_faith`, `best_layer`, `best_beta` | Best LRE grid-cell faithfulness and its parameters |
| `per_subject` | Binary LRE/full-model agreement for each subject |
| `split_fingerprint` | Ordered demonstration/reference/test subject fingerprint |
| `prior_modeltop1`, `prior_gold` | Mean neutral-context log-probability of answer tokens |
| `rels`, `faith`, `lens`, `k`, `prior` | Aligned arrays in the Figure 1 input |

Legacy lens files may have a null fingerprint. In multi-trial LRE runs, the grid
is averaged across trials while `per_subject` and the fingerprint describe the
last trial; fact-rank reconstruction is intended for the single-trial shared
protocol used in the retained snapshots.
