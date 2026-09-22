# Mechanistic experiment findings — 19 September 2026

The experiments support useful, causally relevant information in neutral
subject states. They do **not** establish that the subject representation is
the dominant explanation of LRE performance, or that neutral and
relation-specific transformations are nearly identical.

All five experiment types were completed on OLMo-3-1025-7B: **24 relations,
426 held-out subject–relation pairs, and layers 5, 11, 17 and 23**. All final
model runs used float32 and eager attention on RTX A6000 GPUs on `rho1` and
`rho2`, without quantization or CPU offloading. Primary comparisons below are
at layer 17, with equal weight per relation. Faithfulness means matching the
original model's first predicted token, not necessarily the dataset answer.

## 1. Comparing the affine transformations

The neutral and relation-specific affine predictions had mean cosine
similarity **0.371**. Shuffling subject identities produced **0.366**. Thus,
almost all of this raw similarity was also present for mismatched subjects.

There is shared subject-dependent signal: after removing common offsets,
native-input cosine was **0.225**, versus **0.014** for shuffled subjects.
Applying both operators to exactly the same subject directions gave cosine
**0.301** (95% CI 0.268–0.334), with relative prediction-vector error **1.127**.
The reference-centred differential vectors also aligned above their shuffled
control: **0.201 versus 0.048**.

**Interpretation:** there is shared structure, but these results do not support
"the transformations are very similar." These are comparisons of operator
actions on held-out subject directions, not equality tests over all 4,096
input dimensions.

## 2. Transplanting a neutral subject representation

Original LRE faithfulness was **57.1%**. Replacing its subject state by the
same subject's neutral state retained **50.4%**, a **6.7 percentage-point**
drop (paired 95% CI: 2.5–11.4 points). Using another held-out subject's neutral
state instead gave **15.7%**.

In the full model, the correct neutral donor preserved the original answer
**79.3%** of the time, compared with **56.0%** for a wrong-subject donor. The
paired advantage was **23.3 points**. On dataset-correct cases, neutral-state
LRE faithfulness was **52.8%**, compared with **59.1%** originally.

As another control, removing the subject-dependent term entirely reduced
original LRE faithfulness from **57.1% to 21.1%**.

**Interpretation:** this is the strongest evidence for useful information
already present without a relation prompt. It does not distinguish a stored
fact from an entity identifier that enables downstream retrieval, and it
does not compare every other possible information pathway.

## 3. Removing and restoring attribute directions

This analysis was identifiable in **13 relations / 264 test pairs**, where
training data contained at least two replicated attribute classes. Directions
were learned only from training subjects' neutral states. Three random
subspaces supplied controls with matched rank and per-example perturbation norm.

Targeted removal reduced full-model answer agreement by **2.9 points**, versus
**0.5 points** for random controls. The LRE faithfulness reduction was **1.9
points**, with a 95% interval spanning an improvement of 1.9 to a reduction
of 5.4 points. Probability-based measures showed small extra targeted damage:
**0.060 nats** for the LRE and **0.008 nats** for the full model.

However, alternate relations were also affected. The paired target-minus-
alternate specificity contrast was **−0.014 nats** (95% CI −0.035 to +0.001).
A supplementary held-out classifier in the learned subspace scored **61.1%**
against a **42.5%** training-majority baseline on covered attribute values,
but the between-relation interval on this advantage included zero.

**Interpretation:** some causal effect exists, but clean attribute-specific
erasure was not established. These interventions do not supply a strong proof
that a particular factual attribute is the decisive content. Adding the
removed vector back restores the computation by construction and is a
numerical correctness check, not independent evidence.

## 4. Replacing relation-specific matrices with one neutral matrix

For this experiment, all matrices received the same training-based centering
and relation-specific intercept; their scalar gains were selected on validation
subjects. Consequently its baseline differs from the original Taylor LRE above.

With the initial gain grid, the calibrated relation matrix achieved **61.9%**
faithfulness and one globally shared neutral matrix **49.6%**. Expanding the
gain range through 10,000 changed these to **61.9% and 50.5%**. The remaining
paired gap was **11.4 points** (95% CI 4.7–18.7 points). Bias alone achieved
**24.6%**. Matched-reference neutral matrices achieved **47.4%** with the wider
gain range.

**Interpretation:** a shared neutral transformation can recover substantial
performance beyond the bias baseline, but relation-specific matrix choice
still matters. The gap is not explained solely by the initial gain limit.
Success here still uses a relation-specific intercept and calibration.

## 5. Layer tracing and component interventions

| Hidden-state layer | LRE faithfulness | Neutral differential object rank* |
|---|---:|---:|
| 5 | 40.2% | 4,635 |
| 11 | 47.2% | 1,554 |
| 17 | 57.1% | 845 |
| 23 | 37.9% | 903 |

*Geometric mean vocabulary rank; lower is better. The differential readout
uses no relation-selected gain or restored prompt-output prior.*

At layer 17, replacing the subject-token MLP update with its training mean
reduced LRE faithfulness by **4.3 points**, versus **0.2 points** for the
attention update. Full-model answer retention was **94.9%** and **99.7%**,
respectively. Restoring the complete subject state restores the altered
pathway by construction.

**Interpretation:** the selected MLP updates contribute more under this
intervention. Neutral answer accessibility improves toward the middle layers,
but the similar neutral ranks at layers 17 and 23 accompany a substantial LRE
faithfulness decline. This pattern cautions against explaining layerwise
faithfulness from neutral accessibility alone. Four sampled blocks do not constitute a complete circuit
localization or a comparison of every causal pathway.

## What the paper can claim

> LREs exploit information available in neutral subject representations.
> Replacing relation-conditioned subject states preserves much of their
> faithfulness, while incorrect-subject controls substantially reduce it.
> Relation-specific decoding transformations nevertheless make an additional
> contribution, and our interventions do not establish subject-representation
> dominance.

These are exploratory results from one model and one split. Four datasets
were excluded for insufficient filtered examples; the smallest included
relations have only 4–8 test subjects. Intervals bootstrap relations and do not
capture split variation or all dependencies between related relations. All
measurements concern identifying first tokens, not complete object strings.

## Reproduction and artifacts

- [Protocol and commands](mechanistic-experiments.md)
- [All measurements and confidence intervals](../generated/mechanistic_20260919_rho2/analysis/REPORT.md)
- [Per-relation CSV](../generated/mechanistic_20260919_rho2/analysis/by_relation.csv)
- [Transformation and transplant figure](../generated/mechanistic_20260919_rho2/analysis/comparison_and_transfer.pdf)
- [Gain-range control figure](../generated/mechanistic_20260919_rho2/analysis/gain_control.pdf)
- [Erasure and intervention figure](../generated/mechanistic_20260919_rho2/analysis/erasure_and_layers.pdf)
- [Layer accessibility figure](../generated/mechanistic_20260919_rho2/analysis/neutral_layer_access.pdf)
- [Provenance and checksums](../generated/mechanistic_20260919_rho2/PROVENANCE.json)

The eight implementation checks passed. Every completed measurement passed
split, identity-patch, restoration, and intercept checks. Recomputing the
original gain-grid predictions during the scale control produced **zero
top-token mismatches** and maximum log-probability error **2.87 × 10⁻⁶ nats**.
Interrupted local runs are retained separately and excluded from these results.
