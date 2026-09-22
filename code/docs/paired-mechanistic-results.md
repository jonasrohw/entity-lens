# Same subject, different relations: completed mechanistic analysis

**We found a relation-sensitive attention head and a partial causal connection
between subject-state directions and that head.** We did not establish two
independent factual channels, or a definitive explanation of the LRE performance
gap. The larger paired dataset produced a weaker LRE contrast than the original
small, unmatched samples.

The principal evidence concerns country capital versus currency in
`allenai/Olmo-3-1025-7B`. Runs used the pinned checkpoint
`a81bae42db3975be1671e27b9c9a56da1a9f980f`, float32, eager attention, and disabled
TF32 on eight RTX A6000 GPUs on `rho2`. No quantization, mixed precision,
attribution approximation, or CPU offloading was used.

## Data and controls

The sourced corpus contains 109 paired countries. Both relations have identical
subject partitions: 8 demonstrations, 8 Jacobian references, 24 direction-training
subjects, 16 validation subjects, and **53 confirmation subjects**. Confirmation
countries were absent from the previous country datasets. Source-agreement and
first-token filtering details are in the [feasibility audit](paired-relation-feasibility.md).

The controlled prompts share the complete prefix through the subject and put
the relation afterward, for example `France has the capital city of` and
`France uses the currency called`. Both receive the same combined demonstration
prefix. Subject states were **numerically identical, with maximum absolute
difference zero**, at layers 5, 11, 17, and 23 for every reference, training,
validation, and confirmation subject.

Faithfulness means matching the original model's first predicted answer token.
It is distinct from factual accuracy or semantic equivalence of complete answers.
The model matched the dataset first token for 52/53 capital cases and 36/53
currency cases in the controlled condition; **35 subjects were correct for both**.
All confidence intervals below resample paired subjects. Layers, prompt variants,
heads, and repeated interventions are not treated as independent observations.

## 1. Does the same subject have high LRE performance for one relation and low performance for another?

| Prompt condition, layer 17 | Capital LRE | Currency LRE | Paired gap | Exact McNemar p |
|---|---:|---:|---:|---:|
| Original relation-specific prompts | 37.7% | 22.6% | 15.1 points | .096 |
| Identical subject state, relation follows subject | 30.2% | 13.2% | 17.0 points | .078 |
| Shared state plus explicit currency-unit wording | 34.0% | 20.8% | 13.2 points | .143 |

The primary controlled layer-17 gap has a bootstrap 95% interval of **0.0 to
34.0 points**; its exact paired test is not significant at .05. The bootstrap
interval's discrete endpoint at zero should not be interpreted as stronger
evidence than that paired test.

Selecting each relation's layer using validation chooses layer 11 for capitals
and layer 17 for currencies. On untouched confirmation subjects this gives
**34.0% versus 13.2%**, a **20.8-point** gap (95% CI 5.7–37.7; exact paired
p=.0266). This is a secondary analysis. Neither absolute rate is particularly
high, so this cohort is not an extreme high-LRE/low-LRE contrast. Enlarging the
validation gain grid through 10,000 does not change the layer-17 results.

Concrete controlled layer-17 examples, with both original model answers correct:

| Subject | Capital model / LRE first token | Currency model / LRE first token |
|---|---|---|
| Ireland | Dublin / Dublin | Euro / Irish |
| Finland | Helsinki / Helsinki | Euro / Finnish |
| Portugal | Lisbon / Lisbon | Euro / Portuguese |
| Austria | Vienna / Vienna | Euro / Austrian |

These first-token mismatches do not by themselves determine whether a possible
full continuation would be semantically wrong. The larger gap in high-confidence
cases is descriptive; it is not substituted for the primary whole-cohort test.

## 2. Do the relations use different directions in the shared subject representation?

At each sampled layer, we learned two rank-eight subspaces from exact full-model
answer-versus-donor-token gradients on the 24 training subjects. We removed each
subspace under both relations at half and full strength. Three random subspaces
matched dimensionality and each subject's perturbation norm.

At layer 17, full-strength removal caused the following additional full-model
log-probability damage beyond random removal (nats):

| Evaluated relation | Remove capital-associated directions | Remove currency-associated directions |
|---|---:|---:|
| Capital | .0096 | .0052 |
| Currency | .0621 | .1158 |

Within each relation, its own subspace has the larger mean effect. However,
capital's own-versus-other advantage is uncertain at layer 17. The overall
interaction is .0581 nats, with **Holm-adjusted p=.0795** across the four layers.
Layer 5's interaction survives this correction (**p=.0192**).

In the prespecified 35-subject both-model-correct subset, the layer-17 interaction
is .0985 nats (95% CI .0438–.1572; Holm p=.0036). Currency's own-direction advantage
also survives correction across the eight directional tests; capital's does not.
The margin metric provides additional evidence of relation-dependent sensitivity,
but it also does not establish a robust two-way dissociation across the complete
cohort. At layer 17, the targeted removals preserve all capital top tokens and
change only 1/53 currency top tokens for either subspace.

**Conclusion:** useful directions are partly relation-selective. The data do not
establish two independent channels or isolated stored attributes. The learned
subspaces overlap, and effects depend on the metric and model-correctness subset.

## 3. Is the low-LRE relation more nonlinear?

For each confirmation subject, we interpolated its subject state toward a training
donor with different answers for both relations. The donor and input displacement
were identical across the two relation conditions. We compared actual outputs
with exact local Taylor predictions and the reference-averaged Jacobian response.

At layer 17, after a full donor-state displacement, local Taylor residual divided
by actual output-change norm was **.917 for capital and .923 for currency**.
The paired difference was .0063 (95% CI −.0097 to .0227). Full-displacement
comparisons at the other three layers also did not establish greater currency
nonlinearity. Some smaller-step differences were small and condition-dependent.

Local predictions were substantially better than the reference-averaged response
for small displacements. For example, at a quarter displacement at layer 17,
relative residuals were approximately .36/.37 locally versus 1.03/1.03 for the
averaged operator. This suggests limited transfer of a single averaged operator
across subjects/contexts, in addition to curvature along long state changes.

**Conclusion:** both relations are nonlinear over large changes. Greater currency
nonlinearity is not a convincing general explanation for its lower LRE score.
The experiment examines particular state paths, not all possible perturbations.

## 4. Do they depend on different model components?

We replaced clean attention or MLP updates with matched-donor updates at all 32
blocks, separately at the final subject token, earlier subject tokens, relation
suffix, and answer position. Empty or unequal-length token groups were excluded
with denominators retained. Validation nominated attention at the answer position
in blocks 26 and 22; every head at those two sites was then tested on confirmation
subjects.

**Block 22, head 18 stood out** (block numbering is one-based; head numbering is
zero-based, so this is `model.layers.21`, attention head 18):

* Donor-head replacement changed capital target log probability by approximately
  **+.0010 nats** and currency target log probability by **−.1529 nats**.
* Currency suffered **.1539 nats more damage** (95% CI .0974–.2246), surviving
  Holm correction across all 64 tested heads (**p=.0064**).
* The difference remained in the both-model-correct subset: **.1225 nats**
  (95% CI .0704–.1777; Holm p=.0064).
* The answer-versus-donor logit-margin contrast points the same way: currency
  suffers .6873 more logit units of damage (95% CI .4534–.9303).
* The layer-17 subject representation is unchanged by this downstream patch.

Only **one of 64 heads** passes the corrected probability-contrast test, both in
the full cohort and in the model-correct subset. The whole-block 26 probability
contrast does not persist in the model-correct subset, and its margin contrast
points in the opposite direction. Even block 22's whole-attention margin contrast
is uncertain. Thus the head-level evidence is more specific than a general claim
that currency relies more on entire attention blocks.

**Conclusion:** there is an identifiable relation-sensitive downstream component
operating on a shared subject representation. This does not show wholly separate
circuits or prove that this head causes the baseline LRE performance gap. Head
effects are not additive fractions of the whole block's effect.

## 5. Does the subject-direction effect pass through that head? Exploratory follow-up

After selecting head 22.18 from the confirmation scan, we removed the learned
layer-17 directions and restored that head's clean input to the attention output
projection at the answer position. Three seeded same-block heads (7, 23, 24)
provided restoration controls.

For currency predictions after removing currency-associated directions:

* Removal caused **.1158 nats** of mean target log-probability loss.
* Restoring head 22.18 recovered **.01975 nats**, approximately **17.1% of that
  mean intervention-induced loss**.
* Random heads recovered **.00054 nats** on average.
* Excess recovery was **.01921 nats** (descriptive 95% CI .00957–.02992).
* Recovery also appeared in the margin metric and both-model-correct subset.

The analogous restoration for capital predictions was near zero. The head also
recovered some currency damage caused by removal of the capital-associated
subspace, consistent with overlapping rather than independent channels.

**Conclusion:** this supplies a partial causal connection from the tested subject
directions through a relation-sensitive head. It does not account for most of
the removed signal, and 17.1% is not a fraction of all currency knowledge or the
LRE accuracy gap. Because the head was chosen using this cohort, the follow-up
is explicitly exploratory and requires independent replication.

## 6. What does the affine prediction appear to miss? Exploratory readout analysis

In the controlled condition, **28 of 46 currency LRE mismatches** begin with a
country name or a source-listed demonym. Explicit unit wording reduces this to
**11 of 42**, showing that output wording changes the error pattern substantially.

We decoded the real model's answer-position state at every block with the final
normalization and unembedding. At block 26, the readout matches the final model
token for **34.0% of capital cases and 7.5% of currency cases**. Country/demonym
tokens are top-ranked for **62.3% of currency cases**. At that same block, the
readout matches the currency LRE's incorrect token in **23 of its 46 failures**.
The currency answer becomes directly readable mainly in the last few blocks.

This pattern is consistent with some LRE outputs resembling an intermediate,
entity-associated readout that the full model subsequently transforms. It does
not prove that the affine model is literally executing or stopping at that stage.
An untuned logit lens does not measure calibrated intermediate beliefs, and
unreadability at an early layer does not establish absence of information.

## Paper-level interpretation

> Holding subject representations fixed, capital and currency queries exhibit
> partially relation-selective causal sensitivities. A downstream attention head
> contributes more to currency predictions, and exploratory restoration connects
> part of the subject-subspace effect to this head. These findings support
> relation-dependent decoding of a shared subject representation, without
> establishing independent factual channels or subject-representation dominance.

The strongest result is the specific head intervention and its model-correct
robustness. The mechanistic explanation of the LRE gap remains partial: the
primary fixed-layer gap is uncertain, output wording matters, broad curvature
does not distinguish the relations well, and one head explains only part of a
particular intervention effect. Results cover one model, one split, one relation
pair, and first answer tokens. Shared currency classes and a common demonstration
context can introduce dependencies beyond the conditional subject bootstrap.

## Artifacts and verification

The run contains **50,672 intervention measurements** across channel removal,
interpolation, block/head corruption, and exploratory head restoration; these
are repeated measurements on 53 confirmation subjects, with an additional
16 validation subjects in the block scan. There are also 3,392 intermediate
readouts. Identity patches and repeated ablations reproduce saved outputs
exactly; the largest central finite-difference relative discrepancy is 1.3%.
Nine implementation/statistical checks passed; four pre-existing multi-GPU
placement tests were skipped because these runs used one complete model per GPU.

* [Frozen primary protocol](../generated/paired_mechanistic_20260920/PROTOCOL.md)
* [Explicit exploratory addendum](../generated/paired_mechanistic_20260920/EXPLORATORY_ADDENDUM.md)
* [Full numerical report](../generated/paired_mechanistic_20260920/analysis/REPORT.md)
* [Primary measurements and intervals](../generated/paired_mechanistic_20260920/analysis/summary.json)
* [Exploratory measurements and intervals](../generated/paired_mechanistic_20260920/analysis/supplementary.json)
* [All matched layer-17 first-token predictions](../generated/paired_mechanistic_20260920/analysis/matched_cases.csv)
* [Attention-head comparison](../generated/paired_mechanistic_20260920/analysis/attention_heads.pdf)
* [Head restoration](../generated/paired_mechanistic_20260920/analysis/head_restoration.pdf)
* [Intermediate readout trajectories](../generated/paired_mechanistic_20260920/analysis/intermediate_readout.pdf)
* [Completeness audit and file hashes](../generated/paired_mechanistic_20260920/PROVENANCE.json)

Model stages are reproducible with `python -m jlens.paired_mechanistic` using the
saved manifest; exact commands are retained in each `*_jobs.json`. The supervisor
script records the stage order. Rebuild the CPU report with
`python -m jlens.paired_mechanistic_report generated/paired_mechanistic_20260920`
and the supplementary report with `python -m jlens.paired_supplement_report`
followed by the same directory. Validate with `python -m jlens.paired_mechanistic_validate`
followed by that directory. All commands run from `code/` using its existing environment.
