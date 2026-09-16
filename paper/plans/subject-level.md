<!-- edited 2026-09-10 12:22 - stage 1 complete: splits aligned, per-subject outcomes stored, smoke test passes -->
<!-- edited 2026-09-10 12:03 - plan: move the unit of analysis from the relation to the subject -->

<!-- shared with collaborators as paper/plans/subject-level.md -->

# Plan: the subject as the unit of analysis

**Status: proposed.**  Predictions are pre-registered in §6 before any of
it is run, as in
[`lre-expansion-preregistration.md`](lre-expansion-preregistration.md).

## 1. The question

The paper currently pairs one lens score with one faithfulness value per
relation: 28 points.  Both quantities are, however, defined **per
subject**.  For subject $e$ of relation $r$ we have

- $y_{r,e} \in \{0,1\}$ -- did the LRE reproduce the model's own top-1
  for this subject?
- $x_{r,e}$ -- where the lens ranks that same token, as $-\log_{10}$ rank.

The question is whether $x$ predicts $y$ **within** a relation.

Why this is the strongest available next step:

- **Every relation-level confound is controlled by construction.**  Same
  prompt, same answer space, same cardinality, same in-context examples.
  The three paragraphs the paper spends partialling out $k$ and answer
  frequency become unnecessary — the design does that work.
- **n goes from 28 to ~1,000** (see §7).
- **It tests the claim the account actually makes.**  "Relations about
  salient entities have faithful LREs" is about relations; "the subjects
  whose representations carry the answer are the ones the LRE gets right"
  is about entities, which is the thesis.
- **The old per-triple trap does not apply.**  The earlier per-triple work
  ([`lens-discrimination.md`](lens-discrimination.md)) drowned in the
  question of whether a high lens rank meant parentage or mere
  association.  Here the target is the **model's own top-1**, not gold, so
  that question does not arise: if the model says *Zeus* and the lens
  ranks *Zeus* highly, they agree, whatever the reason.

## 2. Prerequisites (blocking)

**(a) Align the splits.**  `lens_vs_faith.py` shuffles with
`Random(17)`; `lre_faithfulness.py` shuffles with `Random(0)` and then
`Random(1000+trial)`.  The in-context, reference and scored sets
therefore differ between the two measurements, and there is currently
**no per-subject pairing at all**.  Both must take the split from one
place.  Suggested: a small shared helper that, given a relation and a
seed, returns `(icl, ref, test)`; both scripts call it.

**(b) Store per-subject outcomes.**  `lre_faithfulness.py` aggregates to
a rate (`trials[t,li,bi] = mean(...)`).  It needs to also write, at the
selected $(\ell, \beta)$ cell, a `per_subject` map
`{subject: 0/1}`.  A few lines.

**(c) Raise the lens subject count.**  `N_SUB = 20` in
`lens_vs_faith.py`; the faithfulness runs score up to 100.  Raise to
match, and confirm the two score *identical* subject sets with an assert,
not by eye.

## 3. Design

For each relation, at the $(\ell, \beta)$ cell already selected:

| | |
|---|---|
| unit | (relation, subject) |
| outcome $y$ | LRE top-1 $=$ model top-1 for that subject |
| predictor $x$ | $-\log_{10}$ rank of the model's top-1 in $v_e$ |
| stratum | relation |

$v_e$ is unchanged: neutral prompt, layer 17, 8 references, exactly as in
the paper.  Only the *scoring* becomes per-subject.

**Inclusion.** A relation enters the primary analysis if it has
$n_\text{test} \ge 20$ **and** both outcomes occur (it is not at floor or
ceiling).  Relations failing this are reported but excluded, with the
count stated — a relation at faithfulness 0.07 has almost no within-
relation variance to predict and would only add noise.

## 4. Statistics

**Primary — per-relation AUC, combined.**  Within relation $r$, compute
$\mathrm{AUC}_r$: the probability that a randomly chosen *correct*
subject has a higher lens score than a randomly chosen *incorrect* one
(Mann--Whitney).  Chance is 0.5.  Test whether $\{\mathrm{AUC}_r\}$ is
centred above 0.5 with a Wilcoxon signed-rank test across relations.
This is assumption-light, immune to the between-relation variation
entirely, and reports one interpretable number per relation.

**Secondary — conditional logistic regression**, stratified by relation:
$$\mathrm{logit}\,P(y_{r,e}=1) = \alpha_r + \gamma\,x_{r,e}$$
with $\alpha_r$ conditioned out.  $\gamma$ is a single pooled
within-relation effect with a confidence interval.  Report the odds ratio
per unit of $-\log_{10}$ rank.

**Also report the variance decomposition.**  Fit the mixed-effects
version with a random intercept per relation and give the share of
variance in $y$ that is between- versus within-relation.  A real but
small within-relation effect is a different finding from a large one, and
the paper should say which it has.

**Never a single pooled correlation across all 1,000 rows.**  That would
be dominated by between-relation differences and would answer the
question the paper has already answered.

## 5. Confounds — what still varies within a relation

Most things are held fixed by stratification.  Three are not.

**(a) Model confidence.  This is the sharp one.**  If the model is
peaked on subject $e$, both the LRE and the lens may find it easier, and
the association would be real but uninformative — "the lens predicts LRE
success because both track confidence."  This is the per-subject
analogue of the frequency control, and a reviewer will ask.  Control:
the model's top-1 probability (or the entropy of its next-token
distribution) at the target position, entered as a covariate in the
conditional logistic model.  **Report the effect with and without it.**
If it fully explains the effect, that is a deflationary but honest
result and should be reported as such.

**(b) Answer-token frequency.**  Different subjects have different
objects.  Reuse the prior from `analysis/object_prior.py`, per subject
rather than averaged per relation.

**(c) Subject familiarity.**  A famous subject may be both easier for the
LRE and more salient.  Proxy: the model's log-probability of the subject
string, or its unigram prior.  Weaker than (a) but cheap.

## 6. Pre-registered predictions

1. $\mathrm{AUC}_r > 0.5$ for a clear majority of usable relations;
   median $\mathrm{AUC}$ in the range 0.60--0.70.  **Moderate
   confidence.**
2. The effect **survives** the answer-frequency control, as it did at the
   relation level.  Moderate confidence.
3. The effect is **attenuated but not eliminated** by the confidence
   control.  **Low confidence** — this is the outcome I am least sure of
   and the one most likely to change the paper's claim.
4. Between-relation variance in $y$ will exceed within-relation variance,
   so the subject-level effect will be real but smaller than the
   relation-level correlation suggests.

**Falsification.**  If median $\mathrm{AUC}_r \approx 0.5$, the mechanism
is relation-level only: the lens tracks some coarse property of a
relation and not anything about individual entities.  That would not
retract the paper's existing result, but it would force the account to be
stated in relation-level terms and would make the entity-level language
in the abstract and conclusion wrong.  **That is a publishable finding
and should be reported, not buried.**

## 7. Power

With `--n-test` capped at 100:

| criterion | relations | subject-level rows |
|---|---|---|
| all | 28 | 1,660 |
| $n_\text{test} \ge 20$ and $0.10 \le$ faith $\le 0.90$ | **11** | **967** |

The 11: `star_constellation`, `person_plays_instrument`,
`landmark_in_country`, `product_by_company`,
`person_plays_position_in_sport`, `company_hq`, `person_occupation`,
`superhero_person`, `superhero_archnemesis`, `person_university`,
`pokemon_evolutions`.

Note that these are mid-cardinality relations, which is where §6 of the
paper predicts the account should work best — so this is a fair test of
the mechanism but *not* of its boundaries.  Say so.

## 8. Cost and staging

1. Prerequisites (a)--(c): code only, no GPU.
2. Re-run faithfulness on the 11 with per-subject output. Same cost as
   before, ~20 min each on a free card, parallelisable.
3. Re-run the lens on the same subjects with `N_SUB=100`:
   $\approx 11 \times 100 \times 8 = 8{,}800$ JVPs, one to two hours.
4. Analysis: minutes.

Roughly a day of wall-clock on free GPUs.

## 9. Traps, from this project's own history

- **Check the split alignment with an assert**, not by inspection.  The
  paper shipped a false claim that the splits already matched.
- **$(\ell,\beta)$ is selected on the mean over the same test subjects**
  that we then score individually.  The selection uses aggregate
  information so the leak is diffuse, but run a robustness check with the
  cell chosen on a held-out half.
- **Do not report a median.**  In this project the signal has repeatedly
  lived in the tail while the median read at ceiling
  (`../e2_k1_findings.md` §8).  AUC uses the whole distribution.
- **Shuffle control:** permute subject labels within each relation; AUC
  must return to 0.5.  Run it before believing anything.
- **First-token scoring.**  Check for objects whose first token is a bare
  space (token 220) before scoring
  ([`year-relations.md`](year-relations.md)).

## 10. Progress log

**Stage 1 (prerequisites) complete, 2026-09-10.**

- `analysis/relsplit.py` — one canonical `split(samples, seed)` giving
  disjoint `(icl, ref, test)`, plus `fingerprint()` over the subject
  names. Both scripts call it; `--legacy-split` reproduces the paper's
  numbers.
- **A second misalignment was found while wiring it, not anticipated
  above.** The LRE fits on a `ref` set *disjoint* from `test`, but the
  lens used its first eight *scored* subjects as anchors and skipped
  self-comparison — so lens subjects fell into two classes with
  different treatment. The lens now takes the same disjoint `ref` set as
  anchors. This changes lens scores slightly relative to the paper.
- `lre_faithfulness.py` now records hit/miss per subject at every grid
  cell and writes the selected cell's map as `per_subject`.
- Both write `split_fingerprint`; the analysis asserts equality.

**Smoke test — `pokemon_evolutions`, the smallest usable relation.**
Fingerprints matched (`81bfbfbdbdaa`), subject sets identical, 28
subjects, 6 hits. Pilot statistic: **AUC = 0.79** (Mann--Whitney
p = 0.033); shuffle control returns a null AUC of 0.501 and matches the
observed value in 4% of 2,000 permutations. One relation proves nothing,
but the pipeline is correct and the effect is in the predicted direction.

**An unanticipated finding worth acting on.** The same relation scores
faithfulness **0.39 under the old split and 0.21 under the new one** —
same relation, same `n_test = 28`, only a different random partition.
Per-relation faithfulness is far noisier at small `n_test` than the
paper currently acknowledges: the concern is not confined to the seven
relations flagged with `†`. Two consequences: the relation-level
correlation is attenuated by noise in `y` (so the true value is probably
above 0.658), and the re-run should use `--trials > 1` and average,
which is what that flag exists for.
