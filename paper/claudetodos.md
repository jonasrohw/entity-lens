<!-- edited 2026-09-18 15:00 — todo 2: direct subject-vs-relation evidence is thin after cutting the within-domain check -->
<!-- edited 2026-09-18 14:47 — started; first item records the accepted risk of dropping the fact-level numbers -->

# Claude's todos for the paper

Open items I am tracking, newest appended at the end. Separate from
`../todos.txt`, which is the collaborators' list.

---

## 1. No quantitative answer to "it's just a property of relations"

**Status: accepted risk, decided 2026-09-18. Not a bug to fix — a
position to be ready to defend.**

### The gap

The central claim rests on **28 aggregate points** — one per relation.
The obvious referee objection is that both the lens score and LRE
faithfulness track some coarse property of a relation (its answer space,
its prompt, its cardinality) rather than anything about individual
entities. Nothing now in `lens.tex` rules that out. `AUC` appears **zero
times** in the file.

§6 currently opens by saying fact-level linearity is worth asking about
and then reports that it could not be done well, so the section does not
close the gap either.

### Why we are accepting it

HS, 2026-09-18: submission is a week away, there is no time for a
satisfying fact-level analysis, and **given the dataset problems we
documented, the fact-level numbers are not trustworthy enough to report**.
Reporting them would also sit awkwardly beside §6's conclusion that
existing datasets cannot support this analysis — we would be arguing both
that the measurement is unsound and that its output is a result.

That reasoning is sound. The risk is real but it is the right trade.

### What is parked, and where

`rawmaterial.tex` (not `\input` anywhere, so it does not compile). It
holds the whole quantitative subject-level analysis:

- median AUC **0.688** over **1,067** rows in 13 relations, Wilcoxon
  **p = 0.011**, within-relation shuffle control 0 of 2,000
- the confidence control: lens odds ratio 6.7× alone, **3.8×** with
  confidence partialled, still p < 10⁻⁴ — but confidence alone is
  **11.5×**, i.e. the stronger predictor
- variance decomposition: 74.7% of the variation is within-relation;
  McFadden pseudo-R² 0.047 lens, 0.075 confidence, 0.095 together
- `table_subject_level`, `tab:conf`, and the synonym-scoring paragraph
  (composer rank 3 vs writer rank 244)

No dangling `\ref`s were left behind — checked `tab:subject`, `par:conf`,
`tab:conf`, `fig:rankrank`. The abstract was already rewritten to promise
only the relation-level result plus the negative fact-level conclusion,
so abstract and body are consistent.

### If a reviewer raises it

Three responses, in order of preference:

1. **Concede the scope.** The claim is relation-level and the paper says
   so. The fact-level question is posed, attempted, and reported as
   blocked by dataset defects we document — which is a contribution, not
   an evasion.
2. **The covariate controls already argue against the coarse-property
   reading** at the relation level: the correlation survives partialling
   out cardinality (ρ rises to 0.744) and answer-token frequency, and
   holds under leave-one-domain-out. A reviewer asking "is it just a
   coarse property?" is partly answered by *which* coarse properties we
   ruled out.
3. **The rebuttal can cite the parked numbers** if pressed, flagged as
   preliminary and subject to the scoring caveats in §6. Better offered
   under duress than volunteered in the submission.

### If time appears before the deadline

The cheapest partial fix is **not** reintegrating the AUC section. It is
one sentence in §5 or §7 making response 2 explicit — that the covariates
we partial out *are* the coarse relation-level properties a skeptic would
name. That costs a line and removes the easiest version of the objection.

---

## 2. Direct evidence for "subject, not relation" is now thin

**Status: open. Raised by cutting the within-domain check (enote 9),
2026-09-18.**

The title claim is that linear decodability is a property of the
*subject's representation* rather than of the relation. Two things that
argued for it directly are now out of the paper:

- the fact-level analysis (todo 1), parked in `rawmaterial.tex`
- the `person_father` / `person_mother` within-domain check, cut here

What remains is indirect: the lens is computed without the relation and
predicts faithfulness anyway, and the covariate controls rule out
cardinality and answer frequency. That is a good argument, but it is an
argument from what the *predictor* does not see, not evidence about what
varies with the subject.

**Why the cut check could not be kept.** Same-subject relation pairs are
the right instrument, but both available pairs sit at an extreme:

| pair | faithfulness | problem |
|---|---|---|
| `person_father` / `person_mother` | 0.09, 0.07 | both at the floor |
| `country_capital_city` / `country_largest_city` | 1.00, 1.00 | both at the ceiling |

Agreement between two relations that are both at the floor is
over-determined --- it follows equally from "faithfulness is a subject
property" and from "both have $k \approx 1$ and answer spaces in the
hundreds". The comparison only discriminates in the middle of the range.

**What would fix it**, if time ever allows: a same-subject relation pair
with mid-range faithfulness. Jonas's note in `../todos.txt` asks the same
question from the other side --- *"if it were true that it's about the
subject only, we could check for same subject different relationship and
see if that holds?"* --- so this is wanted by more than one of us.
