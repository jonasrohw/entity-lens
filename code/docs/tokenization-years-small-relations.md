# Tokenization, year prediction, and three small relations

The unusually large correlation gap between OLMo and several other models is
partly a measurement-comparability issue. For two presidential-year relations,
Mistral, Gemma, and both Qwen models receive perfect LRE faithfulness for
predicting the same first token, `1`, on every test example. An intercept-only
control achieves the same score without using subject-dependent information.
OLMo instead predicts different three-digit prefixes, making its first-token
agreement a more informative task. Both year relations, together with
`name_gender`, also have only three LRE test examples each.

Excluding these three relations raises Mistral's correlation from **0.325 to
0.637** and Qwen3.8-27B's from **0.197 to 0.637**, while OLMo remains at
**0.771**. This is an exploratory sensitivity analysis of the existing results;
it does not involve refitting an LRE or selecting new layers or beta values.

## What the correlation measures

Each relation contributes one equally weighted point. The predictor is the
negative median `log10(rank + 1)` of the full model's next-token argmax under
the differential lens. Higher values mean better lens ranks. The outcome is
LRE faithfulness: the fraction of test examples on which the LRE's top
vocabulary token agrees with the full model's next-token argmax. The reported
statistic is Spearman's correlation across relations.

Crucially, faithfulness measures **first-token agreement with the model**, not
correct recovery of the complete factual answer. A model and an LRE agreeing
on `1` need not agree on the remaining digits of a year.

The original main experiment uses eight in-context demonstrations, eight LRE
reference examples, and up to 100 LRE test examples. Each model's operators
are fitted independently. The selected faithfulness is the best cell in a
six-layer grid with beta values `0.5, 1, 1.5, 2, 3, 5`. These are single-split
results, including the original selection over that grid. The lens and LRE
evaluations have separate sampling procedures in this main protocol.

## The year-tokenization discrepancy

The three birth-year test subjects are the same across the models below.
These are the decoded **actual model target tokens** in the saved prompts,
rather than hypothetical tokenizations of isolated answer strings.

| Test subject | Dataset birth year | OLMo 3 7B target | Mistral, Gemma 2, Qwen3 8B Base, Qwen3.8 27B target |
| --- | --- | --- | --- |
| Martin Van Buren | 1782 | `178` | `1` |
| Franklin D. Roosevelt | 1882 | `188` | `1` |
| William Henry Harrison | 1773 | `177` | `1` |

For the election-year relation, the same subjects have dataset answers 1836,
1932, and 1840. OLMo's target tokens are `183`, `193`, and `184`; the other
four models again target `1` on all three examples. OLMo's prefixes are more
discriminative here, but they are still not complete four-digit years.

We checked whether a constant prediction could already achieve the reported
faithfulness. The intercept-only control decodes the mean reference final
hidden state through the model's final normalization and output head, with
zero Jacobian contribution. It supplies the same prediction for every test
subject.

| Model | Birth-year LRE agreement | Birth-year intercept-only agreement | Election-year LRE agreement | Election-year intercept-only agreement |
| --- | --- | --- | --- | --- |
| OLMo 3 7B | 2/3 | 0/3 | 2/3 | 0/3 |
| Mistral 7B | 3/3 | 3/3 | 3/3 | 3/3 |
| Gemma 2 9B | 3/3 | 3/3 | 3/3 | 3/3 |
| Qwen3 8B Base | 3/3 | 3/3 | 3/3 | 3/3 |
| Qwen3.8 27B | 3/3 | 3/3 | 3/3 | 3/3 |

For those four models, therefore, perfect agreement on these examples provides
no improvement over this subject-independent control. Their year relations
combine poor differential-lens ranks with maximal LRE faithfulness, weakening
the positive relation-level correlation. This does not establish that the
models lack year knowledge; it shows that the chosen outcome cannot measure
that knowledge on these examples.

The [retained target audits](../results/cross_model_audit/raw/targets/) contain
the subjects, token IDs, decoded tokens, LRE hits, and intercept-only predictions
for all seven models. The concrete birth-year comparison can be inspected in
the [OLMo record](../results/cross_model_audit/raw/targets/olmo3_7b/main/presidents_birth_year.json)
and [Mistral record](../results/cross_model_audit/raw/targets/mistral7b/main/presidents_birth_year.json).

## Why the three small relations matter

Each of `presidents_birth_year`, `presidents_election_year`, and `name_gender`
contains 19 input rows. Allocating eight demonstrations and eight reference
examples leaves **three LRE test examples**. Consequently, measured faithfulness
can only be `0`, `1/3`, `2/3`, or `1`: one changed prediction moves the outcome
by a third. Nevertheless, each relation receives the same weight in Spearman's
correlation as a relation evaluated on 100 examples. Input rows are not always
unique subjects in this dataset.

`name_gender` is a separate small-sample issue, not a year-tokenization example.
Its selected faithfulness is 1/3 for OLMo, 2/3 for Mistral, and 3/3 for the
other five models, despite weak lens ranks. Removing it can therefore further
change the correlation. The following table separates that step from removing
the years.

## Effect of excluding relations

All columns below use the **main protocol** and exactly the same saved
per-relation scores and fitted operators. Only the set of relations entering
the correlation changes. Requiring at least four LRE test examples removes
exactly the three named relations, leaving the same 25-relation cohort for
all seven models.

| Model | All 28 relations | Exclude two year relations: 26 | Exclude years and `name_gender`: 25 | Change, 28 → 25 |
| --- | ---: | ---: | ---: | ---: |
| OLMo 3 7B | 0.771 | 0.783 | 0.771 | +0.000 |
| GPT-J 6B | 0.343 | 0.361 | 0.508 | +0.164 |
| Mistral 7B | 0.325 | 0.609 | 0.637 | +0.312 |
| Gemma 2 9B | 0.153 | 0.437 | 0.562 | +0.408 |
| Qwen3 8B Base | 0.060 | 0.299 | 0.448 | +0.388 |
| Qwen3.8 27B | 0.197 | 0.479 | 0.637 | +0.440 |
| Pythia 6.9B | 0.077 | 0.113 | 0.227 | +0.150 |

Values are rounded to three decimals; OLMo's unrounded change is approximately
`+0.000062`. [Companion CSV](tokenization-years-small-relations.csv) retains
full-precision correlations and cohort sizes.

The two year relations account for a substantial part of the observed
sensitivity in Mistral, Gemma, and both Qwens. For GPT-J and Pythia, the change
from removing `name_gender` is larger than the change from removing the years.
The combined removal therefore cannot be attributed entirely to tokenization.
Correlation changes also depend on reranking the retained points, so the two
steps are descriptive, not independent causal effects.

OLMo's observed advantage over Mistral shrinks from 0.446 to 0.134, and its
advantage over Qwen3.8-27B shrinks from 0.574 to 0.134. The remaining differences
are not uniformly eliminated: Qwen3 8B Base and especially Pythia still have
lower correlations. Paired relation-bootstrap intervals for the remaining
OLMo-minus-Mistral and OLMo-minus-Qwen3.8 gaps include zero, but are wide; this
does not establish equivalence. The [full audit](../results/cross_model_audit/README.md#paired-uncertainty)
reports the exploratory intervals and their limitations.

## A separate gold-token labeling bug

The audit also found a concrete Mistral labeling error. Encoding a bare year
in isolation can prepend an empty dummy-prefix token. For a prompt that already
ends in a space, the correct next gold token is `1`, whereas the old helper
selected that empty token. The helper now skips empty/whitespace-only prefix
tokens in this trailing-space mode. The contextual audit found six such
stable-prefix mismatches, all Mistral year examples, and zero after the fix.

This correction changes auxiliary gold-token accuracy labels. **It does not
change the LRE faithfulness or correlation above**, whose target is the model's
own argmax. Nor did it change the selected examples or filter membership here.
The labeling bug and the unequal information carried by first tokens are
distinct findings. See the [label-fix validation](../results/cross_model_audit/provenance/label_fix_validation.json).

GPT-J has another prompt detail: its six saved year examples use a trailing-space
variant whose tokenized prompt is not a stable prefix of the prompt plus answer.
The table preserves that saved variant. Excluding both year relations raises
GPT-J only from 0.343 to 0.361, so that issue alone does not explain its gap.

## Interpretation and reporting

These checks support a narrower conclusion than an implementation failure:
the original cross-model comparison mixes first-token tasks of different
information content, and three very small relations have substantial influence
on the result. OLMo's original correlation also reproduces in the common
implementation: all 1,944 overlapping LRE grid cells agree, and its main
correlation is unchanged. The [broader implementation audit](cross-model-audit.md)
documents those checks.

The exclusions were examined after observing the results. The 25-relation
analysis should therefore accompany the original 28-relation result as an
**exploratory sensitivity analysis**, rather than replace it as a validated
primary result. It changes both sample size and relation composition. It does
not prove that excluding these relations is the uniquely correct evaluation,
or identify an architectural cause for the remaining model differences.

The main 25-relation analysis is also not the existing shared-protocol result:
the shared protocol changes splits and applies tokenizer-dependent identifying
first-token filtering. Its 25-relation cohort retains `name_gender` and omits
`person_band_lead_singer`, as well as the two year relations.

A follow-up evaluation should define its minimum test-set size in advance,
report intercept-only agreement alongside LRE faithfulness, and explicitly
handle answers whose first token carries little identifying information.
Full-answer or answer-discriminating evaluation would require a specified
scoring procedure; it has not been measured by the present first-token test.
Fresh splits or additional relations would then test whether the improvement
generalizes beyond this exploratory exclusion.

## Reproducing the correlation table

The 28- and 25-relation results are retained in
[analysis.json](../results/cross_model_audit/analysis.json). The intermediate
26-relation column is recomputed from the same original relation records:

```sh
cd code
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, "scripts")
from report_cross_model_audit import LABELS, corr, records

years = {"presidents_birth_year", "presidents_election_year"}
for model, protocols in records().items():
    values = []
    for excluded in (set(), years, years | {"name_gender"}):
        rows = [r for name, r in protocols["main"].items() if name not in excluded]
        result = corr([r["x"] for r in rows], [r["y"] for r in rows])
        values.append((result["n"], result["rho"]))
    print(LABELS[model], values)
PY
```
