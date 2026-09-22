# Corpus co-occurrence and LRE faithfulness: results

> Historical configuration: the numerical comparisons below predate the current matched newline/shared-split appendix. Use [the current baseline tables](../results/shared_baselines/README.md) and [curation guide](shared-baseline-appendix.md) for up-to-date results. Original measurements are preserved for provenance.

Measured 2026-09-18 on the full published OLMo 3 training split. This document
reports what the `jlens corpus-conditionals` run found and how it relates to
[Merullo et al. (2025)](https://arxiv.org/abs/2504.12459), whose headline
correlation is much higher. The method, matching rules, and run options are in
[corpus-conditionals.md](corpus-conditionals.md); this is the result and its
interpretation.

**Summary.** Directional conditional probability correlates with LRE
faithfulness across the 28 relations (`P(o|s)`: *r* = +0.52, *p* = 0.005;
`P(s|o)`: *r* = −0.55, *p* = 0.003; their difference: *r* = +0.59, *p* = 0.001),
while raw co-occurrence and the subject marginal do not reach significance.
The direction effect is, however, largely an object-frequency effect (§5), and
the result is not comparable to the *r* = 0.82 reported by Merullo et al. (§6).

## 1. What was measured

| | |
|---|---|
| Corpus | `allenai/dolma3_mix-6T-1025-7B`, `dolma3_dolmino_mix-100B-1025`, `dolma3_longmino_mix-50B-1025` |
| Scope | `training-split`, 175,877 shards, 3.502 TB compressed / 24.3 TB text |
| Documents | 3,861,363,584 (2,329,747 redacted and therefore uncountable) |
| Facts | 9,725 across 28 relations; 11,415 distinct subject/object strings |
| LRE scores | `best_faith` from `results/olmo/results/main/faithfulness_*.json` (OLMo-3-1025-7B), corrected `_ts` year runs preferred |
| Runtime | ~4h45m on 256 workers, ~207 compressed MB/s |

Corpus revisions are recorded in `report.json` under `corpus_revisions`.

## 2. Definition of the count

For a fact (s, o), **co-occurrence is the number of documents containing both
literal strings at least once**. Each document contributes 0 or 1: a document
mentioning "Paris" forty times counts once. This is document frequency, not
term frequency, and matches the WIMBD convention.

- **Unit** — one document is one JSONL record in the published corpus.
- **No proximity constraint** — both strings anywhere in the same record count,
  adjacent or thousands of tokens apart. There is no sentence or window limit.
- **Undirected** — the pair key is sorted, so the joint count is symmetric and a
  pair shared by two relations is counted once. Direction enters only through
  the denominator: `P(o|s) = joint / count(s)`, `P(s|o) = joint / count(o)`.
- **Literal matching** — Aho-Corasick over the raw relation strings, NFC
  normalization, whitespace collapse, casefold, Unicode word boundaries. No
  alias expansion (`US` ≠ `United States`), no disambiguation (every `Jordan`
  matches), no coreference. Substring overlap (`Mexico` in `Mexico City`) is
  matched and flagged; the `no_lexical_overlap` subset drops those pairs.
- **Aggregation** — each relation contributes one observation, the unweighted
  mean across its eligible facts. `log1p_cooccurrence` averages `log1p(joint)`
  per fact, so it is a mean of logs, not the log of a mean. Micro variants are
  ratios of summed counts.

Example rows: France/Paris — 85,036,785 / 52,846,739 documents, 19,044,875 joint,
`P(o|s)` = 0.224, `P(s|o)` = 0.360. Japan/Tokyo — 84,409,991 / 21,666,655,
12,337,654 joint, `P(o|s)` = 0.146, `P(s|o)` = 0.569.

## 3. Correlations

Relation-level, n = 28, all corpus stages, `all_facts` cohort. Two-sided
unadjusted p-values.

| Feature | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| **direction_difference** | **+0.587** | **0.0010** | **+0.523** | **0.0043** |
| p_object_given_subject | +0.518 | 0.0047 | +0.497 | 0.0072 |
| p_subject_given_object | −0.546 | 0.0026 | −0.470 | 0.0117 |
| micro_p_object_given_subject | +0.482 | 0.0093 | +0.541 | 0.0029 |
| micro_p_subject_given_object | −0.110 | 0.577 | −0.211 | 0.281 |
| cooccurrence | +0.195 | 0.320 | +0.365 | 0.056 |
| log1p_cooccurrence | +0.304 | 0.116 | +0.340 | 0.077 |
| subject_count | +0.187 | 0.340 | +0.044 | 0.823 |
| object_count | +0.068 | 0.732 | +0.572 | 0.0015 |

**Robustness.** The pattern is stable under both available controls:

| Cohort / subset | direction_difference | P(o\|s) | P(s\|o) |
|---|---|---|---|
| all facts | +0.587 (p = 0.0010) | +0.518 | −0.546 |
| no lexical overlap | +0.596 (p = 0.0008) | +0.526 | −0.561 |
| evaluated subjects | +0.564 (p = 0.0018) | +0.502 | −0.523 |
| evaluated + no overlap | +0.565 (p = 0.0017) | +0.506 | −0.527 |

**Per corpus stage.** Not driven by one mix: `direction_difference` is +0.587
(`dolma3_mix-6T`), +0.590 (`dolmino-100B`), +0.579 (`longmino-50B`).

## 4. Per-relation values

Sorted by faithfulness. Co-occurrence is the mean joint document count.

| Relation | facts | best_faith | P(o\|s) | P(s\|o) | diff | mean co-occ |
|---|---|---|---|---|---|---|
| country_language | 24 | 1.00 | 0.1943 | 0.1932 | +0.0011 | 10,496,951 |
| country_largest_city | 24 | 1.00 | 0.1173 | 0.5177 | −0.4004 | 6,267,753 |
| person_native_language | 919 | 0.95 | 0.6827 | 0.0005 | +0.6822 | 28,504 |
| name_birthplace | 31 | 0.93 | 0.3175 | 0.0159 | +0.3016 | 1,309,213 |
| name_religion | 31 | 0.93 | 0.1262 | 0.0495 | +0.0766 | 668,909 |
| landmark_on_continent | 947 | 0.93 | 0.5784 | 0.0036 | +0.5748 | 372,215 |
| person_plays_pro_sport | 318 | 0.93 | 0.5219 | 0.0017 | +0.5202 | 43,818 |
| person_plays_instrument | 513 | 0.88 | 0.5237 | 0.0009 | +0.5229 | 22,590 |
| landmark_in_country | 836 | 0.86 | 0.6553 | 0.0008 | +0.6545 | 26,075 |
| city_in_country | 27 | 0.82 | 0.5728 | 0.0764 | +0.4965 | 2,708,050 |
| person_band_lead_singer | 21 | 0.80 | 0.5381 | 0.1303 | +0.4078 | 234,331 |
| person_occupation | 821 | 0.76 | 0.3509 | 0.0003 | +0.3506 | 7,404 |
| country_capital_city | 24 | 0.75 | 0.0940 | 0.5858 | −0.4919 | 4,632,789 |
| product_by_company | 502 | 0.67 | 0.5886 | 0.0069 | +0.5817 | 583,280 |
| presidents_birth_year | 18 | 0.67 | 0.0421 | 0.0170 | +0.0251 | 71,869 |
| presidents_election_year | 19 | 0.67 | 0.1883 | 0.0205 | +0.1678 | 421,180 |
| person_plays_position_in_sport | 952 | 0.61 | 0.4175 | 0.0019 | +0.4156 | 5,495 |
| person_university | 91 | 0.59 | 0.0079 | 0.0757 | −0.0678 | 10,069 |
| company_hq | 674 | 0.58 | 0.3242 | 0.0023 | +0.3219 | 31,672 |
| star_constellation | 362 | 0.50 | 0.5741 | 0.0123 | +0.5618 | 13,399 |
| superhero_person | 100 | 0.49 | 0.1450 | 0.6539 | −0.5089 | 153,087 |
| country_currency | 30 | 0.36 | 0.0350 | 0.2705 | −0.2355 | 1,721,822 |
| pokemon_evolutions | 44 | 0.36 | 0.2733 | 0.3171 | −0.0438 | 27,124 |
| name_gender | 18 | 0.33 | 0.3265 | 0.0271 | +0.2994 | 9,500,190 |
| superhero_archnemesis | 96 | 0.30 | 0.0762 | 0.3337 | −0.2575 | 117,458 |
| person_father | 991 | 0.11 | 0.0390 | 0.6323 | −0.5934 | 12,710 |
| person_mother | 994 | 0.10 | 0.0117 | 0.6466 | −0.6349 | 4,816 |
| company_ceo | 298 | 0.07 | 0.0192 | 0.5446 | −0.5254 | 40,916 |

## 5. Interpretation, and why the direction effect is weaker than it looks

Raw exposure does not explain faithfulness: co-occurrence (+0.195) and the
subject marginal (+0.187) are both insignificant, and `name_gender` has the
second-highest mean co-occurrence in the table (9.5M) with faithfulness 0.33.
What separates the relations is the *asymmetry* of the conditionals.

That asymmetry is, however, close to a restatement of object frequency. When the
joint count is positive, `P(o|s) / P(s|o) = count(o) / count(s)` identically, so
`direction_difference` largely encodes *how common the object string is relative
to the subject*. The bottom of the table is relations whose objects are rare
named entities — mothers, fathers, CEOs, archnemeses — and the top is dominated
by relations whose objects are drawn from small, high-frequency closed sets:
languages, continents, sports, instruments, occupations. A relation with a
low-cardinality object set is both easier to fit with a single affine map and
guaranteed a high `P(o|s)`, which is a plausible common cause rather than
evidence that exposure produces linearity.

Two further reasons for caution:

- `object_count` has a significant **Spearman** (+0.572, p = 0.0015) while its
  Pearson is flat (+0.068), i.e. object marginal frequency is a rank-level
  confounder for the forward direction.
- Clear counterexamples exist in both directions: `country_largest_city`
  (faith 1.00, diff −0.40) and `country_capital_city` (faith 0.75, diff −0.49)
  are highly faithful with *negative* direction difference, while
  `star_constellation` (diff +0.56) sits at 0.50.

The honest reading is a directional association at relation level, confounded
with object cardinality, not a demonstration that co-occurrence causes linear
structure.

## 6. Relation to Merullo et al. (2025)

Merullo, Smith, Wiegreffe and Elazar report *r* = 0.82 between subject-object
co-occurrence and linear-representation quality. That number is **not**
comparable to the ones above; five differences, ordered by likely impact.

1. **Different dependent variable.** Their 0.82 is *causality*. This repository
   measures only *faithfulness* (`best_faith`); no causality scores exist here.
   Their own permutation test separates the two: hard causality contributed
   ~15% accuracy to their regression, the faithfulness measures ~5%.
2. **Different independent variable.** Their axis is **log** co-occurrence. The
   comparable quantity here is `log1p_cooccurrence` = **+0.304 (p = 0.12)**, or
   +0.346 (p = 0.07) using log10 of the relation mean — not the +0.52 headline,
   which is a normalized conditional probability they do not test.
3. **Their scatter pools pretraining checkpoints.** Their figure plots 41B-token,
   intermediate and final checkpoints together, and the claim is explicitly
   about all phases of pretraining, so each point is a relation × checkpoint.
   Exposure and causality both grow with training, which supplies a shared trend
   that a single-checkpoint cross-relation correlation (n = 28) cannot contain.
   Their text does not state whether 0.82 is restricted to the final checkpoint;
   this should be checked against their code before the number is cited.
4. **Different counting window.** For OLMo they count co-occurrence within a
   *training sequence inside a batch*; we count within a whole *document*. They
   use document-level WIMBD counts only for GPT-J, and report that WIMBD
   over-counts because documents exceed sequence length (best-fit slope 0.94).
   Our procedure therefore matches their GPT-J variant, not their OLMo variant.
5. **Different model and corpus.** Theirs is OLMo-7B 0424 on Dolma (~2T tokens);
   this is OLMo-3-1025-7B on the dolma3 mixes (3.86B documents).

**Range restriction is the most likely reason our correlation is lower.** Their
central claim is a threshold: causality is reliably high once mean subject-object
co-occurrence passes ~1k. Every relation here clears that threshold by one to
four orders of magnitude — the minimum relation mean is 4,816 (`person_mother`) —
yet faithfulness still spans 0.07 to 1.00.

| Relation | mean co-occ | facts ≥1k | best_faith |
|---|---|---|---|
| country_language | 10,496,951 | 100% | 1.00 |
| name_gender | 9,500,190 | 100% | 0.33 |
| country_largest_city | 6,267,753 | 100% | 1.00 |
| … | | | |
| company_ceo | 40,916 | 54% | 0.07 |
| person_father | 12,710 | 38% | 0.11 |
| person_mother | 4,816 | 24% | 0.10 |

Means are over eligible facts, as in §4; the `facts ≥1k` share is over all facts
in the relation, which is why 5,050 of 9,725 individual facts (51.9%) fall below
1k even though every relation *mean* is far above it.

We are sampling the flat, saturated part of their curve. Document-level counting
over 3.86B documents also inflates counts relative to sequence-level counting
over 2T tokens, so their 1k threshold does not transfer numerically.

**To make the comparison properly**, causality would have to be measured on
OLMo-3-1025-7B and correlated against log mean co-occurrence at the final
checkpoint; the OLMo-3 intermediate checkpoints would additionally allow the
pooled-over-training version. Neither is done here.

## 7. Limitations

- Counts describe the published document multiset, not packed training
  sequences or exposure weights; duplicates within and across mixes count as
  published, and 2,329,747 redacted documents are uncountable.
- Literal strings are proxies for entities: no aliases, no disambiguation.
- Retained `best_faith` values come from a hyperparameter sweep; that selection
  bias is not removed here.
- p-values are unadjusted across 36 tests per scope, and relations sharing
  entities are not independent observations.
- 28 relations is a small sample; single relations visibly move the coefficients.

## 8. Reproduction

```bash
cd code
uv sync --locked --extra corpus --inexact
uv run --no-sync jlens corpus-conditionals --workers 256 \
  --faith-dir results/olmo/results/main \
  --out generated/corpus_conditionals
```

`--faith-dir` is required: per-model results live under `results/<model>/results/`,
so the historical `results/main` default no longer resolves. Prefer it over
`jlens --root`, which also moves `data/relations` — part of the checkpoint
fingerprint — and would force a full rescan. Counts are checkpointed in
`generated/corpus_conditionals/counts.sqlite3`; rerunning with `--report-only`
regenerates the report from them in about two minutes.

Artifacts (under `generated/`, not tracked in git): `report.json` (full
correlations, coverage, per-relation LRE sources), `report.txt` (readable
summary), and `{all_stages,<stage>}.{facts,relations}.csv`.
