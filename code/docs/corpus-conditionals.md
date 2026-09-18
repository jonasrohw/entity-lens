# Directional corpus co-occurrence versus LRE faithfulness

`jlens corpus-conditionals` streams the downloaded OLMo 3 corpus, counts the
subject/object strings in all 28 relation datasets, and reports Pearson and
Spearman correlations with the retained LRE scores. No model or GPU is needed.
It writes sufficient statistics, not a copy of the filtered corpus.

## Run

From `code/`:

```bash
uv sync --locked --extra corpus --inexact
uv run --no-sync jlens corpus-conditionals --plan
uv run --no-sync jlens corpus-conditionals --workers 64
```

The default corpus directory is
`/mounts/work/jonas/data/olmo-3-1025-7b-training`; override with `--corpus-root`.
The input `manifest.json` is produced by `download_olmo_corpus.py`. Output defaults
to `code/generated/corpus_conditionals/`; override with `--out`. The common
`jlens --root` option selects the checkout/rerun containing `data/relations` and
`results`, and must precede `corpus-conditionals`.

A bounded real-data smoke run (explicitly labelled **PARTIAL**):

```bash
uv run --no-sync jlens corpus-conditionals --workers 16 --max-shards 32 \
  --out generated/corpus_smoke
```

For a long run, use a persistent session, for example:

```bash
tmux new-session -s corpus-conditionals
uv run --no-sync jlens corpus-conditionals --workers 64 \
  > generated/corpus-conditionals.log 2>&1
```

Rerun the same command to resume. `--max-shards` limits *new* shards, so removing
it resumes toward the full inventory. It selects shards by a deterministic hash
for smoke tests; this is not a representative document sample or a basis for
full-corpus conclusions. The production run schedules larger shards first.

## Corpus scope

The default `--corpus-scope training-split` follows the published 6T dataset
card's `data/**/*.jsonl.zst` split and includes the two later stages' JSONL
shards. The pinned download also contains **52,471 gzip science-PDF files** and
**336 numerically suffixed Zstandard files** outside that declared 6T split.
They are excluded from this default, with every excluded path recorded in
`inventory.json`. The default inventory is **175,877 shards, about 3.502 TB**.
The downloader fetched every repository file, a broader scope than this split.

For a sensitivity analysis that includes all canonically named JSONL files,
including those gzip PDFs, use a separate output directory:

```bash
uv run --no-sync jlens corpus-conditionals --corpus-scope published-jsonl \
  --workers 64 --out generated/corpus_all_published_jsonl
```

Numeric-suffixed files are still excluded when their canonical `.jsonl.zst`
counterpart exists. Other unknown formats inside `data/` fail loudly. Supported
formats are JSONL, gzip JSONL, and Zstandard JSONL, including concatenated frames.
For a custom manifest/repository, all canonically named JSONL files are included.
The 6T split rule is specific to `allenai/dolma3_mix-6T-1025-7B`.

This measures the **published document multiset**, once per selected JSONL
record. It does not reconstruct training batch packing, token windows, mixture
weights, or the number of training exposures. Records repeated within or across
stages remain repeated in the combined result; each stage is reported separately
as well. Upstream `[REMOVED]` texts are counted and disclosed, but unavailable
entity occurrences cannot be recovered. File sizes are checked against the
manifest; content hashes are not recomputed during the scan, so run against the
checksum-verified downloader output. Size/mtime changes invalidate checkpoints.

## Counting and matching

For each directed subject/object fact `(s, o)`, the scanner counts:

- `subject_count`: documents containing `s`, including documents without `o`.
- `object_count`: documents containing `o`, including documents without `s`.
- `cooccurrence`: documents containing both, at most once per document.

The directional estimates are:

```text
P(o | s) = cooccurrence / subject_count
P(s | o) = cooccurrence / object_count
direction_difference = P(o | s) - P(s | o)
```

An absent denominator is missing, not zero. An observed subject whose object
never co-occurs has a valid forward estimate of zero. `--min-count` defaults to
one; raise it to check sensitivity to rarely observed entities. Both marginal
counts must meet this threshold for a fact to enter the directional correlation
comparison. The fact CSV retains the individual estimates and all raw counts.

Matching uses full strings from the relation JSONs, Unicode NFC normalization,
case folding, collapsed whitespace, and Unicode word boundaries (letters,
digits, and underscore). Use `--case-sensitive` for another matching policy;
this requires a separate checkpoint directory. The matcher retains overlapping
names and flags substring overlap. The report also repeats relation correlations
excluding these pairs, e.g. `Mexico City` / `Mexico`. Exact duplicate facts within
a relation are removed. Shared entities and unordered co-occurrence pairs across
relations are counted once and reused in both directions.

Aliases such as `US` for `United States` are not expanded; ambiguous strings
such as `Jordan` are not disambiguated. These are lexical occurrence estimates.
For nonzero joint counts, `P(o|s) / P(s|o) = count(o) / count(s)`: the probability
ratio itself is a marginal-frequency ratio. A stronger forward correlation
would support a directional association with LRE quality, not by itself a causal
explanation of what information is represented.

## LRE matching and correlation

The default target is `best_faith` in `results/main/faithfulness_*.json`, with
corrected `_ts` year runs preferred over their originals, matching the existing
paper reproduction. `--faith-dir` and `--lre-field` select another directory and
numeric field. These are this repository's **faithfulness** scores, not the
added paper's causality scores. Best hyperparameters were selected in the retained
experiment; the corpus analysis does not remove that selection bias.

The primary relation-level predictors are **macro averages across facts**, using
the same eligible facts for both directions. Each relation contributes one
observation to a correlation. CSVs report total facts, observed facts, and
coverage. Results include raw co-occurrence, log co-occurrence, and both marginal
frequencies as baselines, plus micro probabilities (ratio of summed counts).
Micro probabilities weight facts by marginal frequency; shared entities can
contribute once per fact, so these are not probabilities of a union of entities.

The JSON additionally contains an `evaluated_subjects` cohort when LRE files
provide `per_subject`: averages are restricted to those measured subjects.
Within-relation correlations against their LRE outcomes are reported separately;
subjects are never pooled across relations as independent observations. A
relation with multiple objects for one subject is excluded from this latter
analysis to avoid duplicating the outcome. Too few observations and constant
variables produce explicit undefined results. Each defined coefficient includes
sample size and its two-sided, unadjusted p-value; these are exploratory tests,
and shared entities can induce dependence between relations.

Reuse counts without rereading corpus text, for example:

```bash
uv run --no-sync jlens corpus-conditionals --report-only --min-count 100
uv run --no-sync jlens corpus-conditionals --report-only \
  --faith-dir results/shared_split
```

These commands overwrite reports but retain the full count checkpoint. Copy
reports elsewhere before changing settings if you need both versions. Matching,
entity vocabulary, corpus path, manifest, and corpus scope are fingerprinted;
changing them requires a separate `--out`. LRE sources and minimum count can
change without rescanning.

## Outputs and performance

- `report.txt`: readable correlation summary, coverage, and limitations.
- `report.json`: all stages, cohorts, overlap controls, LRE source paths,
  correlations, p-values, and explicit completeness status.
- `*.facts.csv`: one directed fact per row, counts, conditional probabilities,
  overlap flag, and available LRE values.
- `*.relations.csv`: aggregates, support/coverage, and matched LRE scores.
- `counts.sqlite3`: compressed 64-bit counts per shard; resumable checkpoint.
- `inventory.json`: selected scope and excluded/missing file lists.
- `progress.json`: progress updated approximately every 30 seconds.

A C-backed [Aho–Corasick matcher](https://pyahocorasick.readthedocs.io/en/2.0.0/)
searches all names in one pass per document. Only requested fact pairs are checked,
using an adjacency list oriented toward lower-degree entities. Separate processes
stream shards, with at most twice the worker count queued. Workers return count
arrays rather than text. Native math-library thread pools are limited to one
thread per worker. Memory scales with worker count and the largest document,
not the size of the corpus. Start with 16–64 workers and adjust for shared CPU
and storage capacity; the default is at most 32.

SQLite commits each completed shard atomically, and an output lock prevents two
runs from sharing a checkpoint. Missing/wrong-size shards fail by default;
`--allow-partial` explicitly permits partial analysis. Invalid JSON, missing text,
and decompression errors fail the run rather than silently biasing counts. No
full-corpus success is reported until every selected shard has a valid checkpoint.
On interruption, already committed shards remain reusable; bounded in-flight
work can be rescanned.

The added paper's counting method is in
`papers/2504.12459-source/iclr2025_conference.tex`, sections “Counting Frequencies
Throughout Training” and “Batch Search.” It counts within packed training
sequences, unlike the document-level estimates here. The pinned dataset cards
are under the corpus directory; the [6T dataset card](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B)
also documents the redactions.
