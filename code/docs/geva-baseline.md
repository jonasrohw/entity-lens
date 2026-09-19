# Wikipedia attributes baseline

This implements the **attributes rate** of Geva et al., *Dissecting Recall of
Factual Associations in Auto-Regressive Language Models* (EMNLP 2023, §6.1 and
appendix C), following Sebastian Gerstner's reconstruction in *Weakening Neurons:
An Input-Output Functionality in Transformers with Outsize Influence* (EMNLP 2026,
appendix G.2).

Sources:

- [Geva paper](https://aclanthology.org/2023.emnlp-main.751/)
- [Geva code](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions)
- [Gerstner paper](https://arxiv.org/abs/2609.18612)
- [Gerstner code, pinned revision](https://github.com/sjgerstner/RW_functionalities/tree/18860447f7735d773d76329a619af25be46db6ac)
- [Wikipedia dump](https://archive.org/details/enwiki-20211020)

Downloaded papers and source checkouts are under `../papers/geva_baseline/`
(ignored by Git). The implementation is in `jlens/wiki_corpus.py` and
`jlens/geva_baseline.py`; this guide distinguishes the published score from our
adaptation to this repository's experiment.

## Protocol

1. Extract paragraphs from the **entire** English Wikipedia dump of October 20,
   2021. Use WikiExtractor's cleaning, split on blank lines, preserve article and
   section titles, and exclude titles containing `:`, as in Gerstner's code.
2. Query BM25 with each subject. Retrieve **100 paragraphs first**, then keep
   paragraphs whose text or title contains the subject exactly, or its Python
   `capitalize()` variant. Retain the original strings, including any trailing
   spaces in this repository's subjects. Do not prefilter the corpus by subjects,
   objects, known answers, or retrieval success.
3. Concatenate the selected paragraphs with leading spaces and tokenize with the
   evaluated model's tokenizer. Deduplicate decoded token strings; discard strings
   shorter than three characters after stripping spaces, and NLTK English
   stopwords. The corpus-side stopword test is case-sensitive, matching both
   source implementations.
4. At the last token of the subject, project the hidden state through the final
   normalization and output head. Take the **top 50 first**, then remove short
   strings and stopwords (case-insensitive on this prediction side). The attribute
   rate is `100 * |clean predictions ∩ attributes| / len(clean predictions)`.
   An empty retrieved set scores zero; no eligible predictions yields an undefined
   score, explicitly counted rather than replaced by zero.
5. Average attribute rates over the **same retained evaluation rows** in each
   relation. Report Spearman and Pearson correlations against the retained LRE
   faithfulness. This relation-level correlation is our application of the
   published subject-enrichment metric, not a number reported by either source
   paper.

For the OLMo experiment we use `allenai/Olmo-3-1025-7B`, hidden state **17** (output
of zero-based block 16), the neutral prompt `{} is known for`, 28 relations,
513 subject-relation rows, and 456 distinct subjects. The primary outcome and
layer are fixed before computing the correlations. No per-relation layer tuning
or selection by retrieval coverage is performed. The faithfulness values come
from `results/olmo/results/paper_figure_data.json`.

The original experiments use other models/prompts and known-fact subsets. Our
run uses this repository's model, neutral prompt, and retained evaluation subjects
to make the baseline comparable to the existing Jacobian/logit-lens results.

### Retrieval details inherited from the reconstruction

Gerstner's `wiki_retrieve.py` initializes the BM25 tokenizer's vocabulary using
the subject queries, then tokenizes Wikipedia with `update_vocab=False`. We
preserve this behavior, including its effect on document lengths and the empty
token sentinel. Consequently this index is **specific to its subject-query set**;
it is not a conventional full-vocabulary Lucene index. BM25S 0.3.11 implements the
Lucene scoring variant with `k1=1.5`, `b=0.75`. All paragraphs contribute to the
document count and average length, including paragraphs with no query words.
Retrieval ties follow the pinned BM25S/Numba implementation.

To keep the full-corpus run practical, our implementation builds sparse counts
in parallel and vectorizes the same BM25 formula. Tests compare its float32 sparse
score arrays and retrieved IDs **exactly** with the upstream BM25S scorer,
including repeated tokens, empty documents, multi-token queries, and unseen
query words. The extraction fixes upstream XML-wrapper handling and checkpoint
restart problems without changing paragraph cleaning.

### Frequency ranking extension

Neither source paper defines a frequency-ranked Wikipedia vocabulary as its
attribute-rate metric. We additionally retain such a ranking to evaluate the
baseline suggested in the task:

- Count each model token in the concatenated retrieved text; rank the **full
  output vocabulary** by decreasing frequency. Use zero-based average ranks for
  ties. Absent tokens share the bottom rank (they do not receive rank zero).
- Score the same model-answer token IDs as the retained logit-lens baseline.
  Aggregate as `-median(log10(rank + 1))`, then correlate with LRE faithfulness.
  This extension keeps all tokens, including short numeric tokens, so all 28
  relations are comparable.
- Also correlate Wikipedia token frequencies with logit-lens logits within each
  subject, over the vocabulary passing the prediction-side attribute filter.
  Spearman here compares rankings with ties; report the mean and median across
  unique subjects, counting constant-frequency cases as undefined.

## Run

From `code/`:

```sh
uv sync --locked --extra gpu --extra wiki --inexact
uv run --no-sync jlens geva-baseline subjects

mkdir -p /mounts/work/jonas/data/geva-wikipedia-20211020
curl -fL --retry 4 -C - \
  https://archive.org/download/enwiki-20211020/enwiki-20211020-pages-articles-multistream-index.txt.bz2 \
  -o /mounts/work/jonas/data/geva-wikipedia-20211020/enwiki-20211020-pages-articles-multistream-index.txt.bz2
curl -fL --retry 4 -C - \
  https://archive.org/download/enwiki-20211020/enwiki-20211020-pages-articles-multistream.xml.bz2 \
  -o /mounts/work/jonas/data/geva-wikipedia-20211020/enwiki-20211020-pages-articles-multistream.xml.bz2

uv run --no-sync jlens wiki-corpus prepare \
  --dump-dir /mounts/work/jonas/data/geva-wikipedia-20211020 \
  --subjects generated/geva_baseline/subjects.json \
  --out /mounts/work/jonas/data/geva-wikipedia-20211020/olmo-main --workers 96
uv run --no-sync jlens wiki-corpus retrieve \
  --corpus /mounts/work/jonas/data/geva-wikipedia-20211020/olmo-main \
  --out generated/geva_baseline/retrieval.json --workers 32
uv run --no-sync jlens geva-baseline lens
uv run --no-sync jlens geva-baseline score \
  --retrieval generated/geva_baseline/retrieval.json
uv run --no-sync jlens geva-baseline report \
  --results generated/geva_baseline/results.json
```

The data directory is configurable. Preparation can overlap a sequential download
with `--wait-for-download`; completion still requires the full dump's checksum.
Completed extraction shards, count matrices, BM25 index, and per-subject logits
are reusable. Extraction configuration fingerprints prevent reuse with another
query set. Logit caches are tied to the model, layer, prompt, top-k, and SHA-256 of
the reference file. Disk space is needed for the 20.15 GB compressed dump plus the
paragraph shards and index. Corpus processing is CPU-only; readouts require a GPU
with space for the 7B model in float32.

Input SHA-1 checksums (from Internet Archive metadata):

```text
index  7acba6cb378de9fd0a71b8a2a31f81573d03b02e
dump   64714517eae47fcce24803feb90b4a440e963fe0
```

The result JSON retains all relation rows, subject overlap counts and denominators,
top predictions, reference/data hashes, the stopword list, model revision, corpus
coverage, and numerical readout validation. The adjacent `*_vocabulary.json`
contains ranked tokens with counts. `retrieval.json` retains IDs, scores, titles,
text, and keep/drop decisions for all 100 retrieved paragraphs per subject.

Recompute the retained correlations without model weights or Wikipedia:

```sh
uv run jlens geva-baseline report --plot-dir generated/geva_baseline
```

The adapted source is attributed in both modules; its [Apache-2.0 license](licenses/geva-gerstner-APACHE-2.0.txt) is retained.
