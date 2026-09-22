# Gemini paired triplets for anchor/subject interchange

This is a new research corpus for the operator-term swapping experiment in
`paper/lens.tex`. Every included subject has two different relations. Corpus
construction is independent of OLMo outcomes; poor-performing relations remain
in the results.

- `PROTOCOL.md`: specification frozen before OLMo measurements.
- `*_candidates.json`: generated pairs, first audit, and exclusions.
- `*_sourced.json`: independent audit against fetched Wikipedia evidence;
  author writing languages are supplemented by Wikidata P6886.
- `manifest.json`: exact demonstration/reference/validation/test assignments.
- `triples.jsonl`: flattened subject/relation/object facts and split labels.
- `API_PROVENANCE.json`: returned Gemini version, response IDs, hashes, tokens.

Wikipedia pointers are fetched before inclusion. The source-backed audit checks
both labels and entity identity, and rejects uncertain or ambiguous pairs. This
is automated corroboration, not a guarantee that every label is correct.
The generated corpus has selection biases toward recognizable entities and
accessible, sufficiently informative English-language source articles.

The local `generated/anchor_swap_gemini/` directory retains exact prompts,
Gemini JSON responses, source excerpts, gzip-compressed HTML snapshots, Wikidata
snapshots, model vectors and execution logs. These larger working artifacts are
not tracked by Git. API responses contain no credentials. The small retained
records identify their sources and include audit reasons; URLs and source hashes
are not a substitute for the underlying evidence. Wikipedia text is available
under its applicable CC BY-SA license and Wikidata structured data under CC0;
keep attribution when redistributing source excerpts.

From `code/`, with the existing GPU environment and Google ADC credentials:

```sh
python scripts/build_anchor_swap_gemini.py
python scripts/source_audit_anchor_swap.py
python scripts/supplement_author_sources.py
python scripts/run_anchor_swap_gemini.py --freeze
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_anchor_swap_gemini.py --domain countries
# Repeat the last command for each of the six domains.
.venv/bin/python scripts/report_anchor_swap_gemini.py
```

`launch_anchor_swap_gemini.py` waits for all source audits, freezes the manifest,
and runs all six domains using four idle 80 GB local GPUs. It refuses to start
if those GPUs lack sufficient free memory. The manifest cannot be silently
replaced; re-running against different source audits requires a new run directory.
API calls reuse cached responses, so reproduction with the local artifacts does
not charge again. Fresh generation is stochastic and may yield another corpus.
The retained manifest and model revision define the reported experiment.

Results are in `results/anchor_swap_gemini/`; the narrative report is linked from
`docs/anchor-swap-gemini-results.md`. Predictions include all controls and raw
first-token IDs/text. Never interpret a first-token hit as complete-answer
semantic correctness, or a swap result as proof of isolated factual circuits.
