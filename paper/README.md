# Full-evaluation manuscript

The submission source is `lens.tex`; build `lens.pdf` with:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error lens.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error shared_baselines.tex
```

The paper reports the differential/contrastive entity lens (CEL) as its primary readout. The raw/noncontrastive lens (NEL) is a separate baseline. The latest GitLab prose had defaulted to NEL while the verified headline measurements were CEL; the distinction is now explicit.

All benchmark results use the original 28 relations and all 9,747 rows per model, including demonstrations, references and duplicate rows. Gender alternatives are enabled; original continent reference sets are retained. Mistral, Gemma and both Qwen models exclude the two standalone-`1` year relations from the primary correlation, leaving 26 relations and 9,709 scored rows. The appendix also reports the all-28 sensitivity. GPT-J, Pythia and both Qwen models use a leading newline and carry a dagger in the model tables.

The canonical inputs are:

- `../code/results/all_models_gender_original_continents/`: full-row LRE and original approved lens measurements.
- `../code/results/gptj_pythia_full_row_diagnosis/`: full-row newline lens replacements for GPT-J and Pythia.
- `../code/results/paper_full_evaluation/`: matching baseline readouts, full-subject Wikipedia retrieval, corpus reaggregation, generated statistics and validation.

Regenerate the numerical TeX and main figure from the code directory:

```sh
.venv/bin/python scripts/build_full_evaluation_paper.py
```

Active generated tables live in `generated/full_evaluation/`. The paper does not import `generated/shared_baselines/` or historical capped/shared-split figures. The standalone appendix builds from the same new inputs as the main manuscript.

The pre-update manuscript and tables are preserved in `archive/pre_full_evaluation_2026-09-24/`, with source hashes and the merged GitLab revision. Its prominence, reference-composition, anchor-swap, held-out/shared-split and capped numerical sections are historical material, not current full-evaluation evidence. Corresponding numerical claims were removed from the compiled paper; their code and archived text remain available for a separate compatible rerun.

The evaluation is explicitly in sample: demonstration/reference rows are evaluated and LRE parameters are selected on those rows. PDF checks do not constitute author approval or a submission.

Git retains the per-row measurements, statistics, code snapshots, generated tables and PDFs. The multi-gigabyte Wikipedia retrieval/index cache remains local; its configuration and hashes are retained. Rebuilding attribute retrieval requires the source Wikipedia corpus and `scripts/retrieve_shared_wikipedia.py --all-rows`. Regenerating corpus-frequency statistics also requires the saved scans under `code/generated/corpus_conditionals/`, which are local generated artifacts. The committed tables and PDFs can be compiled without these caches.
