# Jacobian lens and relational faithfulness

Code and retained measurements for studying whether a relation-agnostic Jacobian
lens predicts the faithfulness of linear relational embeddings (LREs) in
`allenai/Olmo-3-1025-7B`.

The repository includes 28 named-entity relation datasets and the small result
snapshots needed to regenerate the paper's statistics, LaTeX tables, and figures.
CPU reproduction does not download model weights or require PyTorch.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from this checkout:

```sh
uv sync --locked
uv run jlens reproduce
```

Python 3.12 is selected by `.python-version`. Dependencies and development tools
are declared in `pyproject.toml`; `uv.lock` pins the resolved environment.

Artifacts are written to `generated/`:

| Artifact | Contents |
| --- | --- |
| `statistics.txt` | Correlations and baselines (including the restored zeroth-order term), controls, permutation test, and fact-level counts |
| `paper_figure_data.json` | Reconstructed input, checked against the retained snapshot |
| `table_relations.tex` | All 28 relations, with corrected numeric-object scores |
| `table_prominence.tex` | Object-prominence analysis for 14 relations |
| `relation_agreement.pdf`, `.png` | Main relation-level scatter plot |
| `product_fact_ranks.png` | Product relation's subject ranks |
| `outlier_fact_ranks.png` | Three outlier relations' subject ranks |

Expected headline results are Spearman **ρ = 0.658** across 28 relations,
**ρ = 0.143** for the matched logit-lens baseline, and **ρ = 0.644** across
25 relations under the shared-split, identifying-first-token protocol.

```sh
uv run jlens statistics
uv run jlens reproduce --supplementary
uv run jlens --output-dir /tmp/jlens-figures reproduce
uv run jlens --help
```

Common `--root` and `--output-dir` options go **before** the command. `--root`
selects a directory containing `data/` and `results/`; it defaults to the
checkout. `JLENS_ROOT` and `JLENS_OUTPUT_DIR` are equivalent environment
settings. Commands work independently of the current working directory after
installation. A wheel installation requires an explicit `--root` pointing to
this checkout because the datasets are distributed with the repository.

## Repository layout

```text
jlens/             Flat Python package: measurements, analyses, plots, and CLI
data/relations/   28 relation JSON files (no duplicate symlink tree)
results/           Retained measurement snapshots
  main/            Legacy paper LRE sweep, including corrected year runs
  shared_split/    Identifying-first-token protocol and per-subject ranks
  unfiltered/      Shared-split comparison without the token filter
tests/             Scientific regressions and protocol checks
docs/              Methods, command reference, and provenance
pyproject.toml     Dependencies, packaging, CLI, lint, and test configuration
uv.lock            Resolved dependency versions
```

See the [command reference](docs/commands.md),
[protocol details](docs/methods.md), and
[source and data provenance](docs/provenance.md).
The historical JSON field names are preserved so the retained measurements
remain unchanged; their meaning is documented in the methods guide.

## Qwen: bug diagnosis and ordinary model predictions

The [consolidated bug diagnosis](docs/qwen3-8b-base-bug-diagnosis.md) brings together
the implementation checks, the separate normalization fix, positional controls,
raw/differential comparisons, and the exact token-score decomposition for Germany.
It includes the newly published per-reference scores, coordinate ablations,
JVP tensors, and reproduction scripts, with links to all earlier analyses.

[Ordinary Qwen predictions](docs/qwen3-8b-base-neutral-prompt-predictions.md) show
the next-token probabilities and 48-token greedy continuations for all 16 prompts
`{country} is known for`. Every prompt predicts ` its` first (51.67–74.15%);
the six inspected artifact tokens occur in neither the next top 10 nor the
generated continuations. This separates the lens artifact from the remaining
question of whether the neutral prompt is suitable for the measurement.

## Qwen3 8B Base: 26-relation outliers

The [raw J-lens experiment](docs/qwen3-8b-base-raw-jlens.md) drops reference-state
subtraction, using `mean_i J_i h_subject`. At the original subject position,
Spearman ρ is **0.294661** (versus 0.298768 with subtraction), and `-Semit`
remains the top prediction for 10/16 country-language subjects. At the final
`for` position, raw ρ is **0.364819** (p = 0.066881), but no target reaches the
top 10. The report includes both positions, all 26 relation points, and raw
country-language predictions. Subtraction alone does not explain the artifact.

The [country-capital prediction tables](docs/qwen3-8b-base-country-capital-predictions.md)
show raw and differential top-five tokens and target ranks for all 16 countries
at both positions. CSV/JSON downloads include exact token IDs, input-state norms,
and all 240 included reference contributions. At the subject position, removing
subtraction increases top-10 hits from 2/16 to 4/16; `-Semit` remains top-1 for
the same ten countries.

The [final-prompt-position experiment](docs/last-position-lre-correlation.md)
moves the lens input and Jacobian intervention to `for` in `{subject} is known for`,
keeping the LRE scores, targets, layers and reference subjects fixed. Across the
same 26 relations, Spearman ρ changes from **0.298768 to −0.067077** for Qwen
(new p = 0.744746), and from **0.782758 to 0.645912** for OLMo
(new p = 0.000365077). Qwen's extreme norm disparity and repeated `-Semit`
predictions disappear, but target rankings worsen: **0/491 targets are in the
top 10**. The report includes all relation points, refitted residuals, a plot,
per-subject evidence and reproduction scripts. This ablation does not replace
the historical results below.

**Review finding (20 September):** the original Qwen differential lens can mix
large first-token activations with much smaller later-token states. This produces
the repeated `-Semit` / `(Initialized` outputs in the country relations and
compromises their interpretation as semantic-salience scores. The
[implementation review](docs/qwen3-jlens-implementation-review.md) includes
independent derivative checks, reference/prefix controls, an OLMo comparison,
and a separate fixed normalization bug. The historical scores below are retained;
removing the year relations does not remove this positional artifact.

The [updated Figure-1 analysis](docs/qwen3-8b-base-figure1.md) excludes only
`presidents_birth_year` and `presidents_election_year`, retaining `name_gender`.
The OLS line and all residuals are refitted on the remaining **26 relations**.
Qwen's Spearman correlation increases from **0.060390 to 0.298768**
(two-sided p = 0.138179); the OLS slope increases from **0.046492 to 0.095745**.

The largest positive residuals are `name_gender` (+53.24 percentage points),
`name_religion` (+42.94), and `landmark_on_continent` (+36.30). The largest
negative residuals are `company_ceo` (−52.46), `person_mother` (−44.44), and
`superhero_person` (−43.84). Faithfulness is a success rate against the model's
first answer token: `0.02` means 2% agreement, not a 2% error rate.

The report includes all 26 points ranked by absolute residual, changes from the
old fit, unrounded CSVs, and a plot comparing Qwen and OLMo on the same relation
set. The current OLMo snapshot gives **ρ = 0.782758** on these 26 relations.
`name_gender` and `name_religion` have the largest cross-model residual
differences; these identify follow-up candidates rather than establish a causal
explanation. `name_gender` still has only three LRE test examples.

The [original 28-relation report](docs/qwen3-8b-base-figure1-28-relations.md)
and its CSV remain available. Reproduce the new fit, tables, and figures on CPU:

```sh
uv run python ../paper/repro/qwen3_26_relations.py
```

Concrete prediction tables are available for
[`company_ceo`](docs/qwen3-8b-base-company-ceo-predictions.md) and
[`country_language`](docs/qwen3-8b-base-country-language-predictions.md).
For `country_language`, the selected LRE reproduces 6/8 model targets;
the differential lens has 0/16 top-1 and 1/16 top-10 matches.
The report includes every prediction, the five countries shared between the
two test sets, and a JSON snapshot with token IDs and validation results.

## Corpus conditional probabilities

Measure `P(object | subject)` and `P(subject | object)` in the full published
training split and correlate both with retained LRE faithfulness:

```sh
uv sync --locked --extra corpus --inexact
uv run --no-sync jlens corpus-conditionals --workers 64 \
  --faith-dir results/olmo/results/main
```

The scan streams compressed shards in parallel, checkpoints completed work, and
writes counts, per-stage results, and Pearson/Spearman correlations under
`generated/corpus_conditionals/`. See the [method and run guide](docs/corpus-conditionals.md)
for corpus scope, matching rules, partial smoke runs, and resume/report options,
and the [measured results](docs/corpus-conditionals-results.md) for the 2026-09-18
full-corpus run and its relation to Merullo et al. (2025).

## Wikipedia attributes baseline

The full-Wikipedia Geva/Gerstner baseline has been run on the 28 OLMo relations
(513 evaluation rows). Attribute rate versus LRE faithfulness gives Spearman
**ρ = −0.408**; the separately defined Wikipedia frequency-rank extension gives
**ρ = +0.240**. See the [results and validation](docs/geva-baseline-results.md)
and [method and full rerun instructions](docs/geva-baseline.md).

Recompute the retained correlations and plots on CPU:

```sh
uv run jlens geva-baseline report --plot-dir generated/geva_baseline
```

## Rerun the model measurements

Use a CUDA machine with enough memory for a 7B model in float32 plus
Jacobian-vector products. Install the optional GPU dependencies:

```sh
uv sync --locked --extra gpu
uv run --extra gpu jlens rerun --dry-run
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu jlens rerun
```

The runner downloads the model through Transformers if it is not cached,
copies the small relation inputs into `rerun/`, executes both paper protocols,
and regenerates their artifacts under `rerun/generated/`. It records successful
stages in `rerun/completed_stages.json` so an interrupted run can resume.
Choose another destination with `rerun --run-dir /path/to/run`.

The full measurement workflow is expensive. The dry run prints every command
without creating files or downloading weights. Standalone measurement commands
write to `generated/` by default; use their `--out` or `--out-dir` options to
choose another location.

The refactor has been checked against the retained CPU results and with small
CPU Jacobian tests. A full OLMo GPU reproduction has **not** been run as part of
this cleanup. New measurements can vary with hardware and dependency versions;
the lockfile records the environment used for the refactor, not the original
measurement environment.

## Development

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

The optional Jacobian tests run when PyTorch is installed (the tiny OLMo
integration test also needs Transformers):

```sh
uv run --extra gpu pytest
```

GitLab CI runs lint, formatting, CPU tests, artifact reproduction, and packaging.

Retained data and results are tracked; environments, caches, generated figures,
and reruns are ignored. Change tests when intentionally changing a scientific
protocol, and preserve the original snapshots for comparison.

## Larger same-subject anchor/term swap experiment

A Gemini-generated, source-checked corpus adds 1,672 triples across six domains,
with 644 subjects held out from fitting and tuning. Swapping another relation's
subject term lowers equal-domain first-token faithfulness from 57.4% to 34.1%
(23.3-point loss; paired 95% CI 20.9–25.6), while still outperforming wrong-subject
controls. See the [full results and examples](docs/anchor-swap-gemini-results.md)
and [dataset with frozen splits](data/anchor_swap_gemini/README.md).
