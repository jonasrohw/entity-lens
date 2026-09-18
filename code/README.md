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
| `statistics.txt` | Correlations, controls, permutation test, and fact-level counts |
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

## Corpus conditional probabilities

Measure `P(object | subject)` and `P(subject | object)` in the full published
training split and correlate both with retained LRE faithfulness:

```sh
uv sync --locked --extra corpus --inexact
uv run --no-sync jlens corpus-conditionals --workers 64
```

The scan streams compressed shards in parallel, checkpoints completed work, and
writes counts, per-stage results, and Pearson/Spearman correlations under
`generated/corpus_conditionals/`. See the [method and run guide](docs/corpus-conditionals.md)
for corpus scope, matching rules, partial smoke runs, and resume/report options.

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
