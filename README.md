# Entity lens

Python implementation of linear relational decoding and entity readout experiments,
with pinned model and dataset revisions, saved measurements, and statistical reports.

## Setup

Use Linux, Python 3.12.12, and uv 0.9.27. The Python version is specified in
`.python-version`; `uv.lock` pins the runtime and development dependencies.
The launcher installs uv 0.9.27 when uv is absent and rejects other versions.
`uv sync --locked` fails if the dependency declarations and lockfile disagree.

```sh
./reproduce.sh check
uv run --locked entity-lens verify-artifacts
```

The checks run Ruff formatting and linting, strict mypy, Vulture, ShellCheck,
and pytest with branch coverage. Tests use synthetic data and small local models.
The CUDA test runs when a GPU is available; CPU environments skip it. CI runs the
same checks and validates saved artifact checksums without downloading the dataset
or model weights.

Copy `.env.example` to `.env` to configure local storage. Alternatively, export
`ENTITY_LENS_*` variables or pass `ENV_FILE=/path/to/settings.env` to the launcher.
Environment files accept `KEY=VALUE` assignments, optional enclosing quotes,
blank lines, and comment lines. Values are literal; shell expressions are not
expanded. Exported variables take precedence. Keep credentials and local paths
out of version control; `.env` and `.env.*` are ignored except `.env.example`.
For gated model downloads, accept the upstream licences and export `HF_TOKEN`.

## Commands

Run commands from the repository root.

| Command | Purpose |
|---|---|
| `./reproduce.sh check` | Static analysis and tests |
| `uv run --locked entity-lens verify-artifacts` | Verify all saved files against `artifacts/SHA256SUMS` |
| `./reproduce.sh analysis` | Verify saved artifacts, obtain the pinned dataset, rebuild reports, and verify the results against the existing manifest |
| `./reproduce.sh measure` | Run data acquisition and measurement stages |
| `./reproduce.sh all` | Run measurements, build reports, and write a new manifest |
| `uv run --locked entity-lens --help` | List individual pipeline stages |

`analysis` stops on a checksum failure and does not replace the manifest.
Creating a new baseline requires an explicit `entity-lens manifest` command or
an `all` run. Review numerical differences before accepting a new baseline.

Some measurement stages reuse existing files. To perform a fresh run, set
`ENTITY_LENS_ARTIFACTS_DIR` to a new, empty directory. Existing outputs are not
proof that the current source revision produced them.

## Resources

GPU measurements use float32 and eager attention for second derivatives.
The configured models require GPUs with substantial memory; the largest model is
configured for two GPUs. The saved worker records identify the hardware and
library versions used for those measurements.

Corpus scans need several terabytes of storage. Wikipedia processing also requires
substantial disk space and CPU time. CPU report generation can require downloading
pinned tokenizers even when saved measurements are available. Dataset and model
availability depend on their upstream repositories and access requirements.

## Configuration and outputs

| Path | Contents |
|---|---|
| `configs/protocol.toml` | Dataset revision and checksums, prompts, split seeds, readout settings, and numerical tolerances |
| `configs/models.toml` | Model repositories, revisions, layers, and GPU requirements |
| `configs/experiments.toml` | Experimental protocols and statistical settings |
| `configs/corpus.toml`, `configs/wikipedia.toml` | Corpus sources, checksums, extraction, and retrieval settings |
| `configs/tables.toml`, `configs/figures.toml` | Table and figure formatting |
| `configs/paper_values.toml` | Published comparison values and precision rules |
| `src/entity_lens/` | Measurement, storage, scheduling, analysis, and CLI implementation |
| `tests/` | Unit and integration tests |
| `artifacts/` | Saved measurements, worker provenance, statistics, tables, and figures |

Public dataset and model identifiers are retained so their pinned sources remain
accessible. Names within research data are part of the dataset and are preserved.

## Verification scope

The saved comparison reports 413 matching values at the configured printed
precision. This is a property of the saved artifacts, not a certification of a
fresh end-to-end run. Static checks and synthetic-model tests also do not establish
scientific validity or identical results across GPU architectures.

Several constraints remain relevant to a fresh reproduction:

- Worker source hashes span multiple source revisions. Reusing existing outputs
  does not establish provenance for the current revision.
- Wikipedia extraction expands date-related magic words using the current clock,
  so newly extracted paragraphs can differ from the saved extraction.
- The reference-set experiment evaluates a configured 100-row slice. Some swap
  and diagnostic analyses also select cohorts or filter rows. These paths do not
  establish full-dataset evaluation.
- Year exclusions currently propagate to entire relations when any audited row
  is ignored. This differs from excluding only affected predictions.
- Reference selection uses seeded shuffling without guaranteeing answer-category
  diversity. Some diagnostic statistics use literal gender targets.

These protocol differences must be resolved, with fresh measurements and updated
baselines, before claiming compliance with full-dataset, per-prediction year
exclusion, gender-alternative, and diverse-reference evaluation requirements.
