# Entity lens

Python implementation of linear relational decoding and entity readout experiments,
with pinned model and dataset revisions, saved measurements, and statistical reports.

## Setup

Use Linux, Python 3.12.12, and uv 0.9.27. Dependencies are pinned in `uv.lock`.
The launcher installs uv when absent and requires version 0.9.27.

```sh
./reproduce.sh check
uv run --locked entity-lens verify-artifacts
```

Checks run Ruff, strict mypy, Vulture, ShellCheck, and pytest with branch coverage.
Tests use synthetic data and small local models; the CUDA test requires a GPU.
CI also verifies saved artifact checksums without downloading datasets or weights.

Copy `.env.example` to `.env`, export `ENTITY_LENS_*` variables, or pass
`ENV_FILE=/path/to/settings.env` to configure storage. Environment files use literal
`KEY=VALUE` assignments; exported variables take precedence. Local environment
files are ignored by Git. For gated models, accept the upstream licences and export
`HF_TOKEN`.

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

`analysis` stops on checksum failures. To replace the manifest, run
`entity-lens manifest` or `all` after reviewing numerical differences.

For fresh measurements, set `ENTITY_LENS_ARTIFACTS_DIR` to an empty directory;
some stages reuse existing files.

## Resources

Measurements use float32 and eager attention on GPUs with substantial memory;
the largest model uses two GPUs. Worker records list hardware and library versions.

Corpus scans need several terabytes of storage. Wikipedia processing also needs
substantial disk space and CPU time. Reports may download pinned tokenizers even
with saved measurements. Upstream dataset and model access is required.

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

## Verification scope

The saved artifacts match 413 published values at the configured printed precision.
Fresh runs and agreement across GPU architectures remain unverified. Static checks
and synthetic-model tests do not establish scientific validity. Known limitations:

- Worker source hashes span multiple revisions; saved outputs may predate this source.
- Wikipedia date expansion uses the current clock, so extracted text can differ.
- Reference-set evaluation uses 100 rows. Swap and diagnostic analyses also filter
  rows or select cohorts, limiting dataset coverage.
- Year exclusions affect entire relations rather than individual predictions.
- Seeded reference selection does not guarantee answer-category diversity.
  Some diagnostics use literal gender targets.

Full-protocol reproduction requires resolving these differences and rerunning
measurements before updating the baseline.
