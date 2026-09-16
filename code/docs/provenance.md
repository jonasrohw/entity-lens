# Source and data provenance

## Source extraction

The starting bundle was extracted on 2026-09-15 from
`beta:/mounts/Users/cisintern/hs/auth/linz` for the `jlens` paper. Its README
identified the paper source as `/home/jonas/Code/jlens/lens.tex`, with paper
checkout commits `97c54a1b` on beta and `1a59b8bb` locally. These are historical
identifiers, not runtime requirements or commits of this refactored repository.

The original bundle retained the measurement scripts, 28 named-entity relation
files, paper result snapshots, and reproduction helpers. It excluded unrelated
training experiments and model checkpoints. The relation inputs originated in
the Hernandez et al. (2024) LRE datasets according to the source scripts; the
bundle did not include a dataset license or a complete bibliographic record.

Two historical checksum lists are retained verbatim:

- `UPSTREAM_SOURCE_SHA256SUMS.txt` records copied source before the original
  portability edit that introduced `JLENS_ROOT`.
- `SNAPSHOT_SHA256SUMS.txt` records the pre-refactor bundle and its old paths.

These lists describe historical files. They are **not** checksums of the current
refactored Python code and cannot be checked against the new layout directly.
`asset_manifest.json` records the current paths and SHA-256 hashes of all 174
retained dataset and measurement JSON files. Each file's bytes were checked
against the original bundle during the refactor. `pytest` verifies the manifest.

## Refactor

The code now lives in one flat `jlens` package. Dependencies are declared in
`pyproject.toml`, locked with uv, and split into CPU requirements, an optional
GPU extra, and development tools. Cluster-specific shell launchers are replaced
by the portable `jlens rerun` workflow. The duplicate symlink dataset view and
the duplicate Figure 1 JSON in `repro/` were removed.

The [migration map](migration.md) records module and directory changes. Existing
JSON contents and historical schema keys are preserved. Generated artifact
filenames use descriptive names. Runtime file access, CLI parsing, and model
loading have been moved out of module import paths.

The main table numbers and printed paper statistics were compared with the
pre-refactor programs in the same dependency environment. Figure 1's five arrays
match the retained snapshot exactly. Jacobian refactoring is checked against an
analytic derivative and a tiny randomly initialized OLMo model on CPU. The
large-model GPU experiments have not been rerun during this cleanup.

The source extraction lacked the original plotting programs for `fig_product.png`
and `blacksheep.png`. The retained reconstruction helpers reproduce the quoted
numbers; their typography and pixels can differ from the paper's original PNGs.
The refactor exposes these reconstructions as `product_fact_ranks.png` and
`outlier_fact_ranks.png`.

The supplementary domain analysis now consistently prefers corrected-year
faithfulness values. Its old implementation selected whichever file happened
to appear last in a filesystem glob. This fixes nondeterministic supplementary
output without changing the retained snapshots or main paper statistics.

## Publication metadata

The supplied bundle did not specify a repository license, author list, final
paper title, DOI, or citation record. No license grant or bibliographic metadata
has been invented. Those fields should be supplied by the authors before public
release; the historical attribution above is preserved.
