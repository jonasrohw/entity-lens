# Command reference

Run commands as `uv run jlens COMMAND`. Use
`uv run jlens COMMAND --help` for command-specific arguments. Put common
`--root PATH` and `--output-dir PATH` options before `COMMAND`.

## CPU reproduction and analysis

| Command | Purpose |
| --- | --- |
| `reproduce` | Check rebuilt Figure 1 inputs, then generate statistics, both tables, and paper figures |
| `reproduce --supplementary` | Also generate subject panels, aggregate rank plots, and outlier reports |
| `statistics` | Print the paper's numerical results |
| `build-figure-data` | Reconstruct Figure 1's JSON; accepts `--output` |
| `table-relations` | Write relation table LaTeX to stdout |
| `table-prominence` | Write prominence table LaTeX to stdout |
| `plot-agreement` | Main relation-level figure |
| `plot-facts` | Product and three outlier fact-rank figures |
| `plot-relation-ranks` | Mean or median ranks across relations; accepts `--stat` |
| `plot-subject-ranks` | Continuous per-subject rank panels |
| `plot-subject-outcomes` | Binary hit/miss panels ordered by AUC |
| `subject-statistics` | Within-relation AUC, Wilcoxon test, and shuffle control |
| `relation-statistics` | Supplementary correlations by subject domain |
| `rank-outliers` | Signed residual rankings for both relation-level plots |
| `prominence-statistics` | Recompute the object-prominence and leakage analysis; accepts `--out` |

For standalone LaTeX tables, create the output directory before redirecting:

```sh
mkdir -p generated
uv run jlens table-relations > generated/table_relations.tex
uv run jlens table-prominence > generated/table_prominence.tex
```

Compare the retained unfiltered ranks with the identifying-token protocol:

```sh
uv run jlens plot-subject-ranks \
  --lens results/unfiltered/jacobian_lens.json \
  --rank-dir results/unfiltered --tag _unfiltered
```

Explicit file arguments are interpreted relative to the working directory,
except `build-figure-data --output`, which is relative to `--root`. Defaults
always resolve from `--root` or `--output-dir`.

## GPU measurements

Use `uv run --extra gpu jlens COMMAND`. All individual measurements accept
`--device` (default `cuda:0`). `CUDA_VISIBLE_DEVICES` chooses which physical
GPU that logical device refers to.

| Command | Purpose |
| --- | --- |
| `rerun` | Execute all measurements into an isolated directory and reproduce artifacts |
| `measure-faithfulness --relation NAME` | Six-layer by six-beta LRE sweep |
| `measure-jacobian-lens --rels NAME,NAME` | Differential and affine lens scores |
| `measure-subject-ranks --relation NAME` | Recompute ranks at an existing selected LRE cell |
| `measure-logit-lens` | Layer-17 direct logit-lens baseline matched to stored subjects and model answers |
| `measure-logit-lens-diff` | Same, but on the reference-subtracted state: the lens without its Jacobian |
| `measure-jacobian-lens-raw` | Averaged Jacobian applied to the raw activation: the lens without its difference |
| `measure-answer-prior` | Neutral-context answer-token log-probability control |
| `measure-prominence` | Greedy generation from two prose prompts |
| `probe-names "Barry Sobel" "Sobel"` | Inspect neutral-prompt lens associations for arbitrary names |

The faithfulness and Jacobian-lens commands default to the **shared** split.
Use `--legacy-split` for the main paper protocol. `--distinct-first-token`
filters ambiguous object prefixes. `--trailing-space` corrects numeric-object
targets. The runner supplies these flags separately for the appropriate stages.

The subject-rank scorer requires matching shared splits and the stored
faithfulness files specified by `--faith-dir`. Its `--no-distinct-first-token`
option selects the unfiltered comparison. All three baselines require the
main and corrected-year Jacobian-lens files under the selected root's `results/`;
they rebuild the legacy split and refuse to score a relation whose template,
layer, subjects, or model answers differ from those files.

`--model-correct` is an advanced cached-screen protocol. Neither its screen
files nor the original screening producer were included in the source
extraction. Supply an external screen directory with `--model-correct-dir`;
missing or mismatched screens fail explicitly. This option is not used by the
paper reproduction workflow.
