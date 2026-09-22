# Five-model comparison, started 2026-09-19

The retained [results table](../results/five_model_sweep/README.md) reports only
completed model/protocol combinations. The [CSV](../results/five_model_sweep/correlations.csv)
and per-relation JSON retain the corresponding numerical results.

The comparison adds five checkpoints to the OLMo/GPT-J experiments. Model
selection was fixed before viewing correlation results. Four checkpoints are
base language models; Qwen3.8-27B is a recent post-trained hybrid model, included
to test a larger, current Qwen. All use the same raw completion prompts; this
is not a comparison of their chat interfaces.

| Name | Checkpoint | Layers | Fixed lens state | LRE state grid | Workers |
| --- | --- | --- | --- | --- | --- |
| Qwen3 8B Base | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | 36 | 19 | 6, 12, 16, 19, 22, 26 | rho1, GPUs 0–3 |
| Mistral 7B | [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) | 32 | 17 | 5, 11, 14, 17, 20, 23 | rho1, GPUs 4–7 |
| Pythia 6.9B | [EleutherAI/pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b) | 32 | 17 | 5, 11, 14, 17, 20, 23 | rho2, GPUs 0–3 |
| Gemma 2 9B | [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) | 42 | 22 | 7, 14, 18, 22, 26, 30 | rho2, GPUs 4–7 |
| Qwen3.8 27B | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 64 | 34 | 10, 22, 28, 34, 40, 46 | beta, pairs 0–1 and 2–3 |

Layer choices are the OLMo state indices scaled by model depth and rounded,
not selected using the observed correlations. The newer Qwen is larger than
the original 6–7B experiments, but model size, architecture, training data, and
post-training differ together; a difference cannot be attributed to size alone.

## Protocol

Both the main and shared-split protocols are measured over all 28 input
relations, retaining their normal sample limits and skip rules:

- LRE: eight demonstrations, eight references, up to 100 test subjects, one
  trial, six state indices, beta in 0.5, 1, 1.5, 2, 3, 5. Select the first best
  grid cell in layer/beta order, matching the original scorer.
- Main lens: `Random(17)` split, up to 20 subjects, eight references within the
  test set, excluding self-comparisons. The main LRE uses its original
  independent `Random(0)` then `Random(1000)` split.
- Shared lens/LRE: `Random(0)`, identifying-first-token filtering, disjoint
  reference and test rows, matching fingerprints. Relations with fewer
  than 17 usable samples are skipped; the lens also needs two test subjects.
- Report negative median log10(rank+1) versus LRE faithfulness using Spearman
  correlation. Differential, affine beta=1, and the full affine beta grid are
  retained. Beta tuning is in-sample and is not an independent evaluation.
- Year queries receive a trailing space only when the model's tokenizer
  produces a bare-space first token for every gold year. This applies to both
  Qwens, Mistral, and Gemma; Pythia uses the original query. This avoids blindly
  importing the GPT-J year-prompt problem found in the previous audit.

The tokenizer audit checked all 19,494 neutral/relation subject positions per
model against offsets, with zero mismatches for every model.

A later input audit found duplicate subjects in nine relations. Disjoint rows
do not ensure disjoint subjects, and some subjects also have multiple gold
objects. These original runs preserve the input rows for comparability. The
[focused Mistral report](../results/mistral_grid_search/README.md) includes
an additional check with identical duplicates collapsed and ambiguous subjects
removed before splitting, using frozen hyperparameters.

## Speed and correctness

[`jlens/model_sweep.py`](../jlens/model_sweep.py) keeps each model resident and
shares the forward/reverse graph for each reference across batches of subject
directions. It computes exact double-backward products rather than finite
difference estimates. Float32 and eager attention are retained; Qwen3.8 uses
HF's differentiable PyTorch linear-attention implementation. Its width-four
depthwise convolution is evaluated as four shifted elementwise products,
avoiding a very slow grouped-convolution double-backward path. This substitution
is checked against HF in float64 primitive tests and in pretrained-model logits
and JVPs before measurement. Each Qwen3.8 replica
initially spans two 80-GB GPUs without CPU/disk offloading. Completed small-model
allocations are transferred to additional Qwen replicas spanning three 48-GB
GPUs. Cross-host advisory locks prevent duplicate relation work.

For the large Qwen, `PrefixIntervention` caches the unchanged tokens strictly
before the intervention. Causality makes their derivatives zero. HF evaluates
the suffix with its ordinary attention/recurrent cache; disposable recurrent
updates avoid overwriting constants needed by autograd. A pretrained ICL check
found relative state error 2.2e-6 and JVP error 7.7e-6 versus the full prompt.
Each new replica repeats this check before using the optimization. Direction
batches are increased from two to eight after the initial validation runs;
this changes execution only, not sample sizes or the estimator.

Decoding invokes each model's own final normalization and output head. This
matters for Gemma2 and Qwen3.8, whose RMSNorm uses `1 + weight`; the older manual
readout helper assumes a directly multiplicative weight. The new runner also
handles Pythia's `gpt_neox.layers`, final LayerNorm, and output-embedding layout.
These architecture paths are isolated in the new runner rather than assuming
the older individual measurement commands support every new model.

Every worker gates measurements on:

1. Reconstructed logits agreeing with an ordinary HF forward pass.
2. Reinserting an unmodified state preserving the final state exactly.
3. Batched products agreeing with the original scalar JVP.
4. A central finite-difference derivative check.

All five pretrained models passed. Forty-three new CPU tests cover the five
architectures, both protocols, batched/scalar agreement, and uneven/zero
direction batches, convolution forward/derivative equivalence, and cached
prefixes across ordinary, hybrid, single-token, and chunk-boundary cases.
Cached/full-prompt scorers also agree for both protocols on a tiny hybrid model.
As an additional pretrained regression, the faster runner
reproduced GPT-J's 72 capital-city LRE grid cells, all selected outcomes, and
both lens aggregates. One deep affine rank differed by one position, matching
the earlier verification's float32 variation. The two protocols together took
50.1 seconds of measurement time on an A100, excluding model loading.

## Running and inspecting

From `code/`:

```sh
.venv/bin/python scripts/status_model_sweep.py
.venv/bin/python -m jlens.model_sweep \
  --root generated/five_model_sweep/results --summarize
```

`results/five_model_sweep/models.json` records the chosen models and layers;
`workers.json` records exact worker commands and host/GPU assignments. Each
model's output contains its checkpoint revision, verification values, host,
dtype, layers, per-relation elapsed time, LRE grid/outcomes, and lens ranks.
Tokenizer audits, downloaded checkpoint revisions (including the composite
Qwen config's revision), and source hashes live beside the worker logs. Existing
OLMo/GPT-J snapshots are not overwritten.

Each relation is written atomically. The launcher
[`scripts/launch_model_sweep.py`](../scripts/launch_model_sweep.py) leaves live
workers alone and restarts exited workers, which skip completed relation files.
Only run it with the original host/GPU allocation available. The main protocol
is processed first so its full-model correlations become available sooner.

`scripts/scale_model_sweep.py` migrates allocations after completed relation
writes and after all small-model measurements in an allocation finish. Its
`dynamic_worker_*.json` manifests record the additional worker commands.
`scripts/report_model_sweep.py` refreshes the retained results and table.
The GPU dependency group includes Accelerate for placement across GPUs.

This focused comparison does not rerun prominence generation, answer-prior
controls, or the separate raw-Jacobian/logit-lens ablations. Its results answer
the requested cross-model lens/faithfulness comparison.
