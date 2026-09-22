# GPT-J replication verification — 2026-09-19

The checks found no parallel-residual or Jacobian-product error that explains
the low correlations in the retained GPT-J results. A final-layer cache bug was
found and fixed, but it does not affect these snapshots: their lens uses state
15 and their LRE sweep uses states 4, 10, 12, 15, 18, and 20, all before GPT-J's
last state, 28. This audit does not establish why the scientific association is
weak, or prove correctness for every possible prompt/configuration.

## Hugging Face comparison

Reference: installed `transformers==5.17.0`,
`transformers/models/gptj/modeling_gptj.py`, and the
[HF GPT-J implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/modeling_gptj.py).
HF computes each block as

```text
n = ln_1(x)
y = attn(n) + mlp(n) + x
```

Both branches consume the same normalized input. The measurement code runs
HF's blocks directly and replaces a token at the complete block's output;
it does not reconstruct an attention-then-MLP residual path. Hidden-state index
15 corresponds to the output of zero-based block 14, after both branches and
the residual have been added.

The independent checks explicitly execute that equation using HF's attention
and MLP modules, without calling the block's forward method or using the
intervention hook. They compare it with normal HF forward execution and with
the hooked intervention. A deliberately sequential equation produces a
different answer in the regression test.

## Numerical checks

Pretrained checkpoint: `EleutherAI/gpt-j-6b`, revision
`47e169305d2e8376be1d31e765533382721b2cc1`; float32; eager attention;
PyTorch `2.14.0+cu130`; NVIDIA A100 80 GB.

Two prompts were tested: `France is known for` (4 tokens) and an eight-shot
country-capital prompt (77 tokens). Every retained sweep layer was checked.

| Check | Result |
| --- | --- |
| Explicit parallel tail vs. HF raw final state, all 12 prompt/layer cases | Exact equality |
| Reinserting the original subject state | Exact equality |
| Perturbed intervention vs. explicit parallel tail | Passed |
| Manual LayerNorm + biased unembedding vs. HF logits | Maximum absolute errors 4.77e-6 and 3.34e-5 |
| Double-backward JVP vs. central finite difference at epsilon 0.01 | Relative L2 error at most 8.10e-4 (0.081%) |
| JVP/independent reverse-gradient adjoint identity | Passed |
| Subject-token position vs. tokenizer offsets, all samples in neutral and relation templates | 19,494 checks; zero mismatches |
| Stored shared-split fingerprints, subject sets, and rank-zero/hit agreement | 25 relations, 1,536 subject outcomes; zero mismatches |

Finite-difference directions were normalized to the reference state's L2 norm.
Steps 0.003 and 0.001 were also recorded; cancellation increases float32 error
at the smallest steps, particularly on the longer prompt.

Tiny-GPT-J tests additionally compare the JVP with an explicit dense Jacobian
computed by ordinary reverse mode. They cover multiple layers, both a subject
before the last token and the last token itself, zero directions, cache on/off,
nontrivial final-norm weights/biases, and the final-block identity/zero derivative
limits. These tests do not rely only on two production scorers agreeing.

## Reproduction of saved measurements

The production entry points were rerun with the cached pretrained checkpoint,
writing into `generated/gptj_verification/`:

- `country_capital_city`: all 36 layer/beta cells reproduced exactly under each
  of the main and shared protocols (72 total). Best layer/beta, faithfulness,
  all 16 subject outcomes across the two protocols, LM-correct counts, and split
  fingerprints also matched.
- Main-protocol neutral lens for `country_capital_city`, `city_in_country`, and
  `person_occupation`: all 55 differential ranks matched exactly. Of 55 affine
  ranks, two differed by one vocabulary position: China, 1144 → 1145, and
  William Hardy Wilson, 18267 → 18268. All aggregate scores matched exactly.

This was a representative GPU reproduction, not a fresh all-relation sweep.
It ran before the unrelated final-layer fix. The corrected cache was then
verified on the pretrained checkpoint, including exact no-op restoration at
the last layer for both subject and output positions; the six-layer numerical
checks were repeated successfully.

## Bug fixed: caching the final intervention layer

`HF_output.hidden_states[-1]` is post-final-LayerNorm, whereas a forward hook on
the last transformer block replaces a pre-final-LayerNorm state. Previously,
selecting that last layer cached one boundary and intervened at another. For
an intervention at the output token, even reinserting the original cached value
changed the result. The previous final-output capture fix did not fix this
separate input-cache problem.

`forward_with_raw_hidden_states` now returns a separate state tuple whose final
entry is captured from the actual final block output, at every token position.
It preserves the original HF output. All measurement caches, the logit-lens
baselines, and the name probe use this helper. Intermediate states and the
retained GPT-J results are unaffected.

See [the implementation](../jlens/model.py) and
[regression tests](../tests/test_gptj_reference.py).

## Year-token protocol and correlation sensitivity

The OLMo trailing-space workaround is not required for GPT-J's example year
tokens. GPT-J encodes `" 1743"` as `[" 17", "43"]`, `" 1932"` as
`[" 1932"]`, and `" 2000"` as `[" 2000"]`; there is no initial bare-space
target in these examples. Appending a space to the query changes the GPT-J
prediction task. The `_ts` files should therefore be described as a prompt
variant, not automatically as a GPT-J tokenization correction.

Recomputing correlations from the retained original/variant measurements gives:

| Protocol | Relations | Spearman rho | p |
| --- | --- | --- | --- |
| Saved main, trailing-space year variants | 28 | 0.343468 | 0.073536 |
| Main, original year prompts | 28 | 0.312887 | 0.104983 |
| Main, excluding both year relations | 26 | 0.360992 | 0.070012 |
| Shared split | 25 | 0.154943 | 0.459580 |

Thus the year-prompt choice does not explain the weak association. Neither the
retained data nor the analysis policy selecting `_ts` files was changed in
this audit; choosing a scientific reporting protocol is separate from fixing
the verified cache error. Fixed lens-layer/prompt choices and sampling
variability remain possible contributors, not demonstrated causes. The saved
one-trial measurements also do not replicate the original LRE paper's
24-trial experimental setting.

## Reproduce this audit

From `code/`, with the GPU dependencies and checkpoint already installed:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_model.py tests/test_gptj_reference.py tests/test_baseline_measurements.py -q
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/verify_gptj.py
```

`--math-only` skips the production measurement reruns. `--out-dir` selects a
separate artifact directory. The script uses cached weights, records versions,
and asserts its checks. It writes measurements and `verification.json` without
changing retained snapshots. This audit's original reproduction report is in
`generated/gptj_verification/verification.json`; post-fix math/protocol results
are in `generated/gptj_verification/post_fix/verification.json`.

Targeted tests: **28 passed**, including 14 new GPT-J tests. Ruff lint/format
checks passed for all changed Python files, and `git diff --check` was clean.
Full suite: **84 passed, 14 failed**. The failures concern the previous
top-level result layout and asset manifest, which do not match the current
`results/olmo/`, `results/gptj/`, and `results/olmo-old/` organization. Running
the relevant tests from an untouched `HEAD` source archive against the same
retained files reproduced exactly the same 14 failures (baseline log:
`generated/gptj_verification/baseline_tests.txt`).
