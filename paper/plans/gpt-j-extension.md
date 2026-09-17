<!-- shared with collaborators as paper/plans/gpt-j-extension.md -->

# Plan: extend the Jacobian-lens measurements to GPT-J-6B

**Status: proposed.** No code changes have been made yet.

## 1. Motivation

The paper currently measures one model, `allenai/Olmo-3-1025-7B`. GPT-J-6B
(`EleutherAI/gpt-j-6B`) is a stronger extension target than a same-family
model swap for two reasons:

- It is one of the models Hernandez et al. (2024) validated the original LRE
  method on, so this checks whether the lens-vs-faithfulness relationship
  holds on the exact model where LRE faithfulness was first characterized,
  rather than only on a model nobody has run LRE sweeps on before.
- It is architecturally about as far from OLMo as an open decoder-only model
  gets (GPT-2-style LayerNorm, parallel attention+MLP, partial rotary), so a
  positive result is a stronger generality claim than a same-family swap.

The cost is a correspondingly larger implementation lift: several parts of
the codebase hardcode assumptions specific to OLMo's (Llama-style)
architecture, not just its name.

## 2. Confirmed architecture differences (EleutherAI/gpt-j-6B)

| | OLMo-3-1025-7B (current) | GPT-J-6B |
|---|---|---|
| Depth | ~32 layers (layer defaults tuned to this) | 28 layers, hidden size 4096 |
| Final norm | RMSNorm (`model.norm`, `.variance_epsilon`, weight only) | **LayerNorm** (`transformer.ln_f`, `.eps`, weight **and bias**) |
| Block path | `model.model.layers[i]` | `model.transformer.h[i]` |
| Block topology | Sequential attention → MLP | **Parallel** attention + MLP (both read the same pre-block LayerNorm output, sum into the residual) |
| Position encoding | full rotary (assumed) | **partial** rotary, `rotary_dim=64` of 256 per-head dims |
| `lm_head` | tied / no bias assumed | `tie_word_embeddings=False`, separate `nn.Linear` — confirm whether it carries a bias term |

Source: HF GPT-J config/docs (`n_layer=28`, `n_embd=4096`, `rotary_dim=64`,
`layer_norm_epsilon`, `tie_word_embeddings=False`); EleutherAI's own
description of the parallel attention+FFN design.

## 3. Required code changes

### 3.1 Generalize the final-norm / logit reimplementation (the change that matters most)

Four files each **manually reimplement the final norm** as pure RMSNorm —
`v / sqrt(mean(v²) + eps) * weight`, with no mean-centering and no bias —
then multiply by the unembedding:

- [`jlens/measure_jacobian_lens.py:106-111`](../../code/jlens/measure_jacobian_lens.py#L106-L111) (`rank_of`)
- [`jlens/measure_faithfulness.py:126-128,254`](../../code/jlens/measure_faithfulness.py#L126)
- [`jlens/measure_subject_ranks.py:100-102,167`](../../code/jlens/measure_subject_ranks.py#L100)
- [`jlens/probe_names.py:62-64,83`](../../code/jlens/probe_names.py#L62)

For GPT-J this must become real LayerNorm math —
`(v - mean(v)) / sqrt(var(v) + eps) * weight + bias` — and the logit line
needs `+ lm_head.bias` if the checkpoint has one. Getting this wrong would
silently corrupt every reported rank without raising an error.

**Approach:** factor a single `final_logits(hidden_state, model) -> Tensor`
helper (e.g. in `jlens/model.py`) that dispatches on whether the model's
final norm is `LayerNorm` or an RMSNorm-style module, used by all four call
sites instead of each reimplementing the arithmetic inline. Validate it
against `model.lm_head(model.some_norm(hidden))`-style ground truth (i.e.
compare to a real forward pass) for both architectures before trusting any
downstream number.

### 3.2 Layer / norm attribute paths

Same four files reference `model.model.layers[...]` and
`model.model.norm.variance_epsilon` directly. For GPT-J this is
`model.transformer.h[...]` and `model.transformer.ln_f.eps`. Should be
covered by whatever abstraction 3.1 introduces, plus an equivalent
indirection for locating the block list (`model.transformer.h` vs.
`model.model.layers`).

### 3.3 `MODEL_ID`

Currently hardcoded in [`jlens/config.py:12`](../../code/jlens/config.py#L12).
Parameterize (e.g. `JLENS_MODEL` env var, following the existing
`JLENS_ROOT`/`JLENS_OUTPUT_DIR` pattern) rather than editing in place, so
both models can be run from the same checkout.

### 3.4 Layer index defaults

The fixed lens layer (17) and the LRE sweep (`5, 11, 14, 17, 20, 23`,
[`measure_faithfulness.py:62`](../../code/jlens/measure_faithfulness.py#L62))
were picked relative to OLMo's depth. For GPT-J's 28 layers, prefer reusing
the layer(s) Hernandez et al. themselves reported as best for GPT-J
relations over re-deriving a rescaled set from scratch — more defensible
given the "same model as the original paper" framing (see §1).

### 3.5 Tokenizer boundary effects

GPT-J uses GPT-2 BPE (50257 real tokens + 143 padding tokens). The
identifying-first-token filter and the `presidents_*` trailing-space
workaround exist because of leading-space tokenization quirks under OLMo's
tokenizer specifically. GPT-2 BPE has its own (different) leading-space
conventions, so:

- which relations pass `--distinct-first-token` may differ from OLMo's set,
- other relations beyond the two year relations may need a similar
  trailing-space (or other) workaround.

Do not assume the existing filtered/unfiltered relation lists transfer;
re-derive them for GPT-J's tokenizer.

### 3.6 Attention implementation

The double-backward JVP (`jlens/model.py`'s `HiddenStateIntervention.jvp`)
needs `attn_implementation="eager"`. Confirm the pinned `transformers`
version still supports eager attention for `GPTJForCausalLM` (GPT-J predates
flash-attention support, so this is likely fine, but verify rather than
assume).

### 3.7 Tests

[`code/tests/test_model.py`](../../code/tests/test_model.py)'s tiny-model JVP
sanity check is built on an OLMo-shaped config. GPT-J's parallel-block
residual topology is different enough (both attention and MLP branches read
the *same* pre-block input rather than feeding sequentially) that it
deserves its own tiny-GPT-J-config check, not just reuse of the existing
one — this is the cheapest place to catch a wrong JVP before spending GPU
time on real relations.

## 4. Resource note

GPT-J-6B in float32 (required for the Jacobian double-backward, matching the
current OLMo measurement scripts) is ~24GB for weights alone, before JVP
activations — comparable to the current OLMo-7B budget. Confirm against the
target GPU before scheduling the full sweep.

## 5. Staging

1. **Code only, no GPU:** parameterize `MODEL_ID` (3.3); build and unit-test
   the generalized final-norm/logit helper against both architectures (3.1,
   3.2); add the tiny-GPT-J JVP test (3.7).
2. **Small GPU check:** run `measure-jacobian-lens` and `measure-faithfulness`
   on one or two relations, confirm ranks/logits from the generalized helper
   match a real forward pass exactly (not just "looks plausible").
3. **Re-derive splits:** determine GPT-J's identifying-first-token relation
   set and any needed trailing-space corrections (3.5) before trusting the
   shared-split protocol.
4. **Full pipeline:** run the same `measurement_plan()` stages as OLMo
   (main split + shared split, all measurement types) into a separate
   `--root`/`--run-dir`, per the existing `jlens rerun` mechanism — do not
   overwrite the retained OLMo snapshots.
5. **Analysis:** `jlens reproduce` against the new root, compare headline
   correlation and layer/beta selections against Hernandez et al.'s
   published GPT-J results as an external sanity check.

## 6. Progress log

**Stage 1 (code-only) complete, 2026-09-16.**

- Added `resolve_blocks`, `resolve_final_norm`, and `final_logits` to
  [`jlens/model.py`](../../code/jlens/model.py): a single architecture-dispatch
  helper that locates the transformer block list (`model.model.layers` vs.
  `model.transformer.h`) and applies the correct final norm (RMSNorm,
  weight-only vs. LayerNorm, mean-centered with weight and bias, plus an
  optional `lm_head` bias) before the unembedding. `final_logits` reduces
  only over the last axis, so it works on both a single hidden vector and a
  batch (needed by `measure_faithfulness.py`'s vectorized beta sweep).
- Replaced the four duplicated hardcoded-RMSNorm reimplementations (3.1) and
  `model.model.layers[...]` hook sites (3.2) with calls to these helpers in
  `measure_jacobian_lens.py`, `measure_faithfulness.py`,
  `measure_subject_ranks.py`, and `probe_names.py`.
- Parameterized `MODEL_ID` via a `JLENS_MODEL` env var (3.3), set through a
  new `--model` flag in `cli.py`, following the existing `--root`/
  `--output-dir` pattern.
- Added `test_measurements_agree_on_tiny_gptj` to `tests/test_model.py`
  (3.7): a tiny random `GPTJConfig`/`GPTJForCausalLM`, exercising the full
  faithfulness → subject-ranks → jacobian-lens pipeline through the
  LayerNorm/parallel-block dispatch path, mirroring the existing tiny-OLMo
  test. Confirmed against the real tiny model that `transformer.h` /
  `transformer.ln_f` are the right attribute paths, `ln_f` is `LayerNorm`
  with a bias, and `lm_head.bias` is present (not `None`) — resolves the
  "needs confirming" note in §2's architecture table.
- **Unanticipated:** the local `.venv` (from an earlier `uv sync --extra
  gpu`) had a corrupted numpy install — `f2py2e.py` was listed in numpy's
  RECORD but missing from disk, breaking `scipy.stats` and hence
  `test_reproduction.py`. Root cause looked like a bad hardlink/copy during
  install. Fixed by clearing numpy from the uv cache and rebuilding the
  venv from scratch (`rm -rf .venv && uv sync --extra gpu`).
- **Verification:** `ruff check .` clean; full suite `43 passed`; `uv run
  jlens reproduce` still reproduces the OLMo headline numbers exactly
  (rho=+0.658 main, rho=0.143 logit-lens baseline, rho=0.644 shared split)
  — the refactor is behavior-preserving for the existing RMSNorm path.

**Stage 3 (partial — tokenizer-only re-derivation), 2026-09-16.**

Downloaded only the GPT-J tokenizer (`EleutherAI/gpt-j-6B`, a few MB, not
the 6B-parameter weights) and OLMo's tokenizer, both cached under
`/mounts/work/adrian/models` (`HF_HOME`), and reran
`splits.filter_identifying_objects` across all 28 relations under each.

- **Every relation's usable/not-usable verdict against the
  `n_icl+n_ref+1=17` threshold matches exactly between the two
  tokenizers** — 26/28 relations usable, with both `presidents_birth_year`
  and `presidents_election_year` failing under both (as in the paper).
  Per-relation filtered *counts* differ slightly by tokenizer BPE
  boundaries (e.g. `company_ceo`: 162 under OLMo vs. 165 under GPT-J;
  `person_father`: 233 vs. 237), but never enough to flip a relation across
  the threshold. §3.5's concern (GPT-J's BPE having different leading-space
  conventions that change which relations are usable) does not appear to
  bite in practice — the paper's relation set should transfer as-is.
- **Traced why the paper reports 25 relations, not the 26 with
  faithfulness files**, which §1/§3.5 had left unexplained:
  `person_band_lead_singer` has exactly 17 filtered samples — just enough
  for `measure-faithfulness`'s `n_icl+n_ref+1` requirement, but the
  jacobian-lens split (`n_icl=8, n_ref=8`) then has only 1 disjoint test
  subject left, below `measure_jacobian_lens.py`'s `len(subjects) < 2`
  floor, so it's silently dropped from the correlation. GPT-J's filtered
  count for this relation is also exactly 17, so the same relation will
  very likely drop out for GPT-J for the same reason — worth confirming
  once real per-subject data exists, not assuming.

**Critical bug found and fixed, 2026-09-16: `hidden_states[-1]` is post-norm,
not pre-norm, on the pinned transformers version.**

Added a direct test (`test_final_logits_matches_real_forward_pass`) comparing
`final_logits` against a real model's own `.logits` output with no JVP or
intervention involved — the simplest possible correctness check, and one
this codebase apparently never had. It failed on **both** the tiny-OLMo and
tiny-GPT-J models, off by ~2-5% relative to the logit scale.

Root cause, confirmed by reading `Olmo3Model.forward()`'s source: it ends
with `hidden_states = self.norm(hidden_states)` before returning, so on
`transformers>=5.14`, `output_hidden_states=True`'s **last** tuple entry is
already post-final-norm — not the raw block output the rest of this
codebase (both the original, pre-refactor code and this refactor) assumed.
Every place that reads `hidden_states[-1]` and later normalizes it again —
`HiddenStateIntervention.forward()` (hence every JVP, via `jvp()`), and
`measure_jacobian_lens.py`'s separate `reference_output` cache for the
affine/"lre" lens form — was silently double-normalizing. This is **not**
specific to GPT-J or to this refactor's architecture-dispatch work; it would
equally corrupt a fresh OLMo measurement run on the currently pinned
transformers version, including `measure_faithfulness.py`'s `base` term,
i.e. the actual LRE-faithfulness ground truth that is the paper's y-axis.
Whether it affected the *retained* snapshots depends on which transformers
version originally produced them — worth checking, but not resolved here.

**Fix:** added `forward_capturing_final_block()` to `jlens/model.py`, which
captures the last transformer block's own output via a forward hook
(mirroring how `HiddenStateIntervention` already hooks intermediate layers
for interventions) instead of trusting the `hidden_states` tuple's last
entry. `HiddenStateIntervention` now takes an optional `final_block`
(`resolve_blocks(model)[-1]`) and uses the hook-based capture when given one
— all four measurement scripts now pass it; `TinyModel`'s minimal-mock test
still works unchanged via the `final_block=None` fallback.

**Verification:** `test_final_logits_matches_real_forward_pass` now passes
exactly (not just within loose tolerance) for both tiny models. Added
`test_hidden_states_last_entry_is_post_norm_not_raw` as an explicit
regression guard/doc of *why* the hook-based workaround exists, so it isn't
silently deleted as dead code if a future transformers version reverts the
behavior. Full suite: 46 passed.


