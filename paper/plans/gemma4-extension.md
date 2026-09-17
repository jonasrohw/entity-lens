<!-- shared with collaborators as paper/plans/gemma4-extension.md -->

# Plan: extend the Jacobian-lens measurements to Gemma 4 (12B dense, fallback E4B)

**Status: Stage 1 (code-only) and the Stage-2 config-only check are done.**
No GPU work (Stage 3+) has been run yet. See §6.

## 1. Motivation

The paper currently measures `allenai/Olmo-3-1025-7B` (Llama-style baseline) and
has a completed extension to `EleutherAI/gpt-j-6B` (LayerNorm, parallel
attention+MLP, partial rotary — see `paper/plans/gpt-j-extension.md`). Gemma 4
(Google, released 2026-04-03) is proposed as a third model:

- It is new and currently prominent — a flagship open-weight release from a
  major lab, not a niche variant — so a positive result carries more weight
  as a generality claim than another same-family swap.
- It is architecturally distinct from both existing models along axes neither
  has exercised: hybrid local/global sliding-window attention (two different
  RoPE frequencies depending on layer type), QK-norm, and — the detail that
  matters most for correctness — a **final-logit softcap**, which neither
  OLMo nor GPT-J applies. Getting the softcap wrong would silently corrupt
  every reported number the same way the GPT-J work's `hidden_states[-1]`
  post-norm bug did (see that plan's progress log).
- Unlike GPT-J, Gemma 4 was not one of Hernandez et al. (2024)'s validated
  models, so there is no external "same model as the original LRE paper" or
  "reuse their reported best layer" argument available here — layer choices
  and relation splits have to be derived from scratch (§3.4, §3.5).
- Gemma 4 also ships MoE (`26B-A4B`) and multimodal variants. Both are
  explicitly out of scope: MoE routing raises the same discontinuity and
  full-expert-weight-residency concerns that ruled out Qwen3.5's MoE
  checkpoints for this project (discussed in conversation, not written up
  separately), and multimodal towers add irrelevant memory/complexity for a
  text-only measurement pipeline.

**Staged plan, per the user's request:** attempt the dense **12B** checkpoint
first; if it proves infeasible (most likely on resource grounds — see §4),
fall back to the smaller elastic **E4B** checkpoint. The two are *not*
assumed to share every architectural detail (§2 flags where this matters).

## 2. Confirmed architecture facts

Confirmed directly from **(a)** `google/gemma-4-12B`'s own `config.json`
(downloaded, not the `-it` checkpoint this plan originally guessed from) and
**(b)** reading `transformers==5.17.0`'s actual
`models/gemma4/modeling_gemma4.py` and `models/gemma4_unified/modeling_gemma4_unified.py`
source — both installed in this project's `.venv`, so every row below is
ground truth, not inference from documentation.

| | OLMo-3-1025-7B / GPT-J-6B (current) | Gemma 4 12B |
|---|---|---|
| Depth | 32 / 28 layers | 48 layers, hidden size 3840 |
| Final norm | RMSNorm, weight-only (OLMo) or LayerNorm w/ weight+bias (GPT-J) | RMSNorm, weight-only (`rms_norm_eps=1e-6`) — same *branch* jlens already supported |
| **Final-logit softcap** | **None** | **`final_logit_softcapping=30.0`**. Confirmed exact order from source (`modeling_gemma4.py:1863-1867` / `modeling_gemma4_unified.py`): `logits = lm_head(hidden); logits = logits/cap; logits = tanh(logits); logits = logits*cap` — no bias involved (`lm_head` has none). Implemented and tested (§3.1/§6). |
| **Model class** | plain causal-LM classes for both | **`google/gemma-4-12B`'s own `config.json` shows `architectures: Gemma4UnifiedForConditionalGeneration`, `model_type: gemma4_unified` — there is no separate plain-text checkpoint.** `AutoModelForCausalLM.from_pretrained` still resolves it correctly (`Gemma4UnifiedForConditionalGeneration` is registered in `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`), but the text decoder is nested two levels down: `model.model` is `Gemma4UnifiedModel` (holds `vision_tower`/`audio_tower`/`language_model`), and `model.model.language_model.layers` / `.norm` is where the actual blocks and final norm live. `lm_head` stays at the top level. Required a third `resolve_blocks`/`resolve_final_norm` branch (§3.2, done). |
| **Config nesting** | flat config for both | `model.config` is the composite `Gemma4UnifiedConfig` (`text_config`/`vision_config`/`audio_config`); reading `final_logit_softcapping` (or any text-model field) must go through `model.config.get_text_config()`, exactly as the real `forward()` does (`modeling_gemma4_unified.py:1334`), not `model.config` directly — the latter would silently resolve to `None` for this checkpoint even though softcapping is really active. `get_text_config()` is a no-op identity on OLMo's/GPT-J's plain configs, confirmed safe to call unconditionally. |
| Attention | full rotary (OLMo) / partial rotary (GPT-J), single RoPE base | hybrid: sliding-window local attention (window=1024, `rope_theta=10000`, `rope_type=default`) interleaved with periodic full attention (6 of 48 layers, `rope_theta=1000000`, `rope_type=proportional`, `partial_rotary_factor=0.25`) — internal to each block's own attention module, transparent to jlens's block-level hooking (confirmed by the tiny-model tests in §6; no `resolve_blocks`/`final_logits` change needed for this row) |
| QK-norm (and V-norm) | absent in both | confirmed in source (`Gemma4TextAttention.__init__`: separate `q_norm`/`k_norm`/`v_norm` `Gemma4RMSNorm` modules per head) — internal to attention, same "transparent" reasoning as above |
| Per-layer embeddings (PLE) | absent in both | confirmed real and non-optional in the `Gemma4TextConfig` schema (`hidden_size_per_layer_input`, `vocab_size_per_layer_input`, `per_layer_config`). Computed once per forward pass from `input_ids` (not from the evolving hidden-state stream) and threaded into each decoder layer as a side-channel argument — confirmed from source and by the passing tiny-model tests that it needs no special-casing in `resolve_blocks`/`final_logits`. |
| Cross-layer KV-cache sharing | absent in both | confirmed real in source (`Gemma4TextAttention`: a `shared_kv_states` dict populated by earlier layers, read by later ones, within a single forward call) and gated by `num_kv_shared_layers`. **Real `google/gemma-4-12B` config confirms `num_kv_shared_layers: 0`** — inactive for this checkpoint, so the leak scenario in §3.4 does not currently apply, but must be rechecked if E4B (§4) turns out to use it. |
| `lm_head` | tied (OLMo) / untied with bias (GPT-J) | `tie_word_embeddings=true`, `bias=False` — simplest case, same as OLMo |
| Tokenizer | OLMo BPE / GPT-2 BPE | Gemma SentencePiece, `vocab_size=262144` — far larger than either existing tokenizer |

Source: `google/gemma-4-12B`'s `config.json` (downloaded via
`huggingface_hub.hf_hub_download`, config only, no weights — see §6), and
direct reads of `transformers==5.17.0`'s `modeling_gemma4.py` /
`modeling_gemma4_unified.py` / `configuration_gemma4*.py` in this project's
`.venv`.

## 3. Required code changes

### 3.1 Final-logit softcapping (the change that matters most)

[`jlens/model.py`'s `final_logits()`](../../code/jlens/model.py#L32-L57)
applies norm → unembedding matmul → optional `lm_bias`, with no softcap step.
For Gemma 4 this must become `logits = cap * tanh(logits / cap)` *after* the
unembedding matmul (and after any `lm_bias`, matching whatever order the real
`forward()` uses — confirm from source, don't guess), dispatched on whether
the model config exposes `final_logit_softcapping` (`None`/absent for
OLMo and GPT-J; `30.0` for Gemma 4).

**Done.** [`final_logits()`](../../code/jlens/model.py#L32-L61) takes an
optional `softcap` argument applied after the unembedding matmul and any
`lm_bias`: `logits = softcap * tanh(logits / softcap)`, matching the exact
order confirmed from `Gemma4ForCausalLM.forward`/
`Gemma4UnifiedForConditionalGeneration.forward` source. All four call sites
now extract `softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)`
and pass it through (`measure_jacobian_lens.py`, `measure_faithfulness.py`,
`measure_subject_ranks.py`, `probe_names.py`) — `get_text_config()` rather
than a direct attribute read, because the real checkpoint's config is
composite (§2, §3.2) and a direct read would silently resolve to `None`.
Validated against `test_final_logits_matches_real_forward_pass` on a tiny
random Gemma4 config, matching to ~1e-8 (§6) — exactly the kind of check the
GPT-J work's "`hidden_states[-1]` is post-norm, not pre-norm" fix showed is
necessary: this bug class is silent and non-crashing without it.

### 3.2 Resolve the model class and attribute path

**Done, and the original assumption in this section was wrong: there is no
plain text-only Gemma 4 checkpoint to load.** `google/gemma-4-12B`'s own
`config.json` (§2) confirms `architectures: Gemma4UnifiedForConditionalGeneration`
— `AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)` still resolves
correctly (that class is registered under `AutoModelForCausalLM`), but the
text decoder is nested as `model.model.language_model.layers` /
`model.model.language_model.norm`, not `model.model.layers` /
`model.model.norm` directly.
[`resolve_blocks`](../../code/jlens/model.py#L15-L23) and
[`resolve_final_norm`](../../code/jlens/model.py#L26-L34) now have a third
branch for this nesting, checked after the existing Llama-style and
GPT-2-style branches, added the same way the GPT-J branch was: as a shared
dispatch helper rather than a Gemma-specific path hardcoded at each of the
four measurement call sites. Validated against a tiny
`Gemma4UnifiedForConditionalGeneration` built with `vision_config`/
`audio_config` left `None` (`_tiny_gemma4_unified_model` in
`tests/test_model.py`) — full pipeline and `final_logits` both pass (§6).
`query_pre_attn_scalar` does not exist in the real schema (confirmed absent
from `Gemma4TextConfig`'s field list directly); `q_norm`/`k_norm`/`v_norm`
are confirmed present as real per-head `Gemma4RMSNorm` modules inside
`Gemma4TextAttention` (§2).

### 3.3 `MODEL_ID`

Already parameterized via `JLENS_MODEL`/`--model` from the GPT-J work
([`config.py:12`](../../code/jlens/config.py#L12)). Point it at
`google/gemma-4-12B` (base, not `-it`) — confirmed to be the right choice:
its `config.json` was downloaded directly (§2, §6) and is what §2's table
now describes.

### 3.4 Layer index defaults and the KV-sharing caveat

[`LENS_LAYER=17` and `LRE_LAYERS="5,11,14,17,20,23"`](../../code/jlens/config.py#L16-L20)
are tuned to OLMo's 32-layer depth and don't transfer to Gemma 4's 48 layers.
Unlike GPT-J (where Hernandez et al.'s own reported best layers could be
reused, given the "same model as the original paper" framing), there is no
equivalent external reference for Gemma 4 — new layer choices need deriving
(e.g. proportionally rescaled from OLMo's, then empirically validated), not
borrowed from either existing model.

Separately: if a later config check (§2) turns up `num_kv_shared_layers > 0`
for the checkpoint actually used (12B or E4B), avoid placing `LENS_LAYER` or
any `LRE_LAYERS` entry such that a downstream "shared" layer's KV source sits
*upstream* of the intervention point. In that configuration, the shared
layer would attend using K/V computed from the un-intervened hidden state
(computed before the intervention hook fired) combined with a query derived
from the intervened one — a correctness leak specific to cross-layer KV
sharing that the existing hook-based block design (correct for GPT-J's
parallel-block topology, which doesn't share state across blocks) has never
had to account for. Confirmed unnecessary for 12B given `num_kv_shared_layers:
0` in the fetched config, but must be rechecked if that changes or if E4B is
used instead.

### 3.5 Tokenizer boundary effects

**Done (§6).** Re-ran `splits.filter_identifying_objects` across all 28
relations under Gemma 4's tokenizer (`google/gemma-4-12B`, GemmaTokenizer /
SentencePiece), alongside OLMo's and GPT-J's for comparison, following the
same method used for GPT-J. Result: **every relation's usable/not-usable
verdict against the `n_icl+n_ref+1=17` threshold matches exactly across all
three tokenizers** — 26/28 usable, with both `presidents_birth_year` and
`presidents_election_year` failing under all three (as in the paper). The
existing relation set transfers as-is; §3.5's concern (different leading-
space conventions flipping a relation across the threshold) does not bite
in practice, same conclusion as for GPT-J.

One new, fully-explained wrinkle: **both year relations score exactly 0
filtered samples under Gemma 4** (vs. 10/8 for OLMo, 1/13 for GPT-J), even
with the trailing-space workaround applied. Root cause, confirmed by
tokenizing the actual year strings: Gemma's SentencePiece tokenizer encodes
multi-digit numbers **digit-by-digit** (`1743` → `1`, `7`, `4`, `3`), unlike
GPT-2/GPT-J-style BPE which merges 2-3 digits per token. Since nearly every
year in this dataset starts with `1`, every object's first token is
identical regardless of the trailing-space workaround — nothing can survive
the filter. This changes *why* these two relations are excluded (a
tokenizer design choice, not a borderline sample-count issue) but not
*whether* they are — both were already below the usable threshold under
every model. `person_band_lead_singer` again lands at exactly 17 filtered
samples under Gemma 4, matching OLMo and GPT-J — the same likely later
drop-out from `measure_jacobian_lens.py`'s disjoint-test-subject floor noted
in the GPT-J plan applies here too, to be confirmed once real per-subject
data exists rather than assumed.

### 3.6 Attention implementation

**Confirmed at tiny-model scale (§6): `attn_implementation="eager"` works**
and double-backward runs cleanly through both the sliding-window and
full-attention layer types, on both `Gemma4ForCausalLM` and
`Gemma4UnifiedForConditionalGeneration`. Not yet confirmed at real 12B scale
or on GPU (only CPU, tiny random weights so far) — the eager-attention
mechanics themselves are the same regardless of scale, but a real-scale,
real-GPU run is still the genuine test per §5 Stage 3.

### 3.7 Tests

**Done.** Added to
[`tests/test_model.py`](../../code/tests/test_model.py):
`_tiny_gemma4_model` (plain `Gemma4ForCausalLM`, exercises the softcap
dispatch, sliding/full attention mix, QK/V-norm, and PLE at minimal size) and
`_tiny_gemma4_unified_model` (`Gemma4UnifiedForConditionalGeneration` with
`vision_config`/`audio_config` left `None`, exercising the nested
`language_model` attribute path and the `get_text_config()`-based softcap
lookup from §3.2). Both feed `test_measurements_agree_on_tiny_*` (full
faithfulness → subject-ranks → jacobian-lens pipeline) and are added to
`test_final_logits_matches_real_forward_pass`'s parametrization. Not yet
added: the KV-sharing-leak regression case from the original plan — deferred
since the real 12B config has `num_kv_shared_layers: 0` (§2), so there is
nothing to exercise yet; revisit if E4B (§4) turns out to use it.

## 4. Resource note

12B params in float32 (required for the Jacobian double-backward, matching
the current OLMo/GPT-J measurement scripts) is ≈48GB for weights alone,
before JVP activations — noticeably larger than the current OLMo-7B/GPT-J-6B
budget (~24-28GB). Confirm against the target GPU (likely needs an
80GB-class card) before scheduling the full sweep.

**Fallback:** if 12B doesn't fit, or another blocker specific to it appears,
switch `MODEL_ID` to `google/gemma-4-E4B` (~8B raw parameters underlying the
"4.5B effective" MatFormer/PLE-style checkpoint, ≈32GB fp32 — closer to the
existing GPT-J budget). Treat this as a distinct model requiring its own
pass through §2's verification, not a drop-in swap: its `num_kv_shared_layers`
value, and any structural effect of the per-layer-embedding (PLE) /
elastic-parameterization scheme on the block/norm attribute paths, need
confirming independently — nothing about the 12B investigation guarantees
they're the same.

## 5. Staging

1. **Code only, no GPU — done (§6).** Implemented the final-logit softcap
   dispatch (3.1) and the third block/norm attribute-path branch (3.2)
   against tiny random Gemma4 configs; added the tiny-model tests (3.7). Full
   test suite (61 tests) and `ruff check` both pass.
2. **Config-only check — done (§6).** Downloaded `google/gemma-4-12B`'s
   `config.json` directly (no weights). This is what revealed 3.2's
   correction: the base repo is the unified multimodal architecture, not a
   plain text checkpoint. Confirmed `num_kv_shared_layers: 0`, the
   `layer_types` pattern (6 full-attention layers of 48, evenly spaced), and
   `final_logit_softcapping: 30.0`.
3. **Small GPU check — not started.** Requires downloading the full 12B
   checkpoint (§4: ≈48GB fp32) and an eligible GPU; confirm `final_logits`
   (softcap included) matches a real forward pass exactly on a few prompts —
   non-negotiable given §3.1's stakes — then run
   `measure-jacobian-lens`/`measure-faithfulness` on one or two relations.
4. **Re-derive splits — done (§6).** Confirmed Gemma 4's identifying-first-token
   relation set matches OLMo's/GPT-J's exactly (26/28 usable, same two
   excluded) and root-caused a new digit-tokenization wrinkle for the two
   excluded year relations (3.5) — this step did not require the 12B weights
   (tokenizer-only, like GPT-J's Stage 3), so it was done ahead of step 3
   rather than after it.
5. **Full pipeline:** run the same `measurement_plan()` stages as OLMo/GPT-J
   into a separate `--root`/`--run-dir`, per the existing `jlens rerun`
   mechanism.
6. **Fallback branch:** if step 3 shows 12B doesn't fit the target GPU (or
   another blocker appears), repeat step 2's config-only check specifically
   for `google/gemma-4-E4B` before assuming anything from the 12B
   investigation carries over, then resume from step 3 with E4B.
7. **Analysis:** `jlens reproduce` against the new root, compare headline
   correlation and layer/beta selections against the OLMo and GPT-J results.

## 6. Progress log

**Stage 1 (code-only) and partial Stage 2 (config-only) complete, 2026-09-17.**

- Confirmed `transformers==5.17.0` (already installed in `.venv`) ships full
  Gemma 4 support: `Gemma4Config`/`Gemma4TextConfig`/`Gemma4ForCausalLM` (plain
  text) and `Gemma4UnifiedConfig`/`Gemma4UnifiedForConditionalGeneration`
  (multimodal). Read both `modeling_gemma4.py` and
  `modeling_gemma4_unified.py` directly rather than relying on docs/blog
  posts, since this model family postdates this session's training data.
- Added `softcap` to
  [`final_logits()`](../../code/jlens/model.py#L32-L61): `logits = softcap *
  tanh(logits / softcap)` after the unembedding matmul and any `lm_bias`,
  matching the exact order in the real `forward()` source. `None` (OLMo,
  GPT-J) skips it entirely.
- Updated all four measurement scripts (`measure_jacobian_lens.py`,
  `measure_faithfulness.py`, `measure_subject_ranks.py`, `probe_names.py`) to
  extract `softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)`
  and pass it to `final_logits(...)`. `get_text_config()` rather than a
  direct attribute read, because of the unified-config finding below.
- Downloaded `google/gemma-4-12B`'s `config.json` directly via
  `huggingface_hub.hf_hub_download` (no weights) to verify assumptions before
  committing to a full download, per the staging plan.
  **This overturned §3.2's original assumption:** the base repo's
  `architectures` field is `Gemma4UnifiedForConditionalGeneration`, not a
  plain text-only class — there is no separate non-multimodal Gemma 4 12B
  checkpoint. `AutoModelForCausalLM.from_pretrained` still resolves it
  correctly (confirmed via `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`), but the text
  decoder lives at `model.model.language_model.layers` /
  `model.model.language_model.norm`, two levels deeper than the Llama-style
  branch `resolve_blocks`/`resolve_final_norm` already had.
- Added a third branch to
  [`resolve_blocks`](../../code/jlens/model.py#L15-L23) and
  [`resolve_final_norm`](../../code/jlens/model.py#L26-L34) for this nested
  layout, checked after the existing two branches.
- Added `_tiny_gemma4_model` (plain `Gemma4ForCausalLM`) and
  `_tiny_gemma4_unified_model` (`Gemma4UnifiedForConditionalGeneration` with
  `vision_config=None`, `audio_config=None` — keeps the tiny model
  text-only) to `tests/test_model.py`, each exercising real Gemma 4 features
  at minimal size rather than stubbing them out: hybrid sliding/full
  attention, QK/V-norm, per-layer embeddings (PLE), and (for the unified
  variant) the nested attribute path and composite-config softcap lookup.
  Both feed `test_measurements_agree_on_tiny_*` (the full
  faithfulness → subject-ranks → jacobian-lens pipeline) and
  `test_final_logits_matches_real_forward_pass`.
- **Verification:** confirmed by direct computation (not just passing tests)
  that Gemma 4 reproduces the same `hidden_states[-1]`-is-post-norm behavior
  the GPT-J work found and fixed for OLMo/GPT-J — `forward_capturing_final_block`
  already handles this architecture-agnostically, so no new fix was needed,
  but it was checked rather than assumed. Also confirmed the softcap
  implementation matches the real forward pass to ~1e-8 (vs. ~1e-6 error if
  omitted, at tiny-model logit magnitudes — the gap grows with logit scale,
  so this understates real-model impact). Full test suite: 61 passed;
  `ruff check .` clean.
- **Not yet done:** the GPU-dependent parts of Stage 3 in §5 — no full-weight
  download, no real forward-pass validation against the actual 12B
  checkpoint, no layer-index re-derivation (§3.4). The KV-sharing-leak test
  flagged in the original §3.7 was skipped since `num_kv_shared_layers: 0`
  for the confirmed 12B config leaves nothing to exercise yet.
- Also fixed `scripts/verify_real_model.py` (the GPU-side real-forward-pass
  check the GPT-J/OLMo launch scripts both gate the full pipeline on): it
  computed `final_logits(...)` without the softcap argument, which would
  have made it spuriously fail (or worse, mask the real check it exists to
  perform) the first time it ran against a real Gemma 4 checkpoint. No
  behavior change for OLMo/GPT-J (softcap is `None` there). Added
  `scripts/run_gemma4_extension.sh`, mirroring the OLMo/GPT-J launch
  scripts, defaulting to `google/gemma-4-12B` with the rescaled layer
  defaults from §3.4 and a documented `JLENS_MODEL=google/gemma-4-E4B`
  fallback.

**Stage 4 (tokenizer-only relation-split re-derivation) complete, 2026-09-17.**

Downloaded only the Gemma 4 tokenizer (`google/gemma-4-12B`, a few MB, not
the 12B-parameter weights) and re-ran `splits.filter_identifying_objects`
across all 28 relations under it, alongside OLMo's and GPT-J's tokenizers
for a three-way comparison (script not committed — a one-off analysis, not
part of the pipeline).

- **Every relation's usable/not-usable verdict against the
  `n_icl+n_ref+1=17` threshold matches exactly across all three
  tokenizers** — 26/28 relations usable, with both `presidents_birth_year`
  and `presidents_election_year` failing under all three (as in the paper
  and as found for GPT-J). Per-relation filtered *counts* differ more than
  the OLMo-vs-GPT-J gap did (e.g. `person_father`: 233 OLMo / 237 GPT-J /
  315 Gemma 4), plausibly Gemma's much larger vocabulary (262k vs. ~32k/50k)
  producing fewer first-token collisions — but never enough to flip a
  relation across the threshold. §3.5's concern does not bite in practice,
  same conclusion as GPT-J's.
- **New, fully root-caused wrinkle:** both year relations score exactly 0
  filtered samples under Gemma 4 (vs. 10/8 for OLMo, 1/13 for GPT-J), even
  with the trailing-space workaround. Confirmed by direct tokenization:
  Gemma's SentencePiece tokenizer encodes multi-digit numbers digit-by-digit
  (`1743` → `1`,`7`,`4`,`3`), unlike GPT-2/GPT-J-style BPE's 2-3-digit
  chunks; since nearly every year in this dataset starts with `1`, every
  object's first token collides regardless of the trailing-space fix. This
  changes *why* the two year relations are excluded (a tokenizer design
  choice, not a borderline sample count) but not *whether* — both were
  already below threshold under every model.
- `person_band_lead_singer` again lands at exactly 17 filtered samples
  (matching OLMo and GPT-J) — the same likely later drop-out from
  `measure_jacobian_lens.py`'s disjoint-test-subject floor (noted in the
  GPT-J plan) plausibly applies here too; to be confirmed once real
  per-subject data exists, not assumed.
