# Fixed: the final norm was being applied twice

**Status: fixed 2026-09-16.** This note documents a real correctness bug
found and fixed.

## The bug

Several functions here need "the raw hidden state coming out of the last
transformer block, before the model's final normalization." They obtained it
as `model(...).hidden_states[-1]` — the last entry of `output_hidden_states=True`'s
tuple — and then applied the final norm (RMSNorm or LayerNorm, depending on
architecture) to it themselves before computing logits or ranks.

On the Transformers version this project depends on, `hidden_states[-1]`
is **already** the post-final-norm state. Normalizing it again is a real,
silent double-normalization, not a rounding-level issue: it changes the
resulting logits, and hence which token wins an argmax or where a target
token ranks.

This affected every place that used
`HiddenStateIntervention.forward()`/`.jvp()`'s output as if it were pre-norm:

- The Jacobian lens's "diff" score (`rank_diff` — the value behind the
  paper's headline correlation) and "lre"/affine score (`rank_lre`), in
  `measure_jacobian_lens.py`.
- `base`, the LRE ground-truth prediction, in `measure_faithfulness.py` and
  `measure_subject_ranks.py` — i.e. the paper's y-axis (`best_faith`), not
  just the lens x-axis.
- `probe_names.py`'s name-association lens.

## How it was found

While adding architecture support for GPT-J, a direct correctness test was
added that had never existed before: take a hidden state from a *real*
forward pass, run it through this codebase's norm+unembed reconstruction,
and compare directly against that same forward pass's own `.logits` output —
no JVP, no intervention, nothing else involved
(`test_final_logits_matches_real_forward_pass` in
[`../tests/test_model.py`](../tests/test_model.py)). Every prior test only
checked *internal* consistency between `measure_faithfulness`,
`measure_subject_ranks`, and `measure_jacobian_lens` (do they agree with
each other), never against model ground truth — so this had gone
undetected. The new test failed, on both a tiny synthetic OLMo config and a
tiny synthetic GPT-J config.

Isolating the cause: reapplying the model's own norm module to
`hidden_states[-1]` matched a from-scratch reimplementation exactly (ruling
out an arithmetic bug), but applying **no** norm at all —
`hidden_states[-1] @ unembedding.T` — matched the real logits to float32
precision. That proves `hidden_states[-1]` was already fully normalized.

## Evidence this is real, current Transformers behavior

- **Source, GPT-J** (classic hand-written loop,
  `transformers/models/gptj/modeling_gptj.py`): `self.ln_f` is applied, then
  the result is appended as the last `hidden_states` tuple entry (comment:
  `# Add last hidden state`). This exact pattern is present as far back as
  `transformers==4.0.0` (Dec 2020) for GPT-2, GPT-J's direct ancestor, and
  in Llama's original addition (`transformers==4.28.0`, March 2023) — this
  is a long-standing library convention, not new or version-specific.
- **Source, OLMo3** (newer decorator-based mechanism,
  `transformers/utils/output_capturing.py`): `capture_outputs(...,
  tie_last_hidden_states=True)`'s docstring states plainly: "Whether to
  overwrite `out.hidden_states[-1]` with the `out.last_hidden_state`. **This
  is true for all language models** ...".
- **Official docs**
  ([huggingface.co/docs/transformers/main_classes/output](https://huggingface.co/docs/transformers/main_classes/output)):
  "You may expect `outputs.hidden_states[-1]` to match `outputs.last_hidden_state`
  exactly. ... Some models apply normalization ... [as an exception]" — i.e.
  matching (post-norm) is the documented default; not matching is the
  unusual case, and neither OLMo nor GPT-J is among the named exceptions.
- **`huggingface/transformers` PR #48087** ("Allow untying
  `hidden_states[-1]` from `last_hidden_state` via the model config"),
  opened by an HF maintainer, still in review as of this writing: it exists
  specifically because there is currently *no* way to get the pre-norm
  final state from `hidden_states[-1]` on these architectures.

## The fix

Added `forward_capturing_final_block()` in [`model.py`](model.py): it
registers a temporary forward hook directly on the actual last transformer
block module (found via `resolve_blocks(model)[-1]`), capturing that
block's raw output before anything downstream touches it — independent of
whatever convention `output_hidden_states` happens to follow.
`HiddenStateIntervention` now takes an optional `final_block` argument and
uses this hook-based capture when one is given, instead of trusting
`hidden_states[-1]`. All measurement scripts (`measure_faithfulness.py`,
`measure_jacobian_lens.py`, `measure_subject_ranks.py`, `probe_names.py`)
now construct it with `final_block=resolve_blocks(model)[-1]`.

The same refactor also introduced `resolve_blocks()`, `resolve_final_norm()`,
and `final_logits()` (RMSNorm/LayerNorm dispatch, with or without a bias
term) so the four call sites stop each reimplementing this arithmetic
inline — see `paper/plans/gpt-j-extension.md` §3.1 for that part of the
story.

## Verifying it

- `test_final_logits_matches_real_forward_pass` — the test that caught
  this — now passes at essentially float32 machine precision on both a
  tiny OLMo and tiny GPT-J config, not just "close."
- `test_hidden_states_last_entry_is_post_norm_not_raw` — a documented
  regression guard proving the bug is real (`hidden_states[-1]` measurably
  differs from the raw block output), so the hook-based workaround isn't
  mistaken for dead code and deleted later.
- `scripts/verify_real_model.py` (in `code/scripts/`) runs the same check
  against a real checkpoint (not a tiny synthetic one) on a GPU host, and
  reports the actual magnitude of the discrepancy before any full
  measurement run is allowed to proceed.

