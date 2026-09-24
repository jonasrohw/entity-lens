"""Shared hidden-state interventions for double-backward Jacobian products.

This module is imported only by GPU commands. Layer numbers refer to the
Transformers hidden-state tuple: state 17 is the output of transformer block 16.

Supports three block/final-norm layouts: Llama-style (``model.model.layers``,
RMSNorm at ``model.model.norm``, e.g. OLMo, Gemma4ForCausalLM), GPT-2-style
(``model.transformer.h``, LayerNorm at ``model.transformer.ln_f``, e.g.
GPT-J), and the nested multimodal-wrapper style
(``model.model.language_model.layers`` / ``.norm``, e.g.
``Gemma4UnifiedForConditionalGeneration`` — the class ``AutoModelForCausalLM``
actually resolves a plain ``google/gemma-4-*`` checkpoint id to, since those
repos ship the unified text+vision+audio config even for text-only use).
"""

import torch


def resolve_blocks(model):
    """Return the ordered list of transformer blocks, across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            return language_model.layers
    raise ValueError(f"Unrecognized transformer block layout: {type(model).__name__}")


def resolve_final_norm(model):
    """Return the model's pre-unembedding normalization module, across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "norm"):
            return language_model.norm
    raise ValueError(f"Unrecognized final-norm layout: {type(model).__name__}")


def final_logits(hidden, norm, unembedding, lm_bias=None, softcap=None):
    """Apply a final-norm module, then the unembedding, over the last dimension.

    ``hidden`` may be a single vector or a batch; the supplied module controls
    normalization. Call the model's own module rather than inferring its
    formula from the presence of a bias: Gemma2 RMSNorm uses ``1 + weight``,
    and a bias-free LayerNorm must still subtract the mean.

    ``softcap`` applies Gemma-style final-logit softcapping,
    ``cap * tanh(logits / cap)``, after the unembedding matmul and any
    ``lm_bias`` — matching ``model.config.final_logit_softcapping`` and the
    order used by ``Gemma4ForCausalLM.forward``. ``None`` (the default, and
    the value on OLMo/GPT-J configs) skips this step entirely.
    """
    normed = norm(hidden)
    logits = normed.float() @ unembedding.T
    if lm_bias is not None:
        logits = logits + lm_bias
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def forward_capturing_final_block(
    model,
    final_block,
    input_ids,
    output_hidden_states=False,
    *,
    last_position_only=True,
):
    """Run one forward pass, returning ``(output, raw final-block hidden state)``.

    Captures ``final_block``'s own output at the last position via a forward
    hook, rather than trusting ``output.hidden_states[-1]``: on
    transformers>=5.14, that tuple entry is the POST-final-norm state for
    some models (confirmed for Olmo3 and GPT-J's ``forward()``), not the raw
    block output. Downstream code (``final_logits``) expects a pre-norm
    vector and would otherwise normalize it a second time. Does not detach
    the captured tensor, so gradients still flow through it for the JVP.
    Set ``last_position_only=False`` to capture every token, including the
    subject position when the final block is also the intervention layer.
    """
    captured = {}

    def _capture(module, inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        captured["value"] = state[0, -1, :] if last_position_only else state

    handle = final_block.register_forward_hook(_capture)
    try:
        output = model(input_ids=input_ids, output_hidden_states=output_hidden_states)
    finally:
        handle.remove()
    return output, captured["value"]


def forward_with_raw_hidden_states(model, input_ids):
    """Return HF output plus states at consistent, pre-final-norm boundaries.

    State 0 is the embedding; state k is the output of block k-1, including
    the last block. HF's own final hidden-state entry is post-norm on GPT-J
    and OLMo. Injecting that entry at a block-output hook changes even an
    otherwise unmodified reference. Keep the original HF output intact and
    return a separate tuple with the raw final block output substituted.
    """
    output, raw_final = forward_capturing_final_block(
        model,
        resolve_blocks(model)[-1],
        input_ids,
        output_hidden_states=True,
        last_position_only=False,
    )
    return output, (*output.hidden_states[:-1], raw_final)


class HiddenStateIntervention:
    """Replace one subject state and differentiate the final prompt state.

    Register ``hook`` on a transformer block only after caching unmodified
    states. Remove the returned hook handle before any unmodified forward pass.
    """

    def __init__(self, model, final_block=None):
        """``final_block`` should be ``resolve_blocks(model)[-1]`` for a real
        transformer model, so the final state is captured via a hook rather
        than trusted from ``hidden_states[-1]`` (see
        ``forward_capturing_final_block``). Left as ``None`` only for tests
        against models too minimal to have a resolvable block list.
        """
        self.model = model
        self.hidden = None
        self.position = None
        self._final_block = final_block

    def hook(self, module, inputs, output):
        """Replace a single hidden state without modifying the original tensor."""
        if self.hidden is None or self.position is None:
            raise RuntimeError(
                "Set the intervention state before a hooked forward pass"
            )
        states = output[0] if isinstance(output, tuple) else output
        states = states.clone()
        states[0, self.position, :] = self.hidden
        return (states, *output[1:]) if isinstance(output, tuple) else states

    def forward(self, hidden, input_ids, position):
        """Map an intervened subject state to the raw final-block state."""
        self.hidden = hidden
        self.position = position
        if self._final_block is None:
            return self.model(
                input_ids=input_ids, output_hidden_states=True
            ).hidden_states[-1][0, -1, :]
        _, final_state = forward_capturing_final_block(
            self.model, self._final_block, input_ids
        )
        return final_state

    def jvp(self, reference, direction, input_ids, position):
        """Compute J(reference) @ direction with the original double backward."""
        hidden = reference.clone().requires_grad_(True)
        output = self.forward(hidden, input_ids, position)
        cotangent = torch.zeros_like(output, requires_grad=True)
        (gradient,) = torch.autograd.grad(
            output, hidden, grad_outputs=cotangent, create_graph=True
        )
        (product,) = torch.autograd.grad(gradient, cotangent, grad_outputs=direction)
        return product.detach()
