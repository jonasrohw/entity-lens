"""Shared hidden-state interventions for double-backward Jacobian products.

This module is imported only by GPU commands. Layer numbers refer to the
Transformers hidden-state tuple: state 17 is the output of transformer block 16.

Supports two block/final-norm layouts: Llama-style (``model.model.layers``,
RMSNorm at ``model.model.norm``, e.g. OLMo) and GPT-2-style (``model.transformer.h``,
LayerNorm at ``model.transformer.ln_f``, e.g. GPT-J).
"""

import torch


def resolve_blocks(model):
    """Return the ordered list of transformer blocks, across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unrecognized transformer block layout: {type(model).__name__}")


def resolve_final_norm(model):
    """Return the model's pre-unembedding normalization module, across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise ValueError(f"Unrecognized final-norm layout: {type(model).__name__}")


def final_logits(hidden, norm, unembedding, lm_bias=None):
    """Apply a final-norm module, then the unembedding, over the last dimension.

    ``hidden`` may be a single vector or a batch; only its last axis is
    normalized. Dispatches on the norm module's shape: RMSNorm (e.g.
    Llama/OLMo) has a weight only and no mean-centering; LayerNorm (e.g.
    GPT-2/GPT-J) mean-centers and has both a weight and a bias.
    """
    weight = norm.weight.detach()
    bias = getattr(norm, "bias", None)
    if bias is not None:
        eps = float(norm.eps)
        mean = hidden.mean(dim=-1, keepdim=True)
        var = hidden.var(dim=-1, unbiased=False, keepdim=True)
        normed = (hidden - mean) / torch.sqrt(var + eps) * weight + bias.detach()
    else:
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
        normed = (
            hidden
            * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight
        )
    logits = normed.float() @ unembedding.T
    if lm_bias is not None:
        logits = logits + lm_bias
    return logits


def forward_capturing_final_block(
    model, final_block, input_ids, output_hidden_states=False
):
    """Run one forward pass, returning ``(output, raw final-block hidden state)``.

    Captures ``final_block``'s own output at the last position via a forward
    hook, rather than trusting ``output.hidden_states[-1]``: on
    transformers>=5.14, that tuple entry is the POST-final-norm state for
    some models (confirmed for Olmo3 and GPT-J's ``forward()``), not the raw
    block output. Downstream code (``final_logits``) expects a pre-norm
    vector and would otherwise normalize it a second time. Does not detach
    the captured tensor, so gradients still flow through it for the JVP.
    """
    captured = {}

    def _capture(module, inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        captured["value"] = state[0, -1, :]

    handle = final_block.register_forward_hook(_capture)
    try:
        output = model(input_ids=input_ids, output_hidden_states=output_hidden_states)
    finally:
        handle.remove()
    return output, captured["value"]


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
