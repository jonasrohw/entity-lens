from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from entity_lens.config import ChecksConfig
from entity_lens.modeling import HiddenStateIntervention, LanguageModel, PrefixIntervention
from entity_lens.records import ConvolutionCheck, DerivativeCheck, PrefixCheck, RuntimeChecks


def _relative(a: Tensor, b: Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _scalar_jvp(
    intervention: HiddenStateIntervention, hidden: Tensor, direction: Tensor, ids: Tensor, position: int
) -> Tensor:
    leaf = hidden.clone().requires_grad_()
    output = intervention.forward(leaf, ids, position)
    cotangent = torch.zeros_like(output, requires_grad=True)
    (gradient,) = torch.autograd.grad(output, leaf, grad_outputs=cotangent, create_graph=True)
    (product,) = torch.autograd.grad(gradient, cotangent, grad_outputs=direction)
    return product.detach()


def shifted_causal_conv1d(
    hidden_states: Tensor, weight: Tensor, bias: Tensor | None = None, activation: str | None = None, **kwargs: Any
) -> Tensor:
    from transformers.activations import ACT2FN

    length = hidden_states.shape[-1]
    padded = functional.pad(hidden_states.to(weight.dtype), (weight.shape[-1] - 1, 0))
    out = padded[:, :, :length] * weight[None, :, 0, None]
    for offset in range(1, weight.shape[-1]):
        out = out + padded[:, :, offset : offset + length] * weight[None, :, offset, None]
    if bias is not None:
        out = out + bias[None, :, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def patch_hybrid_convolution(lm: LanguageModel, layer: int, config: ChecksConfig) -> ConvolutionCheck | None:
    if lm.model.config.get_text_config().model_type != config.hybrid_model_type:
        return None
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    original = implementation.causal_conv1d_fn
    before = lm.cache_prompt(config.prompt, config.subject, [layer])
    generator = torch.Generator(device="cpu").manual_seed(config.convolution_seed)
    direction = torch.randn(before.states[layer].shape, generator=generator).to(before.states[layer].device)
    intervention = HiddenStateIntervention(lm)
    with lm.hooked(layer, intervention):
        reference = _scalar_jvp(intervention, before.states[layer], direction, before.ids, before.position)
        implementation.causal_conv1d_fn = shifted_causal_conv1d
        try:
            patched = _scalar_jvp(intervention, before.states[layer], direction, before.ids, before.position)
            torch.testing.assert_close(
                patched, reference, rtol=config.convolution_jvp[0], atol=config.convolution_jvp[1]
            )
        except Exception:
            implementation.causal_conv1d_fn = original
            raise
    after = lm.cache_prompt(config.prompt, config.subject, [layer])
    torch.testing.assert_close(
        after.logits, before.logits, rtol=config.convolution_logits[0], atol=config.convolution_logits[1]
    )
    return ConvolutionCheck(
        logits_max_abs_error=float((after.logits - before.logits).abs().max()),
        jvp_relative_error=_relative(patched, reference),
    )


def _directions(reference: Tensor, count: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn((count, reference.numel()), generator=generator).to(reference.device)
    scaled: Tensor = directions * reference.norm() / directions.norm(dim=1, keepdim=True)
    return scaled


def check_derivatives(lm: LanguageModel, layer: int, config: ChecksConfig) -> DerivativeCheck:
    cache = lm.cache_prompt(config.prompt, config.subject, [layer])
    with torch.no_grad():
        logits = lm.decode(cache.final)
    torch.testing.assert_close(logits, cache.logits, rtol=config.decode[0], atol=config.decode[1])
    directions = _directions(
        cache.states[layer], max(config.min_directions, lm.runtime.jvp_batch_size), config.derivative_seed
    )
    intervention = lm.intervention()
    with lm.hooked(layer, intervention):
        with torch.no_grad():
            unchanged = intervention.forward(cache.states[layer], cache.ids, cache.position)
        torch.testing.assert_close(unchanged, cache.final, rtol=0, atol=0)
        batched = lm.jvps(intervention, cache, layer, directions)
        scalar = _scalar_jvp(intervention, cache.states[layer], directions[0], cache.ids, cache.position)
        batched_error = _relative(batched[0], scalar)
        if batched_error > config.batched_jvp_tolerance:
            message = f"Batched and scalar JVPs differ: relative error {batched_error}"
            raise AssertionError(message)
        step = config.finite_difference_step
        with torch.no_grad():
            plus = intervention.forward(cache.states[layer] + step * directions[0], cache.ids, cache.position)
            minus = intervention.forward(cache.states[layer] - step * directions[0], cache.ids, cache.position)
    error = _relative((plus - minus) / (2 * step), scalar)
    if error >= config.finite_difference_tolerance:
        message = f"Finite-difference check failed: relative error {error}"
        raise AssertionError(message)
    return DerivativeCheck(
        decode_max_abs_error=float((logits - cache.logits).abs().max()),
        batched_scalar_relative_error=batched_error,
        finite_difference_relative_error=error,
    )


def enable_prefix_cache(lm: LanguageModel, layer: int, config: ChecksConfig) -> tuple[PrefixCheck, ...]:
    checks = []
    context_prompt = config.context_demonstration * config.context_repeats + config.context_query
    for prompt, state_layer in ((config.prompt, layer), (context_prompt, config.context_layer)):
        cache = lm.cache_prompt(prompt, config.subject, [state_layer])
        count = max(config.min_directions, lm.runtime.jvp_batch_size)
        directions = _directions(cache.states[state_layer], count, config.prefix_seed)
        results = []
        for intervention in (HiddenStateIntervention(lm), PrefixIntervention(lm)):
            with lm.hooked(state_layer, intervention):
                with torch.no_grad():
                    state = intervention.forward(cache.states[state_layer], cache.ids, cache.position)
                results.append((state, lm.jvps(intervention, cache, state_layer, directions)))
        (full_state, full_products), (cached_state, cached_products) = results
        torch.testing.assert_close(cached_state, full_state, rtol=config.prefix[0], atol=config.prefix[1])
        torch.testing.assert_close(cached_products, full_products, rtol=config.prefix[0], atol=config.prefix[1])
        checks.append(
            PrefixCheck(
                n_tokens=cache.ids.shape[1],
                prefix_length=cache.position,
                layer=state_layer,
                state_relative_error=_relative(cached_state, full_state),
                jvp_relative_error=_relative(cached_products, full_products),
            )
        )
    lm.prefix_cache = True
    return tuple(checks)


def prepare(lm: LanguageModel, config: ChecksConfig) -> RuntimeChecks:
    layer = lm.spec.lens_layer
    convolution = patch_hybrid_convolution(lm, layer, config)
    derivative = check_derivatives(lm, layer, config)
    prefix = enable_prefix_cache(lm, layer, config)
    return RuntimeChecks(convolution=convolution, derivative=derivative, prefix=prefix)
