"""Independent checks of GPT-J residual boundaries and Jacobian products."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import GPTJConfig, GPTJForCausalLM  # noqa: E402

from jlens.model import (  # noqa: E402
    HiddenStateIntervention,
    final_logits,
    forward_capturing_final_block,
    forward_with_raw_hidden_states,
)


def tiny_gptj():
    torch.manual_seed(41)
    config = GPTJConfig(
        vocab_size=32,
        n_embd=8,
        n_inner=16,
        n_layer=3,
        n_head=2,
        rotary_dim=2,
        n_positions=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = GPTJForCausalLM(config).eval().requires_grad_(False)
    # Default random-model norms/biases can hide missing affine terms.
    with torch.no_grad():
        model.transformer.ln_f.weight.uniform_(0.5, 1.5)
        model.transformer.ln_f.bias.normal_(std=0.2)
        model.lm_head.bias.normal_(std=0.2)
    return model


def parallel_tail(model, states, start):
    """Execute HF's parallel equation directly, without block/model hooks."""
    length = states.shape[1]
    positions = torch.arange(length, device=states.device).unsqueeze(0)
    mask = torch.full(
        (length, length),
        torch.finfo(states.dtype).min,
        device=states.device,
        dtype=states.dtype,
    ).triu(1)[None, None]
    for block in model.transformer.h[start:]:
        normed = block.ln_1(states)
        attention = block.attn(
            normed, attention_mask=mask, position_ids=positions, use_cache=False
        )[0]
        states = attention + block.mlp(normed) + states
    return states


def test_parallel_equation_matches_hf_and_sequential_equation_does_not():
    model = tiny_gptj()
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    with torch.no_grad():
        output, raw = forward_capturing_final_block(
            model, model.transformer.h[-1], ids, output_hidden_states=True
        )
        manual = parallel_tail(model, model.transformer.wte(ids), 0)
        torch.testing.assert_close(manual[0, -1], raw, rtol=0, atol=0)
        torch.testing.assert_close(
            model.lm_head(model.transformer.ln_f(manual)), output.logits, rtol=0, atol=0
        )
        for layer in (1, 2):
            prefix = parallel_tail_prefix(model, ids, layer)
            torch.testing.assert_close(
                prefix, output.hidden_states[layer], rtol=0, atol=0
            )
        block = model.transformer.h[0]
        states = model.transformer.wte(ids)
        length = ids.shape[1]
        mask = torch.full((length, length), torch.finfo(states.dtype).min).triu(1)
        attn = block.attn(
            block.ln_1(states),
            attention_mask=mask[None, None],
            position_ids=torch.arange(length)[None],
            use_cache=False,
        )[0]
        parallel = states + attn + block.mlp(block.ln_1(states))
        sequential = states + attn + block.mlp(block.ln_1(states + attn))
        assert (parallel - sequential).abs().max() > 1e-5


def parallel_tail_prefix(model, ids, layer):
    states = model.transformer.wte(ids)
    positions = torch.arange(ids.shape[1])[None]
    mask = torch.full((ids.shape[1], ids.shape[1]), torch.finfo(states.dtype).min).triu(
        1
    )[None, None]
    for block in model.transformer.h[:layer]:
        normalized = block.ln_1(states)
        states = (
            block.attn(
                normalized, attention_mask=mask, position_ids=positions, use_cache=False
            )[0]
            + block.mlp(normalized)
            + states
        )
    return states


@pytest.mark.parametrize("layer", [1, 2, 3])
@pytest.mark.parametrize("position", [1, 4])
def test_intervention_and_jvp_against_independent_tail(layer, position):
    model = tiny_gptj()
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    states = parallel_tail_prefix(model, ids, layer).detach()
    reference = states[0, position].clone()
    direction = torch.randn_like(reference) * reference.norm() / 10

    def independent(hidden):
        replaced = states.clone()
        replaced[0, position] = hidden
        return parallel_tail(model, replaced, layer)[0, -1]

    intervention = HiddenStateIntervention(model, model.transformer.h[-1])
    handle = model.transformer.h[layer - 1].register_forward_hook(intervention.hook)
    try:
        for hidden in (reference, reference + direction):
            torch.testing.assert_close(
                intervention.forward(hidden, ids, position),
                independent(hidden),
                rtol=0,
                atol=0,
            )
        product = intervention.jvp(reference, direction, ids, position)
        # Full Jacobian uses ordinary reverse mode, not the double-backward trick.
        jacobian = torch.autograd.functional.jacobian(independent, reference)
        torch.testing.assert_close(product, jacobian @ direction, rtol=2e-5, atol=1e-8)
        epsilon = 0.01
        finite_difference = (
            independent(reference + epsilon * direction)
            - independent(reference - epsilon * direction)
        ) / (2 * epsilon)
        torch.testing.assert_close(product, finite_difference, rtol=3e-3, atol=3e-7)
        if layer == 3:
            expected = direction if position == 4 else torch.zeros_like(direction)
            torch.testing.assert_close(product, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            intervention.jvp(reference, torch.zeros_like(direction), ids, position),
            torch.zeros_like(direction),
        )
    finally:
        handle.remove()
    output = model(ids)
    raw = independent(reference)
    decoded = final_logits(
        raw, model.transformer.ln_f, model.lm_head.weight, model.lm_head.bias
    )
    torch.testing.assert_close(decoded, output.logits[0, -1], rtol=2e-5, atol=1e-7)


def test_cache_does_not_change_jvp():
    model = tiny_gptj()
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    reference = model(ids, output_hidden_states=True).hidden_states[1][0, 1].detach()
    direction = torch.randn_like(reference)
    intervention = HiddenStateIntervention(model, model.transformer.h[-1])
    handle = model.transformer.h[0].register_forward_hook(intervention.hook)
    try:
        uncached = intervention.jvp(reference, direction, ids, 1)
        model.config.use_cache = True
        cached = intervention.jvp(reference, direction, ids, 1)
        torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    finally:
        handle.remove()


@pytest.mark.parametrize("layer", [1, 2, 3])
@pytest.mark.parametrize("position", [1, 4])
def test_cached_state_is_at_the_intervention_boundary(layer, position):
    model = tiny_gptj()
    ids = torch.tensor([[3, 5, 7, 9, 11]])
    output, states = forward_with_raw_hidden_states(model, ids)
    assert len(states) == len(model.transformer.h) + 1
    torch.testing.assert_close(
        states[layer], parallel_tail_prefix(model, ids, layer), rtol=0, atol=0
    )
    torch.testing.assert_close(
        output.hidden_states[-1], model.transformer.ln_f(states[-1])
    )
    assert not torch.allclose(states[-1], output.hidden_states[-1])
    intervention = HiddenStateIntervention(model, model.transformer.h[-1])
    handle = model.transformer.h[layer - 1].register_forward_hook(intervention.hook)
    try:
        raw = intervention.forward(states[layer][0, position], ids, position)
        torch.testing.assert_close(raw, states[-1][0, -1], rtol=0, atol=0)
        # The previous cache read HF's post-norm final entry. Demonstrate
        # that it corrupts the reference when hooked at the final position.
        if layer == 3 and position == 4:
            wrong = intervention.forward(
                output.hidden_states[layer][0, position], ids, position
            )
            assert not torch.allclose(wrong, raw)
    finally:
        handle.remove()
