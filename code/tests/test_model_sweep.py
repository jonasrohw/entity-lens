"""Guard faster products against the original scalar scorer across architectures."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from jlens import model_sweep  # noqa: E402
from jlens.model import HiddenStateIntervention  # noqa: E402


class Tokenizer:
    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % 127 for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

    def encode(self, text, add_special_tokens=False):
        return self(text).input_ids

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def tiny(kind):
    common = dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        head_dim=8,
    )
    cls, config = {
        "qwen3": (transformers.Qwen3ForCausalLM, transformers.Qwen3Config),
        "mistral": (transformers.MistralForCausalLM, transformers.MistralConfig),
        "olmo3": (transformers.Olmo3ForCausalLM, transformers.Olmo3Config),
        "gemma2": (transformers.Gemma2ForCausalLM, transformers.Gemma2Config),
        "pythia": (transformers.GPTNeoXForCausalLM, transformers.GPTNeoXConfig),
        "qwen3_5": (transformers.Qwen3_5ForCausalLM, transformers.Qwen3_5TextConfig),
    }[kind]
    if kind == "qwen3_5":
        common.update(
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"] * 2,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000,
                partial_rotary_factor=0.5,
                mrope_section=[1, 1, 0],
            ),
        )
    torch.manual_seed(31)
    cfg = config(**common)
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    return cls(cfg).eval().requires_grad_(False)


@pytest.mark.parametrize(
    "kind", ["qwen3", "mistral", "gemma2", "pythia", "qwen3_5", "olmo3"]
)
def test_batched_products_and_actual_norm_match_hf(kind):
    model = tiny(kind)
    result = model_sweep.verify(model, Tokenizer(), 2, 2)
    assert result["finite_difference_relative_error"] < 0.02


@pytest.mark.parametrize("protocol", ["main", "shared_split"])
@pytest.mark.parametrize("neutral_prompt", ["{} is known for", "{}", "Facts about {}:"])
def test_fast_scorer_matches_scalar_products(monkeypatch, protocol, neutral_prompt):
    model = tiny("qwen3")
    tokenizer = Tokenizer()
    samples = [dict(subject=f"name{i}", object=chr(65 + i)) for i in range(20)]
    fast = model_sweep.measure_faith(
        model, tokenizer, samples, "{}:", protocol, [1, 2], False, 2, 3
    )
    fast_lens = model_sweep.measure_lens(
        model,
        tokenizer,
        samples,
        "{}:",
        protocol,
        2,
        False,
        2,
        3,
        neutral_prompt=neutral_prompt,
    )

    def scalar(intervention, reference, directions, ids, position, batch_size):
        return torch.stack(
            [intervention.jvp(reference, d, ids, position) for d in directions]
        )

    monkeypatch.setattr(model_sweep, "jvps", scalar)
    slow = model_sweep.measure_faith(
        model, tokenizer, samples, "{}:", protocol, [1, 2], False, 2, 3
    )
    slow_lens = model_sweep.measure_lens(
        model,
        tokenizer,
        samples,
        "{}:",
        protocol,
        2,
        False,
        2,
        3,
        neutral_prompt=neutral_prompt,
    )
    assert fast == slow
    assert fast_lens == slow_lens


def test_jvps_uneven_batch_and_zero_direction():
    model = tiny("mistral")
    tokenizer = Tokenizer()
    blocks, _ = model_sweep.layout(model)
    cache = model_sweep.cache_prompt(
        model, tokenizer, "France is known for", "France", [2]
    )
    reference = cache["states"][2]
    directions = torch.stack([torch.zeros_like(reference), reference, reference * 2])
    intervention = HiddenStateIntervention(model, blocks[-1])
    handle = blocks[1].register_forward_hook(intervention.hook)
    try:
        result = model_sweep.jvps(
            intervention, reference, directions, cache["ids"], cache["position"], 2
        )
        torch.testing.assert_close(result[0], torch.zeros_like(reference))
        torch.testing.assert_close(result[2], result[1] * 2)
    finally:
        handle.remove()


@pytest.mark.parametrize("length", [1, 3, 17])
@pytest.mark.parametrize("bias", [False, True])
def test_shifted_convolution_matches_hf_forward_and_derivative(length, bias):
    from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

    torch.manual_seed(11)
    x = torch.randn(1, 7, length, dtype=torch.float64)
    weight = torch.randn(7, 4, dtype=torch.float64)
    bias = torch.randn(7, dtype=torch.float64) if bias else None
    direction = torch.randn_like(x)

    def original(h):
        return causal_conv1d_fn(h, weight, bias, "silu")

    def shifted(h):
        return model_sweep.causal_conv1d_shifts(h, weight, bias, "silu")

    expected, expected_product = torch.autograd.functional.jvp(original, x, direction)
    actual, actual_product = torch.autograd.functional.jvp(shifted, x, direction)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_product, expected_product, rtol=1e-12, atol=1e-12)


def test_measurement_claim_excludes_other_workers_and_releases(tmp_path):
    path = tmp_path / "main/relation.json"
    with model_sweep.measurement_claim(path) as first:
        assert first
        with model_sweep.measurement_claim(path) as second:
            assert not second
    with model_sweep.measurement_claim(path) as after_release:
        assert after_release


@pytest.mark.parametrize(
    "kind", ["qwen3", "mistral", "gemma2", "pythia", "qwen3_5", "olmo3"]
)
@pytest.mark.parametrize(
    "length,position", [(12, 0), (12, 1), (12, 5), (75, 70), (75, 74)]
)
def test_prefix_cache_matches_full_prompt_forward_and_derivative(
    kind, length, position
):
    from jlens.prefix_intervention import PrefixIntervention

    model = tiny(kind)
    blocks, _ = model_sweep.layout(model)
    ids = torch.randint(0, 128, (1, length))
    states = []

    def capture(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        states.append(hidden[0, position].detach())

    handle = blocks[1].register_forward_hook(capture)
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    reference = states[0]
    directions = torch.randn(2, reference.numel())
    values = []
    for cls in (HiddenStateIntervention, PrefixIntervention):
        intervention = cls(model, blocks[-1])
        handle = blocks[1].register_forward_hook(intervention.hook)
        try:
            state = intervention.forward(reference, ids, position).detach()
            products = model_sweep.jvps(
                intervention, reference, directions, ids, position, 2
            )
            values.append((state, products))
            # Repeated use must leave the saved prefix unchanged.
            torch.testing.assert_close(
                intervention.forward(reference, ids, position), state
            )
        finally:
            handle.remove()
    for full, cached in zip(*values):
        torch.testing.assert_close(full, cached, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("protocol", ["main", "shared_split"])
def test_cached_hybrid_scorer_matches_full_prompt(protocol):
    model = tiny("qwen3_5")
    tokenizer = Tokenizer()
    samples = [dict(subject=f"name{i}", object=chr(65 + i)) for i in range(20)]
    args = (model, tokenizer, samples, "{}:", protocol)
    faith = model_sweep.measure_faith(*args, [1, 3], False, 2, 3)
    lens = model_sweep.measure_lens(*args, 2, False, 2, 3)
    model._sweep_prefix_cache = True
    assert model_sweep.measure_faith(*args, [1, 3], False, 8, 3) == faith
    assert model_sweep.measure_lens(*args, 2, False, 8, 3) == lens


def test_larger_direction_batches_match_scalar_hybrid_products():
    from jlens.prefix_intervention import PrefixIntervention

    model = tiny("qwen3_5")
    blocks, _ = model_sweep.layout(model)
    cache = model_sweep.cache_prompt(
        model, Tokenizer(), "France is known for", "France", [2]
    )
    reference = cache["states"][2]
    directions = torch.randn(9, reference.numel())
    outputs = []
    for cls, batch_size in [(HiddenStateIntervention, 1), (PrefixIntervention, 8)]:
        intervention = cls(model, blocks[-1])
        handle = blocks[1].register_forward_hook(intervention.hook)
        try:
            outputs.append(
                model_sweep.jvps(
                    intervention,
                    reference,
                    directions,
                    cache["ids"],
                    cache["position"],
                    batch_size,
                )
            )
        finally:
            handle.remove()
    torch.testing.assert_close(*outputs, atol=2e-5, rtol=2e-4)
