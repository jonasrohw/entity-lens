"""Scientific checks for the independent mechanistic experiment runner."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from jlens.mechanistic import Runner, attribute_basis, removal  # noqa: E402


def tiny_runner():
    from transformers import Olmo3Config, Olmo3ForCausalLM

    torch.manual_seed(3)
    config = Olmo3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Olmo3ForCausalLM(config).double().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Runner(model, None, [1, 2])


def test_reused_jacobian_graph_matches_finite_difference():
    runner = tiny_runner()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        _, states, _ = runner.forward(ids, 1, capture=True)
    ref = {"ids": ids, "pos": 1, "states": states}
    directions = torch.randn(3, 8, dtype=torch.double)
    actual = runner.products([ref], 1, directions)
    # OLMo's norms cast to float32 internally even for a double model.
    eps = 1e-3
    for i, d in enumerate(directions):
        with torch.no_grad():
            plus = runner.forward(ids, 1, state_patch=(1, states[1] + eps * d))[0]
            minus = runner.forward(ids, 1, state_patch=(1, states[1] - eps * d))[0]
        torch.testing.assert_close(
            actual[i], (plus - minus) / (2 * eps), atol=2e-5, rtol=2e-3
        )
    assert all(not block._forward_hooks for block in runner.blocks)


def test_identity_patch_and_component_rescue():
    runner = tiny_runner()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        clean, states, _ = runner.forward(ids, 1, capture=True)
        identity = runner.forward(ids, 1, state_patch=(1, states[1]))[0]
        rescued = runner.forward(
            ids,
            1,
            component_patch=(1, "attention", torch.zeros(8)),
            state_patch=(1, states[1]),
        )[0]
        _, _, components = runner.forward(
            ids,
            1,
            capture=True,
            component_patch=(1, "attention", torch.zeros(8)),
            state_patch=(1, states[1]),
        )
    torch.testing.assert_close(clean, identity)
    torch.testing.assert_close(clean, rescued)
    torch.testing.assert_close(
        components[(1, "attention")], torch.zeros(8, dtype=torch.double)
    )


def test_attribute_erasure_and_singleton_rejection():
    x = torch.tensor(
        [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-2.0, 0.0, 0.0], [-2.0, 1.0, 0.0]]
    )
    q = attribute_basis(x, [0, 0, 1, 1])
    assert q.shape == (3, 1)
    erased = x - removal(x, x.mean(0), q)
    torch.testing.assert_close(erased[:2].mean(0), erased[2:].mean(0))
    assert attribute_basis(x, [0, 1, 2, 3]).shape == (3, 0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs needed")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device_count", [2, 3])
def test_two_gpu_placement_preserves_forward_and_jacobian(dtype, device_count):
    import copy

    from accelerate import dispatch_model

    if torch.cuda.device_count() < device_count:
        pytest.skip("Not enough GPUs for this placement")

    single = tiny_runner()
    single.model.to(dtype=dtype)
    other = copy.deepcopy(single.model)
    single.model.to("cuda:0")
    single = Runner(single.model, None, [1, 2])
    other = dispatch_model(
        other,
        device_map={
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.layers.2": device_count - 1,
            "model.norm": device_count - 1,
            "lm_head": device_count - 1,
        },
    )
    split = Runner(other, None, [1, 2])
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    with torch.no_grad():
        final_a, states_a, _ = single.forward(ids, 1, capture=True)
        final_b, states_b, _ = split.forward(ids, 1, capture=True)
    torch.testing.assert_close(final_a.cpu(), final_b.cpu(), atol=1e-7, rtol=1e-6)
    dirs = torch.randn(3, 8, dtype=dtype, device="cuda:0")
    for layer in [1, 2]:
        a = single.products([{"ids": ids, "pos": 1, "states": states_a}], layer, dirs)
        b = split.products(
            [{"ids": ids, "pos": 1, "states": states_b}], layer, dirs.to(split.device)
        )
        torch.testing.assert_close(a.cpu(), b.cpu(), atol=1e-7, rtol=1e-6)


def test_reference_centered_taylor_decomposition():
    runner = tiny_runner()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        _, states, _ = runner.forward(ids, 1, capture=True)
    refs = [
        {"ids": ids, "pos": 1, "states": {1: states[1] + offset}}
        for offset in [0.0, 0.03]
    ]
    queries = torch.randn(3, 8, dtype=torch.double)
    mean = torch.randn(8, dtype=torch.double)
    direct = runner.products(refs, 1, queries, subtract_reference=True)
    centered = runner.products(refs, 1, queries - mean)
    offset = runner.products(refs, 1, mean[None], subtract_reference=True)
    # Internal float32 norm arithmetic sets the numerical precision floor.
    torch.testing.assert_close(direct, centered + offset, atol=1e-7, rtol=1e-6)
