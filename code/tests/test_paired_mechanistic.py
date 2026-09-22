"""Numerical and causal-position checks for paired interventions."""

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

from jlens.mechanistic import Runner
from jlens.paired_mechanistic import capture_components, patched_component


def tiny():
    torch.manual_seed(20)
    config = Olmo3Config(vocab_size=64, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=23, num_attention_heads=4,
                        num_key_value_heads=4, max_position_embeddings=64,
                        eos_token_id=2)
    config._attn_implementation = "eager"
    model = Olmo3ForCausalLM(config).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Runner(model, None, [5, 11, 17, 23])


def cached(runner, ids):
    with torch.no_grad():
        final, states, _ = runner.forward(ids, 2, capture=True)
    return {"ids": ids, "pos": 2, "states": states, "final": final,
            "groups": {"subject_last": [2], "subject_other": [1], "suffix": [3], "answer": [4]}}


def test_causal_prefix_is_identical_across_relations():
    runner = tiny()
    a = cached(runner, torch.tensor([[1, 2, 3, 4, 5]]))
    b = cached(runner, torch.tensor([[1, 2, 3, 9, 10]]))
    for layer in runner.layers:
        torch.testing.assert_close(a["states"][layer], b["states"][layer], atol=1e-7, rtol=1e-6)
    assert not torch.allclose(a["final"], b["final"])


def test_components_and_heads_identity_and_downstream_state_control():
    runner = tiny()
    a = cached(runner, torch.tensor([[1, 2, 3, 4, 5]]))
    b = cached(runner, torch.tensor([[1, 6, 7, 4, 5]]))
    for heads in [False, True]:
        clean = capture_components(runner, a, heads=heads, blocks=[18])
        donor = capture_components(runner, b, heads=heads, blocks=[18])
        for kind in (["heads"] if heads else ["attention", "mlp"]):
            head = 1 if heads else None
            for group in a["groups"]:
                identity, state = patched_component(runner, a, 18, kind, group, clean[(18,kind,group)], head=head)
                torch.testing.assert_close(identity, a["final"], atol=1e-7, rtol=1e-6)
                changed, state = patched_component(runner, a, 18, kind, group, donor[(18,kind,group)], head=head)
                torch.testing.assert_close(state, a["states"][17], atol=1e-7, rtol=1e-6)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in runner.model.modules())


def test_basis_product_reconstructs_exact_intervention_response():
    runner = tiny()
    ref = cached(runner, torch.tensor([[1, 2, 3, 4, 5]]))
    q = torch.linalg.qr(torch.randn(16, 3)).Q
    coeff = torch.tensor([.2, -.3, .1])
    products = runner.products([ref], 17, q.T)
    direct = runner.products([ref], 17, (q @ coeff)[None])[0]
    torch.testing.assert_close(coeff @ products, direct, atol=2e-6, rtol=2e-4)
    # A larger step avoids subtractive cancellation in the tiny float32 model.
    eps = .01
    with torch.no_grad():
        plus = runner.forward(ref["ids"], ref["pos"], state_patch=(17, ref["states"][17] + eps*q@coeff))[0]
        minus = runner.forward(ref["ids"], ref["pos"], state_patch=(17, ref["states"][17] - eps*q@coeff))[0]
    assert (((plus-minus)/(2*eps)-direct).norm()/direct.norm()) < .01
