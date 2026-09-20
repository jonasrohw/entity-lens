"""Readouts must use the architecture's actual final normalization."""

import pytest

torch = pytest.importorskip("torch")

from jlens.model import final_logits, forward_capturing_final_block  # noqa: E402


def test_final_logits_preserves_gemma2_offset_weight_parameterization():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(23)
    config = transformers.Gemma2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        sliding_window=8,
    )
    config._attn_implementation = "eager"
    model = transformers.Gemma2ForCausalLM(config).eval()
    # Gemma2 stores an offset from one; a nonuniform weight catches wrong ranks,
    # not only a constant scaling difference.
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(-0.5, 0.5, 8))
        output, raw = forward_capturing_final_block(
            model, model.model.layers[-1], torch.tensor([[2, 5, 7, 3]])
        )
        result = final_logits(
            raw,
            model.model.norm,
            model.lm_head.weight,
            softcap=config.final_logit_softcapping,
        )
    torch.testing.assert_close(result, output.logits[0, -1], rtol=1e-5, atol=1e-6)


def test_final_logits_supports_layernorm_without_bias():
    torch.manual_seed(24)
    norm = torch.nn.LayerNorm(8, bias=False)
    hidden = torch.randn(3, 8) + 4
    unembedding = torch.randn(20, 8)
    expected = norm(hidden) @ unembedding.T
    actual = final_logits(hidden, norm, unembedding)
    torch.testing.assert_close(actual, expected)
