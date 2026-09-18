"""Check the refactored Jacobian against an analytic derivative on CPU."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from jlens.model import (  # noqa: E402
    HiddenStateIntervention,
    final_logits,
    forward_capturing_final_block,
    resolve_blocks,
    resolve_final_norm,
)


class TokenBatch(dict):
    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 127 + 1 for character in text]

    def __call__(self, text, return_tensors=None):
        tokens = self.encode(text)
        if return_tensors:
            return TokenBatch(input_ids=torch.tensor([tokens]))
        return SimpleNamespace(input_ids=tokens)

    def decode(self, ids):
        return "".join(chr(index - 1) if index else "?" for index in ids)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Identity()

    def forward(self, input_ids, output_hidden_states=True):
        states = self.block(input_ids)
        final = states.pow(3).sum(dim=1, keepdim=True)
        return SimpleNamespace(hidden_states=(states, final))


def test_jvp_matches_analytic_derivative_and_removes_intervention():
    model = TinyModel()
    intervention = HiddenStateIntervention(model)
    input_ids = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
    reference = torch.tensor([2.0, 3.0], dtype=torch.float64)
    direction = torch.tensor([0.5, -1.0], dtype=torch.float64)
    original = model(input_ids).hidden_states[-1].clone()
    handle = model.block.register_forward_hook(intervention.hook)
    try:
        product = intervention.jvp(reference, direction, input_ids, 0)
        torch.testing.assert_close(product, 3 * reference.pow(2) * direction)
        assert not product.requires_grad
        torch.testing.assert_close(
            input_ids[0, 0], torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
    finally:
        handle.remove()
    torch.testing.assert_close(model(input_ids).hidden_states[-1], original)


def test_tuple_hook_preserves_additional_outputs():
    intervention = HiddenStateIntervention(None)
    intervention.hidden = torch.tensor([7.0, 8.0])
    intervention.position = 0
    original = torch.zeros(1, 2, 2)
    marker = object()
    result = intervention.hook(None, None, (original, marker))
    assert result[1] is marker
    assert original.sum() == 0
    torch.testing.assert_close(result[0][0, 0], intervention.hidden)


def _tiny_olmo_model():
    """RMSNorm final norm, sequential attention/MLP blocks (Llama-style)."""
    from transformers import Olmo3Config, Olmo3ForCausalLM

    torch.manual_seed(17)
    config = Olmo3Config(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    return Olmo3ForCausalLM(config)


def _tiny_gptj_model():
    """LayerNorm final norm (weight and bias), parallel attention/MLP blocks."""
    from transformers import GPTJConfig, GPTJForCausalLM

    torch.manual_seed(17)
    config = GPTJConfig(
        vocab_size=128,
        n_embd=8,
        n_inner=16,
        n_layer=2,
        n_head=2,
        rotary_dim=2,
        n_positions=512,
        bos_token_id=None,
        eos_token_id=None,
    )
    config._attn_implementation = "eager"
    return GPTJForCausalLM(config)


def _tiny_gemma4_model():
    """RMSNorm final norm w/ final-logit softcap, hybrid sliding/full attention.

    Also exercises per-layer embeddings (PLE) and QK-norm, both real Gemma 4
    features, at minimal size — kept small so the tiny model stays cheap, not
    to skip these code paths.
    """
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(17)
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=512,
        sliding_window=4,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=4,
        final_logit_softcapping=30.0,
    )
    config._attn_implementation = "eager"
    return Gemma4ForCausalLM(config)


def _tiny_gemma4_unified_model():
    """Nested ``model.model.language_model.layers``/``.norm`` layout.

    ``google/gemma-4-12B`` (and every other plain ``google/gemma-4-*`` repo,
    confirmed by downloading its ``config.json``) ships the multimodal
    ``Gemma4UnifiedForConditionalGeneration`` architecture even for text-only
    use — there is no separate plain-text checkpoint. That class nests the
    text decoder as ``model.model.language_model`` rather than
    ``model.model`` directly, and reads ``final_logit_softcapping`` via
    ``config.get_text_config()`` rather than as a direct attribute (see
    ``Gemma4UnifiedForConditionalGeneration.forward`` in
    ``modeling_gemma4_unified.py``). ``vision_config``/``audio_config`` are
    left ``None`` here to keep the tiny model text-only and cheap.
    """
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

    torch.manual_seed(17)
    text_config = dict(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=512,
        sliding_window=4,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=30.0,
    )
    config = Gemma4UnifiedConfig(
        text_config=text_config, vision_config=None, audio_config=None
    )
    config.get_text_config()._attn_implementation = "eager"
    return Gemma4UnifiedForConditionalGeneration(config)


def _exercise_tiny_pipeline(monkeypatch, tmp_path, transformers, load_model):
    """Run the complete sweep, rank scorer, and lens against a tiny random model.

    Shared across architectures so both the Llama-style (RMSNorm) and
    GPT-2-style (LayerNorm) final-norm paths get the same end-to-end check.
    """
    from jlens import measure_faithfulness, measure_jacobian_lens, measure_subject_ranks
    from jlens.io import read_json, write_json

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: CharacterTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", load_model
    )
    data_dir = tmp_path / "data"
    result_dir = tmp_path / "results"
    write_json(
        data_dir / "toy.json",
        {
            "prompt_templates": ["{}:"],
            "samples": [
                {"subject": f"name{index}", "object": chr(65 + index)}
                for index in range(20)
            ],
        },
    )
    common = [
        "--data-dir",
        str(data_dir),
        "--device",
        "cpu",
        "--n-test",
        "3",
        "--trailing-space",
    ]

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "faithfulness",
            "--relation",
            "toy",
            "--out-dir",
            str(result_dir),
            "--layers",
            "1,2",
            "--distinct-first-token",
            *common,
        ],
    )
    measure_faithfulness.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ranks",
            "--relation",
            "toy",
            "--out-dir",
            str(result_dir),
            "--faith-dir",
            str(result_dir),
            *common,
        ],
    )
    measure_subject_ranks.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lens",
            "--rels",
            "toy",
            "--out",
            str(result_dir / "lens.json"),
            "--layer",
            "1",
            "--distinct-first-token",
            *common,
        ],
    )
    measure_jacobian_lens.main()
    faithfulness = read_json(result_dir / "faithfulness_toy.json")
    ranks = read_json(result_dir / "subject_ranks_toy.json")
    lens = read_json(result_dir / "lens.json")["toy"]
    assert (
        faithfulness["split_fingerprint"]
        == ranks["split_fingerprint"]
        == lens["split_fingerprint"]
    )
    assert (
        set(faithfulness["per_subject"])
        == set(ranks["ranks"])
        == {row["subject"] for row in lens["rows"]}
    )
    assert all(
        int(row["rank"] == 0) == faithfulness["per_subject"][subject]
        for subject, row in ranks["ranks"].items()
    )


def test_measurements_agree_on_tiny_olmo(monkeypatch, tmp_path):
    """RMSNorm final norm, sequential attention/MLP blocks (Llama-style)."""
    transformers = pytest.importorskip("transformers")
    _tiny_olmo_model()  # fail fast with a clear error if Olmo3 isn't available
    _exercise_tiny_pipeline(
        monkeypatch, tmp_path, transformers, lambda *a, **k: _tiny_olmo_model()
    )


def test_measurements_agree_on_tiny_gptj(monkeypatch, tmp_path):
    """LayerNorm final norm (weight and bias), parallel attention/MLP blocks.

    Exercises the ``resolve_blocks``/``resolve_final_norm``/``final_logits``
    dispatch added for GPT-J: ``transformer.h`` / ``transformer.ln_f`` instead
    of ``model.layers`` / ``model.norm``, and a biased LayerNorm instead of
    RMSNorm.
    """
    transformers = pytest.importorskip("transformers")
    _tiny_gptj_model()  # fail fast with a clear error if GPT-J isn't available
    _exercise_tiny_pipeline(
        monkeypatch, tmp_path, transformers, lambda *a, **k: _tiny_gptj_model()
    )


def test_measurements_agree_on_tiny_gemma4(monkeypatch, tmp_path):
    """RMSNorm final norm w/ final-logit softcap, hybrid sliding/full attention.

    Exercises the ``final_logits`` softcap dispatch added for Gemma 4, on top
    of the Llama-style ``model.layers``/``model.norm`` path it already shares
    with OLMo (confirmed to need no changes to ``resolve_blocks``/
    ``resolve_final_norm``).
    """
    transformers = pytest.importorskip("transformers")
    _tiny_gemma4_model()  # fail fast with a clear error if Gemma4 isn't available
    _exercise_tiny_pipeline(
        monkeypatch, tmp_path, transformers, lambda *a, **k: _tiny_gemma4_model()
    )


def test_measurements_agree_on_tiny_gemma4_unified(monkeypatch, tmp_path):
    """Nested ``model.model.language_model`` layout, the shape real Gemma 4
    checkpoints actually ship (see ``_tiny_gemma4_unified_model``).

    Exercises the third ``resolve_blocks``/``resolve_final_norm`` branch and
    the ``config.get_text_config()``-based softcap lookup added for it.
    """
    transformers = pytest.importorskip("transformers")
    _tiny_gemma4_unified_model()  # fail fast if Gemma4Unified isn't available
    _exercise_tiny_pipeline(
        monkeypatch,
        tmp_path,
        transformers,
        lambda *a, **k: _tiny_gemma4_unified_model(),
    )


@pytest.mark.parametrize(
    "build_model",
    [
        _tiny_olmo_model,
        _tiny_gptj_model,
        _tiny_gemma4_model,
        _tiny_gemma4_unified_model,
    ],
)
def test_final_logits_matches_real_forward_pass(build_model):
    """The manual norm+unembed reimplementation must match the model's own logits.

    No JVP or intervention involved: this isolates ``final_logits`` itself
    (RMSNorm vs. LayerNorm dispatch, optional lm_head bias, optional Gemma
    final-logit softcap) from everything else in the pipeline, across a
    Llama-style, a GPT-2-style, and a softcapped-RMSNorm-style model.
    """
    pytest.importorskip("transformers")
    model = build_model().eval()
    input_ids = torch.tensor([[3, 5, 7, 9]])
    final_block = resolve_blocks(model)[-1]
    with torch.no_grad():
        output, raw_final = forward_capturing_final_block(model, final_block, input_ids)
    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach().float()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias = lm_bias.detach().float() if lm_bias is not None else None
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    reconstructed = final_logits(raw_final, norm, unembedding, lm_bias, softcap)
    torch.testing.assert_close(
        reconstructed, output.logits[0, -1, :].float(), rtol=1e-4, atol=1e-4
    )
    text_config = model.config.get_text_config()
    expected_layers = getattr(text_config, "num_hidden_layers", None) or getattr(
        text_config, "n_layer", None
    )
    assert len(resolve_blocks(model)) == expected_layers


def test_hidden_states_last_entry_is_post_norm_not_raw(build_model=_tiny_olmo_model):
    """Regression guard for the bug `forward_capturing_final_block` works around.

    If a future transformers version reverts to returning the raw (pre-norm)
    block output as `hidden_states[-1]`, this documents why the workaround
    exists so it can be safely removed rather than silently left as dead
    weight. If this assertion starts failing, `HiddenStateIntervention`'s
    `final_block=None` fallback path is safe to use unconditionally again.
    """
    pytest.importorskip("transformers")
    model = build_model().eval()
    input_ids = torch.tensor([[3, 5, 7, 9]])
    final_block = resolve_blocks(model)[-1]
    with torch.no_grad():
        output, raw_final = forward_capturing_final_block(
            model, final_block, input_ids, output_hidden_states=True
        )
    assert not torch.allclose(output.hidden_states[-1][0, -1, :], raw_final)
