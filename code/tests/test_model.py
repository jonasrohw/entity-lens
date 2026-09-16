"""Check the refactored Jacobian against an analytic derivative on CPU."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from jlens.model import HiddenStateIntervention  # noqa: E402


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


def test_measurements_agree_on_tiny_olmo(monkeypatch, tmp_path):
    """Exercise the complete sweep, rank scorer, and lens without model downloads."""
    transformers = pytest.importorskip("transformers")
    from transformers import Olmo3Config, Olmo3ForCausalLM

    from jlens import measure_faithfulness, measure_jacobian_lens, measure_subject_ranks
    from jlens.io import read_json, write_json

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

    def load_model(*args, **kwargs):
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
