"""End-to-end runs of the two ablation baselines against a tiny stub model.

The real measurements need a 7B checkpoint and a GPU. Everything except the
checkpoint -- split reconstruction, target verification, the Jacobian--vector
products, the readout, and the output schema -- is exercised here on CPU.
"""

import random
import sys
import types
from types import SimpleNamespace

import pytest

from jlens.io import read_json, write_json

torch = pytest.importorskip("torch")

from jlens import (  # noqa: E402
    matched_baseline,
    measure_jacobian_lens_raw,
    measure_logit_lens_diff,
)

WORDS = [
    "sun moon star cloud river ocean forest desert canyon glacier",
    "valley island meadow tundra prairie lagoon plateau reef fjord marsh",
]
SUBJECTS = WORDS[0].split() + WORDS[1].split()
TEMPLATE = "{} belongs to"
LAYER = 2


class Encoding(dict):
    """Minimal stand-in for a tokenizer's BatchEncoding."""

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class WordTokenizer:
    """Whitespace tokenizer over a fixed vocabulary."""

    def __init__(self, vocabulary):
        self.vocabulary = list(vocabulary)
        self.index = {word: number for number, word in enumerate(self.vocabulary)}

    def __call__(self, text, return_tensors=None):
        ids = [self.index[word] for word in text.split()]
        if return_tensors == "pt":
            return Encoding(input_ids=torch.tensor([ids]))
        return Encoding(input_ids=ids)

    def decode(self, ids):
        return " ".join(self.vocabulary[int(i)] for i in ids)


class RMSNorm(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(width) + 0.5)
        self.variance_epsilon = 1e-5

    def forward(self, states):
        scale = torch.sqrt(states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * states / scale


class Block(torch.nn.Module):
    """Causal prefix-mean mixing, so the last position depends on the subject."""

    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)

    def forward(self, states):
        length = states.shape[1]
        divisor = torch.arange(1, length + 1, dtype=states.dtype).view(1, length, 1)
        mixed = torch.cumsum(states, dim=1) / divisor
        return torch.tanh(self.linear(states + mixed)) + states


class Backbone(torch.nn.Module):
    def __init__(self, vocabulary, width, depth):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocabulary, width)
        self.layers = torch.nn.ModuleList(Block(width) for _ in range(depth))
        self.norm = RMSNorm(width)


class TinyCausalLM(torch.nn.Module):
    """Exposes the attribute surface the measurements reach into."""

    def __init__(self, vocabulary, width=8, depth=3):
        super().__init__()
        self.model = Backbone(vocabulary, width, depth)
        self.lm_head = torch.nn.Linear(width, vocabulary, bias=False)
        self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())

    def forward(self, input_ids, output_hidden_states=False):
        states = self.model.embed_tokens(input_ids)
        hidden = [states]
        for block in self.model.layers:
            states = block(states)
            hidden.append(states)
        return SimpleNamespace(
            hidden_states=tuple(hidden), logits=self.lm_head(self.model.norm(states))
        )


def toy_repository(tmp_path):
    """A one-relation root plus the reference files the baselines are matched to."""
    torch.manual_seed(0)
    samples = [
        {"subject": subject, "object": SUBJECTS[(index + 7) % len(SUBJECTS)]}
        for index, subject in enumerate(SUBJECTS)
    ]
    write_json(
        tmp_path / "data/relations/toy.json",
        {"prompt_templates": [TEMPLATE], "samples": samples},
    )
    vocabulary = sorted(
        {word for subject in SUBJECTS for word in subject.split()}
        | set(TEMPLATE.format("").split())
        | set("is known for".split())
    )
    tokenizer = WordTokenizer(vocabulary)
    model = TinyCausalLM(len(vocabulary)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Rebuild the legacy split here rather than through the helper under test.
    pool = list(samples)
    random.Random(17).shuffle(pool)
    context = "".join(
        TEMPLATE.format(row["subject"]) + " " + row["object"] + "\n" for row in pool[:8]
    )
    subjects = [row["subject"] for row in pool[8:28]]
    rows = []
    with torch.no_grad():
        for subject in subjects:
            ids = tokenizer(context + TEMPLATE.format(subject), return_tensors="pt")
            token = int(model(input_ids=ids["input_ids"]).logits[0, -1].argmax())
            rows.append(dict(subject=subject, model_top1=tokenizer.decode([token])))
    reference = {"toy": dict(n=len(rows), layer=LAYER, template=TEMPLATE, rows=rows)}
    write_json(tmp_path / "results/jacobian_lens.json", reference)
    write_json(tmp_path / "results/jacobian_lens_years.json", {})
    return model, tokenizer, subjects


@pytest.fixture
def toy(tmp_path, monkeypatch):
    model, tokenizer, subjects = toy_repository(tmp_path)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: model
    )
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, device: self)
    for module in (measure_logit_lens_diff, measure_jacobian_lens_raw):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(matched_baseline, "ROOT", tmp_path)
    return SimpleNamespace(root=tmp_path, subjects=subjects, monkeypatch=monkeypatch)


def run(module, toy, name):
    out = toy.root / f"results/{name}.json"
    toy.monkeypatch.setattr(
        sys, "argv", [name, "--out", str(out), "--layer", str(LAYER), "--device", "cpu"]
    )
    module.main()
    return read_json(out)["toy"]


@pytest.mark.parametrize(
    "module,name,field",
    [
        (measure_logit_lens_diff, "logit_lens_diff", "rank_logit_diff"),
        (measure_jacobian_lens_raw, "jacobian_lens_raw", "rank_raw"),
    ],
)
def test_baseline_scores_every_matched_subject(module, name, field, toy):
    result = run(module, toy, name)
    reference = read_json(toy.root / "results/jacobian_lens.json")["toy"]
    assert [row["subject"] for row in result["rows"]] == toy.subjects
    assert [row["model_top1"] for row in result["rows"]] == [
        row["model_top1"] for row in reference["rows"]
    ]
    # The first eight subjects are the references, so they average over seven.
    assert [row["n_references"] for row in result["rows"]] == [7] * 8 + [8] * (
        len(toy.subjects) - 8
    )
    ranks = [row[field] for row in result["rows"]]
    assert result["agg"] == matched_baseline.aggregate(ranks)
    assert result["layer"] == LAYER
    # A readout that ignored the subject would return one vector for all of them.
    assert (
        len({row["top_" + field.removeprefix("rank_")][0] for row in result["rows"]})
        > 1
    )


def test_baseline_refuses_a_reference_it_does_not_reproduce(toy):
    path = toy.root / "results/jacobian_lens.json"
    reference = read_json(path)
    reference["toy"]["rows"][3]["model_top1"] = "glacier"
    write_json(path, reference)
    with pytest.raises(ValueError, match="model answer"):
        run(measure_logit_lens_diff, toy, "logit_lens_diff")
