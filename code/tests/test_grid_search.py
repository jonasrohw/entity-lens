"""Expanded grids preserve the original cells and include boundary states."""

import numpy as np
import pytest
from test_model_sweep import Tokenizer, tiny, torch

from jlens import grid_search as grid
from jlens import model_sweep as sweep


@pytest.mark.parametrize("protocol", ["main", "shared_split"])
def test_expanded_grid_preserves_original_faith_and_lens_cells(protocol):
    model = tiny("mistral")
    tokenizer = Tokenizer()
    samples = [dict(subject=f"name{i}", object=chr(65 + i)) for i in range(20)]
    original = sweep.measure_faith(
        model, tokenizer, samples, "{}:", protocol, [1, 2, 3], False, 2, 100
    )
    products = {}
    expanded = grid.faith_grid(
        model,
        tokenizer,
        samples,
        "{}:",
        protocol,
        list(range(5)),
        False,
        8,
        list(sweep.BETAS),
        tensors=products,
    )
    for layer in [1, 2, 3]:
        for beta in sweep.BETAS:
            assert (
                np.mean(expanded["grid"][str(layer)]["8"][str(beta)]["hits"])
                == original["grid"][str(layer)][str(beta)][0]
            )
    assert expanded["split_fingerprint"] == original["split_fingerprint"]
    torch.testing.assert_close(
        products["faith/4/8/products"], torch.zeros_like(products["faith/4/8/products"])
    )
    old_lens = sweep.measure_lens(
        model, tokenizer, samples, "{}:", protocol, 2, False, 2, 100
    )
    new_lens = grid.lens_grid(
        model,
        tokenizer,
        samples,
        "{}:",
        protocol,
        [0, 2, 4],
        False,
        8,
        list(sweep.BETAS),
        prompts={"known_for": "{} is known for"},
    )
    cell = new_lens["prompts"]["known_for"]["2"]["8"]
    assert cell["diff"]["ranks"] == [r["rank_diff"] for r in old_lens["rows"]]
    assert cell["affine"]["1.0"]["ranks"] == [r["rank_lre"] for r in old_lens["rows"]]


def test_embedding_cache_and_offline_beta_readout():
    model = tiny("mistral")
    tokenizer = Tokenizer()
    c = grid.cache(model, tokenizer, "France is known for", "France", [0, 1, 4])
    expected = model.get_input_embeddings()(c["ids"])[0, c["position"]]
    torch.testing.assert_close(c["states"][0], expected)
    base = torch.randn(1, 16)
    products = torch.randn(3, 16)
    targets = [1, 2, 3]
    batched = grid.decode_grid(model, base, products, targets, grid.BETAS, chunk_size=4)
    scalar = grid.decode_grid(model, base, products, targets, grid.BETAS, chunk_size=1)
    assert batched == scalar
    assert len(grid.BETAS) == 41


def test_relation_selection_does_not_use_evaluation_outcomes():
    from jlens.grid_statistics import select_on_relations

    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 2.0, 1.0, 1.0, 2.0, 3.0]])
    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    second = np.array([1.0, 2.0, 3.0, 6.0, 5.0, 4.0])
    assert (
        select_on_relations(x, first, [0, 1, 2])
        == select_on_relations(x, second, [0, 1, 2])
        == 0
    )


def test_maximum_permutation_test_is_reproducible_and_search_adjusted():
    from jlens.grid_statistics import maximum_permutation_test

    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 1.0, 2.0, 5.0, 6.0, 4.0]])
    y = np.arange(6.0)
    first = maximum_permutation_test(x, y, permutations=250, seed=7)
    assert first == maximum_permutation_test(x, y, permutations=250, seed=7)
    assert first["observed_max_rho"] == pytest.approx(1.0)
    assert 0 < first["adjusted_p"] < 0.1


@pytest.mark.parametrize("protocol", ["main", "shared_split"])
def test_fresh_split_uses_frozen_settings_even_with_three_test_subjects(protocol):
    from jlens import grid_confirmation

    class ObjectTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            return super().encode(text.lstrip(), add_special_tokens=add_special_tokens)

    model = tiny("mistral")
    samples = [dict(subject=f"name{i}", object=chr(65 + i)) for i in range(19)]
    config = dict(
        protocol=protocol,
        seed=1,
        faith_configs={
            "example": {
                "coarse": dict(layer=1, beta=0.5),
                "continuous": dict(layer=3, beta=2.0),
            }
        },
        lens_configs={
            "baseline": dict(
                prompt="known_for",
                layer=2,
                references=8,
                readout="diff",
                beta=None,
                metric="median_logrank",
            )
        },
    )
    result = grid_confirmation.measure(
        model,
        ObjectTokenizer(),
        dict(samples=samples, prompt_templates=["{}:"]),
        "example",
        config,
        8,
    )
    assert result["n_faith"] == 3
    assert result["faith"]["coarse"]["layer"] == 1
    assert result["faith"]["coarse"]["beta"] == 0.5
    assert result["faith"]["continuous"]["layer"] == 3
    assert result["faith"]["continuous"]["beta"] == 2.0
    assert 0 <= result["prior_only_faith"] <= 1
    expected = grid.partition(samples, protocol, seed=1)
    assert result["faith_split_fingerprint"] == grid.splits.split_fingerprint(*expected)
    assert result["faith_split_fingerprint"] != grid.splits.split_fingerprint(
        *grid.partition(samples, protocol, seed=0)
    )
    if protocol == "shared_split":
        assert result["faith_subjects"] == result["lens_subjects"]
        assert result["faith_split_fingerprint"] == result["lens_split_fingerprint"]


def test_subject_cleaning_collapses_duplicates_and_removes_ambiguous_answers():
    from jlens.grid_confirmation import unique_unambiguous_subjects

    samples = [
        dict(subject="A", object="one"),
        dict(subject="B", object="two"),
        dict(subject="A", object="one"),
        dict(subject="C", object="three"),
        dict(subject="B", object="four"),
    ]
    assert unique_unambiguous_subjects(samples) == [samples[0], samples[3]]
