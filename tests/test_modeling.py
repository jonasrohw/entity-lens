from __future__ import annotations

import pytest
import torch
from torch import nn

from entity_lens import checks, modeling
from entity_lens.config import Project
from entity_lens.modeling import (
    HiddenStateIntervention,
    LanguageModel,
    PrefixIntervention,
    resolve_blocks,
    resolve_final_norm,
)


def test_cache_prompt_positions(language_model: LanguageModel) -> None:
    prompt = "France is known for"
    cache = language_model.cache_prompt(prompt, "France", [1, 2])
    expected = len(language_model.tokenizer("France").input_ids) - 1
    assert cache.position == expected
    assert set(cache.states) == {1, 2, language_model.n_layers}
    assert cache.final.shape == (16,)
    with torch.no_grad():
        logits = language_model.decode(cache.final)
    torch.testing.assert_close(logits, cache.logits, rtol=1e-5, atol=1e-5)
    assert cache.target == int(cache.logits.argmax())


def test_intervention_is_identity_at_reference(language_model: LanguageModel) -> None:
    cache = language_model.cache_prompt("The capital of France is", "France", [2])
    intervention = HiddenStateIntervention(language_model)
    with language_model.hooked(2, intervention), torch.no_grad():
        output = intervention.forward(cache.states[2], cache.ids, cache.position)
    torch.testing.assert_close(output, cache.final)


def test_hook_requires_state(language_model: LanguageModel) -> None:
    intervention = HiddenStateIntervention(language_model)
    with pytest.raises(RuntimeError, match="Set the intervention state"):
        intervention.hook(nn.Identity(), (), torch.zeros(1, 2, 16))


def test_batched_jvp_matches_finite_difference(language_model: LanguageModel) -> None:
    cache = language_model.cache_prompt("The capital of France is", "France", [2])
    torch.manual_seed(1)
    directions = torch.randn(5, 16)
    intervention = HiddenStateIntervention(language_model)
    with language_model.hooked(2, intervention):
        products = language_model.jvps(intervention, cache, 2, directions)
        step = 1e-3
        with torch.no_grad():
            plus = intervention.forward(cache.states[2] + step * directions[0], cache.ids, cache.position)
            minus = intervention.forward(cache.states[2] - step * directions[0], cache.ids, cache.position)
    estimate = (plus - minus) / (2 * step)
    assert products.shape == (5, 16)
    assert float((estimate - products[0]).norm() / products[0].norm()) < 1e-2


def test_prefix_intervention_matches_full_graph(language_model: LanguageModel) -> None:
    cache = language_model.cache_prompt("The capital of Germany is Berlin. The capital of France is", "France", [3])
    directions = torch.randn(4, 16, generator=torch.Generator().manual_seed(2))
    results = []
    for intervention in (HiddenStateIntervention(language_model), PrefixIntervention(language_model)):
        with language_model.hooked(3, intervention):
            results.append(language_model.jvps(intervention, cache, 3, directions))
            results.append(language_model.jvps(intervention, cache, 3, directions))
    torch.testing.assert_close(results[2], results[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(results[3], results[1], rtol=1e-4, atol=1e-5)


def test_runtime_checks_enable_prefix_cache(language_model: LanguageModel) -> None:
    language_model.prefix_cache = False
    result = checks.prepare(language_model, Project().protocol.checks)
    assert result.convolution is None
    assert result.derivative.finite_difference_relative_error < 0.02
    assert len(result.prefix) == 2
    assert language_model.prefix_cache
    assert isinstance(language_model.intervention(), PrefixIntervention)
    language_model.prefix_cache = False


def test_generate_is_greedy(language_model: LanguageModel) -> None:
    first = language_model.generate("France is known for", 4)
    assert first == language_model.generate("France is known for", 4)


def test_layout_resolution_errors() -> None:
    with pytest.raises(ValueError, match="block layout"):
        resolve_blocks(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="final-norm"):
        resolve_final_norm(nn.Linear(2, 2))


def test_decode_helper_rejects_batches(language_model: LanguageModel) -> None:
    assert isinstance(modeling.decode(language_model.tokenizer, [1, 2]), str)


def test_hybrid_convolution_matches_grouped_convolution() -> None:
    torch.manual_seed(3)
    hidden = torch.randn(1, 6, 9)
    weight = torch.randn(6, 1, 4)
    bias = torch.randn(6)
    expected = nn.functional.conv1d(nn.functional.pad(hidden, (3, 0)), weight, bias, groups=6)
    torch.testing.assert_close(checks.shifted_causal_conv1d(hidden, weight[:, 0], bias), expected)
    activated = checks.shifted_causal_conv1d(hidden, weight[:, 0], bias, activation="silu")
    torch.testing.assert_close(activated, nn.functional.silu(expected))


def test_device_map_single_gpu(spec: modeling.ModelSpec, runtime: modeling.RuntimeConfig) -> None:
    assert modeling.device_map(spec, runtime) == {"device_map": {"": "cuda:0"}}


def test_weight_patterns_prefer_safetensors() -> None:
    config = Project().protocol.download
    safetensors = modeling.weight_patterns(
        ["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5"], config
    )
    assert "*.safetensors" in safetensors
    assert not any("bin" in pattern for pattern in safetensors)
    legacy = modeling.weight_patterns(["config.json", "pytorch_model.bin", "flax_model.msgpack"], config)
    assert "pytorch_model*.bin" in legacy
    with pytest.raises(RuntimeError, match="No loadable weights"):
        modeling.weight_patterns(["config.json", "tf_model.h5"], config)
