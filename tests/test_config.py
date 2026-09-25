from __future__ import annotations

import pytest
from pydantic import ValidationError

from entity_lens.config import GenderConfig, ModelsConfig, ModelSpec, Project, PromptConfig


def test_repository_configs_validate() -> None:
    project = Project()
    assert len(project.models.models) == 7
    assert project.protocol.split.n_demonstrations == 8
    assert project.experiments.primary.model in {m.key for m in project.models.models}
    assert project.experiments.prompt_variants.model in {m.key for m in project.models.models}


def test_every_experiment_model_is_configured() -> None:
    project = Project()
    experiments = project.experiments
    for model in (
        experiments.prominence.model,
        experiments.reference_sets.model,
        experiments.swap.model,
        experiments.numeric_objects.model,
    ):
        project.models.get(model)


def test_lens_layer_must_be_candidate(spec: ModelSpec) -> None:
    with pytest.raises(ValidationError, match="not an LRE candidate"):
        ModelSpec.model_validate(spec.model_dump() | {"lens_layer": 9})


def test_revision_must_be_pinned(spec: ModelSpec) -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(spec.model_dump() | {"revision": "main"})


def test_model_keys_unique(spec: ModelSpec) -> None:
    with pytest.raises(ValidationError, match="unique"):
        ModelsConfig(models=(spec, spec))


def test_unknown_model(spec: ModelSpec) -> None:
    with pytest.raises(KeyError):
        ModelsConfig(models=(spec,)).get("missing")


def test_prompt_requires_one_slot() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        PromptConfig(neutral="{} {}", newline_prefix="\n", demonstration_separator="\n")


def test_lens_prompt_newline(spec: ModelSpec) -> None:
    prompts = PromptConfig(neutral="{} is known for", newline_prefix="\n", demonstration_separator="\n")
    assert prompts.lens_prompt(spec) == "{} is known for"
    newline = spec.model_copy(update={"newline_prefix": True})
    assert prompts.lens_prompt(newline) == "\n{} is known for"
    assert prompts.lens_prompt(newline, "{} is") == "\n{} is"


def test_gender_alternatives() -> None:
    gender = GenderConfig(relation="name_gender", terms={"woman": ("woman", "her"), "man": ("man", "his")})
    assert gender.alternatives("her") == ("woman", "her")
    assert gender.alternatives("his") == ("man", "his")
    assert gender.alternatives("dog") == ()


def test_models_toml_matches_paper_layers() -> None:
    models = {m.key: m for m in Project().models.models}
    assert models["olmo3_7b"].lens_layer == 17
    assert models["qwen3_8b_base"].lens_layer == 19
    assert models["qwen3_8_27b"].gpus == 2
    assert {k for k, m in models.items() if m.newline_prefix} == {"gptj", "pythia6_9b", "qwen3_8b_base", "qwen3_8_27b"}
