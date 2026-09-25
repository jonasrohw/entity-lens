from __future__ import annotations

from pathlib import Path

import pytest
import torch

from entity_lens import checks, modeling
from entity_lens.config import ModelSpec, Project, RuntimeConfig
from entity_lens.modeling import LanguageModel

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def test_load_places_model_on_gpu(
    tmp_path: Path, language_model: LanguageModel, spec: ModelSpec, runtime: RuntimeConfig
) -> None:
    language_model.model.save_pretrained(tmp_path)
    language_model.tokenizer.save_pretrained(tmp_path)
    local = spec.model_copy(update={"hf_id": str(tmp_path)})
    loaded = modeling.load(local, runtime)
    assert all(parameter.device.type == "cuda" for parameter in loaded.model.parameters())
    assert loaded.output_device.type == "cuda"
    result = checks.prepare(loaded, Project().protocol.checks)
    assert result.derivative.finite_difference_relative_error < 0.02
    cache = loaded.cache_prompt("France is known for", "France", [2])
    assert cache.ids.device.type == "cuda"
