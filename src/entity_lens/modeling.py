from __future__ import annotations

import copy
import fnmatch
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import logging

from entity_lens.config import DownloadConfig, ModelSpec, RuntimeConfig


def decode(tokenizer: PreTrainedTokenizerBase, ids: list[int], *, skip_special_tokens: bool = False) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    if not isinstance(text, str):
        message = "Expected the tokenizer to decode one sequence"
        raise TypeError(message)
    return text


class Intervention(Protocol):
    def forward(self, hidden: Tensor, ids: Tensor, position: int) -> Tensor: ...

    def hook(self, module: nn.Module, inputs: object, output: Any) -> Any: ...


@dataclass(frozen=True)
class PromptCache:
    ids: Tensor
    position: int
    states: dict[int, Tensor]
    final: Tensor
    logits: Tensor

    @property
    def target(self) -> int:
        return int(self.logits.argmax())


def _states(output: Any) -> Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, Tensor):
        message = f"Unexpected block output {type(value).__name__}"
        raise TypeError(message)
    return value


def _replace(output: Any, states: Tensor) -> Any:
    return (states, *output[1:]) if isinstance(output, tuple) else states


def _parameter_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def resolve_blocks(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.language_model.layers"):
        module: Any = model
        for name in path.split("."):
            module = getattr(module, name, None)
        if isinstance(module, nn.ModuleList):
            return module
    message = f"Unrecognized transformer block layout: {type(model).__name__}"
    raise ValueError(message)


def resolve_final_norm(model: nn.Module) -> nn.Module:
    paths = ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.language_model.norm")
    for path in paths:
        module: Any = model
        for name in path.split("."):
            module = getattr(module, name, None)
        if isinstance(module, nn.Module):
            return module
    message = f"Unrecognized final-norm layout: {type(model).__name__}"
    raise ValueError(message)


class HiddenStateIntervention:
    def __init__(self, lm: LanguageModel) -> None:
        self.lm = lm
        self.hidden: Tensor | None = None
        self.position: int | None = None

    def hook(self, module: nn.Module, inputs: object, output: Any) -> Any:
        if self.hidden is None or self.position is None:
            message = "Set the intervention state before a hooked forward pass"
            raise RuntimeError(message)
        states = _states(output).clone()
        states[0, self.position, :] = self.hidden
        return _replace(output, states)

    def _final_state(self, ids: Tensor, **kwargs: Any) -> Tensor:
        captured: list[Tensor] = []

        def capture(module: nn.Module, inputs: object, output: Any) -> None:
            captured.append(_states(output)[0, -1])

        handle = self.lm.blocks[-1].register_forward_hook(capture)
        try:
            self.lm.model(input_ids=ids, **kwargs)
        finally:
            handle.remove()
        return captured[0]

    def forward(self, hidden: Tensor, ids: Tensor, position: int) -> Tensor:
        self.hidden = hidden
        self.position = position
        return self._final_state(ids)


def _keep_recurrent_state(state: Any, *args: object, **kwargs: object) -> Any:
    return state


class PrefixIntervention(HiddenStateIntervention):
    def __init__(self, lm: LanguageModel) -> None:
        super().__init__(lm)
        self.building_prefix = False
        self.cached: tuple[Tensor, int] | None = None
        self.prefix: Any = None

    def hook(self, module: nn.Module, inputs: object, output: Any) -> Any:
        if self.building_prefix:
            return output
        return super().hook(module, inputs, output)

    def forward(self, hidden: Tensor, ids: Tensor, position: int) -> Tensor:
        if self.cached is None or self.cached[0] is not ids or self.cached[1] != position:
            self.building_prefix = True
            try:
                with torch.no_grad():
                    self.prefix = (
                        self.lm.model(input_ids=ids[:, :position], use_cache=True).past_key_values if position else None
                    )
            finally:
                self.building_prefix = False
            self.cached = (ids, position)
        self.hidden = hidden
        self.position = 0
        cache = copy.deepcopy(self.prefix)
        if cache is not None:
            cache.update_recurrent_state = _keep_recurrent_state
        return self._final_state(ids[:, position:], use_cache=True, past_key_values=cache)


@dataclass
class LanguageModel:
    spec: ModelSpec
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    runtime: RuntimeConfig
    blocks: nn.ModuleList = field(init=False)
    norm: nn.Module = field(init=False)
    prefix_cache: bool = False

    def __post_init__(self) -> None:
        self.blocks = resolve_blocks(self.model)
        self.norm = resolve_final_norm(self.model)

    @property
    def head(self) -> nn.Module:
        accessor: Callable[[], nn.Module] = self.model.get_output_embeddings
        return accessor()

    @property
    def input_device(self) -> torch.device:
        embedding: nn.Module = self.model.get_input_embeddings()
        return _parameter_device(embedding)

    @property
    def output_device(self) -> torch.device:
        return _parameter_device(self.head)

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def encode(self, text: str) -> Tensor:
        ids: Tensor = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.input_device)

    def subject_position(self, prompt: str, subject: str, n_tokens: int) -> int:
        end = prompt.rindex(subject) + len(subject)
        return min(len(self.tokenizer(prompt[:end]).input_ids) - 1, n_tokens - 1)

    def decode(self, hidden: Tensor) -> Tensor:
        normalized = self.norm(hidden.to(_parameter_device(self.norm)))
        logits: Tensor = self.head(normalized.to(self.output_device)).float()
        cap = getattr(self.model.config.get_text_config(), "final_logit_softcapping", None)
        return cap * torch.tanh(logits / cap) if cap is not None else logits

    def token(self, token_id: int) -> str:
        return decode(self.tokenizer, [token_id]).strip()

    def generate(self, prompt: str, max_new_tokens: int, *, skip_special_tokens: bool = False) -> str:
        ids = self.encode(prompt)
        generator: Any = self.model
        with torch.no_grad():
            output = generator.generate(
                input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        return decode(self.tokenizer, output[0, ids.shape[1] :].tolist(), skip_special_tokens=skip_special_tokens)

    def next_token_logits(self, prompt: str) -> Tensor:
        with torch.no_grad():
            logits: Tensor = self.model(input_ids=self.encode(prompt)).logits[0, -1]
        return logits.detach().to(self.output_device)

    def cache_prompt(self, prompt: str, subject: str, layers: Sequence[int]) -> PromptCache:
        ids = self.encode(prompt)
        position = self.subject_position(prompt, subject, ids.shape[1])
        captured: dict[int, Tensor] = {}
        final: list[Tensor] = []

        def capture(layer: int) -> Callable[[nn.Module, object, Any], None]:
            def hook(module: nn.Module, inputs: object, output: Any) -> None:
                states = _states(output)
                captured[layer] = states[0, position].detach().clone()
                if layer == self.n_layers:
                    final.append(states[0, -1].detach().clone())

            return hook

        handles = [
            self.blocks[layer - 1].register_forward_hook(capture(layer))
            for layer in sorted(set(layers) | {self.n_layers})
        ]
        try:
            with torch.no_grad():
                logits = self.model(input_ids=ids).logits[0, -1].detach().to(self.output_device)
        finally:
            for handle in handles:
                handle.remove()
        return PromptCache(ids=ids, position=position, states=captured, final=final[0], logits=logits)

    def intervention(self) -> HiddenStateIntervention:
        return PrefixIntervention(self) if self.prefix_cache else HiddenStateIntervention(self)

    @contextmanager
    def hooked(self, layer: int, intervention: Intervention) -> Iterator[None]:
        handle = self.blocks[layer - 1].register_forward_hook(intervention.hook)
        try:
            yield
        finally:
            handle.remove()

    def jvps(self, intervention: Intervention, reference: PromptCache, layer: int, directions: Tensor) -> Tensor:
        hidden = reference.states[layer].detach().clone().requires_grad_()
        output = intervention.forward(hidden, reference.ids, reference.position)
        cotangent = torch.zeros_like(output, requires_grad=True)
        (gradient,) = torch.autograd.grad(output, hidden, grad_outputs=cotangent, create_graph=True)
        batch = self.runtime.jvp_batch_size
        chunks = []
        for start in range(0, len(directions), batch):
            (product,) = torch.autograd.grad(
                gradient,
                cotangent,
                grad_outputs=directions[start : start + batch],
                is_grads_batched=True,
                retain_graph=start + batch < len(directions),
            )
            chunks.append(product.detach())
        return torch.cat(chunks)


def device_map(spec: ModelSpec, runtime: RuntimeConfig) -> dict[str, Any]:
    if spec.gpus == 1 or torch.cuda.device_count() == 1:
        return {"device_map": {"": "cuda:0"}}
    budget = {}
    for index in range(torch.cuda.device_count()):
        total = int(torch.cuda.get_device_properties(index).total_memory / 2**30)
        budget[index] = f"{total - runtime.gpu_memory_headroom_gib}GiB"
    return {"device_map": "balanced", "max_memory": budget}


def load_tokenizer(spec: ModelSpec, *, local_files_only: bool = True) -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        spec.hf_id, revision=spec.revision, local_files_only=local_files_only
    )
    return tokenizer


def load(spec: ModelSpec, runtime: RuntimeConfig, *, local_files_only: bool = True) -> LanguageModel:
    torch.set_num_threads(runtime.torch_threads)
    disable_progress_bar: Callable[[], None] = logging.disable_progress_bar
    disable_progress_bar()
    tokenizer = load_tokenizer(spec, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        revision=spec.revision,
        local_files_only=local_files_only,
        dtype=getattr(torch, runtime.dtype),
        attn_implementation=runtime.attention,
        **device_map(spec, runtime),
    )
    model.train(mode=False)
    model.requires_grad_(requires_grad=False)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        message = f"{spec.key}: every parameter must be placed on a GPU"
        raise RuntimeError(message)
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    return LanguageModel(spec=spec, model=model, tokenizer=tokenizer, runtime=runtime)


def weight_patterns(files: Sequence[str], download: DownloadConfig) -> list[str]:
    for patterns in download.weight_preference:
        if any(fnmatch.fnmatch(name, pattern) for name in files for pattern in patterns):
            return [*download.support_files, *patterns]
    message = f"No loadable weights among {sorted(files)}"
    raise RuntimeError(message)


def download(spec: ModelSpec, config: DownloadConfig) -> str:
    from huggingface_hub import list_repo_files, snapshot_download

    files = list_repo_files(spec.hf_id, revision=spec.revision)
    path: str = snapshot_download(spec.hf_id, revision=spec.revision, allow_patterns=weight_patterns(files, config))
    return path
