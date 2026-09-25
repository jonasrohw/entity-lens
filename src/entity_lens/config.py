from __future__ import annotations

import tomllib
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(Frozen):
    key: str
    label: str
    hf_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    lens_layer: int = Field(gt=0)
    lre_layers: tuple[int, ...]
    newline_prefix: bool
    year_trailing_space: Literal["auto", "always"]
    gpus: int = Field(ge=1)

    @model_validator(mode="after")
    def _lens_layer_is_candidate(self) -> ModelSpec:
        if self.lens_layer not in self.lre_layers:
            message = f"{self.key}: lens layer {self.lens_layer} is not an LRE candidate layer"
            raise ValueError(message)
        return self


class DatasetConfig(Frozen):
    repository: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    categories: tuple[str, ...]
    entity_domains: frozenset[str]
    sha256: dict[str, str]


class RuntimeConfig(Frozen):
    dtype: Literal["float32", "bfloat16"]
    attention: Literal["eager", "sdpa"]
    jvp_batch_size: int = Field(gt=0)
    torch_threads: int = Field(gt=0)
    gpu_memory_headroom_gib: int = Field(ge=0)
    idle_gpu_memory_mib: int = Field(ge=0)


Tolerance = tuple[float, float]


class DownloadConfig(Frozen):
    support_files: tuple[str, ...]
    weight_preference: tuple[tuple[str, ...], ...]


class ChecksConfig(Frozen):
    prompt: str
    subject: str
    context_demonstration: str
    context_repeats: int = Field(gt=0)
    context_query: str
    context_layer: int = Field(gt=0)
    hybrid_model_type: str
    min_directions: int = Field(gt=0)
    convolution_seed: int
    derivative_seed: int
    prefix_seed: int
    finite_difference_step: float = Field(gt=0)
    finite_difference_tolerance: float = Field(gt=0)
    batched_jvp_tolerance: float = Field(gt=0)
    decode: Tolerance
    convolution_jvp: Tolerance
    convolution_logits: Tolerance
    prefix: Tolerance


class PromptConfig(Frozen):
    neutral: str
    newline_prefix: str
    demonstration_separator: str

    @field_validator("neutral")
    @classmethod
    def _one_slot(cls, value: str) -> str:
        if value.count("{}") != 1:
            message = "The neutral prompt must contain exactly one subject slot"
            raise ValueError(message)
        return value

    def lens_prompt(self, model: ModelSpec, template: str | None = None) -> str:
        base = self.neutral if template is None else template
        return (self.newline_prefix if model.newline_prefix else "") + base


class SplitConfig(Frozen):
    n_demonstrations: int = Field(gt=0)
    n_references: int = Field(gt=0)
    lre_seeds: tuple[int, ...]
    lens_seeds: tuple[int, ...]


class LreConfig(Frozen):
    betas: tuple[float, ...]


class YearConfig(Frozen):
    relations: frozenset[str]
    ignored_first_token: str


class GenderConfig(Frozen):
    relation: str
    terms: dict[str, tuple[str, ...]]

    def alternatives(self, token: str) -> tuple[str, ...]:
        for group in self.terms.values():
            if token in group:
                return group
        return ()


class AnswerPriorConfig(Frozen):
    contexts: tuple[str, ...]


class ReadoutConfig(Frozen):
    top_k_tokens: int = Field(gt=0)
    logit_lens_top_k_ids: int = Field(gt=0)
    anchored_betas: tuple[float, ...]


class ProtocolConfig(Frozen):
    dataset: DatasetConfig
    runtime: RuntimeConfig
    download: DownloadConfig
    checks: ChecksConfig
    prompts: PromptConfig
    split: SplitConfig
    lre: LreConfig
    years: YearConfig
    gender: GenderConfig
    answer_prior: AnswerPriorConfig
    readouts: ReadoutConfig


class ModelsConfig(Frozen):
    models: tuple[ModelSpec, ...]

    @model_validator(mode="after")
    def _unique_keys(self) -> ModelsConfig:
        keys = [model.key for model in self.models]
        if len(set(keys)) != len(keys):
            message = "Model keys must be unique"
            raise ValueError(message)
        return self

    def get(self, key: str) -> ModelSpec:
        for model in self.models:
            if model.key == key:
                return model
        message = f"Unknown model {key!r}"
        raise KeyError(message)


class PromptVariantConfig(Frozen):
    model: str
    prompts: tuple[str, ...]


class ProminenceConfig(Frozen):
    model: str
    dtype: Literal["float32", "bfloat16"]
    attention: Literal["eager", "sdpa"]
    max_new_tokens: int = Field(gt=0)
    prompts: tuple[str, ...]
    relations: tuple[str, ...]
    word_pattern: str
    min_word_length: int = Field(gt=0)
    min_lens_token_length: int = Field(gt=0)
    shards: int = Field(gt=0)
    stop_words: frozenset[str]


class ReferenceSetConfig(Frozen):
    model: str
    relation: str
    answer: str
    cross_relations: tuple[str, str]
    n_references: int = Field(gt=0)
    seed: int
    example_subject: str
    top_k: int = Field(gt=0)
    subset_size: int = Field(gt=0)


class SwapFilterConfig(Frozen):
    dtype: Literal["float32", "bfloat16"]
    attention: Literal["eager", "sdpa"]
    max_new_tokens: int = Field(gt=0)
    match_words: int = Field(gt=0)
    aliases: dict[str, tuple[str, ...]]


class SwapConfig(Frozen):
    model: str
    beta: float
    anchor_draws: int = Field(gt=0)
    anchor_references: int = Field(gt=0)
    athlete_anchor_pool: int = Field(gt=0)
    variance_grid: int = Field(gt=0)
    seed: int
    filter: SwapFilterConfig
    currency_betas: tuple[float, ...]
    country_relations: tuple[str, ...]
    person_pair: tuple[str, str]
    country_pair: tuple[str, str]
    currency_relation: str
    composition_relation: str
    composition_object: str
    position_sports: dict[str, str]
    athlete_stream_after_benchmark: bool
    wikipedia_list: str
    wikipedia_sha256: str
    excluded_countries: tuple[str, ...]
    shared_subject_relations: tuple[str, str]


class PrimaryConfig(Frozen):
    model: str


class ResidualConfig(Frozen):
    exclude_year_relations: bool
    literal_scoring: tuple[str, ...]


class CollapseConfig(Frozen):
    threshold: float
    weak_token_relation: str
    weak_token: str
    filtered_relations: tuple[str, ...]


class NumericObjectConfig(Frozen):
    model: str
    example_object: str


class AnchoredConfig(Frozen):
    train_relations: int = Field(gt=0)
    splits: int = Field(gt=0)
    seed: int
    min_distinct: int = Field(gt=0)


class BootstrapConfig(Frozen):
    seed: int
    resamples: int = Field(gt=0)
    level: float = Field(gt=0, lt=1)


class CompetitionConfig(Frozen):
    examples: tuple[tuple[str, str], ...]


class ExperimentsConfig(Frozen):
    primary: PrimaryConfig
    prompt_variants: PromptVariantConfig
    prominence: ProminenceConfig
    reference_sets: ReferenceSetConfig
    swap: SwapConfig
    residuals: ResidualConfig
    collapse: CollapseConfig
    numeric_objects: NumericObjectConfig
    anchored: AnchoredConfig
    bootstrap: BootstrapConfig
    competition: CompetitionConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENTITY_LENS_", env_file=".env", extra="ignore")

    config_dir: Path = Path("configs")
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    corpus_dir: Path = Path("data/olmo-corpus")
    wikipedia_dir: Path = Path("data/wikipedia")
    workers: int = 32


def _read_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


class Project:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def load(self, name: str) -> dict[str, object]:
        return _read_toml(self.settings.config_dir / f"{name}.toml")

    @cached_property
    def protocol(self) -> ProtocolConfig:
        return ProtocolConfig.model_validate(self.load("protocol"))

    @cached_property
    def experiments(self) -> ExperimentsConfig:
        return ExperimentsConfig.model_validate(self.load("experiments"))

    @cached_property
    def models(self) -> ModelsConfig:
        return ModelsConfig.model_validate(self.load("models"))

    @property
    def artifacts(self) -> Path:
        return self.settings.artifacts_dir

    @property
    def data(self) -> Path:
        return self.settings.data_dir

    @property
    def cache(self) -> Path:
        return self.settings.cache_dir
