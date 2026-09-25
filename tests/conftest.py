from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

import pytest
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from entity_lens.config import ModelSpec, Project, RuntimeConfig, Settings
from entity_lens.modeling import LanguageModel

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
GENDER_WORDS = [
    "woman",
    "her",
    "herself",
    "she",
    "hers",
    "female",
    "girl",
    "man",
    "him",
    "himself",
    "he",
    "his",
    "male",
    "boy",
]
CORPUS = [
    "France is known for wine and Paris. Spain is known for Madrid and football.",
    "The capital city of France is Paris. The capital city of Spain is Madrid.",
    "Italy Rome Germany Berlin Japan Tokyo Kenya Nairobi Peru Lima Chile Santiago",
    "Here are some facts about France. First, it is best known for cheese.",
    " ".join(GENDER_WORDS) + " Anna Bob Carla Dan Eva Finn Gina Hugo Iris Jon",
    "Facts about Information Entity can be described as is the city of 1743 1905",
    "The capital of Germany is Berlin. The capital of France is Paris.",
]
COUNTRIES = {
    "France": "Paris",
    "Spain": "Madrid",
    "Italy": "Rome",
    "Germany": "Berlin",
    "Japan": "Tokyo",
    "Kenya": "Nairobi",
    "Peru": "Lima",
    "Chile": "Santiago",
}
ATHLETES = {
    "Anna": "goalkeeper",
    "Bob": "pitcher",
    "Carla": "forward",
    "Dan": "center",
    "Eva": "defender",
    "Finn": "catcher",
    "Gina": "midfielder",
    "Kai": "winger",
}
SPORTS = {
    "Anna": "soccer",
    "Bob": "baseball",
    "Carla": "soccer",
    "Dan": "hockey",
    "Eva": "soccer",
    "Hugo": "basketball",
    "Iris": "soccer",
    "Lea": "tennis",
}
PARENTS = {"Anna": "Bob", "Carla": "Dan", "Eva": "Finn", "Gina": "Hugo", "Iris": "Jon"}
NAMES = {
    "Anna": "woman",
    "Bob": "man",
    "Carla": "woman",
    "Dan": "man",
    "Eva": "woman",
    "Finn": "man",
    "Gina": "woman",
    "Hugo": "man",
    "Iris": "woman",
    "Jon": "man",
    "Kira": "woman",
    "Leo": "man",
    "Mia": "woman",
    "Nils": "man",
    "Olga": "woman",
    "Paul": "man",
    "Rosa": "woman",
    "Sven": "man",
    "Tina": "woman",
    "Udo": "man",
}


def build_tokenizer() -> PreTrainedTokenizerFast:
    trainer = ByteLevelBPETokenizer()
    trainer.train_from_iterator(CORPUS * 4, vocab_size=400, min_frequency=1, special_tokens=["<|endoftext|>"])
    for word in GENDER_WORDS:
        trainer.add_tokens([f" {word}"])
    return PreTrainedTokenizerFast(tokenizer_object=trainer, bos_token="<|endoftext|>", eos_token="<|endoftext|>")


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    return build_tokenizer()


@pytest.fixture(scope="session")
def runtime() -> RuntimeConfig:
    protocol = tomllib.loads((ROOT / "configs" / "protocol.toml").read_text())
    return RuntimeConfig.model_validate(protocol["runtime"] | {"jvp_batch_size": 3})


@pytest.fixture(scope="session")
def spec() -> ModelSpec:
    return ModelSpec(
        key="tiny",
        label="Tiny",
        hf_id="tiny/llama",
        revision="0" * 40,
        lens_layer=2,
        lre_layers=(1, 2, 3),
        newline_prefix=False,
        year_trailing_space="auto",
        gpus=1,
    )


@pytest.fixture(scope="session")
def language_model(tokenizer: PreTrainedTokenizerFast, spec: ModelSpec, runtime: RuntimeConfig) -> LanguageModel:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=12,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
        use_cache=False,
    )
    model = LlamaForCausalLM(config).to(torch.float32)
    model.train(mode=False)
    model.requires_grad_(requires_grad=False)
    return LanguageModel(spec=spec, model=model, tokenizer=tokenizer, runtime=runtime)


def relation_payload(name: str, domain: str, template: str, samples: dict[str, str]) -> dict[str, object]:
    return {
        "name": name.replace("_", " "),
        "prompt_templates": [template, template],
        "prompt_templates_zs": [template],
        "properties": {"relation_type": "factual", "domain_name": domain, "range_name": "thing", "symmetric": False},
        "samples": [{"subject": s, "object": o} for s, o in samples.items()],
    }


def write_relations(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "country_capital_city": relation_payload(
            "country_capital_city", "country", "The capital city of {} is", COUNTRIES
        ),
        "country_largest_city": relation_payload(
            "country_largest_city", "country", "The largest city in {} is", COUNTRIES
        ),
        "name_gender": relation_payload("name_gender", "name", "{} is a", NAMES),
        "country_currency": relation_payload(
            "country_currency", "country", "The official currency of {} is the", dict.fromkeys(COUNTRIES, "euro")
        ),
        "country_language": relation_payload(
            "country_language", "country", "People in {} speak", dict.fromkeys(COUNTRIES, "French")
        ),
        "person_plays_position_in_sport": relation_payload(
            "person_plays_position_in_sport", "person", "{} plays in the position of a", ATHLETES
        ),
        "person_plays_pro_sport": relation_payload("person_plays_pro_sport", "person", "{} plays the sport of", SPORTS),
        "person_father": relation_payload("person_father", "person", "{}'s father is", PARENTS),
        "person_mother": relation_payload(
            "person_mother", "person", "{}'s mother is", {k: v + "a" for k, v in PARENTS.items()}
        ),
        "adj_antonym": relation_payload(
            "adj_antonym", "adjective", "The opposite of {} is", {"big": "small", "hot": "cold"}
        ),
    }
    hashes = {}
    for name, payload in payloads.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=4))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def make_project(tmp_path: Path) -> Project:
    config_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", config_dir)
    hashes = write_relations(tmp_path / "data" / "relations")
    protocol = (config_dir / "protocol.toml").read_text()
    head, rest = protocol.split("[dataset.sha256]\n", 1)
    tail = rest[rest.index("\n\n") :]
    lines = "\n".join(f'{name} = "{digest}"' for name, digest in sorted(hashes.items()))
    tail = tail.replace("n_demonstrations = 8", "n_demonstrations = 2").replace("n_references = 8", "n_references = 3")
    tail = tail.replace("jvp_batch_size = 32", "jvp_batch_size = 3")
    (config_dir / "protocol.toml").write_text(head + "[dataset.sha256]\n" + lines + tail)
    settings = Settings(
        config_dir=config_dir,
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        corpus_dir=tmp_path / "corpus",
        wikipedia_dir=tmp_path / "wikipedia",
        workers=2,
    )
    return Project(settings)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return make_project(tmp_path)


def configure(project: Project, section: str, **values: object) -> None:
    experiments = project.experiments
    updated = getattr(experiments, section).model_copy(update=values)
    project.__dict__["experiments"] = experiments.model_copy(update={section: updated})


@pytest.fixture(scope="session")
def evaluated(tmp_path_factory: pytest.TempPathFactory, language_model: LanguageModel) -> Project:
    from entity_lens import modeling, pipeline

    project = make_project(tmp_path_factory.mktemp("evaluated"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modeling, "load", lambda *args, **kwargs: language_model)
        for key in ("olmo3_7b", "qwen3_8b_base"):
            pipeline.evaluate_model(project, key)
    language_model.prefix_cache = False
    return project


@pytest.fixture
def loaded(language_model: LanguageModel, monkeypatch: pytest.MonkeyPatch) -> LanguageModel:
    from entity_lens import modeling

    monkeypatch.setattr(modeling, "load", lambda *args, **kwargs: language_model)
    return language_model
