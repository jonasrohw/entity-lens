from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from entity_lens.config import DatasetConfig

RAW_URL = "https://raw.githubusercontent.com/{repository}/{commit}/data/{category}/{name}.json"
USER_AGENT = "entity-lens-reproduction/1.0"
TREE_URL = "https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1"


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    object: str


class RelationProperties(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    relation_type: str
    domain_name: str
    range_name: str
    symmetric: bool


class Relation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    key: str = ""
    name: str
    prompt_templates: tuple[str, ...] = Field(min_length=1)
    properties: RelationProperties
    samples: tuple[Sample, ...]

    @property
    def template(self) -> str:
        return self.prompt_templates[0]

    @property
    def domain(self) -> str:
        return self.properties.domain_name

    @property
    def cardinality(self) -> float:
        return len(self.samples) / len({sample.object for sample in self.samples})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        message = f"Refusing non-HTTPS download: {url}"
        raise ValueError(message)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload: bytes = response.read()
    return payload


def remote_paths(config: DatasetConfig) -> dict[str, str]:
    listing = json.loads(fetch(TREE_URL.format(repository=config.repository, commit=config.commit)))
    paths: dict[str, str] = {}
    for entry in listing["tree"]:
        parts = entry["path"].split("/")
        if len(parts) == 3 and parts[0] == "data" and parts[2].endswith(".json"):
            paths[parts[2].removesuffix(".json")] = parts[1]
    return paths


def download(config: DatasetConfig, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    categories = remote_paths(config)
    missing = set(config.sha256) - set(categories)
    if missing:
        message = f"Pinned relations are absent upstream: {sorted(missing)}"
        raise RuntimeError(message)
    written = []
    for name, expected in sorted(config.sha256.items()):
        target = destination / f"{name}.json"
        if not target.exists() or sha256_file(target) != expected:
            url = RAW_URL.format(
                repository=config.repository,
                commit=config.commit,
                category=categories[name],
                name=name,
            )
            payload = fetch(url)
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                message = f"{name}: downloaded SHA-256 {actual} does not match {expected}"
                raise RuntimeError(message)
            target.write_bytes(payload)
        written.append(target)
    return written


def verify(config: DatasetConfig, directory: Path) -> None:
    for name, expected in config.sha256.items():
        actual = sha256_file(directory / f"{name}.json")
        if actual != expected:
            message = f"{name}: SHA-256 {actual} does not match pinned {expected}"
            raise RuntimeError(message)


def load_relation(directory: Path, name: str, config: DatasetConfig) -> Relation:
    path = directory / f"{name}.json"
    actual = sha256_file(path)
    if actual != config.sha256[name]:
        message = f"{name}: SHA-256 {actual} does not match pinned {config.sha256[name]}"
        raise RuntimeError(message)
    relation = Relation.model_validate_json(path.read_bytes())
    return relation.model_copy(update={"key": name})


def all_relation_names(config: DatasetConfig) -> list[str]:
    return sorted(config.sha256)


def entity_relations(directory: Path, config: DatasetConfig) -> list[Relation]:
    relations = [load_relation(directory, name, config) for name in all_relation_names(config)]
    return [relation for relation in relations if relation.domain in config.entity_domains]
