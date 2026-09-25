from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from entity_lens import data
from entity_lens.config import DatasetConfig, Project

TREE = {
    "tree": [
        {"path": "data/factual/alpha.json"},
        {"path": "data/bias/beta.json"},
        {"path": "hparams/gpt2/alpha.json"},
        {"path": "README.md"},
    ]
}


def payload(domain: str) -> bytes:
    relation = {
        "name": "x",
        "prompt_templates": ["{} is"],
        "properties": {"relation_type": "factual", "domain_name": domain, "range_name": "y", "symmetric": False},
        "samples": [{"subject": "a", "object": "b"}, {"subject": "c", "object": "b"}],
    }
    return json.dumps(relation).encode()


def config_for(files: dict[str, bytes]) -> DatasetConfig:
    return DatasetConfig(
        repository="owner/repo",
        commit="1" * 40,
        categories=("factual", "bias"),
        entity_domains=frozenset({"person"}),
        sha256={name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    )


def test_download_verifies_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"alpha": payload("person"), "beta": payload("word")}
    calls: list[str] = []

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        if url.startswith("https://api.github.com"):
            return json.dumps(TREE).encode()
        name = url.rsplit("/", 1)[1].removesuffix(".json")
        return files[name]

    monkeypatch.setattr(data, "fetch", fake_fetch)
    config = config_for(files)
    written = data.download(config, tmp_path)
    assert [p.name for p in written] == ["alpha.json", "beta.json"]
    assert any("/data/factual/alpha.json" in url for url in calls)
    data.verify(config, tmp_path)
    count = len(calls)
    data.download(config, tmp_path)
    assert len(calls) == count + 1
    relations = data.entity_relations(tmp_path, config)
    assert [r.key for r in relations] == ["alpha"]
    assert relations[0].cardinality == 2.0
    assert relations[0].template == "{} is"


def test_download_rejects_tampered_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"alpha": payload("person")}
    config = config_for(files)

    def fake_fetch(url: str) -> bytes:
        if url.startswith("https://api.github.com"):
            return json.dumps(TREE).encode()
        return b"{}"

    monkeypatch.setattr(data, "fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="does not match"):
        data.download(config, tmp_path)


def test_download_requires_upstream_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for({"gamma": payload("person")})
    monkeypatch.setattr(data, "fetch", lambda url: json.dumps(TREE).encode())
    with pytest.raises(RuntimeError, match="absent upstream"):
        data.download(config, tmp_path)


def test_load_relation_rejects_modified_file(tmp_path: Path) -> None:
    files = {"alpha": payload("person")}
    config = config_for(files)
    (tmp_path / "alpha.json").write_bytes(payload("place"))
    with pytest.raises(RuntimeError, match="does not match pinned"):
        data.load_relation(tmp_path, "alpha", config)
    with pytest.raises(RuntimeError, match="does not match pinned"):
        data.verify(config, tmp_path)


def test_fetch_requires_https() -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        data.fetch("http://example.com")


def test_pinned_dataset_selects_paper_relations() -> None:
    project = Project()
    directory = project.data / "relations"
    if not directory.exists():
        pytest.skip("dataset not downloaded")
    relations = data.entity_relations(directory, project.protocol.dataset)
    assert len(data.all_relation_names(project.protocol.dataset)) == 47
    assert len(relations) == 28
    assert sum(len(r.samples) for r in relations) == 9747
