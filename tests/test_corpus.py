from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import orjson
import pytest
import zstandard

from entity_lens import corpus
from entity_lens.config import Project, Settings
from entity_lens.data import Relation, RelationProperties, Sample

REPOSITORY = Path(__file__).resolve().parents[1]
ALPHA = "test/alpha-mix"
BETA = "test/beta-mix"
ALPHA_REVISION = "a" * 40
BETA_REVISION = "b" * 40
CONFIG = f"""
text_field = "text"
casefold = true
data_directory = "data"
duplicate_suffix_pattern = "[0-9]+"
checkpoint_seconds = 0
progress_seconds = 0.05
tasks_per_worker = 1
read_buffer_bytes = 64
hash_block_bytes = 7
download_workers = 2
download_attempts = 2
retry_base_seconds = 0
retry_max_seconds = 0

[formats]
".jsonl.zst" = "zstd"
".jsonl.gz" = "gzip"
".jsonl" = "plain"

[[repos]]
repo_id = "{ALPHA}"
revision = "{ALPHA_REVISION}"
split_suffixes = [".jsonl.zst"]

[[repos]]
repo_id = "{BETA}"
revision = "{BETA_REVISION}"
"""

SAMPLES = {
    "city_in_country": [
        ("Paris", "France"),
        ("Paris", "France"),
        ("Lyon", "France"),
        ("New York City", "United States"),
    ],
    "country_capital_city": [("France", "Paris"), ("Japan", "Tokyo")],
    "same_name": [("Tokyo", "TOKYO")],
}


def relation(key: str, pairs: list[tuple[str, str]]) -> Relation:
    properties = RelationProperties(relation_type="test", domain_name="city", range_name="country", symmetric=False)
    return Relation(
        key=key,
        name=key,
        prompt_templates=("{} is in",),
        properties=properties,
        samples=tuple(Sample(subject=subject, object=obj) for subject, obj in pairs),
    )


RELATIONS = [relation(key, pairs) for key, pairs in SAMPLES.items()]


def jsonl(documents: list[object]) -> bytes:
    return b"".join(orjson.dumps(document) + b"\n" for document in documents)


def zstd_frames(*chunks: bytes) -> bytes:
    compressor = zstandard.ZstdCompressor()
    return b"".join(compressor.compress(chunk) for chunk in chunks)


def documents(*texts: str) -> list[object]:
    return [{"text": text, "id": index} for index, text in enumerate(texts)]


ALPHA_MAIN = documents("Paris is the capital of France.", "Visit PARIS, paris and Lyon!", "Parisian food")
ALPHA_TAIL = documents("Tokyo, Japan", "tokyo_station is in japan", "New  York\nCity, United States")
ALPHA_SECOND = documents("Nothing here", "Lyon; France", "Paris France")
BETA_GZIP = documents("Japan and France", "Tokyo")
BETA_PLAIN = documents("New York City", "the united states of america", "Paris (France)")

FILES: dict[tuple[str, str], bytes] = {
    (ALPHA, ".gitattributes"): b"* binary\n",
    (ALPHA, "README.md"): b"# alpha\n",
    (ALPHA, "data/web/shard-0.jsonl.zst"): zstd_frames(jsonl(ALPHA_MAIN), b"\n  \n" + jsonl(ALPHA_TAIL)),
    (ALPHA, "data/web/shard-0.jsonl.zst12345"): b"duplicate upload",
    (ALPHA, "data/web/shard-1.jsonl.zst"): zstd_frames(jsonl(ALPHA_SECOND)),
    (ALPHA, "data/pdfs/extra.jsonl.gz"): gzip.compress(jsonl(documents("Paris"))),
    (BETA, "README.md"): b"# beta\n",
    (BETA, "mix.png"): b"\x89PNG",
    (BETA, "data/books/part.jsonl.gz"): gzip.compress(jsonl(BETA_GZIP)),
    (BETA, "data/books/plain.jsonl"): jsonl(BETA_PLAIN),
}
ALL_DOCUMENTS = [*ALPHA_MAIN, *ALPHA_TAIL, *ALPHA_SECOND, *BETA_GZIP, *BETA_PLAIN]


def lfs(path: str) -> bool:
    return path.startswith("data/")


def item(repo: str, path: str) -> dict[str, object]:
    payload = FILES[repo, path]
    if lfs(path):
        return {
            "path": path,
            "size": len(payload),
            "algorithm": "sha256",
            "digest": hashlib.sha256(payload).hexdigest(),
        }
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False).hexdigest()
    return {"path": path, "size": len(payload), "algorithm": "git-sha1", "digest": blob}


def manifest_json(files: dict[tuple[str, str], bytes] | None = None) -> dict[str, object]:
    listing = FILES if files is None else files
    return {
        "model": "test",
        "repos": [
            {
                "repo_id": repo,
                "revision": revision,
                "files": [item(repo, path) for owner, path in listing if owner == repo],
            }
            for repo, revision in ((ALPHA, ALPHA_REVISION), (BETA, BETA_REVISION))
        ],
    }


def write_corpus(root: Path) -> None:
    for (repo, path), payload in FILES.items():
        target = root / repo.split("/")[-1] / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (root / "manifest.json").write_text(json.dumps(manifest_json()))


def make_project(tmp_path: Path, name: str = "run", corpus_dir: Path | None = None) -> Project:
    configs = tmp_path / name / "configs"
    configs.mkdir(parents=True)
    (configs / "corpus.toml").write_text(CONFIG)
    shutil.copy(REPOSITORY / "configs" / "protocol.toml", configs / "protocol.toml")
    settings = Settings(
        config_dir=configs,
        artifacts_dir=tmp_path / name / "artifacts",
        data_dir=tmp_path / name / "data",
        cache_dir=tmp_path / name / "cache",
        corpus_dir=corpus_dir or tmp_path / "corpus",
    )
    return Project(settings)


@pytest.fixture(autouse=True)
def fake_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus.data, "entity_relations", lambda *_: RELATIONS)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    write_corpus(tmp_path / "corpus")
    return make_project(tmp_path)


def brute_force(texts: list[str]) -> dict[tuple[str, str, str], tuple[int, int, int]]:
    words = corpus.vocabulary(RELATIONS, casefold=True)
    matcher = corpus.Matcher(corpus.MatcherSpec(terms=words.terms, pairs=words.pairs, casefold=True))
    found = [{words.terms[index] for index in matcher.find(text)} for text in texts]
    expected = {}
    for fact in words.facts:
        subject, obj = words.terms[fact.subject_term], words.terms[fact.object_term]
        expected[fact.relation, fact.subject, fact.object] = (
            sum(subject in terms for terms in found),
            sum(obj in terms for terms in found),
            sum(subject in terms and obj in terms for terms in found),
        )
    return expected


def read_facts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_repository_config_validates() -> None:
    config = corpus.settings(Project(Settings(config_dir=REPOSITORY / "configs")))
    assert [repo.stage for repo in config.repos] == [
        "dolma3_mix-6T-1025-7B",
        "dolma3_dolmino_mix-100B-1025",
        "dolma3_longmino_mix-50B-1025",
    ]
    assert config.text_field == "text"


def test_normalize() -> None:
    assert corpus.normalize("  Café \t\n  STRASSE  ") == "café strasse"
    assert corpus.normalize("Straße") == "strasse"
    assert corpus.normalize(" Mixed  Case ", casefold=False) == "Mixed Case"
    assert corpus.normalize("Å") == corpus.normalize("Å") == "å"


def test_vocabulary() -> None:
    words = corpus.vocabulary(RELATIONS, casefold=True)
    assert list(words.terms) == sorted(words.terms)
    assert set(words.terms) == {"france", "japan", "lyon", "new york city", "paris", "tokyo", "united states"}
    assert len(words.facts) == 6
    paris_france = words.facts[0]
    france_paris = next(fact for fact in words.facts if fact.relation == "country_capital_city")
    assert paris_france.pair == france_paris.pair
    assert words.pairs[paris_france.pair] == tuple(sorted((paris_france.subject_term, paris_france.object_term)))
    same = next(fact for fact in words.facts if fact.relation == "same_name")
    assert same.subject_term == same.object_term
    assert words.size == len(words.terms) + len(words.pairs)
    with pytest.raises(ValueError, match="Empty entity"):
        corpus.vocabulary([relation("bad", [("  ", "x")])], casefold=True)
    with pytest.raises(ValueError, match="No relation samples"):
        corpus.vocabulary([relation("empty", [])], casefold=True)


def matcher(*terms: str) -> corpus.Matcher:
    ordered = tuple(sorted(terms))
    pairs = tuple((first, second) for first in range(len(ordered)) for second in range(first + 1, len(ordered)))
    return corpus.Matcher(corpus.MatcherSpec(terms=ordered, pairs=pairs, casefold=True))


@pytest.mark.parametrize(
    ("text", "present"),
    [
        ("Paris", True),
        ("in Paris.", True),
        ("(paris)", True),
        ("Paris-based", True),
        ("Parisian", False),
        ("Montparis", False),
        ("_paris", False),
        ("paris_", False),
        ("paris2", False),
        ("2paris", False),
        ("éparis", False),
        ("parisé", False),
        ("Parisian and Paris", True),
    ],
)
def test_word_boundaries(text: str, *, present: bool) -> None:
    assert (0 in matcher("paris").find(text)) is present


def test_overlapping_terms_and_whitespace() -> None:
    lens = matcher("new york city", "york")
    assert lens.find("New\n York  City") == {0, 1}
    assert lens.find("yorkshire") == set()


def test_counts_once_per_document_and_pairs() -> None:
    lens = matcher("france", "japan", "paris")
    counts = np.zeros(lens.n_terms + lens.n_pairs, dtype=np.int64)
    lens.count("Paris paris PARIS France france", counts)
    lens.count("Japan", counts)
    lens.count("nothing", counts)
    france, japan, paris = 0, 1, 2
    pair_slots = {(0, 1): 3, (0, 2): 4, (1, 2): 5}
    assert counts[[france, japan, paris]].tolist() == [1, 1, 1]
    assert counts[pair_slots[france, paris]] == 1
    assert counts[pair_slots[france, japan]] == 0
    assert counts[pair_slots[japan, paris]] == 0
    assert counts.dtype == np.int64


def test_inventory_scope(project: Project) -> None:
    config = corpus.settings(project)
    root = project.settings.corpus_dir
    listing = corpus.inventory(config, corpus.read_manifest(root), root)
    assert [entry.key for entry in listing.entries] == [
        "alpha-mix/data/web/shard-0.jsonl.zst",
        "alpha-mix/data/web/shard-1.jsonl.zst",
        "beta-mix/data/books/part.jsonl.gz",
        "beta-mix/data/books/plain.jsonl",
    ]
    assert [entry.codec for entry in listing.entries] == ["zstd", "zstd", "gzip", "plain"]
    assert set(listing.excluded) == {
        "alpha-mix/.gitattributes",
        "alpha-mix/README.md",
        "alpha-mix/data/web/shard-0.jsonl.zst12345",
        "alpha-mix/data/pdfs/extra.jsonl.gz",
        "beta-mix/README.md",
        "beta-mix/mix.png",
    }


def manifest_with(extra: dict[str, object], repo: int = 1) -> corpus.Manifest:
    raw = manifest_json()
    repos = raw["repos"]
    assert isinstance(repos, list)
    repos[repo]["files"].append(extra)
    return corpus.Manifest.model_validate(raw)


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("data/books/table.parquet", "Unsupported data shard format"),
        ("data/books/orphan.jsonl.zst7", "Unsupported data shard format"),
        ("../escape.jsonl", "Unsafe manifest path"),
        ("/absolute.jsonl", "Unsafe manifest path"),
        ("data/books/plain.jsonl", "Duplicate manifest path"),
    ],
)
def test_inventory_rejects(project: Project, path: str, error: str) -> None:
    extra = {"path": path, "size": 1, "algorithm": "sha256", "digest": "0"}
    with pytest.raises(ValueError, match=error):
        corpus.inventory(corpus.settings(project), manifest_with(extra), project.settings.corpus_dir)


def test_inventory_requires_pins_and_data(project: Project) -> None:
    config = corpus.settings(project)
    raw = manifest_json()
    repos = raw["repos"]
    assert isinstance(repos, list)
    repos[0]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="revisions differ"):
        corpus.inventory(config, corpus.Manifest.model_validate(raw), project.settings.corpus_dir)
    empty = corpus.Manifest.model_validate(manifest_json({(ALPHA, "README.md"): b""}))
    with pytest.raises(ValueError, match="no supported data shards"):
        corpus.inventory(config, empty, project.settings.corpus_dir)


def test_scan_and_facts(project: Project) -> None:
    corpus.scan(project, workers=2)
    path = corpus.facts(project)
    assert path == project.artifacts / "corpus" / "facts.csv"
    rows = read_facts(path)
    assert list(rows[0]) == ["relation", "subject", "object", "subject_count", "object_count", "cooccurrence"]
    keys = [(row["relation"], row["subject"], row["object"]) for row in rows]
    assert keys == sorted(keys)
    expected = brute_force([str(document["text"]) for document in ALL_DOCUMENTS if isinstance(document, dict)])
    observed = {
        (row["relation"], row["subject"], row["object"]): (
            int(row["subject_count"]),
            int(row["object_count"]),
            int(row["cooccurrence"]),
        )
        for row in rows
    }
    assert observed == expected
    assert observed["city_in_country", "Paris", "France"] == (4, 5, 3)
    assert observed["city_in_country", "New York City", "United States"] == (2, 2, 1)
    assert observed["same_name", "Tokyo", "TOKYO"] == (2, 2, 2)
    record = corpus.ScanRecord.model_validate_json((project.artifacts / "corpus" / "scan.json").read_bytes())
    assert record.stages["alpha-mix"].shards == 2
    assert record.stages["alpha-mix"].documents == 9
    assert record.stages["beta-mix"].documents == 5
    assert record.stages["beta-mix"].compressed_bytes == len(FILES[BETA, "data/books/part.jsonl.gz"]) + len(
        FILES[BETA, "data/books/plain.jsonl"]
    )
    assert [repo.revision for repo in record.repos] == [ALPHA_REVISION, BETA_REVISION]
    assert (record.terms, record.pairs, record.facts) == (7, 5, 6)


def test_resume_matches_fresh_scan(tmp_path: Path) -> None:
    write_corpus(tmp_path / "corpus")
    fresh = make_project(tmp_path, "fresh")
    corpus.scan(fresh, workers=2)
    resumed = make_project(tmp_path, "resumed")
    corpus.scan(resumed, workers=1, max_shards=1)
    checkpoint_path = resumed.cache / "corpus" / "checkpoint.json"
    partial = corpus.Checkpoint.model_validate_json(checkpoint_path.read_bytes())
    assert len(partial.shards) == 1
    with pytest.raises(RuntimeError, match="not scanned yet"):
        corpus.facts(resumed)
    corpus.scan(resumed, workers=2, max_shards=2)
    assert len(corpus.Checkpoint.model_validate_json(checkpoint_path.read_bytes()).shards) == 3
    corpus.scan(resumed, workers=2)
    corpus.scan(resumed, workers=2)
    assert corpus.facts(resumed).read_bytes() == corpus.facts(fresh).read_bytes()
    first = corpus.Checkpoint.model_validate_json((fresh.cache / "corpus" / "checkpoint.json").read_bytes())
    second = corpus.Checkpoint.model_validate_json(checkpoint_path.read_bytes())
    assert first.counts == second.counts
    assert {key: shard.documents for key, shard in first.shards.items()} == {
        key: shard.documents for key, shard in second.shards.items()
    }


def test_changed_shard_rescans_its_stage(project: Project) -> None:
    corpus.scan(project, workers=2)
    before = {row["subject"]: row for row in read_facts(corpus.facts(project)) if row["relation"] == "city_in_country"}
    replacement = jsonl(documents("Paris France", "Paris France", "Paris"))
    assert len(replacement) < len(FILES[BETA, "data/books/plain.jsonl"])
    root = project.settings.corpus_dir
    target = root / "beta-mix" / "data" / "books" / "plain.jsonl"
    target.write_bytes(replacement.ljust(len(FILES[BETA, "data/books/plain.jsonl"]), b"\n"))
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    corpus.scan(project, workers=2)
    after = {row["subject"]: row for row in read_facts(corpus.facts(project)) if row["relation"] == "city_in_country"}
    assert int(after["Paris"]["subject_count"]) == int(before["Paris"]["subject_count"]) + 2
    assert int(after["Paris"]["cooccurrence"]) == int(before["Paris"]["cooccurrence"]) + 1
    assert int(after["Paris"]["object_count"]) == int(before["Paris"]["object_count"]) + 1
    assert after["Lyon"]["subject_count"] == before["Lyon"]["subject_count"]
    assert after["Lyon"]["cooccurrence"] == before["Lyon"]["cooccurrence"]


def test_checkpoint_rejects_other_vocabulary(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus.scan(project, workers=1, max_shards=1)
    monkeypatch.setattr(corpus.data, "entity_relations", lambda *_: RELATIONS[:1])
    with pytest.raises(ValueError, match="different vocabulary"):
        corpus.scan(project, workers=1)


def test_checkpoint_rejects_wrong_shape(project: Project) -> None:
    corpus.scan(project, workers=1, max_shards=1)
    path = project.cache / "corpus" / "checkpoint.json"
    checkpoint = corpus.Checkpoint.model_validate_json(path.read_bytes())
    broken = checkpoint.model_copy(update={"counts": {"alpha-mix": (1, 2)}})
    path.write_text(broken.model_dump_json())
    with pytest.raises(ValueError, match="unexpected count vector"):
        corpus.scan(project, workers=1)


def test_scan_requires_complete_corpus(project: Project) -> None:
    books = project.settings.corpus_dir / "beta-mix" / "data" / "books"
    (books / "plain.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="run download first"):
        corpus.scan(project, workers=1)
    (books / "plain.jsonl").write_bytes(b"truncated")
    with pytest.raises(FileNotFoundError, match=r"plain\.jsonl"):
        corpus.scan(project, workers=1)
    with pytest.raises(ValueError, match="positive"):
        corpus.scan(project, workers=0)


def shard_for(tmp_path: Path, payload: bytes, codec: corpus.Codec = "plain") -> corpus.Shard:
    path = tmp_path / "shard.jsonl"
    path.write_bytes(payload)
    stat = path.stat()
    return corpus.Shard(key="k", stage="s", path=path, codec=codec, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def test_scan_shard_validates_records(tmp_path: Path) -> None:
    corpus._Worker.matcher = None
    shard = shard_for(tmp_path, b'{"text": "Paris"}\n')
    with pytest.raises(RuntimeError, match="not initialized"):
        corpus.scan_shard(shard, "text", 16)
    corpus.initialize_worker(corpus.MatcherSpec(terms=("paris",), pairs=(), casefold=True))
    assert corpus.scan_shard(shard, "text", 16).counts.tolist() == [1]
    for payload in (b"not json\n", b'{"body": "Paris"}\n', b"[1, 2]\n"):
        with pytest.raises(ValueError, match=r"shard\.jsonl:1"):
            corpus.scan_shard(shard_for(tmp_path, payload), "text", 16)
    with pytest.raises(TypeError, match="not a string"):
        corpus.scan_shard(shard_for(tmp_path, b'\n{"text": 3}\n'), "text", 16)
    stale = shard_for(tmp_path, b'{"text": "Paris"}\n').model_copy(update={"mtime_ns": 0})
    with pytest.raises(RuntimeError, match="changed during scan"):
        corpus.scan_shard(stale, "text", 16)
    corpus._Worker.matcher = None


class FakeApi:
    calls: ClassVar[list[str]] = []
    revision_override: str | None = None

    def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> SimpleNamespace:
        FakeApi.calls.append(repo_id)
        siblings = []
        for (repo, path), payload in FILES.items():
            if repo != repo_id:
                continue
            digest = item(repo, path)["digest"]
            siblings.append(
                SimpleNamespace(
                    rfilename=path,
                    size=len(payload),
                    blob_id=None if lfs(path) else digest,
                    lfs=SimpleNamespace(sha256=digest) if lfs(path) else None,
                )
            )
        return SimpleNamespace(sha=FakeApi.revision_override or revision, siblings=siblings)


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    downloads: list[str] = []
    FakeApi.calls = []
    FakeApi.revision_override = None

    def fake_download(repo_id: str, filename: str, **options: object) -> str:
        assert options["repo_type"] == "dataset"
        repo = ALPHA if repo_id == ALPHA else BETA
        target = Path(str(options["local_dir"])) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(FILES[repo, filename])
        downloads.append(f"{repo_id}/{filename}")
        return str(target)

    monkeypatch.setattr(corpus, "HfApi", FakeApi)
    monkeypatch.setattr(corpus, "hf_hub_download", fake_download)
    return downloads


def test_download_builds_manifest_and_fetches_scope(tmp_path: Path, fake_hub: list[str]) -> None:
    project = make_project(tmp_path)
    corpus.download(project)
    assert FakeApi.calls == [ALPHA, BETA]
    assert sorted(fake_hub) == [
        f"{ALPHA}/data/web/shard-0.jsonl.zst",
        f"{ALPHA}/data/web/shard-1.jsonl.zst",
        f"{BETA}/data/books/part.jsonl.gz",
        f"{BETA}/data/books/plain.jsonl",
    ]
    written = corpus.read_manifest(project.settings.corpus_dir)
    assert written.pins() == [(ALPHA, ALPHA_REVISION), (BETA, BETA_REVISION)]
    assert {file.algorithm for repo in written.repos for file in repo.files} == {"sha256", "git-sha1"}
    fake_hub.clear()
    corpus.download(project)
    assert fake_hub == []
    assert FakeApi.calls == [ALPHA, BETA]


def test_download_reuses_existing_manifest_and_verifies(tmp_path: Path, fake_hub: list[str]) -> None:
    write_corpus(tmp_path / "corpus")
    project = make_project(tmp_path)
    shard = project.settings.corpus_dir / "alpha-mix" / "data" / "web" / "shard-1.jsonl.zst"
    shard.write_bytes(b"x" * shard.stat().st_size)
    corpus.download(project)
    assert FakeApi.calls == []
    assert fake_hub == []
    corpus.download(project, verify_checksums=True)
    assert fake_hub == [f"{ALPHA}/data/web/shard-1.jsonl.zst"]
    assert shard.read_bytes() == FILES[ALPHA, "data/web/shard-1.jsonl.zst"]


def test_download_refreshes_stale_manifest(tmp_path: Path, fake_hub: list[str]) -> None:
    write_corpus(tmp_path / "corpus")
    raw = manifest_json()
    repos = raw["repos"]
    assert isinstance(repos, list)
    repos[0]["revision"] = "c" * 40
    (tmp_path / "corpus" / "manifest.json").write_text(json.dumps(raw))
    corpus.download(make_project(tmp_path))
    assert FakeApi.calls == [ALPHA, BETA]
    assert corpus.read_manifest(tmp_path / "corpus").pins()[0] == (ALPHA, ALPHA_REVISION)


def test_download_rejects_revision_and_bad_checksum(tmp_path: Path, fake_hub: list[str]) -> None:
    project = make_project(tmp_path)
    FakeApi.revision_override = "d" * 40
    with pytest.raises(RuntimeError, match="API returned revision"):
        corpus.download(project)
    FakeApi.revision_override = None
    corrupt = {key: payload[::-1] if key[1].endswith(".jsonl") else payload for key, payload in FILES.items()}
    original = FILES.copy()
    corpus.download(project)
    FILES.update(corrupt)
    try:
        plain = project.settings.corpus_dir / "beta-mix" / "data" / "books" / "plain.jsonl"
        plain.unlink()
        with pytest.raises(ValueError, match="Checksum mismatch"):
            corpus.download(project)
    finally:
        FILES.clear()
        FILES.update(original)


def test_checksum_matches(tmp_path: Path) -> None:
    payload = b"hello world"
    path = tmp_path / "file"
    path.write_bytes(payload)
    sha = corpus.ManifestFile(path="f", size=11, algorithm="sha256", digest=hashlib.sha256(payload).hexdigest())
    git = corpus.ManifestFile(
        path="f", size=11, algorithm="git-sha1", digest="95d09f2b10159347eece71399a7e2e907ea3df4f"
    )
    assert corpus.checksum_matches(path, sha, 4)
    assert corpus.checksum_matches(path, git, 4)
    assert not corpus.checksum_matches(path, sha.model_copy(update={"size": 3}), 4)
    assert not corpus.checksum_matches(tmp_path / "absent", sha, 4)


def test_incomplete_listing_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = corpus.settings(make_project(tmp_path))

    class EmptyApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> SimpleNamespace:
            return SimpleNamespace(sha=revision, siblings=[])

    class SizelessApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> SimpleNamespace:
            sibling = SimpleNamespace(rfilename="data/a.jsonl", size=None, blob_id="x", lfs=None)
            return SimpleNamespace(sha=revision, siblings=[sibling])

    with pytest.raises(RuntimeError, match="empty file listing"):
        corpus._listing(EmptyApi(), config.repos[0])
    with pytest.raises(RuntimeError, match="incomplete metadata"):
        corpus._listing(SizelessApi(), config.repos[0])


def test_scan_shard_reads_every_codec(project: Project) -> None:
    config = corpus.settings(project)
    root = project.settings.corpus_dir
    shards = corpus.shards(corpus.inventory(config, corpus.read_manifest(root), root))
    words = corpus.vocabulary(RELATIONS, casefold=True)
    corpus.initialize_worker(corpus.MatcherSpec(terms=words.terms, pairs=words.pairs, casefold=True))
    try:
        results = [corpus.scan_shard(shard, config.text_field, config.read_buffer_bytes) for shard in shards]
    finally:
        corpus._Worker.matcher = None
    assert [result.documents for result in results] == [6, 3, 2, 3]
    total = sum(result.counts for result in results)
    project_scan = make_project(project.settings.corpus_dir.parent, "reference")
    corpus.scan(project_scan, workers=2)
    checkpoint = corpus.Checkpoint.model_validate_json((project_scan.cache / "corpus" / "checkpoint.json").read_bytes())
    assert total.tolist() == [sum(values) for values in zip(*checkpoint.counts.values(), strict=True)]


def test_retry_recovers_from_transient_errors(project: Project) -> None:
    config = corpus.settings(project)
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            message = "transient"
            raise OSError(message)
        return "ok"

    assert corpus._retry(flaky, config, "label") == "ok"
    assert len(attempts) == 2
