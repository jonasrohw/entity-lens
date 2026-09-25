from __future__ import annotations

import csv
import gzip
import hashlib
import io
import multiprocessing
import os
import re
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

import ahocorasick
import httpx
import numpy as np
import numpy.typing as npt
import orjson
import zstandard
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field

from entity_lens import data

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from entity_lens.config import Project
    from entity_lens.data import Relation

Codec = Literal["zstd", "gzip", "plain"]
Algorithm = Literal["sha256", "git-sha1"]
Counts = npt.NDArray[np.int64]
WHITESPACE = re.compile(r"\s+")
SINGLE_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
TRANSIENT_ERRORS = (OSError, ValueError, httpx.HTTPError)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RepoConfig(Frozen):
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split_suffixes: tuple[str, ...] | None = None

    @property
    def stage(self) -> str:
        return self.repo_id.split("/")[-1]


class CorpusConfig(Frozen):
    text_field: str
    casefold: bool
    data_directory: str
    duplicate_suffix_pattern: str
    checkpoint_seconds: float = Field(ge=0)
    progress_seconds: float = Field(gt=0)
    tasks_per_worker: int = Field(gt=0)
    read_buffer_bytes: int = Field(gt=0)
    hash_block_bytes: int = Field(gt=0)
    download_workers: int = Field(gt=0)
    download_attempts: int = Field(gt=0)
    retry_base_seconds: float = Field(ge=0)
    retry_max_seconds: float = Field(ge=0)
    formats: dict[str, Codec] = Field(min_length=1)
    repos: tuple[RepoConfig, ...] = Field(min_length=1)

    def codec(self, path: str) -> tuple[str, Codec] | None:
        for suffix in sorted(self.formats, key=len, reverse=True):
            if path.endswith(suffix):
                return suffix, self.formats[suffix]
        return None


class ManifestFile(Frozen):
    path: str
    size: int = Field(ge=0)
    algorithm: Algorithm
    digest: str


class ManifestRepo(Frozen):
    repo_id: str
    revision: str
    files: tuple[ManifestFile, ...]


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    repos: tuple[ManifestRepo, ...]

    def pins(self) -> list[tuple[str, str]]:
        return [(repo.repo_id, repo.revision) for repo in self.repos]


class Fact(Frozen):
    relation: str
    subject: str
    object: str
    subject_term: int
    object_term: int
    pair: int


class Vocabulary(Frozen):
    terms: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]
    facts: tuple[Fact, ...]

    @property
    def size(self) -> int:
        return len(self.terms) + len(self.pairs)


class Entry(Frozen):
    repo: RepoConfig
    file: ManifestFile
    key: str
    path: Path
    codec: Codec


class Inventory(Frozen):
    entries: tuple[Entry, ...]
    excluded: tuple[str, ...]


class Shard(Frozen):
    key: str
    stage: str
    path: Path
    codec: Codec
    size: int
    mtime_ns: int


class CompletedShard(Frozen):
    stage: str
    size: int
    mtime_ns: int
    documents: int


class Checkpoint(Frozen):
    fingerprint: str
    counts: dict[str, tuple[int, ...]]
    shards: dict[str, CompletedShard]


class StageSummary(Frozen):
    shards: int
    documents: int
    compressed_bytes: int


class ScanRecord(Frozen):
    repos: tuple[RepoConfig, ...]
    stages: dict[str, StageSummary]
    terms: int
    pairs: int
    facts: int
    fingerprint: str


class MatcherSpec(Frozen):
    terms: tuple[str, ...]
    pairs: tuple[tuple[int, int], ...]
    casefold: bool


@dataclass(frozen=True)
class ShardCounts:
    documents: int
    counts: Counts


def settings(project: Project) -> CorpusConfig:
    return CorpusConfig.model_validate(project.load("corpus"))


def normalize(text: str, *, casefold: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    if casefold:
        text = text.casefold()
    return WHITESPACE.sub(" ", text).strip()


def vocabulary(relations: Sequence[Relation], *, casefold: bool) -> Vocabulary:
    raw: list[tuple[str, str, str, str, str]] = []
    for relation in relations:
        seen: set[tuple[str, str]] = set()
        for sample in relation.samples:
            pair = (sample.subject, sample.object)
            if pair in seen:
                continue
            seen.add(pair)
            subject, obj = normalize(sample.subject, casefold=casefold), normalize(sample.object, casefold=casefold)
            if not subject or not obj:
                message = f"Empty entity in {relation.key}: {pair}"
                raise ValueError(message)
            raw.append((relation.key, sample.subject, sample.object, subject, obj))
    if not raw:
        message = "No relation samples to count"
        raise ValueError(message)
    terms = sorted({term for *_, subject, obj in raw for term in (subject, obj)})
    ids = {term: index for index, term in enumerate(terms)}
    edges = [(ids[subject], ids[obj]) for *_, subject, obj in raw]
    pairs = sorted({(min(edge), max(edge)) for edge in edges})
    pair_ids = {pair: index for index, pair in enumerate(pairs)}
    facts = [
        Fact(
            relation=relation,
            subject=subject,
            object=obj,
            subject_term=edge[0],
            object_term=edge[1],
            pair=pair_ids[min(edge), max(edge)],
        )
        for (relation, subject, obj, *_), edge in zip(raw, edges, strict=True)
    ]
    return Vocabulary(terms=tuple(terms), pairs=tuple(pairs), facts=tuple(facts))


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


class Matcher:
    def __init__(self, spec: MatcherSpec) -> None:
        terms, pairs = spec.terms, spec.pairs
        self.casefold = spec.casefold
        self.n_terms = len(terms)
        self.n_pairs = len(pairs)
        self.automaton = ahocorasick.Automaton()
        for index, term in enumerate(terms):
            self.automaton.add_word(term, (index, len(term)))
        self.automaton.make_automaton()
        degree = np.bincount([term for pair in pairs for term in pair], minlength=len(terms))
        self.adjacency: list[list[tuple[int, int]]] = [[] for _ in terms]
        for pair_index, (first, second) in enumerate(pairs):
            low, high = (second, first) if degree[first] > degree[second] else (first, second)
            self.adjacency[low].append((high, self.n_terms + pair_index))

    def find(self, text: str) -> set[int]:
        text = normalize(text, casefold=self.casefold)
        found: set[int] = set()
        for end, (term, length) in self.automaton.iter(text):
            if term in found:
                continue
            start = end - length + 1
            if start > 0 and _is_word_character(text[start - 1]):
                continue
            if end + 1 < len(text) and _is_word_character(text[end + 1]):
                continue
            found.add(term)
        return found

    def count(self, text: str, counts: Counts) -> None:
        found = self.find(text)
        for term in found:
            counts[term] += 1
            for other, slot in self.adjacency[term]:
                if other in found:
                    counts[slot] += 1


class _Worker:
    matcher: Matcher | None = None


def initialize_worker(spec: MatcherSpec) -> None:
    _Worker.matcher = Matcher(spec)


@contextmanager
def open_lines(path: Path, codec: Codec, buffer_bytes: int) -> Iterator[Iterable[bytes]]:
    if codec == "zstd":
        with (
            path.open("rb") as raw,
            zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as reader,
            io.BufferedReader(reader, buffer_size=buffer_bytes) as stream,
        ):
            yield stream
    elif codec == "gzip":
        with gzip.open(path, "rb") as stream:
            yield stream
    else:
        with path.open("rb", buffering=buffer_bytes) as stream:
            yield stream


def scan_shard(shard: Shard, text_field: str, buffer_bytes: int) -> ShardCounts:
    matcher = _Worker.matcher
    if matcher is None:
        message = "Worker matcher is not initialized"
        raise RuntimeError(message)
    counts = np.zeros(matcher.n_terms + matcher.n_pairs, dtype=np.int64)
    documents = 0
    with open_lines(shard.path, shard.codec, buffer_bytes) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                text = orjson.loads(line)[text_field]
            except (orjson.JSONDecodeError, KeyError, TypeError) as error:
                message = f"{shard.path}:{line_number}: {error!r}"
                raise ValueError(message) from error
            if not isinstance(text, str):
                message = f"{shard.path}:{line_number}: {text_field!r} is not a string"
                raise TypeError(message)
            documents += 1
            matcher.count(text, counts)
    stat = shard.path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (shard.size, shard.mtime_ns):
        message = f"Shard changed during scan: {shard.path}"
        raise RuntimeError(message)
    return ShardCounts(documents=documents, counts=counts)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _safe_path(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        message = f"Unsafe manifest path: {path}"
        raise ValueError(message)
    return relative


def _is_duplicate(path: str, published: set[str], config: CorpusConfig) -> bool:
    for suffix in config.formats:
        head, separator, tail = path.rpartition(suffix)
        if separator and re.fullmatch(config.duplicate_suffix_pattern, tail) and head + suffix in published:
            return True
    return False


def inventory(config: CorpusConfig, manifest: Manifest, root: Path) -> Inventory:
    if manifest.pins() != [(repo.repo_id, repo.revision) for repo in config.repos]:
        message = "Manifest repositories or revisions differ from configs/corpus.toml"
        raise ValueError(message)
    entries: list[Entry] = []
    excluded: list[str] = []
    keys: set[str] = set()
    for repo, listing in zip(config.repos, manifest.repos, strict=True):
        published = {item.path for item in listing.files}
        for item in listing.files:
            relative = _safe_path(item.path)
            key = f"{repo.stage}/{relative}"
            if key in keys:
                message = f"Duplicate manifest path: {key}"
                raise ValueError(message)
            keys.add(key)
            if relative.parts[0] != config.data_directory:
                excluded.append(key)
                continue
            match = config.codec(item.path)
            if match is None:
                if _is_duplicate(item.path, published, config):
                    excluded.append(key)
                    continue
                message = f"Unsupported data shard format: {key}"
                raise ValueError(message)
            suffix, codec = match
            if repo.split_suffixes is not None and suffix not in repo.split_suffixes:
                excluded.append(key)
                continue
            entries.append(Entry(repo=repo, file=item, key=key, path=root / key, codec=codec))
    if not entries:
        message = "Manifest contains no supported data shards"
        raise ValueError(message)
    return Inventory(entries=tuple(entries), excluded=tuple(excluded))


def read_manifest(root: Path) -> Manifest:
    return Manifest.model_validate_json((root / "manifest.json").read_bytes())


def _retry[T](call: Callable[[], T], config: CorpusConfig, label: str) -> T:
    for attempt in range(1, config.download_attempts):
        try:
            return call()
        except TRANSIENT_ERRORS as error:
            delay = min(config.retry_max_seconds, config.retry_base_seconds * 2 ** (attempt - 1))
            print(
                f"{label}: {type(error).__name__}; retry {attempt}/{config.download_attempts} in {delay}s", flush=True
            )
            time.sleep(delay)
    return call()


def _listing(api: HfApi, repo: RepoConfig) -> ManifestRepo:
    info = api.dataset_info(repo.repo_id, revision=repo.revision, files_metadata=True)
    if info.sha != repo.revision:
        message = f"{repo.repo_id}: API returned revision {info.sha}, expected {repo.revision}"
        raise RuntimeError(message)
    files = []
    for sibling in info.siblings or []:
        _safe_path(sibling.rfilename)
        digest = sibling.lfs.sha256 if sibling.lfs else sibling.blob_id
        if sibling.size is None or not digest:
            message = f"{repo.repo_id}: incomplete metadata for {sibling.rfilename}"
            raise RuntimeError(message)
        algorithm: Algorithm = "sha256" if sibling.lfs else "git-sha1"
        files.append(ManifestFile(path=sibling.rfilename, size=sibling.size, algorithm=algorithm, digest=digest))
    if not files:
        message = f"{repo.repo_id}: empty file listing"
        raise RuntimeError(message)
    return ManifestRepo(repo_id=repo.repo_id, revision=repo.revision, files=tuple(files))


def manifest(config: CorpusConfig, root: Path) -> Manifest:
    path = root / "manifest.json"
    pins = [(repo.repo_id, repo.revision) for repo in config.repos]
    if path.exists():
        existing = read_manifest(root)
        if existing.pins() == pins:
            return existing
        print(f"{path} lists {existing.pins()}; rebuilding it for {pins}", flush=True)
    api = HfApi()
    listings = [_retry(partial(_listing, api, repo), config, repo.repo_id) for repo in config.repos]
    fresh = Manifest(repos=tuple(listings))
    _atomic_write(path, fresh.model_dump_json(indent=1).encode())
    return fresh


def checksum_matches(path: Path, item: ManifestFile, block_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size != item.size:
        return False
    digest = hashlib.sha256() if item.algorithm == "sha256" else hashlib.sha1(usedforsecurity=False)
    if item.algorithm == "git-sha1":
        digest.update(f"blob {item.size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest() == item.digest


def _fetch(entry: Entry, root: Path, config: CorpusConfig) -> None:
    def transfer() -> None:
        hf_hub_download(
            repo_id=entry.repo.repo_id,
            filename=entry.file.path,
            repo_type="dataset",
            revision=entry.repo.revision,
            local_dir=root / entry.repo.stage,
            force_download=entry.path.exists(),
        )
        if not checksum_matches(entry.path, entry.file, config.hash_block_bytes):
            message = f"Checksum mismatch after download: {entry.path}"
            raise ValueError(message)

    _retry(transfer, config, entry.key)


def _needs_download(entry: Entry, config: CorpusConfig, *, verify_checksums: bool) -> bool:
    if not entry.path.is_file() or entry.path.stat().st_size != entry.file.size:
        return True
    return verify_checksums and not checksum_matches(entry.path, entry.file, config.hash_block_bytes)


def download(project: Project, *, verify_checksums: bool = False) -> None:
    config = settings(project)
    root = project.settings.corpus_dir
    root.mkdir(parents=True, exist_ok=True)
    listing = inventory(config, manifest(config, root), root)
    with ThreadPoolExecutor(max_workers=config.download_workers) as pool:
        checks = pool.map(
            lambda entry: _needs_download(entry, config, verify_checksums=verify_checksums), listing.entries
        )
        pending = [entry for entry, needed in zip(listing.entries, checks, strict=True) if needed]
        total = sum(entry.file.size for entry in pending)
        print(f"{len(pending)}/{len(listing.entries)} shards to download ({total / 1e9:.1f} GB)", flush=True)
        for done, _ in enumerate(pool.map(lambda entry: _fetch(entry, root, config), pending), 1):
            if done % 100 == 0 or done == len(pending):
                print(f"downloaded {done}/{len(pending)} shards", flush=True)


def _stat(entry: Entry) -> Shard | None:
    try:
        stat = entry.path.stat()
    except FileNotFoundError:
        return None
    if stat.st_size != entry.file.size:
        return None
    return Shard(
        key=entry.key,
        stage=entry.repo.stage,
        path=entry.path,
        codec=entry.codec,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def shards(listing: Inventory) -> list[Shard]:
    present = [_stat(entry) for entry in listing.entries]
    missing = [entry.key for entry, shard in zip(listing.entries, present, strict=True) if shard is None]
    if missing:
        message = f"{len(missing)} shards are absent or have the wrong size (e.g. {missing[0]}); run download first"
        raise FileNotFoundError(message)
    return [shard for shard in present if shard is not None]


def fingerprint(config: CorpusConfig, words: Vocabulary) -> str:
    scope = config.model_dump(
        include={"text_field", "casefold", "data_directory", "duplicate_suffix_pattern", "formats", "repos"}
    )
    payload = orjson.dumps({"scope": scope, "terms": words.terms, "pairs": words.pairs}, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


class Progress:
    def __init__(self, path: Path, signature: str, stages: Sequence[str], size: int) -> None:
        self.path = path
        self.signature = signature
        self.counts = {stage: np.zeros(size, dtype=np.int64) for stage in stages}
        self.completed: dict[str, CompletedShard] = {}
        if path.exists():
            self._restore(Checkpoint.model_validate_json(path.read_bytes()), size)

    def _restore(self, checkpoint: Checkpoint, size: int) -> None:
        if checkpoint.fingerprint != self.signature:
            message = f"{self.path} was written for a different vocabulary or corpus scope; move it away to rescan"
            raise ValueError(message)
        for stage, values in checkpoint.counts.items():
            if stage not in self.counts or len(values) != size:
                message = f"{self.path} has an unexpected count vector for {stage}"
                raise ValueError(message)
            self.counts[stage] = np.array(values, dtype=np.int64)
        self.completed = dict(checkpoint.shards)

    def reconcile(self, current: Sequence[Shard]) -> None:
        state = {shard.key: (shard.stage, shard.size, shard.mtime_ns) for shard in current}
        stale = {
            done.stage
            for key, done in self.completed.items()
            if state.get(key) != (done.stage, done.size, done.mtime_ns)
        }
        for stage in sorted(stale):
            print(f"{stage}: a completed shard changed or disappeared; rescanning the stage", flush=True)
            self.counts[stage][:] = 0
        self.completed = {key: done for key, done in self.completed.items() if done.stage not in stale}

    def add(self, shard: Shard, result: ShardCounts) -> None:
        self.counts[shard.stage] += result.counts
        self.completed[shard.key] = CompletedShard(
            stage=shard.stage, size=shard.size, mtime_ns=shard.mtime_ns, documents=result.documents
        )

    def save(self) -> None:
        checkpoint = Checkpoint(
            fingerprint=self.signature,
            counts={stage: tuple(values.tolist()) for stage, values in self.counts.items()},
            shards=self.completed,
        )
        _atomic_write(self.path, checkpoint.model_dump_json().encode())


def _checkpoint_path(project: Project) -> Path:
    return project.cache / "corpus" / "checkpoint.json"


def _prepare(project: Project) -> tuple[CorpusConfig, Vocabulary, list[Shard], Progress]:
    config = settings(project)
    words = vocabulary(
        data.entity_relations(project.data / "relations", project.protocol.dataset), casefold=config.casefold
    )
    root = project.settings.corpus_dir
    current = shards(inventory(config, read_manifest(root), root))
    stages = [repo.stage for repo in config.repos]
    progress = Progress(_checkpoint_path(project), fingerprint(config, words), stages, words.size)
    progress.reconcile(current)
    return config, words, current, progress


def _run(
    pool: ProcessPoolExecutor, pending: Sequence[Shard], progress: Progress, config: CorpusConfig, workers: int
) -> None:
    started = last_checkpoint = last_progress = time.monotonic()
    scanned_bytes = scanned = 0
    queue = iter(pending)
    running: dict[Future[ShardCounts], Shard] = {}
    try:
        while True:
            for shard in islice(queue, workers * config.tasks_per_worker - len(running)):
                running[pool.submit(scan_shard, shard, config.text_field, config.read_buffer_bytes)] = shard
            if not running:
                break
            finished, _ = wait(running, timeout=config.progress_seconds, return_when=FIRST_COMPLETED)
            for future in finished:
                shard = running.pop(future)
                progress.add(shard, future.result())
                scanned += 1
                scanned_bytes += shard.size
            now = time.monotonic()
            if now - last_checkpoint >= config.checkpoint_seconds:
                progress.save()
                last_checkpoint = now
            if now - last_progress >= config.progress_seconds:
                rate = scanned_bytes / 1e6 / max(now - started, 1e-9)
                print(f"{scanned}/{len(pending)} shards; {rate:.1f} compressed MB/s", flush=True)
                last_progress = now
    finally:
        for future in running:
            future.cancel()
        progress.save()
    print(f"scanned {scanned} shards in {time.monotonic() - started:.0f}s", flush=True)


def scan(project: Project, workers: int, max_shards: int | None = None) -> None:
    if workers < 1:
        message = "workers must be positive"
        raise ValueError(message)
    config, words, current, progress = _prepare(project)
    pending = sorted(
        (shard for shard in current if shard.key not in progress.completed), key=lambda shard: (-shard.size, shard.key)
    )
    pending = pending[:max_shards] if max_shards is not None else pending
    print(f"{len(progress.completed)} shards cached; scanning {len(pending)} with {workers} workers", flush=True)
    if not pending:
        progress.save()
        return
    for variable in SINGLE_THREAD_VARIABLES:
        os.environ[variable] = "1"
    spec = MatcherSpec(terms=words.terms, pairs=words.pairs, casefold=config.casefold)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=initialize_worker, initargs=(spec,)) as pool:
        _run(pool, pending, progress, config, workers)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    _atomic_write(path, buffer.getvalue().encode())


def facts(project: Project) -> Path:
    config, words, current, progress = _prepare(project)
    unfinished = [shard.key for shard in current if shard.key not in progress.completed]
    if unfinished:
        message = f"{len(unfinished)} shards are not scanned yet (e.g. {unfinished[0]}); run scan first"
        raise RuntimeError(message)
    total = sum(progress.counts.values(), start=np.zeros(words.size, dtype=np.int64))
    documents = sum(done.documents for done in progress.completed.values())
    rows = []
    for fact in sorted(words.facts, key=lambda fact: (fact.relation, fact.subject, fact.object)):
        subject_count = int(total[fact.subject_term])
        object_count = int(total[fact.object_term])
        cooccurrence = int(total[len(words.terms) + fact.pair])
        if not 0 <= cooccurrence <= min(subject_count, object_count) or max(subject_count, object_count) > documents:
            message = f"Inconsistent document counts for {fact}"
            raise RuntimeError(message)
        rows.append((fact.relation, fact.subject, fact.object, subject_count, object_count, cooccurrence))
    output = project.artifacts / "corpus"
    header = ("relation", "subject", "object", "subject_count", "object_count", "cooccurrence")
    _write_csv(output / "facts.csv", header, rows)
    stages = {
        repo.stage: StageSummary(
            shards=sum(shard.stage == repo.stage for shard in current),
            documents=sum(done.documents for done in progress.completed.values() if done.stage == repo.stage),
            compressed_bytes=sum(shard.size for shard in current if shard.stage == repo.stage),
        )
        for repo in config.repos
    }
    record = ScanRecord(
        repos=config.repos,
        stages=stages,
        terms=len(words.terms),
        pairs=len(words.pairs),
        facts=len(words.facts),
        fingerprint=progress.signature,
    )
    _atomic_write(output / "scan.json", (record.model_dump_json(indent=1) + "\n").encode())
    return output / "facts.csv"
