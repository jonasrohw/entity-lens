import bisect
import bz2
import gzip
import hashlib
import http.client
import itertools
import json
import math
import multiprocessing
import os
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from pydantic import Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from entity_lens import data, storage
from entity_lens.config import Frozen, Project
from entity_lens.records import LensRow, Record, read, sha256, write

SHA1 = r"^[0-9a-f]{40}$"
SHA256 = r"^[0-9a-f]{64}$"
HTTP_PARTIAL = 206
HTTP_RANGE_NOT_SATISFIABLE = 416


class DumpConfig(Frozen):
    base_url: str = Field(pattern=r"^https://")
    dump: str
    index: str
    dump_sha1: str = Field(pattern=SHA1)
    index_sha1: str = Field(pattern=SHA1)
    partial_suffix: str
    chunk_bytes: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    retries: int = Field(ge=0)


class ExtractionConfig(Frozen):
    directory: str
    streams_per_shard: int = Field(gt=0)
    shard_name: str
    shard_glob: str
    stats_suffix: str
    manifest: str
    compression_level: int = Field(ge=0, le=9)
    paragraph_separator: str
    excluded_title_marker: str
    title_separator: str
    untitled: str


class RetrievalConfig(Frozen):
    k: int = Field(gt=0)
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    query_batch_size: int = Field(gt=0)
    threads: int = Field(gt=0)
    artifact_dir: str
    artifact: str
    cache_dir: str
    counts_dir: str
    paragraphs_cache: str
    cache_compression_level: int = Field(ge=0, le=9)


class AttributeConfig(Frozen):
    top_k: int = Field(gt=0)
    min_token_length: int = Field(ge=0)
    encode_batch_size: int = Field(gt=0)
    nltk_dir: str
    stopwords_package: str
    stopwords_language: str
    stopwords_sha256: str = Field(pattern=SHA256)
    artifact_dir: str


class WikipediaConfig(Frozen):
    dump: DumpConfig
    extraction: ExtractionConfig
    retrieval: RetrievalConfig
    attributes: AttributeConfig


class ShardStats(Record):
    pages: int
    paragraphs: int
    bytes: int


class CorpusManifest(Record):
    dump_sha1: str
    index_sha1: str
    complete: bool
    shards: int
    pages: int
    paragraphs: int
    config_fingerprint: str | None = None


class ShardJob(Record):
    dump: Path
    offsets: tuple[int, ...]
    output: Path
    config: ExtractionConfig


class CountJob(Record):
    source: Path
    output: Path
    vocabulary: dict[str, int]


class SelectJob(Record):
    source: Path
    offset: int
    wanted: frozenset[int]


class Corpus(Record):
    manifest: CorpusManifest
    shards: tuple[Path, ...]
    sizes: tuple[int, ...]


class SubjectHits(Record):
    ids: tuple[int, ...]
    scores: tuple[float, ...]
    keep: tuple[bool, ...]


class Retrieval(Record):
    corpus: CorpusManifest
    method: str
    k: int
    k1: float
    b: float
    bm25s_version: str
    vocabulary_size: int
    zero_score_hits: int
    subjects: dict[str, SubjectHits]


class ParagraphCache(Record):
    retrieval_sha256: str
    paragraphs: dict[int, tuple[str, str]]


class AttributeRow(Record):
    subject: str
    object: str
    rate: float
    overlap: int
    filtered_predictions: int


class RelationAttributeRates(Record):
    n_rows: int
    mean_rate: float
    empty_rows: int
    rows: tuple[AttributeRow, ...]


class AttributeRates(Record):
    model: str
    hf_id: str
    revision: str
    retrieval_sha256: str
    stopwords_sha256: str
    top_k: int
    min_token_length: int
    n_subjects: int
    relations: dict[str, RelationAttributeRates]


def load_config(project: Project) -> WikipediaConfig:
    return WikipediaConfig.model_validate(project.load("wikipedia"))


def corpus_dir(project: Project, config: WikipediaConfig) -> Path:
    return project.settings.wikipedia_dir / config.extraction.directory


def retrieval_path(project: Project, config: WikipediaConfig) -> Path:
    return project.artifacts / config.retrieval.artifact_dir / config.retrieval.artifact


def paragraphs_cache_path(project: Project, config: WikipediaConfig) -> Path:
    return project.cache / config.retrieval.cache_dir / config.retrieval.paragraphs_cache


def attribute_rates_path(project: Project, config: WikipediaConfig, model_key: str) -> Path:
    return project.artifacts / config.attributes.artifact_dir / f"{model_key}.json"


def checked_sha1(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha1").hexdigest()
    if actual != expected:
        message = f"SHA-1 mismatch for {path}: {actual} != {expected}"
        raise ValueError(message)


def _fetch_remainder(url: str, partial: Path, config: DumpConfig) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=config.timeout_seconds)
    except urllib.error.HTTPError as error:
        if error.code == HTTP_RANGE_NOT_SATISFIABLE:
            return
        raise
    with response:
        resumed = response.status == HTTP_PARTIAL
        expected = int(response.headers["Content-Length"]) + (offset if resumed else 0)
        with partial.open("ab" if resumed else "wb") as sink:
            while chunk := response.read(config.chunk_bytes):
                sink.write(chunk)
    if partial.stat().st_size != expected:
        message = f"Incomplete download of {url}: {partial.stat().st_size} of {expected} bytes"
        raise ConnectionError(message)


def download(url: str, target: Path, sha1: str, config: DumpConfig) -> None:
    if target.exists():
        checked_sha1(target, sha1)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + config.partial_suffix)
    for attempt in itertools.count():
        try:
            _fetch_remainder(url, partial, config)
            break
        except (OSError, http.client.HTTPException) as error:
            if attempt >= config.retries:
                raise
            print(f"Retrying {url} after {error!r}")
    try:
        checked_sha1(partial, sha1)
    except ValueError:
        partial.unlink()
        raise
    partial.replace(target)


def paragraphs(title: str, text: str, extractor: Any, config: ExtractionConfig) -> Iterator[tuple[str, str]]:
    if config.excluded_title_marker in title or not text:
        return
    titles = [title]
    for block in text.split(config.paragraph_separator):
        paragraph = block
        if paragraph.startswith("="):
            heading, *body = paragraph.split("\n", 1)
            titles = [*titles[: heading.count("=") // 2 - 1], heading.strip(" =\n")]
            if not body:
                continue
            paragraph = body[0]
        cleaned = " ".join(extractor.clean_text(paragraph)).strip().replace("\n", " ")
        if not cleaned:
            continue
        heading_text = " ".join(extractor.clean_text(config.title_separator.join(titles))).strip()
        yield heading_text or config.untitled, cleaned


def stats_path(shard: Path, config: ExtractionConfig) -> Path:
    return shard.with_name(shard.name + config.stats_suffix)


def extract_shard(job: ShardJob) -> ShardStats:
    from defusedxml import ElementTree
    from wikiextractor.extract import Extractor

    extractor = Extractor(id=None, revid=None, urlbase=None, title="", page=None)
    pages = count = 0
    temporary = job.output.with_name(f"{job.output.name}.{os.getpid()}.tmp")
    with (
        job.dump.open("rb") as source,
        gzip.open(temporary, "wb", compresslevel=job.config.compression_level) as sink,
    ):
        for start, end in itertools.pairwise(job.offsets):
            source.seek(start)
            xml = bz2.decompress(source.read(end - start)).decode("utf-8")
            for fragment in xml.split("<page>")[1:]:
                tree = ElementTree.fromstring("<page>" + fragment.split("</page>", 1)[0] + "</page>")
                pages += 1
                title = tree.findtext("title") or ""
                text = tree.findtext("revision/text") or ""
                for heading, content in paragraphs(title, text, extractor, job.config):
                    sink.write(orjson.dumps([heading, content]) + b"\n")
                    count += 1
    temporary.replace(job.output)
    stats = ShardStats(pages=pages, paragraphs=count, bytes=job.output.stat().st_size)
    write(stats_path(job.output, job.config), stats)
    return stats


def stream_offsets(index: Path, dump_size: int) -> list[int]:
    offsets: list[int] = []
    with bz2.open(index, "rt", encoding="utf-8") as lines:
        for line in lines:
            offset = int(line.split(":", 1)[0])
            if not offsets or offsets[-1] != offset:
                offsets.append(offset)
    offsets.append(dump_size)
    return offsets


def shard_paths(directory: Path, config: ExtractionConfig) -> list[Path]:
    return sorted(directory.glob(config.shard_glob))


def extract(config: WikipediaConfig, wikipedia_dir: Path, workers: int) -> CorpusManifest:
    settings = config.extraction
    directory = wikipedia_dir / settings.directory
    directory.mkdir(parents=True, exist_ok=True)
    dump = wikipedia_dir / config.dump.dump
    offsets = stream_offsets(wikipedia_dir / config.dump.index, dump.stat().st_size)
    step = settings.streams_per_shard
    jobs = []
    for start in range(0, len(offsets) - 1, step):
        output = directory / settings.shard_name.format(start)
        if not (output.exists() and stats_path(output, settings).exists()):
            jobs.append(
                ShardJob(dump=dump, offsets=tuple(offsets[start : start + step + 1]), output=output, config=settings)
            )
    print(f"{len(offsets) - 1} streams; {len(jobs)} pending shards")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for done, _ in enumerate(pool.map(extract_shard, jobs), start=1):
            if done % 100 == 0 or done == len(jobs):
                print(f"Extracted {done}/{len(jobs)} shards")
    shards = shard_paths(directory, settings)
    totals = [read(stats_path(shard, settings), ShardStats) for shard in shards]
    manifest = CorpusManifest(
        dump_sha1=config.dump.dump_sha1,
        index_sha1=config.dump.index_sha1,
        complete=True,
        shards=len(shards),
        pages=sum(total.pages for total in totals),
        paragraphs=sum(total.paragraphs for total in totals),
    )
    write(directory / settings.manifest, manifest)
    return manifest


def load_corpus(config: WikipediaConfig, directory: Path) -> Corpus:
    settings = config.extraction
    manifest_path = directory / settings.manifest
    if not manifest_path.exists():
        message = f"No extracted corpus at {directory}"
        raise FileNotFoundError(message)
    manifest = read(manifest_path, CorpusManifest)
    if not manifest.complete or (manifest.dump_sha1, manifest.index_sha1) != (
        config.dump.dump_sha1,
        config.dump.index_sha1,
    ):
        message = f"Incomplete or mismatched corpus at {directory}"
        raise ValueError(message)
    shards = shard_paths(directory, settings)
    sizes = tuple(read(stats_path(shard, settings), ShardStats).paragraphs for shard in shards)
    if len(shards) != manifest.shards or sum(sizes) != manifest.paragraphs:
        message = f"Shards at {directory} do not match their manifest"
        raise ValueError(message)
    return Corpus(manifest=manifest, shards=tuple(shards), sizes=sizes)


def corpus_complete(config: WikipediaConfig, directory: Path) -> bool:
    try:
        load_corpus(config, directory)
    except (FileNotFoundError, ValueError):
        return False
    return True


def prepare(project: Project) -> None:
    config = load_config(project)
    root = project.settings.wikipedia_dir
    if corpus_complete(config, root / config.extraction.directory):
        return
    for name, sha1 in ((config.dump.index, config.dump.index_sha1), (config.dump.dump, config.dump.dump_sha1)):
        download(f"{config.dump.base_url}/{name}", root / name, sha1, config.dump)
    extract(config, root, project.settings.workers)


def bm25_tokenizer(vocabulary: Mapping[str, int] | None = None) -> Any:
    from bm25s.tokenization import Tokenizer

    tokenizer = Tokenizer()
    if vocabulary is not None:
        tokenizer.word_to_id = dict(vocabulary)
    return tokenizer


def count_shard(job: CountJob) -> Path:
    from scipy import sparse

    if job.output.exists():
        return job.output
    tokenizer = bm25_tokenizer(job.vocabulary)
    indices, pointers = array("i"), array("q", [0])
    with gzip.open(job.source, "rb") as lines:
        for line in lines:
            text = orjson.loads(line)[1]
            indices.extend(next(tokenizer.streaming_tokenize([text], update_vocab=False)))
            pointers.append(len(indices))
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(pointers, dtype=np.int64),
        ),
        shape=(len(pointers) - 1, len(job.vocabulary)),
    )
    matrix.sum_duplicates()
    temporary = job.output.with_name(f"{job.output.stem}.{os.getpid()}.tmp.npz")
    sparse.save_npz(temporary, matrix)
    temporary.replace(job.output)
    return job.output


def index_from_counts(counts: Any, vocabulary: Mapping[str, int], config: RetrievalConfig) -> Any:
    import bm25s

    n_docs, n_vocab = counts.shape
    lengths = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    frequencies = np.bincount(counts.indices, minlength=n_vocab)
    idf = np.array(
        [math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) if df else 0.0 for df in frequencies],
        dtype=np.float32,
    )
    denominator = config.k1 * ((1 - config.b) + config.b * lengths / lengths.mean())
    repeated = np.repeat(denominator, np.diff(counts.indptr))
    counts.data = (idf[counts.indices] * (counts.data / (repeated + counts.data))).astype(np.float32)
    scores = counts.tocsc()
    retriever = bm25s.BM25(k1=config.k1, b=config.b, method="lucene", backend="numba")
    retriever.nonoccurrence_array = None
    retriever.scores = {"data": scores.data, "indices": scores.indices, "indptr": scores.indptr, "num_docs": n_docs}
    retriever.vocab_dict = dict(vocabulary)
    retriever.unique_token_ids_set = set(vocabulary.values())
    return retriever


def select_paragraphs(job: SelectJob) -> dict[int, tuple[str, str]]:
    selected: dict[int, tuple[str, str]] = {}
    last = max(job.wanted)
    with gzip.open(job.source, "rb") as lines:
        for local, line in enumerate(lines):
            if local in job.wanted:
                row = orjson.loads(line)
                selected[job.offset + local] = (row[0], row[1])
            if local == last:
                break
    return selected


def read_paragraphs(corpus: Corpus, ids: Iterable[int], workers: int) -> dict[int, tuple[str, str]]:
    wanted = sorted(set(ids))
    jobs, offset = [], 0
    for shard, size in zip(corpus.shards, corpus.sizes, strict=True):
        local = wanted[bisect.bisect_left(wanted, offset) : bisect.bisect_left(wanted, offset + size)]
        if local:
            jobs.append(SelectJob(source=shard, offset=offset, wanted=frozenset(i - offset for i in local)))
        offset += size
    paragraphs_by_id: dict[int, tuple[str, str]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for selected in pool.map(select_paragraphs, jobs):
            paragraphs_by_id.update(selected)
    return paragraphs_by_id


def keeps(subject: str, title: str, text: str, config: ExtractionConfig) -> bool:
    combined = title + config.title_separator + text
    return subject in combined or subject.capitalize() in combined


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_retrieval(
    config: WikipediaConfig,
    corpus: Corpus,
    cache: Path,
    subjects: Sequence[str],
    workers: int,
) -> tuple[Retrieval, dict[int, tuple[str, str]]]:
    import bm25s
    from scipy import sparse

    settings = config.retrieval
    tokenizer = bm25_tokenizer()
    queries = tokenizer.tokenize(list(subjects), show_progress=False)
    vocabulary: dict[str, int] = tokenizer.word_to_id
    counts_dir = cache / settings.counts_dir / fingerprint([vocabulary, corpus.manifest.model_dump(mode="json")])
    counts_dir.mkdir(parents=True, exist_ok=True)
    jobs = [CountJob(source=s, output=counts_dir / f"{s.name}.npz", vocabulary=vocabulary) for s in corpus.shards]
    print(f"Counting {len(vocabulary)} query terms over {len(jobs)} shards for {len(subjects)} subjects")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        outputs = list(pool.map(count_shard, jobs))
    matrix = sparse.vstack([sparse.load_npz(path) for path in outputs], format="csr")
    if matrix.shape[0] != corpus.manifest.paragraphs:
        message = f"Counted {matrix.shape[0]} paragraphs, manifest has {corpus.manifest.paragraphs}"
        raise ValueError(message)
    retriever = index_from_counts(matrix, vocabulary, settings)
    del matrix
    ids, scores = [], []
    for start in range(0, len(queries), settings.query_batch_size):
        batch_ids, batch_scores = retriever.retrieve(
            queries[start : start + settings.query_batch_size],
            k=settings.k,
            return_as="tuple",
            n_threads=min(settings.threads, workers),
            show_progress=False,
        )
        ids.extend(batch_ids)
        scores.extend(batch_scores)
    id_matrix, score_matrix = np.array(ids), np.array(scores)
    texts = read_paragraphs(corpus, (int(i) for i in id_matrix.flat), workers)
    hits = {}
    for subject, row_ids, row_scores in zip(subjects, id_matrix.tolist(), score_matrix.tolist(), strict=True):
        keep = tuple(keeps(subject, *texts[i], config.extraction) for i in row_ids)
        hits[subject] = SubjectHits(ids=tuple(row_ids), scores=tuple(row_scores), keep=keep)
    retrieval = Retrieval(
        corpus=corpus.manifest,
        method="lucene",
        k=settings.k,
        k1=settings.k1,
        b=settings.b,
        bm25s_version=bm25s.__version__,
        vocabulary_size=len(vocabulary),
        zero_score_hits=int(np.sum(score_matrix == 0)),
        subjects=hits,
    )
    return retrieval, texts


def write_paragraph_cache(path: Path, cache: ParagraphCache, level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(gzip.compress(cache.model_dump_json().encode(), compresslevel=level, mtime=0))
    temporary.replace(path)


def query_subjects(project: Project) -> list[str]:
    relations = data.entity_relations(project.data / "relations", project.protocol.dataset)
    return sorted({sample.subject for relation in relations for sample in relation.samples})


def retrieve(project: Project, workers: int) -> None:
    config = load_config(project)
    corpus = load_corpus(config, corpus_dir(project, config))
    subjects = query_subjects(project)
    retrieval, texts = build_retrieval(config, corpus, project.cache / config.retrieval.cache_dir, subjects, workers)
    path = retrieval_path(project, config)
    write(path, retrieval)
    cache = ParagraphCache(retrieval_sha256=sha256(path), paragraphs=texts)
    write_paragraph_cache(paragraphs_cache_path(project, config), cache, config.retrieval.cache_compression_level)
    print(f"Wrote retrievals for {len(subjects)} subjects to {path}")


def retrieved_texts(project: Project, config: WikipediaConfig, workers: int) -> dict[int, tuple[str, str]]:
    path = retrieval_path(project, config)
    digest = sha256(path)
    cache_path = paragraphs_cache_path(project, config)
    if cache_path.exists():
        cache = ParagraphCache.model_validate_json(gzip.decompress(cache_path.read_bytes()))
        if cache.retrieval_sha256 == digest:
            return cache.paragraphs
    retrieval = read(path, Retrieval)
    corpus = load_corpus(config, corpus_dir(project, config))
    if corpus.manifest != retrieval.corpus:
        message = "The extracted corpus differs from the one used for retrieval"
        raise ValueError(message)
    texts = read_paragraphs(corpus, (i for hits in retrieval.subjects.values() for i in hits.ids), workers)
    cache = ParagraphCache(retrieval_sha256=digest, paragraphs=texts)
    write_paragraph_cache(cache_path, cache, config.retrieval.cache_compression_level)
    return texts


def stopwords(directory: Path, config: AttributeConfig) -> frozenset[str]:
    import nltk
    from nltk.corpus.reader import WordListCorpusReader

    root = directory / "corpora" / config.stopwords_package
    path = root / config.stopwords_language
    if not path.exists() and not nltk.download(config.stopwords_package, download_dir=str(directory), quiet=True):
        message = f"Could not download NLTK {config.stopwords_package!r}"
        raise RuntimeError(message)
    actual = sha256(path)
    if actual != config.stopwords_sha256:
        message = f"NLTK stopwords SHA-256 {actual} does not match pinned {config.stopwords_sha256}"
        raise ValueError(message)
    data_path = str(directory.resolve())
    if data_path not in nltk.data.path:
        nltk.data.path.insert(0, data_path)
    return frozenset(WordListCorpusReader(str(root.resolve()), [config.stopwords_language]).words())


def attribute_tokens(decoded: Iterable[str], stop: frozenset[str], min_length: int) -> frozenset[str]:
    return frozenset(s for s in decoded if len(s.strip()) >= min_length and s.strip() not in stop)


def attribute_rate(
    predictions: Sequence[str], attributes: frozenset[str], stop: frozenset[str], min_length: int
) -> tuple[float, int, int]:
    clean = [s for s in predictions if len(s.strip()) >= min_length and s.strip().lower() not in stop]
    if not clean:
        return 0.0, 0, 0
    overlap = len(set(clean) & attributes)
    return 100.0 * overlap / len(clean), overlap, len(clean)


def decoded_vocabulary(tokenizer: PreTrainedTokenizerBase, rows: Iterable[LensRow]) -> list[str]:
    size = max(len(tokenizer), 1 + max((i for row in rows for i in row.logit_lens_top_ids), default=-1))
    decoded: list[str] = tokenizer.batch_decode([[i] for i in range(size)])
    return decoded


def subject_attributes(
    tokenizer: PreTrainedTokenizerBase,
    decoded: Sequence[str],
    documents: Mapping[str, str],
    stop: frozenset[str],
    config: AttributeConfig,
) -> dict[str, frozenset[str]]:
    subjects = sorted(documents)
    attributes = {}
    for start in range(0, len(subjects), config.encode_batch_size):
        batch = subjects[start : start + config.encode_batch_size]
        encoded = tokenizer([documents[s] for s in batch], add_special_tokens=False)["input_ids"]
        for subject, ids in zip(batch, encoded, strict=True):
            attributes[subject] = attribute_tokens((decoded[i] for i in set(ids)), stop, config.min_token_length)
    return attributes


def kept_text(hits: SubjectHits, texts: Mapping[int, tuple[str, str]]) -> str:
    return "".join(" " + texts[i][1] for i, keep in zip(hits.ids, hits.keep, strict=True) if keep)


def relation_rates(
    rows: Sequence[LensRow],
    decoded: Sequence[str],
    attributes: Mapping[str, frozenset[str]],
    stop: frozenset[str],
    config: AttributeConfig,
) -> RelationAttributeRates:
    results = []
    for row in rows:
        if len(row.logit_lens_top_ids) < config.top_k:
            message = f"{row.subject}: only {len(row.logit_lens_top_ids)} logit-lens ids, need {config.top_k}"
            raise ValueError(message)
        predictions = [decoded[i] for i in row.logit_lens_top_ids[: config.top_k]]
        rate, overlap, count = attribute_rate(predictions, attributes[row.subject], stop, config.min_token_length)
        results.append(
            AttributeRow(subject=row.subject, object=row.object, rate=rate, overlap=overlap, filtered_predictions=count)
        )
    return RelationAttributeRates(
        n_rows=len(results),
        mean_rate=float(np.mean([r.rate for r in results])),
        empty_rows=sum(r.filtered_predictions == 0 for r in results),
        rows=tuple(results),
    )


def score_rows(
    rows_by_relation: Mapping[str, Sequence[LensRow]],
    tokenizer: PreTrainedTokenizerBase,
    retrieval: Retrieval,
    texts: Mapping[int, tuple[str, str]],
    stop: frozenset[str],
    config: AttributeConfig,
) -> dict[str, RelationAttributeRates]:
    all_rows = [row for rows in rows_by_relation.values() for row in rows]
    subjects = {row.subject for row in all_rows}
    missing = subjects - set(retrieval.subjects)
    if missing:
        message = f"{len(missing)} subjects were not retrieved, e.g. {sorted(missing)[:3]}"
        raise KeyError(message)
    decoded = decoded_vocabulary(tokenizer, all_rows)
    documents = {s: kept_text(retrieval.subjects[s], texts) for s in subjects}
    attributes = subject_attributes(tokenizer, decoded, documents, stop, config)
    return {key: relation_rates(rows, decoded, attributes, stop, config) for key, rows in rows_by_relation.items()}


def load_tokenizer(hf_id: str, revision: str) -> PreTrainedTokenizerBase:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    return tokenizer


def score(project: Project, model_key: str) -> Path:
    config = load_config(project)
    spec = project.models.get(model_key)
    measurements = storage.load_measurements(project, model_key)
    path = retrieval_path(project, config)
    retrieval = read(path, Retrieval)
    texts = retrieved_texts(project, config, project.settings.workers)
    stop = stopwords(project.cache / config.attributes.nltk_dir, config.attributes)
    rows = {key: measurement.lens.rows for key, measurement in measurements.items()}
    relations = score_rows(rows, load_tokenizer(spec.hf_id, spec.revision), retrieval, texts, stop, config.attributes)
    record = AttributeRates(
        model=model_key,
        hf_id=spec.hf_id,
        revision=spec.revision,
        retrieval_sha256=sha256(path),
        stopwords_sha256=config.attributes.stopwords_sha256,
        top_k=config.attributes.top_k,
        min_token_length=config.attributes.min_token_length,
        n_subjects=len({row.subject for relation_rows in rows.values() for row in relation_rows}),
        relations=relations,
    )
    output = attribute_rates_path(project, config, model_key)
    write(output, record)
    return output
