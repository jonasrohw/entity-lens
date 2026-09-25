import bz2
import gzip
import hashlib
import io
import json
import re
import shutil
import urllib.error
from pathlib import Path

import bm25s
import numpy as np
import orjson
import pytest
from scipy import sparse
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from entity_lens import wikipedia
from entity_lens.config import Project, Settings
from entity_lens.data import Sample
from entity_lens.records import LensMeasurement, LensRow, RelationMeasurement, read

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STOPWORDS = "the\nof\nand\nis\nin\n"
STOPWORDS_SHA256 = hashlib.sha256(STOPWORDS.encode()).hexdigest()

PAGES = [
    (
        "Paris",
        (
            "Paris is in [[France]].\n\n== History ==\nAn old city.\n\n=== Modern ===\n\n"
            "'''Today''' big.\n\n== Culture ==\nArt."
        ),
    ),
    ("Talk:Paris", "Exclude me."),
    ("Empty", ""),
    ("London", "London is in England.\n\nLondon has the Thames."),
]


def replace(text, key, value):
    return re.sub(rf"(?m)^{key} = .*$", f"{key} = {value}", text, count=1)


def make_project(tmp_path, **overrides):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ("models", "protocol"):
        shutil.copy(CONFIGS / f"{name}.toml", config_dir / f"{name}.toml")
    text = (CONFIGS / "wikipedia.toml").read_text()
    values = {"k": "3", "top_k": "4", "stopwords_sha256": f'"{STOPWORDS_SHA256}"', "streams_per_shard": "1"}
    for key, value in {**values, **overrides}.items():
        text = replace(text, key, value)
    (config_dir / "wikipedia.toml").write_text(text)
    settings = Settings(
        config_dir=config_dir,
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        wikipedia_dir=tmp_path / "wikipedia",
        workers=2,
    )
    return Project(settings)


def page_xml(title, text):
    return f"<page><title>{title}</title><ns>0</ns><revision><text>{text}</text></revision></page>\n"


def build_dump(directory, pages=PAGES):
    directory.mkdir(parents=True, exist_ok=True)
    streams = [
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/"><siteinfo><sitename>W</sitename></siteinfo>\n'
        + "".join(page_xml(*p) for p in pages[:3]),
        "".join(page_xml(*p) for p in pages[3:]) + "</mediawiki>\n",
    ]
    compressed = [bz2.compress(s.encode()) for s in streams]
    offsets = [0, len(compressed[0])]
    lines = [f"{offsets[0]}:{i}:{title}" for i, (title, _) in enumerate(pages[:3])]
    lines += [f"{offsets[1]}:{i + 3}:{title}" for i, (title, _) in enumerate(pages[3:])]
    dump = b"".join(compressed)
    index = bz2.compress(("\n".join(lines) + "\n").encode())
    return dump, index


def sha1(payload):
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def dump_project(tmp_path, **overrides):
    dump, index = build_dump(tmp_path / "source")
    project = make_project(tmp_path, dump_sha1=f'"{sha1(dump)}"', index_sha1=f'"{sha1(index)}"', **overrides)
    return project, dump, index


def extractor():
    from wikiextractor.extract import Extractor

    return Extractor(id=None, revid=None, urlbase=None, title="", page=None)


def extraction_config(tmp_path):
    return wikipedia.load_config(make_project(tmp_path)).extraction


def test_config_matches_upstream_protocol():
    config = wikipedia.load_config(Project(Settings(config_dir=CONFIGS)))
    assert config.extraction.directory == "olmo-main"
    assert (config.retrieval.k, config.retrieval.k1, config.retrieval.b) == (100, 1.5, 0.75)
    assert (config.attributes.top_k, config.attributes.min_token_length) == (50, 3)
    assert config.dump.dump_sha1 == "64714517eae47fcce24803feb90b4a440e963fe0"


def test_paragraph_splitting_and_section_titles(tmp_path):
    config = extraction_config(tmp_path)
    rows = list(wikipedia.paragraphs(*PAGES[0], extractor(), config))
    assert rows == [
        ("Paris", "Paris is in France."),
        ("Paris. History", "An old city."),
        ("Paris. History. Modern", "Today big."),
        ("Paris. Culture", "Art."),
    ]
    assert list(wikipedia.paragraphs("Talk:Paris", "Text.", extractor(), config)) == []
    assert list(wikipedia.paragraphs("Paris", "", extractor(), config)) == []
    assert list(wikipedia.paragraphs("", "Orphan text.", extractor(), config)) == [("[UNK]", "Orphan text.")]
    assert list(wikipedia.paragraphs("A", "Line one\nline two\n\n\n\n", extractor(), config)) == [
        ("A", "Line one line two")
    ]


def test_extraction_writes_shards_and_manifest(tmp_path):
    project, dump, index = dump_project(tmp_path)
    config = wikipedia.load_config(project)
    root = project.settings.wikipedia_dir
    root.mkdir()
    (root / config.dump.dump).write_bytes(dump)
    (root / config.dump.index).write_bytes(index)
    manifest = wikipedia.extract(config, root, workers=1)
    assert (manifest.shards, manifest.pages, manifest.paragraphs) == (2, 4, 6)
    corpus = wikipedia.load_corpus(config, root / config.extraction.directory)
    assert [p.name for p in corpus.shards] == ["paragraphs-0000000.jsonl.gz", "paragraphs-0000001.jsonl.gz"]
    assert corpus.sizes == (4, 2)
    with gzip.open(corpus.shards[1], "rb") as lines:
        assert [orjson.loads(line) for line in lines] == [
            ["London", "London is in England."],
            ["London", "London has the Thames."],
        ]
    shard = corpus.shards[0]
    before = shard.stat().st_mtime_ns
    assert wikipedia.extract(config, root, workers=1) == manifest
    assert shard.stat().st_mtime_ns == before


def test_extract_shard_in_process(tmp_path):
    config = extraction_config(tmp_path)
    dump, _ = build_dump(tmp_path)
    (tmp_path / "dump.bz2").write_bytes(dump)
    output = tmp_path / "shard.jsonl.gz"
    job = wikipedia.ShardJob(dump=tmp_path / "dump.bz2", offsets=(0, len(dump)), output=output, config=config)
    stats = wikipedia.extract_shard(job)
    assert (stats.pages, stats.paragraphs) == (4, 6)
    assert read(Path(f"{output}.json"), wikipedia.ShardStats) == stats
    with gzip.open(output, "rb") as lines:
        assert [orjson.loads(line)[0] for line in lines] == [
            "Paris",
            "Paris. History",
            "Paris. History. Modern",
            "Paris. Culture",
            "London",
            "London",
        ]


def test_stream_offsets_deduplicate_consecutive_offsets(tmp_path):
    index = tmp_path / "index.bz2"
    index.write_bytes(bz2.compress(b"0:1:A\n0:2:B:C\n10:3:D\n"))
    assert wikipedia.stream_offsets(index, 25) == [0, 10, 25]


def write_corpus(directory, shards, manifest_extra=None):
    directory.mkdir(parents=True)
    for number, rows in enumerate(shards):
        path = directory / f"paragraphs-{number:07d}.jsonl.gz"
        with gzip.open(path, "wb") as sink:
            for title, text in rows:
                sink.write(orjson.dumps([title, text, [0]]) + b"\n")
        stats = {"pages": len(rows), "paragraphs": len(rows), "bytes": path.stat().st_size}
        Path(f"{path}.json").write_text(json.dumps(stats))
    manifest = {
        "config_fingerprint": "legacy",
        "dump_sha1": "64714517eae47fcce24803feb90b4a440e963fe0",
        "index_sha1": "7acba6cb378de9fd0a71b8a2a31f81573d03b02e",
        "complete": True,
        "shards": len(shards),
        "pages": sum(map(len, shards)),
        "paragraphs": sum(map(len, shards)),
        **(manifest_extra or {}),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


CORPUS = [
    [
        ("Paris", "Paris is the capital of France."),
        ("France", "The French capital is Paris; paris is large."),
        ("Nothing", "Unrelated words only."),
    ],
    [
        ("New York", "New York City is in New York state."),
        ("York", "York is an old English city, not new."),
        ("Empty", "..."),
        ("abraham lincoln", "Lincoln was a president."),
    ],
]


def test_load_corpus_rejects_incomplete_and_mismatched(tmp_path):
    config = extraction_config(tmp_path / "config")
    project_config = wikipedia.load_config(make_project(tmp_path / "config"))
    with pytest.raises(FileNotFoundError):
        wikipedia.load_corpus(project_config, tmp_path / "missing")
    write_corpus(tmp_path / "incomplete", CORPUS, {"complete": False})
    assert not wikipedia.corpus_complete(project_config, tmp_path / "incomplete")
    write_corpus(tmp_path / "count", CORPUS, {"paragraphs": 99})
    with pytest.raises(ValueError, match="do not match"):
        wikipedia.load_corpus(project_config, tmp_path / "count")
    write_corpus(tmp_path / "good", CORPUS)
    assert wikipedia.corpus_complete(project_config, tmp_path / "good")
    assert config.shard_glob == "paragraphs-*.jsonl.gz"


def upstream_index(texts, queries_text):
    tokenizer = bm25s.tokenization.Tokenizer()
    queries = tokenizer.tokenize(queries_text, show_progress=False)
    ids = tokenizer.tokenize(texts, update_vocab=False, show_progress=False)
    upstream = bm25s.BM25(method="lucene", backend="numba")
    upstream.index((ids, tokenizer.word_to_id), show_progress=False)
    return tokenizer, queries, ids, upstream


def test_vectorized_bm25_exactly_matches_upstream(tmp_path):
    config = wikipedia.load_config(make_project(tmp_path)).retrieval
    texts = ["Paris Paris", "New York", "Paris and New York", "Nothing relevant", "New New York", "", "the of"]
    tokenizer, queries, ids, upstream = upstream_index(texts, ["Paris", "New York", "nonexistent", "York Paris"])
    counts = np.zeros((len(ids), len(tokenizer.word_to_id)), dtype=np.float32)
    for row, tokens in enumerate(ids):
        for token in tokens:
            counts[row, token] += 1
    optimized = wikipedia.index_from_counts(sparse.csr_matrix(counts), tokenizer.word_to_id, config)
    for key in ("indices", "indptr", "data"):
        np.testing.assert_array_equal(optimized.scores[key], upstream.scores[key])
    assert optimized.scores["num_docs"] == upstream.scores["num_docs"]
    expected = upstream.retrieve(queries, k=5, show_progress=False, n_threads=1)
    actual = optimized.retrieve(queries, k=5, show_progress=False, n_threads=1)
    np.testing.assert_array_equal(actual.documents, expected.documents)
    np.testing.assert_array_equal(actual.scores, expected.scores)


def test_count_shard_uses_query_vocabulary_only(tmp_path):
    write_corpus(tmp_path / "corpus", CORPUS)
    texts = [text for shard in CORPUS for _, text in shard]
    tokenizer, _, ids, _ = upstream_index(texts, ["Paris", "New York", "Abraham Lincoln"])
    source = tmp_path / "corpus" / "paragraphs-0000001.jsonl.gz"
    output = tmp_path / "counts.npz"
    job = wikipedia.CountJob(source=source, output=output, vocabulary=tokenizer.word_to_id)
    assert wikipedia.count_shard(job) == output
    matrix = sparse.load_npz(output).toarray()
    expected = np.zeros_like(matrix)
    for row, tokens in enumerate(ids[3:]):
        for token in tokens:
            expected[row, token] += 1
    np.testing.assert_array_equal(matrix, expected)
    output.write_bytes(b"cached")
    assert wikipedia.count_shard(job) == output
    assert output.read_bytes() == b"cached"


def test_keep_rule(tmp_path):
    config = extraction_config(tmp_path)
    assert wikipedia.keeps("Paris", "France", "Paris is big.", config)
    assert wikipedia.keeps("Paris", "Paris", "It is big.", config)
    assert wikipedia.keeps("iPhone", "Phones", "The IPhone.", config) is False
    assert wikipedia.keeps("abraham lincoln", "Abraham lincoln", "text", config)
    assert not wikipedia.keeps("abraham lincoln", "Abraham Lincoln", "text", config)
    assert wikipedia.keeps("York ", "York", "York is old.", config)
    assert not wikipedia.keeps("York ", "York", "text", config)
    assert not wikipedia.keeps("Paris France", "Paris", "France", config)
    assert wikipedia.keeps("Paris. France", "Paris", "France", config)


def test_retrieval_matches_upstream_bm25_and_keep_rule(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    write_corpus(project.settings.wikipedia_dir / "olmo-main", CORPUS)
    subjects = ["New York", "Paris", "abraham lincoln", "zzz"]
    monkeypatch.setattr(wikipedia, "query_subjects", lambda _: subjects)
    wikipedia.retrieve(project, workers=2)
    retrieval = read(project.artifacts / "wikipedia" / "retrieval.json.gz", wikipedia.Retrieval)
    texts = [text for shard in CORPUS for _, text in shard]
    titles = [title for shard in CORPUS for title, _ in shard]
    _, queries, _, upstream = upstream_index(texts, subjects)
    expected = upstream.retrieve(queries, k=3, show_progress=False, n_threads=1)
    assert list(retrieval.subjects) == subjects
    for subject, ids, scores in zip(subjects, expected.documents, expected.scores, strict=True):
        hits = retrieval.subjects[subject]
        assert hits.ids == tuple(int(i) for i in ids)
        assert hits.scores == tuple(float(s) for s in scores)
        combined = [titles[i] + ". " + texts[i] for i in ids]
        assert hits.keep == tuple(subject in c or subject.capitalize() in c for c in combined)
    assert retrieval.subjects["Paris"].keep == (True, True, False)
    assert retrieval.subjects["abraham lincoln"].keep[0]
    assert retrieval.zero_score_hits == int(np.sum(expected.scores == 0))
    assert retrieval.corpus.config_fingerprint == "legacy"
    config = wikipedia.load_config(project)
    cached = wikipedia.retrieved_texts(project, config, workers=1)
    assert cached[3] == ("New York", "New York City is in New York state.")
    wikipedia.paragraphs_cache_path(project, config).unlink()
    assert wikipedia.retrieved_texts(project, config, workers=1) == cached
    assert wikipedia.paragraphs_cache_path(project, config).exists()
    manifest = project.settings.wikipedia_dir / "olmo-main" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"legacy"', '"changed"'))
    wikipedia.paragraphs_cache_path(project, config).unlink()
    with pytest.raises(ValueError, match="differs"):
        wikipedia.retrieved_texts(project, config, workers=1)


def test_retrieval_rejects_paragraph_count_mismatch(tmp_path):
    project = make_project(tmp_path)
    config = wikipedia.load_config(project)
    directory = tmp_path / "corpus"
    write_corpus(directory, CORPUS)
    corpus = wikipedia.load_corpus(config, directory)
    broken = corpus.model_copy(update={"manifest": corpus.manifest.model_copy(update={"paragraphs": 8})})
    with pytest.raises(ValueError, match="Counted 7"):
        wikipedia.build_retrieval(config, broken, tmp_path / "cache", ["Paris"], workers=1)


def test_select_paragraphs_accepts_extra_trailing_elements(tmp_path):
    write_corpus(tmp_path / "corpus", CORPUS)
    job = wikipedia.SelectJob(source=tmp_path / "corpus" / "paragraphs-0000001.jsonl.gz", offset=3, wanted={0, 2})
    assert wikipedia.select_paragraphs(job) == {3: CORPUS[1][0], 5: CORPUS[1][2]}


def test_query_subjects_are_sorted_union(tmp_path, monkeypatch):
    project = make_project(tmp_path)

    class FakeRelation:
        def __init__(self, subjects):
            self.samples = [Sample(subject=s, object="x") for s in subjects]

    relations = [FakeRelation(["b", "a "]), FakeRelation(["a ", "c"])]
    monkeypatch.setattr(wikipedia.data, "entity_relations", lambda *_: relations)
    assert wikipedia.query_subjects(project) == ["a ", "b", "c"]


def test_attribute_filters_are_asymmetric():
    stop = frozenset({"the", "and"})
    decoded = [" The", " the", "an", " and", "Paris", " Paris", "  ab ", "abc"]
    assert wikipedia.attribute_tokens(decoded, stop, 3) == frozenset({" The", "Paris", " Paris", "abc"})
    rate, overlap, count = wikipedia.attribute_rate([" The", "Paris", " x", "Paris", "Lyon"], {"Paris"}, stop, 3)
    assert (overlap, count) == (1, 3)
    assert rate == pytest.approx(100 / 3)


def test_empty_filtered_predictions_score_zero():
    assert wikipedia.attribute_rate([" the", "ab", " "], frozenset({"ab"}), frozenset({"the"}), 3) == (0.0, 0, 0)
    assert wikipedia.attribute_rate(["Paris"], frozenset(), frozenset(), 3) == (0.0, 0, 1)


def test_kept_text_joins_kept_hits_with_leading_spaces():
    hits = wikipedia.SubjectHits(ids=(2, 0, 1), scores=(3.0, 2.0, 1.0), keep=(True, False, True))
    texts = {0: ("T0", "zero"), 1: ("T1", "one"), 2: ("T2", "two")}
    assert wikipedia.kept_text(hits, texts) == " two one"


def fake_tokenizer():
    words = ["[UNK]", "the", "Paris", "France", "capital", "is", "of", "ab", "Lyon", "city", "York", "New"]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def lens_row(subject, ids):
    return LensRow.model_construct(subject=subject, object="o", logit_lens_top_ids=tuple(ids))


def measurement(rows):
    return RelationMeasurement.model_construct(lens=LensMeasurement.model_construct(rows=tuple(rows)))


def test_score_rows_and_decoded_vocabulary():
    tokenizer = fake_tokenizer()
    config = wikipedia.WikipediaConfig.model_validate(Project(Settings(config_dir=CONFIGS)).load("wikipedia"))
    settings = config.attributes.model_copy(update={"top_k": 4, "encode_batch_size": 1})
    retrieval = wikipedia.Retrieval(
        corpus=wikipedia.CorpusManifest(dump_sha1="a", index_sha1="b", complete=True, shards=1, pages=1, paragraphs=3),
        method="lucene",
        k=2,
        k1=1.5,
        b=0.75,
        bm25s_version="0.3.11",
        vocabulary_size=1,
        zero_score_hits=0,
        subjects={
            "Paris": wikipedia.SubjectHits(ids=(0, 1), scores=(2.0, 1.0), keep=(True, False)),
            "York": wikipedia.SubjectHits(ids=(2, 1), scores=(2.0, 1.0), keep=(False, False)),
        },
    )
    texts = {0: ("Paris", "Paris is the capital of France"), 1: ("x", "Lyon city"), 2: ("y", "New York")}
    rows = {
        "a": [lens_row("Paris", [2, 3, 4, 1, 9]), lens_row("Paris", [1, 5, 7, 6, 2])],
        "b": [lens_row("York", [10, 11, 2, 3, 12])],
    }
    stop = frozenset({"the", "is", "of"})
    decoded = wikipedia.decoded_vocabulary(tokenizer, [r for rs in rows.values() for r in rs])
    assert len(decoded) == 13
    assert decoded[:3] == ["[UNK]", "the", "Paris"]
    result = wikipedia.score_rows(rows, tokenizer, retrieval, texts, stop, settings)
    first, second = result["a"].rows
    assert (first.overlap, first.filtered_predictions, first.rate) == (3, 3, 100.0)
    assert (second.overlap, second.filtered_predictions, second.rate) == (0, 0, 0.0)
    assert (result["a"].mean_rate, result["a"].empty_rows, result["a"].n_rows) == (50.0, 1, 2)
    third = result["b"].rows[0]
    assert (third.overlap, third.filtered_predictions, third.rate) == (0, 4, 0.0)
    with pytest.raises(KeyError, match="not retrieved"):
        wikipedia.score_rows({"c": [lens_row("Lyon", [1, 2, 3, 4])]}, tokenizer, retrieval, texts, stop, settings)
    with pytest.raises(ValueError, match="need 4"):
        wikipedia.score_rows({"c": [lens_row("Paris", [1, 2])]}, tokenizer, retrieval, texts, stop, settings)


def write_stopwords(directory):
    path = directory / "corpora" / "stopwords" / "english"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STOPWORDS)


def test_stopwords_download_and_checksum(tmp_path, monkeypatch):
    import nltk

    config = wikipedia.load_config(make_project(tmp_path)).attributes
    calls = []

    def fake_download(package, download_dir, quiet):
        calls.append((package, download_dir, quiet))
        write_stopwords(Path(download_dir))
        return True

    monkeypatch.setattr(nltk, "download", fake_download)
    directory = tmp_path / "nltk"
    assert wikipedia.stopwords(directory, config) == frozenset({"the", "of", "and", "is", "in"})
    assert calls == [("stopwords", str(directory), True)]
    assert wikipedia.stopwords(directory, config) == frozenset({"the", "of", "and", "is", "in"})
    assert len(calls) == 1
    (directory / "corpora" / "stopwords" / "english").write_text("changed\n")
    with pytest.raises(ValueError, match="does not match"):
        wikipedia.stopwords(directory, config)
    monkeypatch.setattr(nltk, "download", lambda *_, **__: False)
    with pytest.raises(RuntimeError, match="Could not download"):
        wikipedia.stopwords(tmp_path / "other", config)


def test_score_end_to_end(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    write_corpus(project.settings.wikipedia_dir / "olmo-main", CORPUS)
    monkeypatch.setattr(wikipedia, "query_subjects", lambda _: ["New York", "Paris"])
    wikipedia.retrieve(project, workers=1)
    write_stopwords(project.cache / "nltk")
    measurements = {
        "rel": measurement([lens_row("Paris", [2, 4, 1, 5]), lens_row("New York", [1, 5, 6, 7])]),
    }
    monkeypatch.setattr(wikipedia.storage, "load_measurements", lambda *_: measurements)
    requested = []

    def fake_load(hf_id, revision):
        requested.append((hf_id, revision))
        return fake_tokenizer()

    monkeypatch.setattr(wikipedia, "load_tokenizer", fake_load)
    path = wikipedia.score(project, "gptj")
    assert path == project.artifacts / "attribute_rates" / "gptj.json"
    record = read(path, wikipedia.AttributeRates)
    spec = project.models.get("gptj")
    assert requested == [(spec.hf_id, spec.revision)]
    assert record.n_subjects == 2
    relation = record.relations["rel"]
    assert [(r.overlap, r.filtered_predictions) for r in relation.rows] == [(2, 2), (0, 0)]
    assert (relation.mean_rate, relation.empty_rows) == (50.0, 1)


def test_load_tokenizer_passes_revision(monkeypatch):
    calls = []
    monkeypatch.setattr(
        wikipedia.AutoTokenizer, "from_pretrained", lambda hf_id, revision: calls.append((hf_id, revision)) or "tok"
    )
    assert wikipedia.load_tokenizer("org/model", "a" * 40) == "tok"
    assert calls == [("org/model", "a" * 40)]


class FakeResponse(io.BytesIO):
    def __init__(self, payload, status, length=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}


class FakeServer:
    def __init__(self, files, *, truncate_first=0, fail_first=0, ignore_range=False):
        self.files = files
        self.truncate_first = truncate_first
        self.fail_first = fail_first
        self.ignore_range = ignore_range
        self.ranges = []

    def __call__(self, request, timeout):
        payload = self.files[request.full_url]
        header = request.get_header("Range")
        self.ranges.append(header)
        if self.fail_first:
            self.fail_first -= 1
            message = "connection reset"
            raise ConnectionResetError(message)
        start = int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
        if header and not self.ignore_range:
            if start >= len(payload):
                raise urllib.error.HTTPError(request.full_url, 416, "range", {}, None)
            body, status = payload[start:], 206
        else:
            body, status = payload, 200
        if self.truncate_first:
            self.truncate_first -= 1
            return FakeResponse(body[: len(body) // 2], status, len(body))
        return FakeResponse(body, status)


def download_config(tmp_path, **update):
    return wikipedia.load_config(make_project(tmp_path)).dump.model_copy(update={"chunk_bytes": 7, **update})


def test_download_resumes_and_verifies(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 3
    server = FakeServer({"https://host/file": payload}, truncate_first=1, fail_first=1)
    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", server)
    target = tmp_path / "out" / "file"
    wikipedia.download("https://host/file", target, sha1(payload), download_config(tmp_path))
    assert target.read_bytes() == payload
    assert server.ranges == [None, None, f"bytes={len(payload) // 2}-"]
    assert not (tmp_path / "out" / "file.part").exists()
    wikipedia.download("https://host/unused", target, sha1(payload), download_config(tmp_path))
    with pytest.raises(ValueError, match="SHA-1 mismatch"):
        wikipedia.download("https://host/unused", target, sha1(b"other"), download_config(tmp_path))


def test_download_handles_complete_partial_and_full_responses(tmp_path, monkeypatch):
    payload = b"abcdefghij"
    partial = tmp_path / "file.part"
    partial.write_bytes(payload)
    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", FakeServer({"https://h/f": payload}))
    wikipedia.download("https://h/f", tmp_path / "file", sha1(payload), download_config(tmp_path))
    assert (tmp_path / "file").read_bytes() == payload
    (tmp_path / "second.part").write_bytes(b"XYZ")
    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", FakeServer({"https://h/f": payload}, ignore_range=True))
    wikipedia.download("https://h/f", tmp_path / "second", sha1(payload), download_config(tmp_path))
    assert (tmp_path / "second").read_bytes() == payload


def test_download_gives_up_and_discards_corrupt_files(tmp_path, monkeypatch):
    payload = b"abcdefghij"
    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", FakeServer({"https://h/f": payload}, fail_first=5))
    with pytest.raises(ConnectionResetError):
        wikipedia.download("https://h/f", tmp_path / "file", sha1(payload), download_config(tmp_path, retries=2))
    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", FakeServer({"https://h/f": payload}))
    with pytest.raises(ValueError, match="SHA-1 mismatch"):
        wikipedia.download("https://h/f", tmp_path / "file", sha1(b"x"), download_config(tmp_path))
    assert not (tmp_path / "file.part").exists()

    def forbidden(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, None)

    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", forbidden)
    with pytest.raises(urllib.error.HTTPError):
        wikipedia.download("https://h/f", tmp_path / "file", sha1(payload), download_config(tmp_path, retries=0))


def test_prepare_downloads_extracts_and_then_skips(tmp_path, monkeypatch):
    project, dump, index = dump_project(tmp_path)
    config = wikipedia.load_config(project)
    base = config.dump.base_url
    server = FakeServer({f"{base}/{config.dump.dump}": dump, f"{base}/{config.dump.index}": index})
    monkeypatch.setattr(wikipedia.urllib.request, "urlopen", server)
    wikipedia.prepare(project)
    corpus = wikipedia.load_corpus(config, wikipedia.corpus_dir(project, config))
    assert corpus.manifest.paragraphs == 6
    assert len(server.ranges) == 2
    (project.settings.wikipedia_dir / config.dump.dump).unlink()
    wikipedia.prepare(project)
    assert len(server.ranges) == 2
