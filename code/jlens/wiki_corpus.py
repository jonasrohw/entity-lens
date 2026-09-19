"""Rebuild Gerstner's 2021 Wikipedia paragraph corpus and BM25 retrieval.

The query-vocabulary restriction (including its effect on document lengths)
deliberately matches RW_functionalities/wiki/wiki_retrieve.py. This is not a
general-purpose full-vocabulary Lucene index. No subject/answer filtering is
applied before retrieval. All Wikipedia paragraphs contribute to corpus size.
"""

# Adapted from Gerstner et al., RW_functionalities/wiki (Apache-2.0).
# This version adds parallel extraction, safe checkpoints, and vectorized BM25.
# See docs/licenses/geva-gerstner-APACHE-2.0.txt and docs/geva-baseline.md.

import argparse
import bz2
import gzip
import hashlib
import json
import os
import time
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from jlens.io import read_json, write_json

DUMP_NAME = "enwiki-20211020-pages-articles-multistream"
DUMP_SIZE = 20152226680
DUMP_SHA1 = "64714517eae47fcce24803feb90b4a440e963fe0"
INDEX_SHA1 = "7acba6cb378de9fd0a71b8a2a31f81573d03b02e"
UPSTREAM_COMMIT = "18860447f7735d773d76329a619af25be46db6ac"
VERSION = 1


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked_sha1(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha1").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}: {actual} != {expected}")
    return actual


def paragraphs(title, text, extractor):
    """Match upstream splitting, namespace exclusion, and section-title handling."""
    if ":" in title or not text:
        return
    current_title = [title]
    for paragraph in text.split("\n\n"):
        if paragraph.startswith("="):
            parts = paragraph.split("\n", 1)
            depth = parts[0].count("=") // 2
            current_title = current_title[: depth - 1] + [parts[0].strip(" =\n")]
            if len(parts) == 1:
                continue
            paragraph = parts[1]
        cleaned = " ".join(extractor.clean_text(paragraph)).strip().replace("\n", " ")
        if not cleaned:
            continue
        cleaned_title = " ".join(extractor.clean_text(". ".join(current_title))).strip()
        yield cleaned_title or "[UNK]", cleaned


def process_chunk(job):
    import orjson
    from bm25s.tokenization import Tokenizer
    from defusedxml import ElementTree
    from wikiextractor.extract import Extractor

    dump, offsets, output, vocab = job
    tokenizer = Tokenizer()
    tokenizer.word_to_id = vocab
    extractor = Extractor(id=None, revid=None, urlbase=None, title="", page=None)
    n_pages = n_paragraphs = 0
    temporary = output + ".tmp"
    with (
        open(dump, "rb") as source,
        gzip.open(temporary, "wb", compresslevel=1) as dest,
    ):
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            source.seek(start)
            xml = bz2.decompress(source.read(end - start)).decode("utf-8")
            # First/last streams contain the mediawiki wrapper. Splitting on the
            # page opening tag also avoids the first-stream XML header bug upstream.
            for page in xml.split("<page>")[1:]:
                page = "<page>" + page.split("</page>", 1)[0] + "</page>"
                tree = ElementTree.fromstring(page)
                n_pages += 1
                title = tree.findtext("title") or ""
                text = tree.findtext("revision/text") or ""
                for heading, content in paragraphs(title, text, extractor):
                    # Avoid constructing tqdm (and taking its shared process
                    # lock) for every paragraph. Same tokenizer, same token IDs.
                    tokens = next(
                        tokenizer.streaming_tokenize([content], update_vocab=False)
                    )
                    dest.write(orjson.dumps([heading, content, tokens]) + b"\n")
                    n_paragraphs += 1
    os.replace(temporary, output)
    result = dict(
        pages=n_pages, paragraphs=n_paragraphs, bytes=Path(output).stat().st_size
    )
    write_json(output + ".json", result)
    return result


def prepare(args):
    from bm25s.tokenization import Tokenizer

    subjects = sorted(set(read_json(args.subjects)))
    tokenizer = Tokenizer()
    queries = tokenizer.tokenize(subjects, show_progress=False)
    config = dict(
        version=VERSION,
        subjects=subjects,
        queries=queries,
        vocab=tokenizer.word_to_id,
        upstream_commit=UPSTREAM_COMMIT,
        dump=DUMP_NAME,
        streams_per_shard=args.streams_per_shard,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing corpus configuration differs; use another --out")
    write_json(config_path, config)
    index = args.dump_dir / f"{DUMP_NAME}-index.txt.bz2"
    checked_sha1(index, INDEX_SHA1)
    offsets = []
    with bz2.open(index, "rt") as stream:
        for line in stream:
            offset = int(line.split(":", 1)[0])
            if not offsets or offsets[-1] != offset:
                offsets.append(offset)
    offsets.append(DUMP_SIZE)
    dump = args.dump_dir / f"{DUMP_NAME}.xml.bz2"
    jobs = []
    for i in range(0, len(offsets) - 1, args.streams_per_shard):
        chunk = offsets[i : i + args.streams_per_shard + 1]
        output = args.out / f"paragraphs-{i:07d}.jsonl.gz"
        if output.exists() and Path(str(output) + ".json").exists():
            continue
        jobs.append((str(dump), chunk, str(output), tokenizer.word_to_id))
    print(f"{len(offsets) - 1} streams; {len(jobs)} pending shards", flush=True)
    pending = set()
    next_job = completed = 0
    last_progress = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while next_job < len(jobs) or pending:
            available = dump.stat().st_size if dump.exists() else 0
            while next_job < len(jobs) and len(pending) < args.workers * 2:
                job = jobs[next_job]
                if job[1][-1] > available:
                    if not args.wait_for_download:
                        raise ValueError("Dump incomplete; use --wait-for-download")
                    break
                pending.add(pool.submit(process_chunk, job))
                next_job += 1
            if pending:
                done, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
            else:
                time.sleep(2)
            if time.monotonic() - last_progress > 30:
                print(
                    f"Prepared {completed}/{len(jobs)} shards; downloaded {available / 1e9:.2f} GB",
                    flush=True,
                )
                last_progress = time.monotonic()
    checked_sha1(dump, DUMP_SHA1)
    shards = sorted(args.out.glob("paragraphs-*.jsonl.gz.json"))
    totals = [read_json(p) for p in shards]
    manifest = dict(
        config_fingerprint=fingerprint(config),
        dump_sha1=DUMP_SHA1,
        index_sha1=INDEX_SHA1,
        complete=True,
        shards=len(shards),
        pages=sum(x["pages"] for x in totals),
        paragraphs=sum(x["paragraphs"] for x in totals),
    )
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def count_shard(job):
    """Materialize sparse integer counts; duplicate summation runs in SciPy."""
    import numpy as np
    import orjson
    from scipy import sparse

    path, n_vocab = job
    output = Path(str(path) + ".counts.npz")
    if output.exists():
        return str(output)
    indices, indptr = array("i"), array("q", [0])
    with gzip.open(path, "rb") as stream:
        for line in stream:
            ids = orjson.loads(line)[2]
            indices.extend(ids)
            indptr.append(len(indices))
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(indptr) - 1, n_vocab),
    )
    matrix.sum_duplicates()
    temporary = output.with_suffix(".tmp.npz")
    sparse.save_npz(temporary, matrix)
    temporary.replace(output)
    return str(output)


def index_from_counts(counts, vocab):
    """Vectorized equivalent of bm25s 0.3.11's Lucene scorer (float32 output)."""
    import math

    import bm25s
    import numpy as np

    n_docs, n_vocab = counts.shape
    lengths = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    average = lengths.mean()
    frequencies = np.bincount(counts.indices, minlength=n_vocab)
    idf = np.array(
        [
            math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) if df else 0.0
            for df in frequencies
        ],
        dtype=np.float32,
    )
    # BM25S computes each document's denominator in float64 and stores final
    # scores as float32. Preserve that order to preserve retrieval ties.
    denominator = 1.5 * ((1 - 0.75) + 0.75 * lengths / average)
    repeated = np.repeat(denominator, np.diff(counts.indptr))
    counts.data = (
        idf[counts.indices] * (counts.data / (repeated + counts.data))
    ).astype(np.float32)
    scores = counts.tocsc()
    retriever = bm25s.BM25(method="lucene", backend="numba")
    retriever.nonoccurrence_array = None
    retriever.scores = dict(
        data=scores.data, indices=scores.indices, indptr=scores.indptr, num_docs=n_docs
    )
    retriever.vocab_dict = vocab
    retriever.unique_token_ids_set = set(vocab.values())
    return retriever


def select_documents(job):
    import orjson

    path, offset, wanted = job
    output = {}
    with gzip.open(path, "rb") as stream:
        for local, line in enumerate(stream):
            if local in wanted:
                title, content, _ = orjson.loads(line)
                output[offset + local] = (title, content)
    return output


def retrieve(args):
    import bm25s
    import numpy as np

    config = read_json(args.corpus / "config.json")
    manifest = read_json(args.corpus / "manifest.json")
    if not manifest["complete"] or manifest["config_fingerprint"] != fingerprint(
        config
    ):
        raise ValueError("Incomplete or mismatched corpus")
    index_path = args.corpus / "bm25"
    paths = sorted(args.corpus.glob("paragraphs-*.jsonl.gz"))
    if not (index_path / "params.index.json").exists():
        from scipy import sparse

        print("Building sparse paragraph counts", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            count_paths = list(
                pool.map(count_shard, [(p, len(config["vocab"])) for p in paths])
            )
        counts = sparse.vstack([sparse.load_npz(p) for p in count_paths], format="csr")
        if counts.shape[0] != manifest["paragraphs"]:
            raise ValueError("Paragraph count mismatch")
        print(f"Indexing {counts.shape[0]} paragraphs", flush=True)
        retriever = index_from_counts(counts, config["vocab"])
        retriever.save(str(index_path))
        del counts
    else:
        retriever = bm25s.BM25.load(str(index_path), mmap=True)
    ids, scores = retriever.retrieve(
        config["queries"], k=args.k, return_as="tuple", n_threads=args.workers
    )
    wanted = set(int(i) for i in ids.flat)
    documents = {}
    offset = 0
    jobs = []
    for path in paths:
        n = read_json(str(path) + ".json")["paragraphs"]
        local = {i - offset for i in wanted if offset <= i < offset + n}
        if local:
            jobs.append((path, offset, local))
        offset += n
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for selected in pool.map(select_documents, jobs):
            documents.update(selected)
    results = {}
    for subject, indices, values in zip(config["subjects"], ids, scores, strict=True):
        rows = []
        for index, score in zip(indices, values, strict=True):
            title, text = documents[int(index)]
            combined = title + ". " + text
            keep = subject in combined or subject.capitalize() in combined
            rows.append(
                dict(
                    id=int(index), score=float(score), title=title, text=text, keep=keep
                )
            )
        results[subject] = rows
    output = dict(
        metadata=dict(
            **manifest,
            k=args.k,
            method="lucene",
            k1=1.5,
            b=0.75,
            query_vocabulary_only=True,
            bm25s_version=bm25s.__version__,
            n_subjects=len(results),
            zero_score_hits=int(np.sum(scores == 0)),
        ),
        subjects=results,
    )
    write_json(args.out, output)
    print(f"Wrote retrievals for {len(results)} subjects to {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dump-dir", type=Path, required=True)
    prep.add_argument("--subjects", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--workers", type=int, default=32)
    prep.add_argument("--streams-per-shard", type=int, default=50)
    prep.add_argument("--wait-for-download", action="store_true")
    ret = sub.add_parser("retrieve")
    ret.add_argument("--corpus", type=Path, required=True)
    ret.add_argument("--out", type=Path, required=True)
    ret.add_argument("--k", type=int, default=100)
    ret.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else retrieve)(args)


if __name__ == "__main__":
    main()
