"""Reindex the verified 2021 Wikipedia paragraphs for the shared subject union."""

import argparse
import gzip
import hashlib
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
from scipy import sparse

from jlens import wiki_corpus as wiki
from jlens.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated/shared_baselines/wikipedia"
CORPUS = Path("/mounts/work/jonas/data/geva-wikipedia-20211020/olmo-main")


def count(job):
    from bm25s.tokenization import Tokenizer

    source, output, vocab = job
    if output.exists():
        return str(output)
    tok = Tokenizer()
    tok.word_to_id = vocab
    indices, pointers = array("i"), array("q", [0])
    with gzip.open(source, "rb") as stream:
        for line in stream:
            ids = next(
                tok.streaming_tokenize([orjson.loads(line)[1]], update_vocab=False)
            )
            indices.extend(ids)
            pointers.append(len(indices))
    matrix = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(pointers, dtype=np.int64),
        ),
        shape=(len(pointers) - 1, len(vocab)),
    )
    matrix.sum_duplicates()
    temporary = output.with_suffix(".tmp.npz")
    sparse.save_npz(temporary, matrix)
    temporary.replace(output)
    return str(output)


def main():
    import bm25s
    from bm25s.tokenization import Tokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=48)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "results/all_model_newline").glob("*/shared_split/*.json"))
    subjects = sorted(
        {r["subject"] for f in sources for r in read_json(f)["lens"]["rows"]}
    )
    tok = Tokenizer()
    queries = tok.tokenize(subjects, show_progress=False)
    vocab = tok.word_to_id
    config = dict(
        subjects=subjects,
        queries=queries,
        vocab=vocab,
        query_scope="union_of_seven_models_shared_split_test_subjects",
        source_sha256={
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sources
        },
        corpus=str(CORPUS),
        corpus_manifest_sha256=hashlib.sha256(
            (CORPUS / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    if (OUT / "config.json").exists():
        assert read_json(OUT / "config.json") == config
    write_json(OUT / "config.json", config)
    manifest = read_json(CORPUS / "manifest.json")
    assert manifest["complete"]
    paths = sorted(CORPUS.glob("paragraphs-*.jsonl.gz"))
    index = OUT / "bm25"
    if not (index / "params.index.json").exists():
        counts_dir = OUT / "counts"
        counts_dir.mkdir(exist_ok=True)
        jobs = [(f, counts_dir / (f.name + ".npz"), vocab) for f in paths]
        print(
            "RETOKENIZE",
            len(subjects),
            "subjects",
            len(vocab),
            "terms",
            len(jobs),
            "shards",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            outputs = []
            for i, output in enumerate(pool.map(count, jobs)):
                outputs.append(output)
                if (i + 1) % 100 == 0:
                    print("COUNTS", i + 1, "/", len(jobs), flush=True)
        matrix = sparse.vstack([sparse.load_npz(f) for f in outputs], format="csr")
        assert matrix.shape[0] == manifest["paragraphs"]
        print("INDEX", matrix.shape, matrix.nnz, flush=True)
        retriever = wiki.index_from_counts(matrix, vocab)
        retriever.save(str(index))
        del matrix
    else:
        retriever = bm25s.BM25.load(str(index), mmap=True)
    print("RETRIEVE", len(subjects), flush=True)
    result_ids, result_scores = [], []
    for start in range(0, len(queries), 32):
        ids, scores = retriever.retrieve(
            queries[start : start + 32],
            k=100,
            return_as="tuple",
            n_threads=min(16, a.workers),
            show_progress=False,
        )
        result_ids.extend(ids)
        result_scores.extend(scores)
        print("RETRIEVED", min(start + 32, len(queries)), flush=True)
    ids, scores = np.array(result_ids), np.array(result_scores)
    np.savez_compressed(OUT / "retrieved_ids.npz", ids=ids, scores=scores)
    wanted = sorted(set(int(i) for i in ids.flat))
    documents, jobs, offset = {}, [], 0
    import bisect

    for path in paths:
        n = read_json(str(path) + ".json")["paragraphs"]
        local = {
            i - offset
            for i in wanted[
                bisect.bisect_left(wanted, offset) : bisect.bisect_left(
                    wanted, offset + n
                )
            ]
        }
        if local:
            jobs.append((path, offset, local))
        offset += n
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for selected in pool.map(wiki.select_documents, jobs):
            documents.update(selected)
    results = {}
    for subject, indices, values in zip(subjects, ids, scores, strict=True):
        results[subject] = [
            dict(
                id=int(i),
                score=float(s),
                title=documents[int(i)][0],
                text=documents[int(i)][1],
                keep=subject in (documents[int(i)][0] + ". " + documents[int(i)][1])
                or subject.capitalize()
                in (documents[int(i)][0] + ". " + documents[int(i)][1]),
            )
            for i, s in zip(indices, values, strict=True)
        ]
    write_json(
        OUT / "retrieval.json",
        dict(
            metadata=dict(
                **manifest,
                k=100,
                method="lucene",
                k1=1.5,
                b=0.75,
                query_vocabulary_only=True,
                query_scope=config["query_scope"],
                bm25s_version=bm25s.__version__,
                n_subjects=len(subjects),
                zero_score_hits=int(np.sum(scores == 0)),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            subjects=results,
        ),
    )
    print("COMPLETE", len(results), flush=True)


if __name__ == "__main__":
    main()
