"""Independently compare refreshed sparse counts/BM25 against upstream BM25S."""

import gzip
import hashlib
from pathlib import Path

import bm25s
import numpy as np
import orjson
from scipy import sparse

from jlens.io import read_json, write_json
from jlens.wiki_corpus import index_from_counts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated/shared_baselines/wikipedia"


def main():
    config = read_json(OUT / "config.json")
    texts, matrices = [], []
    for path in sorted(Path(config["corpus"]).glob("paragraphs-*.jsonl.gz")):
        with gzip.open(path, "rb") as stream:
            texts.extend(orjson.loads(line)[1] for line in stream)
        matrices.append(sparse.load_npz(OUT / "counts" / (path.name + ".npz")))
        if len(texts) >= 10000:
            break
    texts = texts[:10000]
    measured = sparse.vstack(matrices, format="csr")[:10000]
    tok = bm25s.tokenization.Tokenizer()
    tok.word_to_id = config["vocab"]
    ids = tok.tokenize(texts, update_vocab=False, show_progress=False)
    for row, tokens in enumerate(ids):
        unique, counts = np.unique(tokens, return_counts=True)
        np.testing.assert_array_equal(measured[row].indices, unique)
        np.testing.assert_array_equal(measured[row].data, counts)
    upstream = bm25s.BM25(method="lucene", backend="numba")
    upstream.index((ids, config["vocab"]), show_progress=False)
    optimized = index_from_counts(measured, config["vocab"])
    for key in ("indices", "indptr", "data"):
        np.testing.assert_array_equal(optimized.scores[key], upstream.scores[key])
    frequency = np.asarray(
        sparse.vstack(matrices, format="csr")[:10000].sum(axis=0)
    ).ravel()
    query_indices = sorted(
        range(len(config["queries"])),
        key=lambda i: -sum(frequency[t] for t in config["queries"][i]),
    )[:20]
    queries = [config["queries"][i] for i in query_indices]
    expected = upstream.retrieve(queries, k=100, n_threads=1, show_progress=False)
    actual = optimized.retrieve(queries, k=100, n_threads=1, show_progress=False)
    np.testing.assert_array_equal(actual.documents, expected.documents)
    np.testing.assert_array_equal(actual.scores, expected.scores)
    result = dict(
        paragraphs=len(texts),
        queries=[config["subjects"][i] for i in query_indices],
        k=100,
        counts_exact=True,
        sparse_scores_exact=True,
        retrieved_ids_exact=True,
        retrieved_scores_exact=True,
        bm25s_version=bm25s.__version__,
        config_sha256=hashlib.sha256((OUT / "config.json").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write_json(ROOT / "results/shared_baselines/wikipedia_validation.json", result)
    print(result)


if __name__ == "__main__":
    main()
