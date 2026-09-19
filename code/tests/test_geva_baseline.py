"""Scientific protocol checks for the Wikipedia retrieval baseline."""

import bz2
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from jlens.geva_baseline import (
    attribute_rate,
    attribute_tokens,
    correlation,
    frequency_ranks,
    validate_reference_rank,
)


@pytest.fixture(autouse=True)
def preserve_torch_thread_count():
    # BM25S restores Numba's thread count through the shared OpenMP runtime,
    # which otherwise changes PyTorch's thread count for subsequent tests.
    torch = sys.modules.get("torch")
    threads = torch.get_num_threads() if torch is not None else None
    yield
    if torch is not None:
        torch.set_num_threads(threads)


def test_upstream_asymmetric_stopwords_and_exact_decoded_token_matching():
    stop = {"the", "and"}
    attributes = attribute_tokens(
        [" the", " The", " Paris", "Paris", "is", " ing"], stop
    )
    assert attributes == {" The", " Paris", "Paris", " ing"}
    # The capitalized stopword is removed on the prediction side. Leading space
    # and case remain significant for intersection, just as upstream.
    rate, overlap, count = attribute_rate(
        [" The", " Paris", " paris", "is"], attributes, stop
    )
    assert (rate, overlap, count) == (50.0, 1, 2)


def test_prediction_denominator_and_empty_retrieval():
    assert attribute_rate([" a", "be"], {" Paris"}, set()) == (None, 0, 0)
    assert attribute_rate([" Paris", " London"], set(), set()) == (0.0, 0, 2)
    # Upstream uses a set in the numerator and a list in the denominator.
    assert attribute_rate([" Paris", " Paris"], {" Paris"}, set()) == (50.0, 1, 2)


def test_retained_full_corpus_result_and_relation_aggregation():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/olmo/results/geva_baseline.json").read_text())
    reference = json.loads((root / "results/olmo/results/logit_lens.json").read_text())
    assert set(result["relations"]) == set(reference)
    assert result["metadata"]["retrieval"]["complete"] is True
    assert result["metadata"]["retrieval"]["paragraphs"] == 40284960
    assert result["metadata"]["n_rows"] == 513
    assert result["metadata"]["n_subjects"] == 456
    assert result["metadata"]["empty_retrieval_subjects"] == 16
    for relation, value in result["relations"].items():
        assert [r["subject"] for r in value["rows"]] == [
            r["subject"] for r in reference[relation]["rows"]
        ]
        assert value["mean_attribute_rate"] == pytest.approx(
            np.mean([r["attribute_rate"] for r in value["rows"]])
        )
    for value in result["subjects"].values():
        assert value["attribute_rate"] == pytest.approx(
            100 * value["overlap"] / value["clean_predictions"]
        )
        if value["n_selected"] == 0:
            assert value["attribute_rate"] == 0
    values = list(result["relations"].values())
    recomputed = correlation(
        [v["mean_attribute_rate"] for v in values], [v["faithfulness"] for v in values]
    )
    assert recomputed["n"] == 28
    assert recomputed["spearman"]["statistic"] == pytest.approx(-0.40769658267165626)


def test_frequency_ties_and_absent_tokens_have_bottom_midranks():
    ranks, counts = frequency_ranks({1: 4, 3: 4, 4: 2}, 6)
    np.testing.assert_array_equal(counts, [0, 4, 0, 4, 2, 0])
    np.testing.assert_array_equal(ranks, [4, 0.5, 4, 0.5, 2, 4])
    empty, _ = frequency_ranks({}, 6)
    np.testing.assert_array_equal(empty, [2.5] * 6)


def test_correlations_report_undefined_instead_of_fabricating_zero():
    result = correlation([0, 0, 0], [1, 2, 3])
    assert result["spearman"]["statistic"] is None
    result = correlation([1, 2, None, 4], [4, 3, 2, 1])
    assert result["n"] == 3 and result["excluded"] == 1
    assert result["spearman"]["statistic"] == -1


def test_reference_validation_allows_only_one_position_near_tie():
    row = dict(subject="Example", model_top1_id=0, rank_logit=0)
    check = validate_reference_rank(np.array([1.0, 1.000001, -2.0]), row)
    assert check["delta"] == 1 and check["logit_gap"] < 0.000002
    with pytest.raises(ValueError, match="rank mismatch"):
        validate_reference_rank(np.array([1.0, 1.1, -2.0]), row)


def test_vectorized_bm25_exactly_matches_upstream_scores_and_retrieval():
    bm25s = pytest.importorskip("bm25s")
    from scipy import sparse

    from jlens.wiki_corpus import index_from_counts

    tokenizer = bm25s.tokenization.Tokenizer()
    queries = tokenizer.tokenize(
        ["Paris", "New York", "nonexistent"], show_progress=False
    )
    texts = [
        "Paris Paris",
        "New York",
        "Paris and New York",
        "Nothing relevant",
        "New New York",
        "",
    ]
    ids = tokenizer.tokenize(texts, update_vocab=False, show_progress=False)
    upstream = bm25s.BM25(method="lucene", backend="numba")
    upstream.index((ids, tokenizer.word_to_id), show_progress=False)
    counts = np.zeros((len(ids), len(tokenizer.word_to_id)), dtype=np.float32)
    for row, tokens in enumerate(ids):
        for token in tokens:
            counts[row, token] += 1
    optimized = index_from_counts(sparse.csr_matrix(counts), tokenizer.word_to_id)
    for key in ["indices", "indptr", "data"]:
        np.testing.assert_array_equal(optimized.scores[key], upstream.scores[key])
    expected = upstream.retrieve(queries, k=3, show_progress=False, n_threads=1)
    actual = optimized.retrieve(queries, k=3, show_progress=False, n_threads=1)
    np.testing.assert_array_equal(actual.documents, expected.documents)
    np.testing.assert_array_equal(actual.scores, expected.scores)


def test_multistream_preparation_matches_upstream_tokenization(tmp_path):
    bm25s = pytest.importorskip("bm25s")
    pytest.importorskip("wikiextractor")
    pytest.importorskip("orjson")
    from jlens.wiki_corpus import process_chunk

    tokenizer = bm25s.tokenization.Tokenizer()
    tokenizer.tokenize(["Paris", "London"], show_progress=False)
    parts = [
        "<mediawiki><siteinfo></siteinfo><page><title>Paris</title><revision><text>"
        "Paris is in [[France]].\n\n== History ==\nAn old city."
        "</text></revision></page>",
        "<page><title>Talk:Paris</title><revision><text>Exclude me.</text></revision></page>"
        "<page><title>London</title><revision><text>London is in England.</text></revision></page></mediawiki>",
    ]
    compressed = [bz2.compress(p.encode()) for p in parts]
    dump = tmp_path / "dump.bz2"
    dump.write_bytes(b"".join(compressed))
    out = tmp_path / "paragraphs.jsonl.gz"
    result = process_chunk(
        (
            str(dump),
            [0, len(compressed[0]), sum(map(len, compressed))],
            str(out),
            tokenizer.word_to_id,
        )
    )
    with gzip.open(out, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert result["pages"] == 3 and result["paragraphs"] == 3
    assert rows[0][:2] == ["Paris", "Paris is in France."]
    assert rows[1][:2] == ["Paris. History", "An old city."]
    expected = tokenizer.tokenize(
        [r[1] for r in rows], update_vocab=False, show_progress=False
    )
    assert [r[2] for r in rows] == expected
    # Restricted vocabulary lengths (including the empty-token sentinel) must
    # match the upstream BM25 scores, not full-document word counts.
    a, b = bm25s.BM25(method="lucene"), bm25s.BM25(method="lucene")
    a.index((expected, tokenizer.word_to_id), show_progress=False)
    b.index(([r[2] for r in rows], tokenizer.word_to_id), show_progress=False)
    q = tokenizer.tokenize(["Paris"], update_vocab=False, show_progress=False)
    np.testing.assert_array_equal(
        a.retrieve(q, k=3, show_progress=False).scores,
        b.retrieve(q, k=3, show_progress=False).scores,
    )
