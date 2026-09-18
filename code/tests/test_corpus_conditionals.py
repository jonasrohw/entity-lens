"""Counting invariants and a real multiprocess resume round trip."""

import gzip
import json
import os
import re
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("ahocorasick")
pytest.importorskip("orjson")
zstd = pytest.importorskip("zstandard")

from jlens.corpus_conditionals import (  # noqa: E402
    Matcher,
    analyze,
    correlation,
    inventory,
    load_lre,
    normalize,
    open_checkpoint,
    records,
    vocabulary,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_matching_against_independent_regex_oracle():
    terms = ["ann", "france", "mexico", "mexico city", "paris", "são paulo", "strasse"]
    pairs = [(0, 0), (1, 4), (2, 3), (4, 5)]
    matcher = Matcher(terms, pairs)
    documents = [
        "Paris Paris in FRANCE; France again.",
        "Parisian Annette annual ann2 _ann",
        "Mexico\nCity, México; SÃO\tPAULO; Sa\u0303o Paulo.",
        "Ann met Ann. Straße.",
        "Paris and São Paulo",
        "France",
        "",
    ]
    counts = np.zeros(len(terms) + len(pairs), dtype=np.int64)
    expected = counts.copy()
    for document in documents:
        normalized = normalize(document)
        seen = {
            i
            for i, term in enumerate(terms)
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized)
        }
        assert matcher.find(document) == seen
        matcher.count(document, counts)
        for i in seen:
            expected[i] += 1
        for i, (s, o) in enumerate(pairs):
            expected[len(terms) + i] += s in seen and o in seen
    np.testing.assert_array_equal(counts, expected)
    assert Matcher(["Paris"], [], True).find("PARIS Parisian") == set()


def test_directional_probabilities_and_unseen_are_not_zero(tmp_path):
    write(
        tmp_path / "data/relations/place.json",
        dict(
            samples=[
                dict(subject="Paris", object="France"),
                dict(subject="France", object="Paris"),
                dict(subject="Unknown", object="France"),
                dict(subject="Paris", object="France"),
            ]
        ),
    )
    terms, pairs, facts = vocabulary(tmp_path)
    assert len(facts) == 3 and len(pairs) == 2
    matcher = Matcher(terms, pairs)
    counts = np.zeros(len(terms) + len(pairs), dtype=np.int64)
    for doc in ["Paris France", "Paris France", "France", "France", "Paris"]:
        matcher.count(doc, counts)
    rows, relations, _, _ = analyze(
        counts, 5, terms, facts, {"place": {"value": 0.8, "subjects": {}}}, 1
    )
    assert rows[0]["p_object_given_subject"] == 2 / 3
    assert rows[0]["p_subject_given_object"] == 1 / 2
    assert rows[1]["p_object_given_subject"] == rows[0]["p_subject_given_object"]
    assert rows[2]["p_object_given_subject"] is None
    assert rows[2]["p_subject_given_object"] == 0
    assert relations[0]["n_observed"] == 2
    assert relations[0]["coverage"] == 2 / 3
    _, filtered, _, _ = analyze(counts, 5, terms, facts, {}, 4)
    assert filtered[0]["n_observed"] == 0


def test_correlation_known_values_and_degenerate_cases():
    assert correlation([1, 2, 3, None], [3, 2, 1, 4])["spearman"] == -1
    assert correlation([1, 1, 1], [1, 2, 3])["pearson"] is None
    assert correlation([1], [2])["n"] == 1


def test_lre_prefers_corrected_run(tmp_path):
    write(tmp_path / "faithfulness_year.json", dict(relation="year", best_faith=0.1))
    write(tmp_path / "faithfulness_year_ts.json", dict(relation="year", best_faith=0.9))
    assert load_lre(tmp_path, "best_faith")["year"]["value"] == 0.9


def test_checkpoint_rejects_changed_configuration(tmp_path):
    path = tmp_path / "cache.sqlite3"
    open_checkpoint(path, "a").close()
    with pytest.raises(ValueError, match="configuration changed"):
        open_checkpoint(path, "b")


def fixture_corpus(tmp_path):
    root, corpus = tmp_path / "repo", tmp_path / "corpus"
    for name, s, o, faith in [
        ("a", "Paris", "France", 0.8),
        ("b", "Berlin", "Germany", 0.5),
        ("c", "Rome", "Italy", 0.2),
    ]:
        write(
            root / f"data/relations/{name}.json",
            dict(samples=[dict(subject=s, object=o)]),
        )
        write(
            root / f"results/main/faithfulness_{name}.json",
            dict(relation=name, best_faith=faith, per_subject={s: int(faith > 0.5)}),
        )
    docs = [
        "Paris France Paris",
        "France",
        "Berlin",
        "Berlin Germany",
        "Italy",
        "[REMOVED]",
        "unrelated",
    ]
    repos = []
    for stage, suffix in [("pretrain", ".zst"), ("midtrain", ".gz"), ("long", "")]:
        raw = b"".join(json.dumps({"text": d}).encode() + b"\n" for d in docs)
        if suffix == ".zst":
            compressor = zstd.ZstdCompressor()
            blob = compressor.compress(raw) + compressor.compress(raw)
        elif suffix == ".gz":
            blob = gzip.compress(raw)
        else:
            blob = raw
        rel = f"data/0.jsonl{suffix}"
        path = corpus / stage / rel
        path.parent.mkdir(parents=True)
        path.write_bytes(blob)
        repos.append(
            dict(
                repo_id=f"test/{stage}",
                revision="fixture",
                files=[dict(path=rel, size=len(blob))],
            )
        )
    write(corpus / "manifest.json", dict(repos=repos))
    return root, corpus


def test_parallel_resume_report_and_serial_equivalence(tmp_path):
    root, corpus = fixture_corpus(tmp_path)
    out = tmp_path / "output"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [
        sys.executable,
        "-m",
        "jlens",
        "--root",
        str(root),
        "corpus-conditionals",
        "--corpus-root",
        str(corpus),
        "--out",
        str(out),
    ]

    def run(*args):
        return subprocess.run(
            command + list(args), env=env, check=True, text=True, capture_output=True
        )

    run("--workers", "2", "--max-shards", "1")
    assert not json.loads((out / "report.json").read_text())["complete"]
    run("--workers", "2")
    report = json.loads((out / "report.json").read_text())
    assert report["complete"] and report["processed_shards"] == 3
    assert report["scopes"]["all_stages"]["coverage"]["documents"] == 28
    assert report["scopes"]["all_stages"]["coverage"]["redacted_documents"] == 4
    before = (out / "all_stages.facts.csv").read_bytes()
    with sqlite3.connect(out / "counts.sqlite3") as db:
        before_cache = db.execute("SELECT * FROM shards ORDER BY key").fetchall()
    assert "0 pending" in run("--workers", "1").stderr
    run("--report-only")
    assert (out / "all_stages.facts.csv").read_bytes() == before
    with sqlite3.connect(out / "counts.sqlite3") as db:
        assert (
            db.execute("SELECT * FROM shards ORDER BY key").fetchall() == before_cache
        )
    serial = tmp_path / "serial"
    run("--workers", "1", "--out", str(serial))
    assert (serial / "all_stages.facts.csv").read_bytes() == before
    path = corpus / "long/data/0.jsonl"
    path.write_bytes(path.read_bytes().replace(b"France", b"FRANCE"))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert "1 pending" in run("--workers", "1").stderr
    assert (out / "all_stages.facts.csv").read_bytes() == before
    path.unlink()
    failed = subprocess.run(command, env=env, text=True, capture_output=True)
    assert failed.returncode != 0 and "absent or wrong size" in failed.stderr
    run("--report-only", "--allow-partial")
    partial = json.loads((out / "report.json").read_text())
    assert not partial["complete"] and partial["processed_shards"] == 2


def test_inventory_rejects_unknown_data_formats(tmp_path):
    _, corpus = fixture_corpus(tmp_path)
    path = corpus / "manifest.json"
    data = json.loads(path.read_text())
    data["repos"][0]["files"].append(dict(path="data/unexpected.parquet", size=0))
    write(path, data)
    with pytest.raises(ValueError, match="Unsupported data shard format"):
        inventory(corpus)


def test_streams_concatenated_zstd_and_gzip(tmp_path):
    _, corpus = fixture_corpus(tmp_path)
    for name, n in [("pretrain/data/0.jsonl.zst", 14), ("midtrain/data/0.jsonl.gz", 7)]:
        with records(corpus / name) as stream:
            assert len(list(stream)) == n


def test_official_split_excludes_extra_pdfs_and_numeric_suffixes(tmp_path):
    _, corpus = fixture_corpus(tmp_path)
    m = json.loads((corpus / "manifest.json").read_text())
    old_stage = corpus / "pretrain"
    stage = "dolma3_mix-6T-1025-7B"
    old_stage.rename(corpus / stage)
    m["repos"][0]["repo_id"] = "allenai/" + stage
    for suffix in [".gz", ".zst1234"]:
        path = corpus / stage / ("data/0.jsonl" + suffix)
        path.write_bytes(b"extra")
        m["repos"][0]["files"].append(dict(path="data/0.jsonl" + suffix, size=5))
    write(corpus / "manifest.json", m)
    _, jobs, missing, excluded = inventory(corpus)
    assert len(jobs) == 3 and not missing and len(excluded) == 2
    _, jobs, missing, excluded = inventory(corpus, "published-jsonl")
    assert len(jobs) == 4 and not missing and len(excluded) == 1


def test_invalid_text_fails_without_committing_shard(tmp_path):
    from jlens.corpus_conditionals import initialize_worker, scan_shard

    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": null}\n')
    st = path.stat()
    initialize_worker(["paris", "france"], [(0, 1)], False)
    with pytest.raises(ValueError, match="not a string"):
        scan_shard(
            dict(
                absolute=str(path),
                key="bad",
                stage="test",
                size=st.st_size,
                mtime=st.st_mtime_ns,
            ),
            "text",
        )
