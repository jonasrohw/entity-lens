"""Parallel, resumable document-frequency analysis of directional LRE facts.

One Aho-Corasick pass finds ALL entity strings in each document. Only edges in
our relation datasets are counted, never a quadratic product of matched terms.
Workers stream compressed shards and return small sufficient-statistic arrays.
"""

import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import re
import sqlite3
import time
import unicodedata
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
from scipy import stats

from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, relation_data, write_json

VERSION = 1
DEFAULT_CORPUS = Path("/mounts/work/jonas/data/olmo-3-1025-7b-training")
SUFFIXES = (".jsonl.zst", ".jsonl.gz", ".jsonl")
SPACE = re.compile(r"\s+")
FEATURES = (
    "p_object_given_subject",
    "p_subject_given_object",
    "direction_difference",
    "cooccurrence",
    "subject_count",
    "object_count",
    "log1p_cooccurrence",
)
MATCHER = None


def normalize(text, case_sensitive=False):
    text = unicodedata.normalize("NFC", text)
    if not case_sensitive:
        text = text.casefold()
    return SPACE.sub(" ", text).strip()


def vocabulary(root, case_sensitive=False):
    facts = []
    for relation, data in relation_data(root).items():
        seen = set()
        for sample in data["samples"]:
            s, o = sample["subject"], sample["object"]
            key = (s, o)
            if key in seen:
                continue
            seen.add(key)
            sn, on = normalize(s, case_sensitive), normalize(o, case_sensitive)
            if not sn or not on:
                raise ValueError(f"Empty entity in {relation}: {key}")
            facts.append(dict(relation=relation, subject=s, object=o, sn=sn, on=on))
    if not facts:
        raise ValueError(f"No relation samples under {root}/data/relations")
    terms = sorted({f[k] for f in facts for k in ("sn", "on")})
    ids = {term: i for i, term in enumerate(terms)}
    pairs = sorted({tuple(sorted((ids[f["sn"]], ids[f["on"]]))) for f in facts})
    pair_ids = {pair: i for i, pair in enumerate(pairs)}
    for f in facts:
        f["sid"], f["oid"] = ids[f.pop("sn")], ids[f.pop("on")]
        f["pid"] = pair_ids[tuple(sorted((f["sid"], f["oid"])))]
        # Exact containment is a diagnostic, not an entity-disambiguation method.
        s, o = terms[f["sid"]], terms[f["oid"]]
        f["lexical_overlap"] = s in o or o in s
    return terms, pairs, facts


class Matcher:
    def __init__(self, terms, pairs, case_sensitive=False):
        import ahocorasick

        self.case_sensitive = case_sensitive
        self.automaton = ahocorasick.Automaton()
        for tid, term in enumerate(terms):
            self.automaton.add_word(term, (tid, len(term)))
        self.automaton.make_automaton()
        self.adjacency = [[] for _ in terms]
        # Store each undirected edge once, on its lower-degree endpoint. This
        # avoids traversing every country edge whenever a common country occurs.
        degree = np.bincount([t for pair in pairs for t in pair], minlength=len(terms))
        for pid, (a, b) in enumerate(pairs):
            if degree[a] > degree[b]:
                a, b = b, a
            self.adjacency[a].append((b, pid))
        self.n_terms, self.n_pairs = len(terms), len(pairs)

    def find(self, text):
        text = normalize(text, self.case_sensitive)
        found = set()
        for end, (tid, length) in self.automaton.iter(text):
            if tid in found:
                continue
            start = end - length + 1
            # Unicode word boundaries, including digits and underscore. Overlaps
            # remain visible (e.g. Mexico inside Mexico City); flagged in reports.
            if start and (text[start - 1].isalnum() or text[start - 1] == "_"):
                continue
            if end + 1 < len(text) and (
                text[end + 1].isalnum() or text[end + 1] == "_"
            ):
                continue
            found.add(tid)
        return found

    def count(self, text, counts):
        found = self.find(text)
        for tid in found:
            counts[tid] += 1
            for other, pid in self.adjacency[tid]:
                if other in found:
                    counts[self.n_terms + pid] += 1
        return bool(found)


def initialize_worker(terms, pairs, case_sensitive):
    global MATCHER
    MATCHER = Matcher(terms, pairs, case_sensitive)


@contextmanager
def records(path):
    """Read JSONL without materializing the decompressed shard."""
    if str(path).endswith(".zst"):
        import zstandard

        with open(path, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(
                raw, read_across_frames=True
            ) as reader:
                with io.BufferedReader(reader, buffer_size=1024 * 1024) as stream:
                    yield stream
    elif str(path).endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            yield stream
    else:
        with open(path, "rb", buffering=1024 * 1024) as stream:
            yield stream


def scan_shard(job, text_field):
    import orjson

    started = time.monotonic()
    counts = np.zeros(MATCHER.n_terms + MATCHER.n_pairs, dtype="<i8")
    docs = matched = redacted = text_bytes = 0
    path = Path(job["absolute"])
    with records(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = orjson.loads(line)
                text = row[text_field]
                if not isinstance(text, str):
                    raise TypeError(f"{text_field} is not a string")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            docs += 1
            text_bytes += len(text.encode("utf-8"))
            redacted += text.strip() == "[REMOVED]"
            matched += MATCHER.count(text, counts)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (job["size"], job["mtime"]):
        raise ValueError(f"Shard changed during scan: {path}")
    return (
        job["key"],
        job["stage"],
        job["size"],
        job["mtime"],
        docs,
        matched,
        redacted,
        text_bytes,
        time.monotonic() - started,
        zlib.compress(counts.tobytes(), level=1),
    )


def inventory(root, scope="training-split"):
    manifest = read_json(root / "manifest.json")
    jobs, missing, excluded = [], [], []
    keys = set()
    for repo in manifest["repos"]:
        stage = repo["repo_id"].split("/")[-1]
        published_paths = {item["path"] for item in repo["files"]}
        for item in repo["files"]:
            rel = PurePosixPath(item["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe manifest path: {rel}")
            key = f"{stage}/{rel}"
            if key in keys:
                raise ValueError(f"Duplicate manifest path: {key}")
            keys.add(key)
            # The pinned 6T card declares data/**/*.jsonl.zst. Gzip PDFs and
            # numeric-suffixed files in that repository are outside its split.
            if (
                scope == "training-split"
                and repo["repo_id"] == "allenai/dolma3_mix-6T-1025-7B"
                and not str(rel).endswith(".jsonl.zst")
            ):
                excluded.append(key)
                continue
            if not str(rel).endswith(SUFFIXES):
                if (
                    re.fullmatch(r".*\.jsonl\.zst\d+", str(rel))
                    and re.sub(r"(?<=\.zst)\d+$", "", str(rel)) in published_paths
                ):
                    excluded.append(key)
                    continue
                # Ignore cards/images, but reject unrecognized data formats.
                if rel.parts[0] == "data":
                    raise ValueError(f"Unsupported data shard format: {key}")
                excluded.append(key)
                continue
            path = root / key
            try:
                st = path.stat()
            except FileNotFoundError:
                st = None
            if st is None or st.st_size != item["size"]:
                missing.append(key)
                continue
            jobs.append(
                dict(
                    key=key,
                    stage=stage,
                    absolute=str(path),
                    size=st.st_size,
                    mtime=st.st_mtime_ns,
                )
            )
    if not jobs and not missing:
        raise ValueError("Manifest contains no supported JSONL data shards")
    return manifest, jobs, missing, excluded


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def open_checkpoint(path, signature):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (signature TEXT NOT NULL)")
    db.execute("""CREATE TABLE IF NOT EXISTS shards (
        key TEXT PRIMARY KEY, stage TEXT, size INTEGER, mtime INTEGER,
        documents INTEGER, matched INTEGER, redacted INTEGER, text_bytes INTEGER,
        seconds REAL, counts BLOB)""")
    old = db.execute("SELECT signature FROM metadata").fetchone()
    if old and old[0] != signature:
        db.close()
        raise ValueError(
            "Checkpoint configuration changed; choose a new --out directory"
        )
    if old is None:
        db.execute("INSERT INTO metadata VALUES (?)", (signature,))
    db.commit()
    return db


def run_scan(db, jobs, terms, pairs, args):
    cached = {k: (s, m) for k, s, m in db.execute("SELECT key,size,mtime FROM shards")}
    pending = [j for j in jobs if cached.get(j["key"]) != (j["size"], j["mtime"])]
    # Invalidate old results BEFORE scanning so interruption cannot expose stale
    # counts through --report-only. Each finished shard is an atomic transaction.
    db.executemany("DELETE FROM shards WHERE key=?", [(j["key"],) for j in pending])
    db.commit()
    logging.info(
        "%s valid cached shards; %s pending", len(jobs) - len(pending), len(pending)
    )
    if args.report_only or not pending:
        return
    if args.max_shards:
        # Reproducible broad sample for a smoke test, not alphabetical domains.
        pending.sort(key=lambda j: hashlib.sha256(j["key"].encode()).digest())
        pending = pending[: args.max_shards]
    else:
        pending.sort(key=lambda j: j["size"], reverse=True)
    started = time.monotonic()
    done = byte_count = docs = 0
    work = iter(pending)
    futures = {}
    # Every worker scans with one CPU; prevent BLAS from spawning a thread pool
    # when numpy/scipy are imported by the spawned interpreter.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    # Spawn prevents inheriting the parent's SQLite connection and BLAS threads.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(terms, pairs, args.case_sensitive),
    ) as pool:

        def submit():
            job = next(work, None)
            if job is not None:
                futures[pool.submit(scan_shard, job, args.text_field)] = job

        for _ in range(min(2 * args.workers, len(pending))):
            submit()
        last_log = 0
        try:
            while futures:
                finished, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = futures.pop(future)
                    result = future.result()  # malformed/truncated shards fail the run
                    with db:
                        db.execute(
                            "INSERT OR REPLACE INTO shards VALUES (?,?,?,?,?,?,?,?,?,?)",
                            result,
                        )
                    done += 1
                    byte_count += job["size"]
                    docs += result[4]
                    submit()
                elapsed = time.monotonic() - started
                if elapsed - last_log >= 30 or not futures:
                    logging.info(
                        "%s/%s new shards; %s docs; %.1f compressed MB/s",
                        done,
                        len(pending),
                        docs,
                        byte_count / 1e6 / max(elapsed, 0.001),
                    )
                    write_json(
                        args.out / "progress.json",
                        dict(
                            state="running" if futures else "scan_finished",
                            new_shards=done,
                            scheduled_shards=len(pending),
                            new_documents=docs,
                            elapsed_seconds=elapsed,
                            compressed_MB_per_second=byte_count
                            / 1e6
                            / max(elapsed, 0.001),
                        ),
                    )
                    last_log = elapsed
        except BaseException:
            for future in futures:
                future.cancel()
            # The pool finishes its bounded in-flight tasks; they can be rescanned.
            raise


def load_lre(directory, field):
    scores = {}
    for path in sorted(directory.glob("faithfulness_*.json")):
        row = read_json(path)
        value = row.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            continue
        relation = row["relation"]
        if relation not in scores or path.stem.endswith("_ts"):
            scores[relation] = dict(
                value=float(value),
                source=str(path),
                subjects=row.get("per_subject") or {},
            )
    if not scores:
        raise ValueError(f"No finite {field} scores in {directory}")
    return scores


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out = dict(n=len(x), pearson=None, pearson_p=None, spearman=None, spearman_p=None)
    if len(x) < 3:
        out["reason"] = "fewer than three observations"
    elif np.ptp(x) == 0 or np.ptp(y) == 0:
        out["reason"] = "constant variable"
    else:
        p, s = stats.pearsonr(x, y), stats.spearmanr(x, y)
        out.update(
            pearson=float(p.statistic),
            pearson_p=float(p.pvalue),
            spearman=float(s.statistic),
            spearman_p=float(s.pvalue),
        )
    return out


def csv_write(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def analyze(counts, documents, terms, facts, lre, min_count):
    fact_rows, relation_rows, correlations, subject_correlations = [], [], [], []
    for f in facts:
        ns, no, joint = (
            int(counts[f["sid"]]),
            int(counts[f["oid"]]),
            int(counts[len(terms) + f["pid"]]),
        )
        if not (0 <= joint <= min(ns, no) and max(ns, no) <= documents):
            raise ValueError(f"Inconsistent document counts for {f}")
        forward = joint / ns if ns >= min_count else None
        reverse = joint / no if no >= min_count else None
        score = lre.get(f["relation"], {})
        fact_rows.append(
            dict(
                relation=f["relation"],
                subject=f["subject"],
                object=f["object"],
                subject_count=ns,
                object_count=no,
                cooccurrence=joint,
                p_object_given_subject=forward,
                p_subject_given_object=reverse,
                direction_difference=(forward - reverse)
                if forward is not None and reverse is not None
                else None,
                log1p_cooccurrence=float(np.log1p(joint)),
                lexical_overlap=f["lexical_overlap"],
                lre_value=score.get("value"),
                lre_subject_value=score.get("subjects", {}).get(f["subject"]),
            )
        )
    # Comparing directions uses the SAME facts, with both denominators observed.
    # Unseen marginals remain missing, not falsely zero probability.
    for cohort in ("all_facts", "evaluated_subjects"):
        for subset in ("all", "no_lexical_overlap"):
            for relation in sorted({f["relation"] for f in facts}):
                all_rows = [
                    r
                    for r in fact_rows
                    if r["relation"] == relation
                    and (subset == "all" or not r["lexical_overlap"])
                    and (cohort == "all_facts" or r["lre_subject_value"] is not None)
                ]
                eligible = [
                    r for r in all_rows if r["direction_difference"] is not None
                ]
                row = dict(
                    relation=relation,
                    cohort=cohort,
                    subset=subset,
                    n_facts=len(all_rows),
                    n_observed=len(eligible),
                    coverage=len(eligible) / len(all_rows) if all_rows else None,
                    lre_value=lre.get(relation, {}).get("value"),
                )
                for feature in FEATURES:
                    row[feature] = (
                        float(np.mean([r[feature] for r in eligible]))
                        if eligible
                        else None
                    )
                row["micro_p_object_given_subject"] = (
                    (
                        sum(r["cooccurrence"] for r in eligible)
                        / sum(r["subject_count"] for r in eligible)
                    )
                    if eligible
                    else None
                )
                row["micro_p_subject_given_object"] = (
                    (
                        sum(r["cooccurrence"] for r in eligible)
                        / sum(r["object_count"] for r in eligible)
                    )
                    if eligible
                    else None
                )
                relation_rows.append(row)
            selected = [
                r
                for r in relation_rows
                if r["cohort"] == cohort
                and r["subset"] == subset
                and r["lre_value"] is not None
                and r["n_observed"] > 0
            ]
            for feature in (
                *FEATURES,
                "micro_p_object_given_subject",
                "micro_p_subject_given_object",
            ):
                correlations.append(
                    dict(
                        cohort=cohort,
                        subset=subset,
                        feature=feature,
                        **correlation(
                            [r[feature] for r in selected],
                            [r["lre_value"] for r in selected],
                        ),
                    )
                )
    for relation in sorted(lre):
        rows = [
            r
            for r in fact_rows
            if r["relation"] == relation
            and r["lre_subject_value"] is not None
            and r["direction_difference"] is not None
        ]
        # Ambiguous subject -> multiple objects would duplicate a single outcome.
        if len({r["subject"] for r in rows}) != len(rows):
            subject_correlations.append(
                dict(relation=relation, reason="multiple objects per subject; skipped")
            )
            continue
        for feature in FEATURES:
            subject_correlations.append(
                dict(
                    relation=relation,
                    feature=feature,
                    **correlation(
                        [r[feature] for r in rows],
                        [r["lre_subject_value"] for r in rows],
                    ),
                )
            )
    return fact_rows, relation_rows, correlations, subject_correlations


def report(db, jobs, missing, manifest, terms, facts, args):
    lre = load_lre(args.faith_dir, args.lre_field)
    stages = {r["repo_id"].split("/")[-1] for r in manifest["repos"]}
    shape = len(terms) + max(f["pid"] for f in facts) + 1
    totals = {s: np.zeros(shape, dtype="<i8") for s in stages}
    coverage = {
        s: dict(
            shards=0,
            documents=0,
            matched_documents=0,
            redacted_documents=0,
            text_bytes=0,
            compressed_bytes=0,
        )
        for s in stages
    }
    valid = {j["key"]: (j["size"], j["mtime"]) for j in jobs}
    for (
        key,
        stage,
        size,
        mtime,
        docs,
        matched,
        redacted,
        text_bytes,
        _,
        blob,
    ) in db.execute("SELECT * FROM shards"):
        if valid.get(key) != (size, mtime):
            continue
        totals[stage] += np.frombuffer(zlib.decompress(blob), dtype="<i8")
        for field, value in zip(
            coverage[stage], (1, docs, matched, redacted, text_bytes, size), strict=True
        ):
            coverage[stage][field] += value
    total = sum(totals.values(), start=np.zeros(shape, dtype="<i8"))
    all_coverage = {
        k: sum(c[k] for c in coverage.values()) for k in next(iter(coverage.values()))
    }
    complete = not missing and all_coverage["shards"] == len(jobs)
    expected = len(jobs) + len(missing)
    output = dict(
        generated_at_unix=time.time(),
        complete=complete,
        expected_shards=expected,
        processed_shards=all_coverage["shards"],
        missing_shards=len(missing),
        unprocessed_shards=len(jobs) - all_coverage["shards"],
        unit="document; one JSONL record, binary presence per entity",
        matching="NFC + whitespace collapse + Unicode word boundaries; "
        + ("case sensitive" if args.case_sensitive else "casefold"),
        min_marginal_count=args.min_count,
        lre_field=args.lre_field,
        faith_dir=str(args.faith_dir),
        lre_sources=lre,
        corpus_revisions={r["repo_id"]: r.get("revision") for r in manifest["repos"]},
        corpus_scope=args.corpus_scope,
        notes=[
            "Counts describe the published document multiset, not exact packed training sequences or exposure weights.",
            "Duplicate records and repeated records across stages are counted as published; redacted text is unavailable.",
            "Literal names are proxies for entities; aliases and entity disambiguation are not inferred.",
            "Relation correlations use macro means on common support (both marginals >= min_count); coverage is reported.",
            "P-values are exploratory and unadjusted; facts sharing subjects or objects are not independent.",
            "Forward/reverse probability ratio equals object/subject marginal frequency when the joint count is positive.",
            "Retained best_faith values involve hyperparameter selection; this analysis does not remove that bias.",
        ],
        scopes={},
    )
    text = [
        f"Corpus conditional probabilities — {'COMPLETE' if complete else 'PARTIAL'}",
        f"Shards: {all_coverage['shards']:,}/{expected:,}; documents: {all_coverage['documents']:,}",
        f"LRE field: {args.lre_field}; source: {args.faith_dir}",
        "P(o|s)=documents(s AND o)/documents(s); P(s|o)=documents(s AND o)/documents(o).",
        "Relation-level Pearson r / Spearman rho (two-sided unadjusted p; matched facts):",
    ]
    for scope, counts in [("all_stages", total), *sorted(totals.items())]:
        c = all_coverage if scope == "all_stages" else coverage[scope]
        rows, relations, corr, subject_corr = analyze(
            counts, c["documents"], terms, facts, lre, args.min_count
        )
        csv_write(args.out / f"{scope}.facts.csv", rows)
        csv_write(args.out / f"{scope}.relations.csv", relations)
        output["scopes"][scope] = dict(
            coverage=c,
            correlations=corr,
            within_relation_subject_correlations=subject_corr,
        )
        text.append(
            f"\n{scope}: {c['documents']:,} documents; {c['redacted_documents']:,} redacted"
        )
        for r in corr:
            if r["cohort"] != "all_facts" or r["subset"] != "all":
                continue
            if r["pearson"] is None:
                text.append(
                    f"  {r['feature']:32s} n={r['n']:2d} undefined ({r['reason']})"
                )
            else:
                text.append(
                    f"  {r['feature']:32s} n={r['n']:2d} r={r['pearson']:+.4f} (p={r['pearson_p']:.3g}) rho={r['spearman']:+.4f} (p={r['spearman_p']:.3g})"
                )
    text.extend(["", *output["notes"]])
    write_json(args.out / "report.json", output)
    (args.out / "report.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--corpus-scope",
        choices=("training-split", "published-jsonl"),
        default="training-split",
        help="Default follows the published 6T split (*.jsonl.zst); published-jsonl also includes gzip PDF files outside that split",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "corpus_conditionals")
    parser.add_argument("--faith-dir", type=Path, default=ROOT / "results/main")
    parser.add_argument("--lre-field", default="best_faith")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            32,
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1,
        ),
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum count for BOTH marginals in correlations; counts are reusable",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Process at most this many new shards; report is explicitly partial",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit absent shards and successful exit for partial reports",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reuse counts without scanning; can change LRE source/min-count",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate inventory and print scope without scanning",
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.min_count < 1
        or (args.max_shards is not None and args.max_shards < 1)
    ):
        parser.error("workers, min-count and max-shards must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if not (args.plan or args.report_only):
        for dependency in ("ahocorasick", "orjson", "zstandard"):
            importlib.import_module(dependency)
    args.corpus_root, args.out = args.corpus_root.resolve(), args.out.resolve()
    terms, pairs, facts = vocabulary(ROOT, args.case_sensitive)
    logging.info("Validating corpus inventory")
    manifest, jobs, missing, excluded = inventory(args.corpus_root, args.corpus_scope)
    logging.info(
        "%s available shards (%.3f TB); %s missing; %s files outside scope; %s terms; %s facts",
        len(jobs),
        sum(j["size"] for j in jobs) / 1e12,
        len(missing),
        len(excluded),
        len(terms),
        len(facts),
    )
    if missing and not args.allow_partial:
        raise SystemExit(
            f"{len(missing)} shards absent or wrong size, e.g. {missing[0]}; finish download or explicitly --allow-partial"
        )
    load_lre(args.faith_dir, args.lre_field)
    if args.plan:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(
        dict(
            version=VERSION,
            corpus_root=str(args.corpus_root),
            manifest=manifest,
            terms=terms,
            pairs=pairs,
            text_field=args.text_field,
            case_sensitive=args.case_sensitive,
            corpus_scope=args.corpus_scope,
        )
    )
    with (args.out / "scan.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "Another scan/report is using this output directory"
            ) from None
        db = open_checkpoint(args.out / "counts.sqlite3", signature)
        try:
            write_json(
                args.out / "inventory.json",
                dict(
                    corpus_scope=args.corpus_scope,
                    available_shards=len(jobs),
                    missing_shards=missing,
                    excluded_files=excluded,
                    selected_compressed_bytes=sum(j["size"] for j in jobs),
                ),
            )
            run_scan(db, jobs, terms, pairs, args)
            complete = report(db, jobs, missing, manifest, terms, facts, args)
            write_json(
                args.out / "progress.json",
                dict(
                    state="complete" if complete else "partial",
                    report=str(args.out / "report.json"),
                ),
            )
        except BaseException as exc:
            write_json(
                args.out / "progress.json",
                dict(
                    state="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    error=str(exc),
                    message="Committed shards are preserved. Rerun to resume; previous reports may be stale.",
                ),
            )
            raise
        finally:
            db.close()
    if not complete and not (args.allow_partial or args.max_shards):
        raise SystemExit("Incomplete corpus report; resume the scan to finish")


if __name__ == "__main__":
    main()
