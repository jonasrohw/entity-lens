"""Geva attribute rate and separate Wikipedia TF controls on current shared rows."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import stopwords
from run_all_model_newline import OUTPUT, ROOT, sha, specs
from transformers import AutoTokenizer

from jlens.geva_baseline import (
    attribute_rate,
    attribute_tokens,
    correlation,
    frequency_ranks,
    subject_key,
)
from jlens.io import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    a = p.parse_args()
    spec = specs()[a.name]
    tok = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"], local_files_only=True
    )
    words = stopwords.words("english")
    stop = set(words)
    retrieval_path = ROOT / "generated/shared_baselines/wikipedia/retrieval.json"
    retrieval = read_json(retrieval_path)
    paths = sorted((OUTPUT / a.name / "shared_split").glob("*.json"))
    sources = {p.stem: read_json(p) for p in paths}
    subjects = sorted(
        {r["subject"] for v in sources.values() for r in v["lens"]["rows"]}
    )
    cache = ROOT / "generated/shared_baselines" / a.name
    first = np.load(cache / f"{subject_key(subjects[0])}.npz")
    vocab_size = len(first["logits"])
    decoded = tok.batch_decode([[i] for i in range(vocab_size)])
    eligible = np.array(
        [len(s.strip()) > 2 and s.strip().lower() not in stop for s in decoded]
    )
    targets = {s: set() for s in subjects}
    for v in sources.values():
        for row in v["lens"]["rows"]:
            targets[row["subject"]].add(row["model_top1_id"])
    results, validation = {}, []
    for i, subject in enumerate(subjects):
        hits = retrieval["subjects"][subject]
        selected = [h for h in hits if h["keep"]]
        text = "".join(" " + h["text"] for h in selected)
        counts = Counter(tok.encode(text, add_special_tokens=False))
        ranks, frequencies = frequency_ranks(counts, vocab_size)
        attributes = attribute_tokens((decoded[t] for t in counts), stop)
        path = cache / f"{subject_key(subject)}.npz"
        with np.load(path) as f:
            assert str(f["subject"]) == subject
            values, top_ids = f["logits"], f["top_ids"]
        top = [decoded[t] for t in top_ids]
        rate, overlap, denominator = attribute_rate(top, attributes, stop)
        result = dict(
            n_retrieved=len(hits),
            n_selected=len(selected),
            n_attributes=len(attributes),
            attribute_rate=rate,
            overlap=overlap,
            clean_predictions=denominator,
            top_predictions=top,
            vocabulary_correlation=correlation(frequencies[eligible], values[eligible]),
            targets={
                str(t): dict(
                    rank_wikipedia=float(ranks[t]),
                    wikipedia_count=counts[t],
                    rank_logit=int(np.sum(values > values[t])),
                )
                for t in targets[subject]
            },
            logits_sha256=sha(path),
        )
        results[subject] = result
        if (i + 1) % 100 == 0 or i + 1 == len(subjects):
            print("SCORED", a.name, i + 1, "/", len(subjects), flush=True)
    relations = {}
    for relation, source in sources.items():
        baseline_path = (
            ROOT
            / "results/shared_baselines"
            / a.name
            / "shared_split"
            / f"{relation}.json"
        )
        baseline = read_json(baseline_path)
        assert baseline["source_sha256"] == sha(
            OUTPUT / a.name / "shared_split" / f"{relation}.json"
        )
        rows = []
        for ref, base in zip(source["lens"]["rows"], baseline["rows"], strict=True):
            subject, target = ref["subject"], ref["model_top1_id"]
            info = results[subject]
            measured = info["targets"][str(target)]
            delta = measured["rank_logit"] - base["rank_logit_lens"]
            if delta:
                # Several vocabulary entries may lie within the same fp32
                # rounding interval. Gate the crossed logit gap, not the
                # number of almost-tied tokens; record both for inspection.
                with np.load(cache / f"{subject_key(subject)}.npz") as f:
                    values = f["logits"]
                gap = abs(
                    float(
                        np.sort(values)[::-1][base["rank_logit_lens"]] - values[target]
                    )
                )
                assert gap <= 1e-5 * max(1, abs(float(values[target]))), (
                    a.name,
                    relation,
                    subject,
                    delta,
                    gap,
                )
            else:
                gap = 0.0
            validation.append(
                dict(relation=relation, subject=subject, delta=delta, gap=gap)
            )
            rows.append(
                dict(
                    subject=subject,
                    model_top1_id=target,
                    **measured,
                    attribute_rate=info["attribute_rate"],
                    n_selected=info["n_selected"],
                    vocabulary_spearman=info["vocabulary_correlation"]["spearman"][
                        "statistic"
                    ],
                )
            )
        rates = [r["attribute_rate"] for r in rows if r["attribute_rate"] is not None]
        vocab_corr = [
            r["vocabulary_spearman"]
            for r in rows
            if r["vocabulary_spearman"] is not None
        ]
        relations[relation] = dict(
            n=len(rows),
            split_fingerprint=source["split_fingerprint"],
            geva_attribute_rate=float(np.mean(rates)) if rates else None,
            rate_missing=len(rows) - len(rates),
            wikipedia_tf=-float(
                np.median(np.log10([r["rank_wikipedia"] + 1 for r in rows]))
            ),
            tf_logit_spearman=float(np.mean(vocab_corr)) if vocab_corr else None,
            rows=rows,
        )
    write_json(
        ROOT / "results/shared_baselines" / a.name / "wikipedia.json",
        dict(
            metadata=dict(
                **spec,
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                script_sha256=sha(Path(__file__)),
                implementation_sha256=sha(ROOT / "jlens/geva_baseline.py"),
                retrieval_sha256=sha(retrieval_path),
                retrieval=retrieval["metadata"],
                stopwords=words,
                stopwords_sha256=hashlib.sha256("\n".join(words).encode()).hexdigest(),
                nltk_version=nltk.__version__,
                vocabulary_size=vocab_size,
                eligible_vocabulary=int(eligible.sum()),
                n_subjects=len(subjects),
                n_rows=sum(v["n"] for v in relations.values()),
                empty_retrieval_subjects=sum(
                    v["n_selected"] == 0 for v in results.values()
                ),
                rank_validation=validation,
            ),
            relations=relations,
            subjects=results,
        ),
    )
    print("COMPLETE", a.name, flush=True)


if __name__ == "__main__":
    main()
