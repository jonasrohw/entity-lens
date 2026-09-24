"""Wikipedia attributes-rate replication and an explicitly separate TF baseline.

Geva et al. (2023), section 6.1 / appendix C, as reconstructed by Gerstner
et al. (2026), appendix G.2. The published metric is set overlap, not a
correlation of two vocabulary rankings. We retain that metric and additionally
measure token-frequency ranks for the matched LRE-faithfulness comparison.
"""

# Attribute filtering/overlap adapted from Geva et al. (Copyright 2023 Google
# LLC) and Gerstner et al., licensed under Apache-2.0. This implementation adds
# the matched OLMo protocol, resumable readouts, TF ranks, and correlations.
# See docs/licenses/geva-gerstner-APACHE-2.0.txt and docs/geva-baseline.md.

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from jlens.config import MODEL_ID, OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def subject_key(subject):
    return hashlib.sha256(subject.encode()).hexdigest()[:24]


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    result = dict(n=len(x), excluded=int((~valid).sum()))
    for label, function in [("spearman", stats.spearmanr), ("pearson", stats.pearsonr)]:
        if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            result[label] = dict(statistic=None, pvalue=None)
        else:
            value = function(x, y)
            result[label] = dict(
                statistic=float(value.statistic), pvalue=float(value.pvalue)
            )
    return result


def attribute_tokens(decoded, stopwords):
    # Upstream intentionally does NOT lowercase the corpus-side stopword check.
    return {s for s in decoded if len(s.strip()) > 2 and s.strip() not in stopwords}


def attribute_rate(top_predictions, attributes, stopwords):
    # Filter AFTER taking the top 50, exactly as Gerstner's enrichment.py does.
    clean = [
        s
        for s in top_predictions
        if len(s.strip()) > 2 and s.strip().lower() not in stopwords
    ]
    if not clean:
        return None, 0, 0
    overlap = len(set(clean) & attributes)
    return 100.0 * overlap / len(clean), overlap, len(clean)


def frequency_ranks(counts, vocab_size):
    """Zero-based midranks over the full vocabulary; absent tokens tie at bottom."""
    vector = np.zeros(vocab_size, dtype=np.int64)
    for token, count in counts.items():
        vector[token] = count
    return stats.rankdata(-vector, method="average") - 1, vector


def validate_reference_rank(values, row):
    """Allow only a one-position float32 near-tie, recording its actual gap."""
    target = values[row["model_top1_id"]]
    rank = int(np.sum(values > target))
    delta = rank - row["rank_logit"]
    gap = 0.0
    if delta:
        ordered = np.sort(values)[::-1]
        gap = float(abs(ordered[row["rank_logit"]] - target))
        if abs(delta) > 1 or gap > 1e-5 * max(1.0, abs(float(target))):
            raise ValueError(
                f"Reference rank mismatch for {row['subject']}: {rank} != {row['rank_logit']}; gap={gap}"
            )
    return dict(rank=rank, reference_rank=row["rank_logit"], delta=delta, logit_gap=gap)


def lens(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import final_logits, resolve_blocks, resolve_final_norm

    reference = read_json(args.reference)
    subjects = sorted({row["subject"] for v in reference.values() for row in v["rows"]})
    layers = {v["layer"] for v in reference.values()}
    prompts = {v["neutral_prompt"] for v in reference.values()}
    if len(layers) != 1 or len(prompts) != 1:
        raise ValueError("Expected a single reference layer and neutral prompt")
    layer, prompt_template = layers.pop(), prompts.pop()
    config = dict(
        model=args.model,
        layer=layer,
        prompt=prompt_template,
        top_k=args.top_k,
        dtype="float32",
        reference_sha256=sha256(args.reference),
        readout="final_norm_then_unembedding",
        position="last_subject_token",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing lens configuration differs; choose another --out")
    write_json(config_path, config)
    pending = [s for s in subjects if not (args.out / f"{subject_key(s)}.npz").exists()]
    if not pending:
        print("All subject logits already cached", flush=True)
        return
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="eager"
        )
        .to(args.device)
        .eval()
    )
    model.requires_grad_(False)
    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach().float()
    bias = getattr(model.lm_head, "bias", None)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    if not 1 <= layer <= len(resolve_blocks(model)):
        raise ValueError("Reference layer outside model")
    captured = {}

    def capture(module, inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = state[0, captured["position"]].detach()

    handle = resolve_blocks(model)[layer - 1].register_forward_hook(capture)
    checked = []
    try:
        with torch.inference_mode():
            for i, subject in enumerate(pending):
                prompt = prompt_template.format(subject)
                encoded = tokenizer(
                    prompt, return_tensors="pt", return_offsets_mapping=True
                )
                end = prompt.rindex(subject) + len(subject)
                offsets = encoded.pop("offset_mapping")[0].tolist()
                position = max(
                    j
                    for j, (start, stop) in enumerate(offsets)
                    if stop > 0 and start < end
                )
                captured["position"] = position
                model(
                    input_ids=encoded["input_ids"].to(args.device),
                    output_hidden_states=True,
                )
                logits = final_logits(
                    captured["hidden"], norm, unembedding, bias, softcap
                )
                top_ids = logits.topk(args.top_k).indices.cpu().numpy()
                values = logits.cpu().numpy()
                for relation, entry in reference.items():
                    for row in entry["rows"]:
                        if row["subject"] == subject:
                            check = validate_reference_rank(values, row)
                            checked.append(
                                dict(relation=relation, subject=subject, **check)
                            )
                path = args.out / f"{subject_key(subject)}.npz"
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream, subject=subject, logits=values, top_ids=top_ids
                    )
                temporary.replace(path)
                if (i + 1) % 25 == 0 or i == len(pending) - 1:
                    print(
                        f"Logit lens: {i + 1}/{len(pending)} new subjects; retained ranks validated",
                        flush=True,
                    )
    finally:
        handle.remove()
    # Revalidate cached rows too, so the final audit covers interrupted runs.
    checked = []
    for relation, entry in reference.items():
        for row in entry["rows"]:
            with np.load(args.out / f"{subject_key(row['subject'])}.npz") as cached:
                check = validate_reference_rank(cached["logits"], row)
            checked.append(dict(relation=relation, subject=row["subject"], **check))
    write_json(
        args.out / "validation.json",
        dict(
            subjects=len(subjects),
            rows_checked=len(checked),
            exact_matches=sum(check["delta"] == 0 for check in checked),
            transformers=transformers.__version__,
            torch=torch.__version__,
            model_revision=getattr(model.config, "_commit_hash", None),
            checks=checked,
        ),
    )


def score(args):
    import nltk
    from nltk.corpus import stopwords
    from transformers import AutoTokenizer

    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True, raise_on_error=True)
        words = stopwords.words("english")
    stop = set(words)
    reference, figure, retrieval = map(
        read_json, [args.reference, args.figure, args.retrieval]
    )
    lens_config = read_json(args.lens / "config.json")
    if lens_config["reference_sha256"] != sha256(args.reference):
        raise ValueError("Lens cache belongs to another reference")
    if set(reference) != set(figure["rels"]):
        raise ValueError("Reference and faithfulness relation sets differ")
    tokenizer = AutoTokenizer.from_pretrained(lens_config["model"])
    first = next(iter(reference.values()))["rows"][0]["subject"]
    with np.load(args.lens / f"{subject_key(first)}.npz") as cached:
        vocab_size = len(cached["logits"])
    decoded = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    eligible = np.array(
        [len(s.strip()) > 2 and s.strip().lower() not in stop for s in decoded]
    )
    subjects = sorted({row["subject"] for v in reference.values() for row in v["rows"]})
    subject_results, ranks_by_subject, counts_by_subject = {}, {}, {}
    vocabulary_rows = {}
    for subject in subjects:
        hits = retrieval["subjects"][subject]
        selected = [hit for hit in hits if hit["keep"]]
        # Leading space and concatenation match the upstream wiki.csv builder.
        text = "".join(" " + hit["text"] for hit in selected)
        ids = tokenizer.encode(text, add_special_tokens=False)
        counts = Counter(ids)
        ranks, frequencies = frequency_ranks(counts, vocab_size)
        ranks_by_subject[subject], counts_by_subject[subject] = ranks, counts
        attributes = attribute_tokens((decoded[i] for i in counts), stop)
        with np.load(args.lens / f"{subject_key(subject)}.npz") as cached:
            if str(cached["subject"]) != subject:
                raise ValueError("Cache subject mismatch")
            logits, top_ids = cached["logits"], cached["top_ids"]
        top = [decoded[i] for i in top_ids]
        rate, overlap, denominator = attribute_rate(top, attributes, stop)
        agreement = correlation(frequencies[eligible], logits[eligible])
        subject_results[subject] = dict(
            n_retrieved=len(hits),
            n_selected=len(selected),
            n_attributes=len(attributes),
            attribute_rate=rate,
            overlap=overlap,
            clean_predictions=denominator,
            top_predictions=top,
            vocabulary_correlation=agreement,
        )
        vocabulary_rows[subject] = [
            dict(
                token_id=int(i),
                token=decoded[i],
                count=int(n),
                rank=float(ranks[i]),
                attribute=decoded[i] in attributes,
            )
            for i, n in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ]
    faith = dict(zip(figure["rels"], figure["faith"], strict=True))
    relations = {}
    for relation, entry in reference.items():
        rows = []
        for row in entry["rows"]:
            subject, token = row["subject"], row["model_top1_id"]
            info = subject_results[subject]
            rows.append(
                dict(
                    subject=subject,
                    model_top1_id=token,
                    model_top1=row["model_top1"],
                    rank_logit=row["rank_logit"],
                    rank_wikipedia=float(ranks_by_subject[subject][token]),
                    wikipedia_count=counts_by_subject[subject].get(token, 0),
                    attribute_rate=info["attribute_rate"],
                    n_selected=info["n_selected"],
                    target_is_attribute=decoded[token]
                    in {v["token"] for v in vocabulary_rows[subject] if v["attribute"]},
                )
            )
        rates = [r["attribute_rate"] for r in rows if r["attribute_rate"] is not None]
        relations[relation] = dict(
            n=len(rows),
            faithfulness=faith[relation],
            mean_attribute_rate=float(np.mean(rates)) if rates else None,
            rate_missing=len(rows) - len(rates),
            wikipedia_score=-float(
                np.median(np.log10([r["rank_wikipedia"] + 1 for r in rows]))
            ),
            logit_score=-entry["agg"]["median_lograng"],
            rows=rows,
        )
    values = [relations[r] for r in figure["rels"]]
    reports = {}
    for feature in ["mean_attribute_rate", "wikipedia_score", "logit_score"]:
        reports[feature + "_vs_faithfulness"] = correlation(
            [v[feature] for v in values], figure["faith"]
        )
    reports["wikipedia_score_vs_logit_score"] = correlation(
        [v["wikipedia_score"] for v in values], [v["logit_score"] for v in values]
    )
    agreements = [
        v["vocabulary_correlation"]["spearman"]["statistic"]
        for v in subject_results.values()
    ]
    valid_agreements = [v for v in agreements if v is not None]
    report = dict(
        metadata=dict(
            **lens_config,
            retrieval=retrieval["metadata"],
            retrieval_sha256=sha256(args.retrieval),
            figure_sha256=sha256(args.figure),
            stopwords=words,
            stopwords_sha256=hashlib.sha256("\n".join(words).encode()).hexdigest(),
            nltk_version=nltk.__version__,
            token_rank_ties="average, zero-based, full vocabulary",
            primary="mean_attribute_rate_vs_faithfulness",
            vocabulary_size=vocab_size,
            eligible_vocabulary=int(eligible.sum()),
            n_subjects=len(subjects),
            n_rows=sum(v["n"] for v in values),
            empty_retrieval_subjects=sum(
                v["n_selected"] == 0 for v in subject_results.values()
            ),
            lens_validation=read_json(args.lens / "validation.json"),
        ),
        correlations=reports,
        vocabulary_agreement=dict(
            n=len(valid_agreements),
            undefined=len(agreements) - len(valid_agreements),
            mean_spearman=float(np.mean(valid_agreements)),
            median_spearman=float(np.median(valid_agreements)),
        ),
        relations=relations,
        subjects=subject_results,
    )
    write_json(args.out, report)
    write_json(args.out.with_name(args.out.stem + "_vocabulary.json"), vocabulary_rows)
    print(json.dumps(reports, indent=2), flush=True)
    print(json.dumps(report["vocabulary_agreement"], indent=2), flush=True)


def report_results(path, plot_dir=None):
    """CPU-only recomputation from the retained per-relation measurements."""
    result = read_json(path)
    values = list(result["relations"].values())
    faith = [v["faithfulness"] for v in values]
    labels = {
        "mean_attribute_rate": "Geva/Gerstner attribute rate",
        "wikipedia_score": "Wikipedia frequency rank (extension)",
        "logit_score": "Matched logit-lens answer rank",
    }
    for feature, label in labels.items():
        computed = correlation([v[feature] for v in values], faith)
        r, p = computed["spearman"]["statistic"], computed["spearman"]["pvalue"]
        pearson = computed["pearson"]["statistic"]
        print(
            f"{label} vs LRE faithfulness: n={computed['n']}, Spearman rho={r:.6f}, p={p:.6g}; Pearson r={pearson:.6f}"
        )
    meta = result["metadata"]
    print(
        f"Coverage: {meta['n_rows']} rows, {meta['n_subjects']} subjects; {meta['empty_retrieval_subjects']} subjects with no retained paragraphs."
    )
    agreement = result["vocabulary_agreement"]
    print(
        f"Wikipedia frequency vs logit-lens vocabulary scores: mean subject Spearman rho={agreement['mean_spearman']:.6f}, n={agreement['n']} (undefined: {agreement['undefined']})."
    )
    if plot_dir is not None:
        import matplotlib.pyplot as plt

        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        panels = [
            ("mean_attribute_rate", "Attribute rate (%)", "Geva/Gerstner metric"),
            (
                "wikipedia_score",
                "−median log₁₀(Wikipedia rank + 1)",
                "Frequency-rank extension",
            ),
        ]
        for ax, (feature, xlabel, title) in zip(axes, panels, strict=True):
            x = np.array([v[feature] for v in values])
            r = correlation(x, faith)["spearman"]
            ax.scatter(
                x, faith, s=35, color="#2166ac", edgecolor="white", linewidth=0.5
            )
            ax.set(xlabel=xlabel, title=title, ylim=(-0.04, 1.06))
            ax.text(
                0.04,
                0.05,
                f"Spearman ρ = {r['statistic']:.3f}\np = {r['pvalue']:.3g}; n = {len(values)}",
                transform=ax.transAxes,
            )
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("LRE faithfulness")
        fig.suptitle("OLMo-3-1025-7B · 28 relations · full Wikipedia retrieval")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(plot_dir / f"geva_baseline.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("lens")
    measure.add_argument(
        "--reference", type=Path, default=ROOT / "results/olmo/results/logit_lens.json"
    )
    measure.add_argument("--model", default=MODEL_ID)
    measure.add_argument("--device", default="cuda:0")
    measure.add_argument("--top-k", type=int, default=50)
    measure.add_argument("--out", type=Path, default=OUTPUT_DIR / "geva_baseline/lens")
    evaluate = sub.add_parser("score")
    evaluate.add_argument(
        "--reference", type=Path, default=ROOT / "results/olmo/results/logit_lens.json"
    )
    evaluate.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results/olmo/results/paper_figure_data.json",
    )
    evaluate.add_argument("--retrieval", type=Path, required=True)
    evaluate.add_argument(
        "--lens", type=Path, default=OUTPUT_DIR / "geva_baseline/lens"
    )
    evaluate.add_argument(
        "--out", type=Path, default=OUTPUT_DIR / "geva_baseline/results.json"
    )
    report_parser = sub.add_parser("report")
    report_parser.add_argument(
        "--results", type=Path, default=ROOT / "results/olmo/results/geva_baseline.json"
    )
    report_parser.add_argument("--plot-dir", type=Path)
    subjects_parser = sub.add_parser("subjects")
    subjects_parser.add_argument(
        "--reference", type=Path, default=ROOT / "results/olmo/results/logit_lens.json"
    )
    subjects_parser.add_argument(
        "--out", type=Path, default=OUTPUT_DIR / "geva_baseline/subjects.json"
    )
    args = parser.parse_args()
    if args.command == "report":
        report_results(args.results, args.plot_dir)
    elif args.command == "subjects":
        reference = read_json(args.reference)
        write_json(
            args.out,
            sorted({row["subject"] for v in reference.values() for row in v["rows"]}),
        )
    else:
        (lens if args.command == "lens" else score)(args)


if __name__ == "__main__":
    main()
