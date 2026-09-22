"""Validate, curate, and typeset current shared-split baseline comparisons."""

import csv
import json
from pathlib import Path

import numpy as np
from report_qwen_prompt_sensitivity import bootstrap_rho
from run_all_model_newline import OUTPUT, PROMPT, ROOT, sha, specs
from scipy.stats import pearsonr, rankdata, spearmanr

from jlens import affine_beta
from jlens.geva_baseline import correlation
from jlens.io import read_json, write_json

DEST = ROOT / "results/shared_baselines"
PAPER = ROOT.parent / "paper"
MODELS = [
    "olmo3_7b",
    "gptj",
    "mistral7b",
    "gemma2_9b",
    "qwen3_8b_base",
    "qwen3_8_27b",
    "pythia6_9b",
]
LABELS = [
    "OLMo 3 7B",
    "GPT-J 6B",
    "Mistral 7B",
    "Gemma 2 9B",
    "Qwen3 8B Base",
    "Qwen3.8 27B",
    "Pythia 6.9B",
]
SHORT = ["OLMo", "GPT-J", "Mistral", "Gemma", "Qwen 8B", "Qwen 27B", "Pythia"]
NAMES = {
    "differential": "Differential Jacobian",
    "raw_jacobian": "Raw Jacobian",
    "logit_lens": "Logit lens",
    "differential_logit_lens": "Differential logit lens",
    "affine": "Affine Jacobian (beta=1)",
    "geva_attribute_rate": "Geva attribute rate",
    "wikipedia_tf": "Wikipedia token-frequency rank",
    "tf_logit_spearman": "Wikipedia TF/logit vocabulary agreement",
    "target_prior": "Model answer-token prior",
    "gold_prior": "Gold answer-token prior",
    "cardinality_full": "Full-data samples/objects",
    "cardinality_filtered": "Filtered samples/objects",
    "cardinality_test": "Test samples/objects",
    "prior_faith": "Relation-prompt intercept-only agreement",
    "target_modal_fraction": "Modal-target fraction (oracle)",
    "target_entropy_bits": "Target entropy",
    "lm_accuracy": "Model gold-token accuracy",
    "corpus_subject_count": "Subject document frequency",
    "corpus_object_count": "Object document frequency",
    "corpus_cooccurrence": "Joint document frequency",
    "corpus_log1p_cooccurrence": "Mean log(1+joint frequency)",
    "corpus_log_mean_cooccurrence": "Log(1+mean joint frequency)",
    "corpus_p_object_given_subject": "P(object | subject)",
    "corpus_p_subject_given_object": "P(subject | object)",
    "corpus_direction_difference": "Forward minus reverse conditional",
    "corpus_micro_p_object_given_subject": "Micro P(object | subject)",
    "corpus_micro_p_subject_given_object": "Micro P(subject | object)",
}
LENS = [
    "differential",
    "raw_jacobian",
    "logit_lens",
    "differential_logit_lens",
    "affine",
    "geva_attribute_rate",
    "wikipedia_tf",
    "tf_logit_spearman",
]
CONTROLS = [
    "target_prior",
    "gold_prior",
    "cardinality_full",
    "cardinality_filtered",
    "cardinality_test",
    "prior_faith",
    "target_modal_fraction",
    "target_entropy_bits",
    "lm_accuracy",
]
CORPUS = [k for k in NAMES if k.startswith("corpus_")]


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def number(x, digits=3):
    return "--" if x is None else f"{x:+.{digits}f}"


def latex(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(path, caption, label, headers, rows, *, size="scriptsize"):
    text = [
        r"\begin{table}[htbp]",
        r"\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\" + size,
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\linewidth}{!}{%",
        "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    path.write_text("\n".join(text))


def partial(x, y, controls):
    x, y = rankdata(x), rankdata(y)
    z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls])
    xr = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    yr = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    return float(pearsonr(xr, yr).statistic)


def corpus_features(rows):
    fields = [
        "subject_count",
        "object_count",
        "cooccurrence",
        "log1p_cooccurrence",
        "p_object_given_subject",
        "p_subject_given_object",
        "direction_difference",
    ]
    eligible = [
        r for r in rows if int(r["subject_count"]) >= 1 and int(r["object_count"]) >= 1
    ]
    out = {
        "corpus_" + k: float(np.mean([float(r[k]) for r in eligible]))
        if eligible
        else None
        for k in fields
    }
    out["corpus_log_mean_cooccurrence"] = (
        float(np.log1p(out["corpus_cooccurrence"])) if eligible else None
    )
    for name, denominator in [
        ("object_given_subject", "subject_count"),
        ("subject_given_object", "object_count"),
    ]:
        out["corpus_micro_p_" + name] = (
            sum(int(r["cooccurrence"]) for r in eligible)
            / sum(int(r[denominator]) for r in eligible)
            if eligible
            else None
        )
    return out, len(eligible)


def main():
    provenance = {}

    def read(path):
        provenance[str(path.relative_to(ROOT.parent))] = sha(path)
        return read_json(path)

    executed = read(DEST / "provenance/executed_sources.json")
    wiki_validation = read(DEST / "wikipedia_validation.json")
    assert all(
        wiki_validation[k]
        for k in (
            "counts_exact",
            "sparse_scores_exact",
            "retrieved_ids_exact",
            "retrieved_scores_exact",
        )
    )
    assert wiki_validation["script_sha256"] == sha(
        ROOT / "scripts/validate_shared_wikipedia.py"
    )

    def checked_source(relative, expected):
        path = ROOT / relative
        if sha(path) != expected:
            archive = executed[relative]
            assert archive["executed_sha256"] == expected
            path = ROOT / archive["snapshot"]
        assert sha(path) == expected, relative
        provenance[str(path.relative_to(ROOT.parent))] = expected

    corpus_report = read(ROOT / "generated/corpus_conditionals/report.json")
    assert (
        corpus_report["complete"]
        and corpus_report["expected_shards"] == corpus_report["processed_shards"]
    )
    corpus_sets = {}
    for path in sorted((ROOT / "generated/corpus_conditionals").glob("*.facts.csv")):
        provenance[str(path.relative_to(ROOT.parent))] = sha(path)
        with path.open() as f:
            corpus_sets[path.name.removesuffix(".facts.csv")] = {
                (r["relation"], r["subject"], r["object"]): r for r in csv.DictReader(f)
            }
    assert len(corpus_sets) == 4
    data, audits, corpus_rows, row_audit = [], {}, [], []
    for name in MODELS:
        spec = specs()[name]
        wiki = read(DEST / name / "wikipedia.json")
        checked_source(
            "scripts/retrieve_shared_wikipedia.py",
            wiki["metadata"]["retrieval"]["script_sha256"],
        )
        assert wiki["metadata"]["revision"] == spec["revision"]
        checked_source(
            "scripts/score_shared_wikipedia.py", wiki["metadata"]["script_sha256"]
        )
        assert wiki["metadata"]["implementation_sha256"] == sha(
            ROOT / "jlens/geva_baseline.py"
        )
        assert wiki["metadata"]["retrieval_sha256"] == sha(
            ROOT / "generated/shared_baselines/wikipedia/retrieval.json"
        )
        audit_metadata = read(
            ROOT / "results/cross_model_audit/raw/targets" / name / "metadata.json"
        )
        assert audit_metadata["revision"] == spec["revision"]
        paths = sorted((OUTPUT / name / "shared_split").glob("*.json"))
        assert len(paths) == 25 and {p.stem for p in paths} == set(wiki["relations"])
        for path in paths:
            current = read(path)
            base = read(DEST / name / "shared_split" / path.name)
            assert base["source_sha256"] == sha(path)
            assert base["prompt"] == current["lens"]["neutral_prompt"] == PROMPT
            assert base["layer"] == current["lens"]["layer"] == spec["lens_layer"]
            assert (
                base["split_fingerprint"]
                == current["split_fingerprint"]
                == wiki["relations"][path.stem]["split_fingerprint"]
            )
            for root, worker in [(DEST, base["worker"]), (OUTPUT, current["worker"])]:
                metadata = read(root / name / f"worker_{worker}.json")
                assert (
                    metadata["revision"] == spec["revision"]
                    and metadata["dtype"] == "float32"
                )
                for rel, expected in metadata["source_sha256"].items():
                    checked_source(rel, expected)
            assert (
                base["data_sha256"]
                == current["data_sha256"]
                == sha(ROOT / "data/relations" / path.name)
            )
            targets = read(
                ROOT
                / "results/cross_model_audit/raw/targets"
                / name
                / "shared_split"
                / path.name
            )
            assert (
                targets["subjects"]
                == [r["subject"] for r in base["rows"]]
                == [r["subject"] for r in current["lens"]["rows"]]
            )
            assert (
                targets["targets"]
                == [r["model_top1_id"] for r in base["rows"]]
                == [r["model_top1_id"] for r in current["lens"]["rows"]]
            )
            assert targets["gold_ids"] == [r["gold_id"] for r in base["rows"]]
            assert targets["split_fingerprint"] == base["split_fingerprint"]
            assert (
                abs(targets["selected_faith"] - current["faithfulness"]["best_faith"])
                < 1e-12
            )
            lens = current["lens"]
            row = dict(
                model=name,
                relation=path.stem,
                n=base["n"],
                n_lre=current["n_lre"],
                faithfulness=current["faithfulness"]["best_faith"],
                split_fingerprint=base["split_fingerprint"],
                differential=-lens["agg"]["diff"]["median_lograng"],
                affine=-lens["agg"]["lre"]["median_lograng"],
            )
            assert row["n"] == row["n_lre"] == len(base["rows"])
            for kind, aggregate in base["agg"].items():
                score = -float(
                    np.median(np.log10([r["rank_" + kind] + 1 for r in base["rows"]]))
                )
                assert abs(score + aggregate["median_lograng"]) < 1e-12
                row[kind] = score
            for beta in lens["beta_sweep"]["betas"]:
                key = f"affine_beta_{beta}"
                NAMES[key] = f"Affine beta={beta:g}"
                row[key] = -lens["beta_sweep"]["agg"][str(beta)]["median_lograng"]
            for metric in affine_beta.SELECTION_METRICS:
                key = "affine_tuned_" + metric
                NAMES[key] = "Affine tuned " + metric + " (in sample)"
                row[key] = affine_beta.tuned_score(lens["beta_sweep"], metric)[1]
            for key in ("target_prior", "gold_prior"):
                row[key] = float(np.mean([r[key] for r in base["rows"]]))
            for key in ("cardinality_full", "cardinality_filtered", "cardinality_test"):
                row[key] = base[key]
            for key in (
                "prior_faith",
                "target_modal_fraction",
                "target_entropy_bits",
                "lm_accuracy",
            ):
                row[key] = targets[key]
            for key in ("geva_attribute_rate", "wikipedia_tf", "tf_logit_spearman"):
                row[key] = wiki["relations"][path.stem][key]
            row["geva_n_defined"] = (
                base["n"] - wiki["relations"][path.stem]["rate_missing"]
            )
            for scope, facts in corpus_sets.items():
                selected = [
                    facts[(path.stem, r["subject"], r["object"])] for r in base["rows"]
                ]
                for subset in ("all", "no_lexical_overlap"):
                    chosen = (
                        selected
                        if subset == "all"
                        else [r for r in selected if r["lexical_overlap"] == "False"]
                    )
                    features, eligible = corpus_features(chosen)
                    corpus_rows.append(
                        dict(
                            model=name,
                            relation=path.stem,
                            scope=scope,
                            subset=subset,
                            n_test=base["n"],
                            n_subset=len(chosen),
                            n_eligible=eligible,
                            faithfulness=row["faithfulness"],
                            **features,
                        )
                    )
                    if scope == "all_stages" and subset == "all":
                        row.update(features)
                        row["corpus_n_eligible"] = eligible
            data.append(row)
            for i, (b, lens_row, w) in enumerate(
                zip(
                    base["rows"],
                    lens["rows"],
                    wiki["relations"][path.stem]["rows"],
                    strict=True,
                )
            ):
                assert b["subject"] == lens_row["subject"] == w["subject"]
                row_audit.append(
                    dict(
                        model=name,
                        relation=path.stem,
                        row=i,
                        **{
                            k: b[k]
                            for k in (
                                "subject",
                                "object",
                                "model_top1_id",
                                "gold_id",
                                "n_references",
                                "rank_raw_jacobian",
                                "rank_logit_lens",
                                "rank_differential_logit_lens",
                                "target_prior",
                                "gold_prior",
                            )
                        },
                        rank_diff=lens_row["rank_diff"],
                        rank_affine=lens_row["rank_lre"],
                        rank_wikipedia=w["rank_wikipedia"],
                        attribute_rate=w["attribute_rate"],
                        n_selected=w["n_selected"],
                        selected_lre_hit=targets["selected_lre_hits"][i],
                    )
                )
        audits[name] = dict(
            spec=spec,
            n_relations=len(paths),
            n_rows=wiki["metadata"]["n_rows"],
            n_subjects=wiki["metadata"]["n_subjects"],
            empty_retrieval_subjects=wiki["metadata"]["empty_retrieval_subjects"],
            rank_near_ties=sum(
                r["delta"] != 0 for r in wiki["metadata"]["rank_validation"]
            ),
            rate_missing=sum(v["rate_missing"] for v in wiki["relations"].values()),
        )
    assert len(data) == 175
    all_correlations = []

    def add(
        name,
        feature,
        rows,
        *,
        cohort="all_25",
        scope="all_stages",
        subset="all",
        outcome="faithfulness",
        category="primary",
    ):
        available = [
            r for r in rows if r.get(feature) is not None and r.get(outcome) is not None
        ]
        c = correlation(
            [r[feature] for r in available], [r[outcome] for r in available]
        )
        result = dict(
            model=name,
            feature=feature,
            label=NAMES.get(feature, feature),
            category=category,
            cohort=cohort,
            scope=scope,
            subset=subset,
            outcome=outcome,
            n=c["n"],
            excluded=len(rows) - c["n"],
            spearman_rho=c["spearman"]["statistic"],
            spearman_p=c["spearman"]["pvalue"],
            pearson_r=c["pearson"]["statistic"],
            pearson_p=c["pearson"]["pvalue"],
        )
        all_correlations.append(result)
        return result

    for name in MODELS:
        rows = [r for r in data if r["model"] == name]
        for cohort in ("all_25", "at_least_4_tests"):
            chosen = [r for r in rows if cohort == "all_25" or r["n_lre"] >= 4]
            for feature in NAMES:
                add(name, feature, chosen, cohort=cohort)
        for scope in corpus_sets:
            for subset in ("all", "no_lexical_overlap"):
                chosen = [
                    r
                    for r in corpus_rows
                    if r["model"] == name
                    and r["scope"] == scope
                    and r["subset"] == subset
                ]
                for feature in CORPUS:
                    add(
                        name,
                        feature,
                        chosen,
                        scope=scope,
                        subset=subset,
                        category="corpus_sensitivity",
                    )
    geva_matched = []
    matched_features = [
        "differential",
        "raw_jacobian",
        "logit_lens",
        "differential_logit_lens",
        "affine",
        "wikipedia_tf",
        "geva_attribute_rate",
    ]
    for name in MODELS:
        for relation in sorted(
            {r["relation"] for r in row_audit if r["model"] == name}
        ):
            rows = [
                r
                for r in row_audit
                if r["model"] == name
                and r["relation"] == relation
                and r["attribute_rate"] is not None
            ]
            if len(rows) < 2:
                continue
            result = dict(
                model=name,
                relation=relation,
                n=len(rows),
                faithfulness=float(np.mean([r["selected_lre_hit"] for r in rows])),
                geva_attribute_rate=float(np.mean([r["attribute_rate"] for r in rows])),
            )
            for feature, rank in [
                ("differential", "rank_diff"),
                ("raw_jacobian", "rank_raw_jacobian"),
                ("logit_lens", "rank_logit_lens"),
                ("differential_logit_lens", "rank_differential_logit_lens"),
                ("affine", "rank_affine"),
                ("wikipedia_tf", "rank_wikipedia"),
            ]:
                result[feature] = -float(
                    np.median(np.log10([r[rank] + 1 for r in rows]))
                )
            geva_matched.append(result)
        rows = [r for r in geva_matched if r["model"] == name]
        for feature in matched_features:
            add(
                name,
                feature,
                rows,
                cohort="geva_defined_at_least_2_rows",
                category="geva_coverage_sensitivity",
            )
    sensitivity, gaps, robustness = [], [], []
    for name in MODELS:
        rows = [r for r in data if r["model"] == name]
        x = np.array([r["differential"] for r in rows])
        y = np.array([r["faithfulness"] for r in rows])
        domains = [
            read(ROOT / "data/relations" / (r["relation"] + ".json"))["properties"][
                "domain_name"
            ]
            for r in rows
        ]
        for excluded in sorted(set(domains)):
            selected = [r for r, d in zip(rows, domains) if d != excluded]
            c = correlation(
                [r["differential"] for r in selected],
                [r["faithfulness"] for r in selected],
            )
            robustness.append(
                dict(
                    model=name,
                    analysis="leave_one_domain_out",
                    excluded_domain=excluded,
                    n=c["n"],
                    rho=c["spearman"]["statistic"],
                )
            )
        xr, yr = rankdata(x), rankdata(y)
        xr -= xr.mean()
        yr -= yr.mean()
        rng = np.random.default_rng(20260921)
        perm = (
            rng.permuted(np.tile(xr, (10000, 1)), axis=1)
            @ yr
            / np.sqrt((xr @ xr) * (yr @ yr))
        )
        observed = float(spearmanr(x, y).statistic)
        robustness.append(
            dict(
                model=name,
                analysis="permutation",
                n=len(rows),
                rho=observed,
                p=float((1 + np.sum(np.abs(perm) >= abs(observed))) / 10001),
                draws=10000,
                seed=20260921,
            )
        )
        correct = []
        for row in rows:
            p = (
                ROOT
                / "results/cross_model_audit/raw/targets"
                / name
                / "shared_split"
                / (row["relation"] + ".json")
            )
            target = read(p)
            if sum(target["correct"]) >= 4:
                correct.append(
                    dict(
                        row,
                        correct_faith=float(
                            np.mean(
                                [
                                    hit
                                    for hit, valid in zip(
                                        target["selected_lre_hits"], target["correct"]
                                    )
                                    if valid
                                ]
                            )
                        ),
                    )
                )
        c = correlation(
            [r["differential"] for r in correct], [r["correct_faith"] for r in correct]
        )
        robustness.append(
            dict(
                model=name,
                analysis="model_correct_outcome_at_least_4",
                n=c["n"],
                rho=c["spearman"]["statistic"],
                p=c["spearman"]["pvalue"],
            )
        )
        controls = {
            "cardinality": [r["cardinality_full"] for r in rows],
            "answer_prior": [r["target_prior"] for r in rows],
            "object_frequency": [r["corpus_object_count"] for r in rows],
        }
        for keys in [
            ("cardinality",),
            ("answer_prior",),
            ("cardinality", "answer_prior"),
            ("object_frequency",),
            ("cardinality", "answer_prior", "object_frequency"),
        ]:
            sensitivity.append(
                dict(
                    model=name,
                    analysis="partial_spearman",
                    controls=" + ".join(keys),
                    n=len(rows),
                    rho=partial(x, y, [controls[k] for k in keys]),
                )
            )
        headroom = [
            dict(
                r,
                headroom=(r["faithfulness"] - r["prior_faith"])
                / (1 - r["prior_faith"]),
            )
            for r in rows
            if r["prior_faith"] < 1
        ]
        result = add(
            name,
            "differential",
            headroom,
            outcome="headroom",
            category="outcome_sensitivity",
        )
        sensitivity.append(
            dict(
                model=name,
                analysis="intercept_adjusted_outcome",
                n=result["n"],
                rho=result["spearman_rho"],
            )
        )
        boot = np.random.default_rng(20260921).integers(
            0, len(rows), size=(10000, len(rows))
        )
        bx = bootstrap_rho(x, y, boot)
        for feature in [
            "raw_jacobian",
            "logit_lens",
            "differential_logit_lens",
            "affine",
            "geva_attribute_rate",
            "wikipedia_tf",
        ]:
            other = np.array([r[feature] for r in rows])
            diff = bx - bootstrap_rho(other, y, boot)
            low, high = np.nanpercentile(diff, [2.5, 97.5])
            gaps.append(
                dict(
                    model=name,
                    baseline=feature,
                    n=len(rows),
                    delta_rho=float(
                        spearmanr(x, y).statistic - spearmanr(other, y).statistic
                    ),
                    ci_low=float(low),
                    ci_high=float(high),
                    draws=10000,
                    seed=20260921,
                )
            )
    prominence = []
    for name in ["olmo3_7b", "gptj"]:
        for path in sorted((DEST / name / "prominence").glob("*.json")):
            p = read(path)
            checked_source(
                "scripts/run_shared_prominence.py", p["metadata"]["script_sha256"]
            )
            assert p["metadata"]["revision"] == specs()[name]["revision"]
            if "skipped" in p:
                continue
            assert p["source_sha256"] == sha(OUTPUT / name / "shared_split" / path.name)
            full = [
                r
                for r in row_audit
                if r["model"] == name and r["relation"] == path.stem
            ]
            for subset in ("all", "no_name_leakage"):
                rows = [r for r in p["rows"] if subset == "all" or not r["leak"]]
                if len(rows) < 5:
                    continue
                hits = []
                for row in rows:
                    matches = [
                        r
                        for r in full
                        if r["subject"] == row["subject"]
                        and r["object"] == row["object"]
                    ]
                    assert (
                        matches and len({r["selected_lre_hit"] for r in matches}) == 1
                    )
                    hits.append(matches[0]["selected_lre_hit"])
                prominence.append(
                    dict(
                        model=name,
                        relation=path.stem,
                        subset=subset,
                        n=len(rows),
                        prominence=float(np.mean([r["hit"] for r in rows])),
                        gold_in_lens=float(np.mean([r["gold_in_lens"] for r in rows])),
                        faithfulness=float(np.mean(hits)),
                    )
                )
        assert len(list((DEST / name / "prominence").glob("*.json"))) == 14
        for subset in ("all", "no_name_leakage"):
            rows = [
                r for r in prominence if r["model"] == name and r["subset"] == subset
            ]
            for outcome in ("faithfulness", "gold_in_lens"):
                add(
                    name,
                    "prominence",
                    rows,
                    cohort="first_40_scorable",
                    subset=subset,
                    outcome=outcome,
                    category="prominence_diagnostic",
                )
    csv_write(DEST / "relation_scores.csv", data)
    csv_write(DEST / "subject_scores.csv", row_audit)
    csv_write(DEST / "corpus_relation_scores.csv", corpus_rows)
    csv_write(DEST / "correlations.csv", all_correlations)
    csv_write(DEST / "sensitivity.csv", sensitivity)
    csv_write(DEST / "paired_gaps.csv", gaps)
    csv_write(DEST / "robustness.csv", robustness)
    csv_write(DEST / "prominence.csv", prominence)
    csv_write(DEST / "geva_matched_relation_scores.csv", geva_matched)
    provenance["code/scripts/report_shared_baselines.py"] = sha(Path(__file__))
    report = dict(
        prompt=PROMPT,
        protocol="shared_split",
        model_order=MODELS,
        audits=audits,
        correlations=all_correlations,
        sensitivity=sensitivity,
        paired_gaps=gaps,
        robustness=robustness,
        corpus_metadata={
            k: corpus_report[k]
            for k in (
                "complete",
                "processed_shards",
                "unit",
                "matching",
                "corpus_revisions",
                "corpus_scope",
                "notes",
            )
        },
        source_sha256=provenance,
    )
    write_json(DEST / "summary.json", report)

    def lookup(
        model,
        feature,
        cohort="all_25",
        category="primary",
        scope="all_stages",
        subset="all",
    ):
        return next(
            r
            for r in all_correlations
            if (
                r["model"],
                r["feature"],
                r["cohort"],
                r["category"],
                r["scope"],
                r["subset"],
            )
            == (model, feature, cohort, category, scope, subset)
        )

    generated = PAPER / "generated/shared_baselines"
    generated.mkdir(parents=True, exist_ok=True)
    differential = [lookup(m, "differential")["spearman_rho"] for m in MODELS]
    raw = [lookup(m, "raw_jacobian")["spearman_rho"] for m in MODELS]
    raw_wins = sum(a > b for a, b in zip(raw, differential))
    raw_gaps = [r for r in gaps if r["baseline"] == "raw_jacobian"]
    qwen_ll = lookup("qwen3_8b_base", "logit_lens")["spearman_rho"]
    findings = f"The differential-Jacobian correlation ranges from {min(differential):.3f} to {max(differential):.3f}. The raw Jacobian has a higher point estimate in {raw_wins} of seven models; thus these results do not establish a general benefit from reference subtraction. For Qwen3 8B Base, ordinary logit lens reaches $\\rho={qwen_ll:.3f}$, compared with $\\rho={lookup('qwen3_8b_base', 'differential')['spearman_rho']:.3f}$ for differential Jacobian. "
    if all(r["ci_low"] <= 0 <= r["ci_high"] for r in raw_gaps):
        findings += "Every paired 95\\% bootstrap interval for differential minus raw Jacobian includes zero. "
    findings += "The tables retain negative as well as positive correlations and do not select a different baseline for each model after observing the results.\n"
    (generated / "findings.tex").write_text(findings)
    gap_rows = []
    for m, label in zip(MODELS, LABELS):
        entries = []
        for feature in ("raw_jacobian", "logit_lens", "affine"):
            g = next(r for r in gaps if r["model"] == m and r["baseline"] == feature)
            entries.append(
                f"{g['delta_rho']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]"
            )
        gap_rows.append([label] + entries)
    table(
        generated / "paired_gaps.tex",
        "Paired differences in Spearman correlation: differential Jacobian minus each baseline, with 95\\% relation-bootstrap intervals. Positive values favor differential Jacobian. All six baseline contrasts are retained in the accompanying CSV.",
        "tab:shared-paired-gaps",
        ["Model", "Minus raw Jacobian", "Minus logit lens", "Minus affine"],
        gap_rows,
    )
    for group, features in [
        ("lenses", LENS),
        ("controls", CONTROLS),
        ("corpus", CORPUS),
        ("beta", [k for k in NAMES if k.startswith("affine_")]),
    ]:
        for statistic in ("spearman", "pearson"):
            key = "spearman_rho" if statistic == "spearman" else "pearson_r"
            caption = (
                "Spearman $\\rho$" if statistic == "spearman" else "Pearson $r$"
            ) + " with shared-split LRE faithfulness across 25 relations per model. "
            if group == "corpus":
                caption += "Document counts come from OLMo's published training split; they are an external corpus proxy for the other models."
            elif group == "beta":
                caption += "Tuned rows select beta on the scored test rows and are exploratory."
            else:
                caption += "Each relation has equal weight; larger lens scores mean better target ranks."
            rows = [
                [latex(NAMES[f])] + [number(lookup(m, f)[key]) for m in MODELS]
                for f in features
            ]
            table(
                generated / f"{group}_{statistic}.tex",
                caption,
                f"tab:shared-{group}-{statistic}",
                ["Predictor"] + SHORT,
                rows,
            )
    for group, features in [
        ("lenses", LENS),
        ("controls", CONTROLS),
        ("corpus", CORPUS),
    ]:
        rows = [
            [latex(NAMES[f])] + [f"{lookup(m, f)['spearman_p']:.3g}" for m in MODELS]
            for f in features
        ]
        table(
            generated / f"{group}_pvalues.tex",
            "Two-sided, unadjusted Spearman $p$-values for the "
            + group
            + " comparisons (25 relations). These are exploratory tests, not multiplicity-adjusted claims.",
            f"tab:shared-{group}-pvalues",
            ["Predictor"] + SHORT,
            rows,
        )
    rows = []
    for name, label in zip(MODELS, LABELS):
        a = audits[name]
        rows.append(
            [
                label,
                str(a["spec"]["lens_layer"]),
                str(a["n_rows"]),
                str(a["n_subjects"]),
                str(a["empty_retrieval_subjects"]),
                str(a["rate_missing"]),
                str(lookup(name, "differential", "at_least_4_tests")["n"]),
                number(
                    lookup(name, "differential", "at_least_4_tests")["spearman_rho"]
                ),
            ]
        )
    table(
        generated / "coverage.tex",
        "Coverage and the sensitivity excluding relations with fewer than four test rows. Empty Wikipedia retrievals remain in the primary evaluation. Undefined Geva rows have no predictions surviving the source filter and are excluded from attribute-rate means.",
        "tab:shared-coverage",
        [
            "Model",
            "Layer",
            "Rows",
            "Subjects",
            "Empty wiki",
            "Undefined Geva",
            "$n_{\\geq4}$",
            "$\\rho_{\\geq4}$",
        ],
        rows,
    )
    rows = []
    for name, label in zip(MODELS, LABELS):
        chosen = [
            r
            for r in all_correlations
            if r["model"] == name and r["category"] == "geva_coverage_sensitivity"
        ]
        rows.append(
            [
                label,
                str(chosen[0]["n"]),
                str(sum(r["n"] for r in geva_matched if r["model"] == name)),
            ]
            + [
                number(
                    next(r["spearman_rho"] for r in chosen if r["feature"] == feature)
                )
                for feature in [
                    "differential",
                    "raw_jacobian",
                    "logit_lens",
                    "geva_attribute_rate",
                    "wikipedia_tf",
                ]
            ]
        )
    table(
        generated / "geva_coverage.tex",
        "Geva coverage sensitivity: restrict every readout and the LRE outcome to the same rows with a defined Geva rate, retaining relations with at least two such rows. Entries are Spearman correlations. Only Pythia loses rows to the Geva prediction filter.",
        "tab:shared-geva-coverage",
        [
            "Model",
            "Relations",
            "Rows",
            "Differential",
            "Raw J",
            "Logit lens",
            "Geva",
            "Wiki TF",
        ],
        rows,
    )
    rows = []
    for keys in [
        "cardinality",
        "answer_prior",
        "cardinality + answer_prior",
        "object_frequency",
        "cardinality + answer_prior + object_frequency",
    ]:
        rows.append(
            [latex(keys)]
            + [
                number(
                    next(
                        r["rho"]
                        for r in sensitivity
                        if r["model"] == m and r.get("controls") == keys
                    )
                )
                for m in MODELS
            ]
        )
    rows.append(
        ["Intercept-adjusted outcome"]
        + [
            number(
                next(
                    r["rho"]
                    for r in sensitivity
                    if r["model"] == m and r["analysis"] == "intercept_adjusted_outcome"
                )
            )
            for m in MODELS
        ]
    )
    table(
        generated / "sensitivity.tex",
        "Sensitivity of the differential-Jacobian association. Partial Spearman residualizes ranked variables on ranked controls; the last row changes the outcome to intercept-adjusted agreement.",
        "tab:shared-sensitivity",
        ["Control / outcome"] + SHORT,
        rows,
    )
    rows = []
    for name, label in zip(MODELS, LABELS):
        domain = [
            r["rho"]
            for r in robustness
            if r["model"] == name and r["analysis"] == "leave_one_domain_out"
        ]
        perm = next(
            r
            for r in robustness
            if r["model"] == name and r["analysis"] == "permutation"
        )
        correct = next(
            r
            for r in robustness
            if r["model"] == name
            and r["analysis"] == "model_correct_outcome_at_least_4"
        )
        rows.append(
            [
                label,
                f"{min(domain):.3f}--{max(domain):.3f}",
                f"{np.median(domain):.3f}",
                f"{perm['p']:.4g}",
                number(correct["rho"]) + f" ({correct['n']})",
            ]
        )
    table(
        generated / "robustness.tex",
        "Differential-Jacobian robustness: leave-one-domain-out Spearman range and median, two-sided permutation $p$ (10,000 draws, plus-one correction), and correlation after restricting the LRE outcome to at least four model-correct test rows per relation. The last column retains the original full-test lens score; relation counts are in parentheses.",
        "tab:shared-robustness",
        ["Model", "Domain range", "Median", "Permutation $p$", "Correct-only outcome"],
        rows,
    )
    rows = []
    for scope in corpus_sets:
        for subset in ("all", "no_lexical_overlap"):
            chosen = [
                lookup(
                    "olmo3_7b",
                    f,
                    category="corpus_sensitivity",
                    scope=scope,
                    subset=subset,
                )
                for f in [
                    "corpus_log1p_cooccurrence",
                    "corpus_object_count",
                    "corpus_p_object_given_subject",
                    "corpus_p_subject_given_object",
                    "corpus_direction_difference",
                ]
            ]
            short = (
                scope.replace("dolma3_", "")
                .replace("-1025-7B", "")
                .replace("-1025", "")
                .replace("all_stages", "All stages")
            )
            rows.append(
                [latex(short) + (" (no overlap)" if subset != "all" else "")]
                + [number(r["spearman_rho"]) + f" ({r['n']})" for r in chosen]
            )
    table(
        generated / "corpus_stages.tex",
        "OLMo-only Spearman sensitivities by corpus stage and literal-overlap exclusion; relation counts in parentheses. All features share support where both marginals are observed.",
        "tab:shared-corpus-stages",
        [
            "Corpus subset",
            "Log joint",
            "Object freq.",
            "$P(o|s)$",
            "$P(s|o)$",
            "Difference",
        ],
        rows,
    )
    rows = []
    for name in ["olmo3_7b", "gptj"]:
        for subset in ("all", "no_name_leakage"):
            cs = [
                r
                for r in all_correlations
                if r["category"] == "prominence_diagnostic"
                and r["model"] == name
                and r["subset"] == subset
            ]
            rows.append(
                [
                    LABELS[MODELS.index(name)],
                    "All" if subset == "all" else "No name leakage",
                    str(cs[0]["n"]),
                ]
                + [number(r["spearman_rho"]) + f" ({r['spearman_p']:.3g})" for r in cs]
            )
    table(
        generated / "prominence.tex",
        "Fresh free-generation prominence diagnostic on identical generation/lens rows. Entries show Spearman $\\rho$ ($p$); LRE faithfulness is recomputed on those rows with the original selected LRE settings.",
        "tab:shared-prominence",
        [
            "Model",
            "Subset",
            "Relations",
            "vs. LRE faithfulness",
            "vs. gold in lens top 5",
        ],
        rows,
    )
    for name, label in zip(MODELS, LABELS):
        rows = []
        for r in [r for r in data if r["model"] == name]:
            rows.append(
                [latex(r["relation"]), str(r["n"]), f"{r['faithfulness']:.3f}"]
                + [
                    f"{r[f]:.3f}" if r[f] is not None else "--"
                    for f in [
                        "differential",
                        "raw_jacobian",
                        "logit_lens",
                        "differential_logit_lens",
                        "affine",
                        "geva_attribute_rate",
                        "wikipedia_tf",
                    ]
                ]
            )
        table(
            generated / f"relations_{name}.tex",
            label
            + ": all shared-split relation measurements. Lens columns are negative median log-ranks; Geva is mean attribute overlap in percent. D: differential Jacobian; J: raw Jacobian; LL: logit lens; DLL: differential logit lens; A: affine (beta=1); W: Wikipedia token-frequency rank.",
            f"tab:shared-relations-{name}",
            ["Relation", "$n$", "Faith", "D", "J", "LL", "DLL", "A", "Geva", "W"],
            rows,
        )
    lines = [
        "# Current shared-split baseline appendix",
        "",
        "All seven models, exact neutral template `\\n{} is known for`, pinned revisions, fp32/eager, 25 relations each. Every statistic below is recomputed from retained current rows. Spearman is primary; Pearson, unadjusted p-values, and exact sample counts are also retained.",
        "",
        "[Paper appendix](../../../paper/appendix_shared_baselines.tex) · [Standalone PDF](../../../paper/shared_baselines.pdf) · [All correlations](correlations.csv) · [Relation measurements](relation_scores.csv) · [Subject measurements](subject_scores.csv) · [Source hashes and validation](summary.json)",
        "",
    ]
    for title, features in [
        ("Lens and Wikipedia baselines", LENS),
        ("Prior and cardinality controls", CONTROLS),
        ("Corpus frequency and direction", CORPUS),
        ("Affine beta sweep", [k for k in NAMES if k.startswith("affine_")]),
    ]:
        lines += [
            "## " + title,
            "",
            "Spearman ρ across 25 relations; columns follow the same model order.",
            "",
            "| Predictor | " + " | ".join(SHORT) + " |",
            "|---|" + "---:|" * 7,
        ]
        for f in features:
            lines.append(
                "| "
                + NAMES[f].replace("|", r"\|")
                + " | "
                + " | ".join(number(lookup(m, f)["spearman_rho"]) for m in MODELS)
                + " |"
            )
        lines.append("")
    lines += [
        "## Interpretation and scope",
        "",
        "Pythia has 413 rows with undefined Geva rates because no top-50 prediction survives the filter. They are excluded from its attribute-rate means. The appendix and [Geva-matched measurements](geva_matched_relation_scores.csv) additionally restrict every readout and the LRE outcome to the same Geva-defined rows. The other models have no undefined attribute rates.",
        "",
        "Raw Jacobian, affine Jacobian, and differential Jacobian are distinct readouts. Geva's attribute rate measures subject-attribute overlap, while Wikipedia TF is our separate answer-token ranking extension. OLMo document counts measure its published training multiset; for six other models they are a common external corpus proxy, not their training frequencies. Tuned affine beta and selected LRE settings use the scored data and are not held-out estimates. P-values are exploratory, two-sided and unadjusted.",
        "",
        "The two-model prominence diagnostic was rerun in fp32 with newline-prefixed prose prompts, 90 generated tokens per prompt and up to 40 test rows per relation. It is kept separate from the 25-relation primary table. [Matched prominence rows](prominence.csv).",
        "",
        "[Paired differences and 95% relation-bootstrap intervals](paired_gaps.csv), [partial/outcome controls](sensitivity.csv), and [all corpus stages and overlap sensitivities](corpus_relation_scores.csv) are included. Bootstrap intervals describe relation resampling, not uncertainty over models or training runs.",
        "",
        "See [curation and protocol notes](../../docs/shared-baseline-appendix.md) for definitions, historical exclusions, and reproduction commands.",
    ]
    (DEST / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                models=len(MODELS),
                relations=len(data),
                subject_rows=len(row_audit),
                correlations=len(all_correlations),
                sources=len(provenance),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
