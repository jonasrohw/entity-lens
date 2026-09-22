"""Audit comparability, reproduce OLMo, and separate predictor/outcome differences."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "generated/cross_model_audit"
DEST = BASE / "results/cross_model_audit"
LABELS = {
    "olmo3_7b": "OLMo 3 7B",
    "gptj": "GPT-J 6B",
    "mistral7b": "Mistral 7B",
    "gemma2_9b": "Gemma 2 9B",
    "qwen3_8b_base": "Qwen3 8B Base",
    "qwen3_8_27b": "Qwen3.8 27B",
    "pythia6_9b": "Pythia 6.9B",
}


def read(path):
    return json.loads(path.read_text())


def corr(x, y):
    result = spearmanr(x, y)
    return dict(n=len(x), rho=float(result.statistic), p=float(result.pvalue))


def records():
    models = {}
    for name in LABELS:
        models[name] = {}
        for protocol in ["main", "shared_split"]:
            rows = {}
            if name in {"olmo3_7b", "gptj"}:
                root = (
                    BASE
                    / "results"
                    / ("olmo" if name == "olmo3_7b" else "gptj")
                    / "results"
                )
                lens = read(
                    root
                    / (
                        "jacobian_lens.json"
                        if protocol == "main"
                        else "shared_split/jacobian_lens.json"
                    )
                )
                if protocol == "main":
                    lens.update(read(root / "jacobian_lens_years.json"))
                for relation, values in lens.items():
                    suffix = (
                        "_ts"
                        if protocol == "main" and relation.startswith("presidents_")
                        else ""
                    )
                    faith = read(
                        root
                        / ("main" if protocol == "main" else "shared_split")
                        / f"faithfulness_{relation}{suffix}.json"
                    )
                    n = (
                        min(
                            100,
                            len(
                                read(BASE / "data/relations" / f"{relation}.json")[
                                    "samples"
                                ]
                            )
                            - 16,
                        )
                        if protocol == "main"
                        else len(values["rows"])
                    )
                    rows[relation] = dict(lens=values, faithfulness=faith, n_test=n)
            else:
                for path in (BASE / "results/five_model_sweep" / name / protocol).glob(
                    "*.json"
                ):
                    row = read(path)
                    if "faithfulness" in row and "agg" in row.get("lens", {}):
                        row["n_test"] = row["faithfulness"]["n_test"]
                        rows[path.stem] = row
            for relation, row in rows.items():
                row["x"] = -row["lens"]["agg"]["diff"]["median_lograng"]
                row["y"] = row["faithfulness"]["best_faith"]
            models[name][protocol] = rows
    return models


def bootstrap_gap(a, b, *, repetitions=10000):
    """Paired relation bootstrap with ties reranked within each resample."""
    rng = np.random.default_rng(20260919)
    ids = rng.integers(0, len(a[0]), size=(repetitions, len(a[0])))
    values = []
    for x, y in [a, b]:
        x = rankdata(np.asarray(x)[ids], axis=1)
        y = rankdata(np.asarray(y)[ids], axis=1)
        x -= x.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        values.append((x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1)))
    delta = values[0] - values[1]
    assert np.isfinite(delta).all()
    return dict(
        delta=corr(*a)["rho"] - corr(*b)["rho"],
        ci95=np.quantile(delta, [0.025, 0.975]).tolist(),
        n=len(a[0]),
        repetitions=repetitions,
    )


def reproduce(models):
    result = dict(
        faith_cells=0, ranks=0, faith_differences=[], rank_differences=[], protocols={}
    )
    for protocol in ["main", "shared_split"]:
        paths = sorted((RUN / "results/olmo3_7b" / protocol).glob("*.json"))
        result["protocols"][protocol] = dict(finished=len(paths), expected=28)
        for path in paths:
            new = read(path)
            if path.stem not in models["olmo3_7b"][protocol]:
                if "faithfulness" not in new:
                    continue
                # Also check the one-subject shared relation with no usable lens.
                old = dict(
                    faithfulness=read(
                        BASE
                        / "results/olmo/results"
                        / protocol
                        / f"faithfulness_{path.stem}.json"
                    )
                )
            else:
                old = models["olmo3_7b"][protocol][path.stem]
            for layer, cells in old["faithfulness"]["grid"].items():
                for beta, value in cells.items():
                    result["faith_cells"] += 1
                    actual = new["faithfulness"]["grid"][layer][beta]
                    if actual != value:
                        result["faith_differences"].append(
                            [protocol, path.stem, layer, beta, value, actual]
                        )
            for key in [
                "split_fingerprint",
                "lm_correct",
                "per_subject",
                "best_layer",
                "best_beta",
                "best_faith",
            ]:
                assert new["faithfulness"][key] == old["faithfulness"][key], (
                    protocol,
                    path.stem,
                    key,
                )
            if "lens" not in old:
                continue
            for a, b in zip(old["lens"]["rows"], new["lens"]["rows"], strict=True):
                assert (
                    a["subject"] == b["subject"] and a["model_top1"] == b["model_top1"]
                )
                for kind in ["diff", "lre"]:
                    result["ranks"] += 1
                    delta = b["rank_" + kind] - a["rank_" + kind]
                    if delta:
                        result["rank_differences"].append(
                            [protocol, path.stem, a["subject"], kind, delta]
                        )
        included = [read(p) for p in paths if p.stem in models["olmo3_7b"][protocol]]
        if len(paths) == 28:
            result["protocols"][protocol]["correlation"] = corr(
                [r["lens"]["agg"]["diff"]["median_lograng"] * -1 for r in included],
                [r["faithfulness"]["best_faith"] for r in included],
            )
    assert not result["faith_differences"]
    return result


def analyze(models):
    continuation = (
        read(RUN / "continuation_tokens.json")
        if (RUN / "continuation_tokens.json").exists()
        else {}
    )
    output = dict(
        comparison={}, cross_axis={}, paired_gaps={}, target_controls={}, fresh_split={}
    )
    for name, protocols in models.items():
        output["comparison"][name] = {}
        output["paired_gaps"][name] = {}
        for label, protocol, minimum in [
            ("main", "main", 1),
            ("main_at_least_four", "main", 4),
            ("shared_split", "shared_split", 1),
        ]:
            keys = sorted(
                r for r, v in protocols[protocol].items() if v["n_test"] >= minimum
            )
            other = protocols[protocol]
            olmo = models["olmo3_7b"][protocol]
            assert set(keys) <= olmo.keys()
            xy = ([other[r]["x"] for r in keys], [other[r]["y"] for r in keys])
            output["comparison"][name][label] = corr(*xy)
            if name != "olmo3_7b":
                output["paired_gaps"][name][label] = bootstrap_gap(
                    ([olmo[r]["x"] for r in keys], [olmo[r]["y"] for r in keys]), xy
                )
        keys = sorted(protocols["main"])
        row = protocols["main"]
        olmo = models["olmo3_7b"]["main"]
        x, y, ox, oy = [
            [data[r][key] for r in keys]
            for data, key in [(row, "x"), (row, "y"), (olmo, "x"), (olmo, "y")]
        ]
        output["cross_axis"][name] = dict(
            lens_agreement=corr(x, ox),
            faith_agreement=corr(y, oy),
            model_lens_olmo_faith=corr(x, oy),
            olmo_lens_model_faith=corr(ox, y),
        )
        output["target_controls"][name] = {}
        for protocol, rows in protocols.items():
            target_root = RUN / "targets" / name / protocol
            if len(list(target_root.glob("*.json"))) != 28:
                continue
            targets = {r: read(target_root / f"{r}.json") for r in rows}
            label_fixes = []
            if name in continuation:
                for relation, target in targets.items():
                    labels = [
                        r
                        for r in continuation[name]["rows"]
                        if r["relation"] == relation and r["protocol"] == protocol
                    ]
                    assert [r["subject"] for r in labels] == target["subjects"]
                    corrected = [
                        int(token == label["continuation_id"])
                        if label["prefix_stable"]
                        else old
                        for token, label, old in zip(
                            target["targets"], labels, target["correct"], strict=True
                        )
                    ]
                    if corrected != target["correct"]:
                        label_fixes.append(
                            dict(
                                relation=relation,
                                before=sum(target["correct"]),
                                after=sum(corrected),
                                n=target["n"],
                            )
                        )
                    target["correct"] = corrected
                    target["lm_accuracy"] = float(np.mean(corrected))
                    target["gold_correct_subject_faith"] = (
                        float(
                            np.mean(
                                np.asarray(target["selected_lre_hits"])[
                                    np.asarray(corrected, dtype=bool)
                                ]
                            )
                        )
                        if any(corrected)
                        else None
                    )
            keys = [r for r in sorted(rows) if targets[r]["prior_faith"] < 1]
            x = [rows[r]["x"] for r in keys]
            y = [
                (rows[r]["y"] - targets[r]["prior_faith"])
                / (1 - targets[r]["prior_faith"])
                for r in keys
            ]
            correct = [r for r in sorted(rows) if sum(targets[r]["correct"]) >= 4]
            output["target_controls"][name][protocol] = dict(
                context_gold_label_corrections=label_fixes,
                normalized_excess=corr(x, y),
                excluded_zero_headroom=sorted(set(rows) - set(keys)),
                correct_subject_faith=corr(
                    [rows[r]["x"] for r in correct],
                    [targets[r]["gold_correct_subject_faith"] for r in correct],
                ),
                correct_subject_relations=correct,
                mean_model_accuracy=float(
                    np.mean([r["lm_accuracy"] for r in targets.values()])
                ),
                mean_prior_faith=float(
                    np.mean([r["prior_faith"] for r in targets.values()])
                ),
                constant_target_relations={
                    r: dict(
                        token=t["target_tokens"][0],
                        n=t["n"],
                        faith=t["selected_faith"],
                        prior=t["prior_faith"],
                    )
                    for r, t in targets.items()
                    if t["target_unique"] == 1
                },
            )
    for protocol in ["main", "shared_split"]:
        paths = list(
            (RUN / f"olmo_confirmation_{protocol}" / "confirmation").glob("*.json")
        )
        if len(paths) == 28:
            rows = [read(p) for p in paths if "faith" in read(p)]
            output["fresh_split"][protocol] = corr(
                [-r["lens"]["baseline"]["result"]["median_logrank"] for r in rows],
                [r["faith"]["coarse"]["score"] for r in rows],
            )
    output["olmo_reproduction"] = reproduce(models)
    return output


def report(output):
    lines = [
        "# OLMo versus other models: implementation and measurement audit",
        "",
        "OLMo reproduces exactly in the common runner and remains strong on a fresh split. Its apparent advantage is much smaller under the shared protocol or after excluding three very small relations. A concrete Mistral gold-token labeling bug was fixed, but it does not change the model-top-1 LRE correlation. Tokenizer-dependent first-token information remains a substantive limitation of the raw metric.",
        "",
        "The original main and shared protocols must be compared separately. The table uses original fixed lens settings and each model's independently fitted LRE. The at-least-four column changes the main cohort without changing any model settings.",
        "",
        "| Model | Main, 28 relations | Main, at least four tests (25) | Shared, 25 relations |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in output["comparison"].items():
        lines.append(
            f"| {LABELS[name]} | {row['main']['rho']:.3f} | {row['main_at_least_four']['rho']:.3f} | {row['shared_split']['rho']:.3f} |"
        )
    lines += [
        "",
        "Qwen3.8-27B is post-trained; the other new checkpoints are base models. GPT-J main preserves the saved trailing-space year prompt variant. Shared filtering uses each tokenizer and therefore can select different subjects across models. Original data retain duplicate subjects; the separate Mistral cleaning sensitivity is not substituted into this table.",
        "",
        "![Protocol comparison](protocol_comparison.png)",
        "",
        "## First-token information is not equivalent",
        "",
        "These are the actual next-token predictions for the same three main birth-year test subjects. Intercept-only agreement uses the averaged reference output with zero Jacobian contribution. A perfect score on the token `1` does not establish recovery of a full year.",
        "",
        "| Model | Three predicted tokens | Intercept-only agreement | Selected LRE agreement |",
        "| --- | --- | --- | --- |",
    ]
    for name in ["olmo3_7b", "mistral7b", "gemma2_9b", "qwen3_8b_base", "qwen3_8_27b"]:
        path = RUN / "targets" / name / "main/presidents_birth_year.json"
        if path.exists():
            row = read(path)
            tokens = ", ".join(f"`{token}`" for token in row["target_tokens"])
            lines.append(
                f"| {LABELS[name]} | {tokens} | {row['prior_faith']:.3f} | {row['selected_faith']:.3f} |"
            )
    lines += [
        "",
        "## Predictor versus outcome",
        "",
        "These crossed correlations are diagnostic only: they pair a lens from one model with faithfulness from another, so they are not performance of an actual model/LRE system.",
        "",
        "| Model | Lens ranking vs OLMo lens | Model lens / OLMo faith | OLMo lens / model faith |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in output["cross_axis"].items():
        if name != "olmo3_7b":
            lines.append(
                f"| {LABELS[name]} | {row['lens_agreement']['rho']:.3f} | {row['model_lens_olmo_faith']['rho']:.3f} | {row['olmo_lens_model_faith']['rho']:.3f} |"
            )
    lines += [
        "",
        "![Axis substitution diagnostic](axis_substitution.png)",
        "",
        "## Paired uncertainty",
        "",
        "The following 95% percentile intervals resample the same relations for both models (10,000 draws, seed 20260919), reranking ties within each draw. They describe the observed dataset and fixed fits, not training or subject-sampling uncertainty. They are exploratory and not adjusted for all model comparisons.",
        "",
        "| OLMo minus model | Main gap [95% interval] | Main 25 gap [95% interval] | Shared gap [95% interval] |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in output["paired_gaps"].items():
        if row:
            values = [
                f"{row[k]['delta']:.3f} [{row[k]['ci95'][0]:.3f}, {row[k]['ci95'][1]:.3f}]"
                for k in ["main", "main_at_least_four", "shared_split"]
            ]
            lines.append(f"| {LABELS[name]} | {' | '.join(values)} |")
    lines += [
        "",
        "## Outcome controls",
        "",
        "Prior-normalized agreement is `(faith - intercept_only_faith)/(1 - intercept_only_faith)`. It changes the outcome definition and excludes relations with no headroom. The model-correct subset diagnostic keeps at least four correctly predicted test subjects per relation, retains the original fitted LRE/settings, and leaves the original aggregate lens score unchanged. Both are sensitivity analyses, not replacements for the primary result.",
        "",
        "| Model | Main normalized rho (n) | Shared normalized rho (n) | Mean main intercept-only faith |",
        "| --- | --- | --- | --- |",
    ]
    for name, row in output["target_controls"].items():
        if set(row) == {"main", "shared_split"}:
            a, b = (
                row["main"]["normalized_excess"],
                row["shared_split"]["normalized_excess"],
            )
            lines.append(
                f"| {LABELS[name]} | {a['rho']:.3f} ({a['n']}) | {b['rho']:.3f} ({b['n']}) | {row['main']['mean_prior_faith']:.3f} |"
            )
    reproduction = output["olmo_reproduction"]
    lines += [
        "",
        "## Fresh numerical reproduction",
        "",
        f"The common runner checked {reproduction['faith_cells']} OLMo grid cells with {len(reproduction['faith_differences'])} faithfulness differences and {reproduction['ranks']} individual ranks. Rank changes are recorded in the JSON rather than discarded. Completion and fresh correlations: `{json.dumps(reproduction['protocols'])}`.",
        "",
        "Frozen-setting seed-1 operator refits: `"
        + json.dumps(output["fresh_split"])
        + "`. These reuse the same dataset and have overlapping subjects; no hyperparameters are selected on seed 1.",
        "",
        "[Full analysis](analysis.json), [audit scope and interpretation](../../docs/cross-model-audit.md). The raw OLMo rerun, target/prior audits, and fresh fits are retained beside this report. Numerical gates compare HF logits, no-op interventions, scalar/batched products, finite differences, and cached/full-prompt derivatives.",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(lines))


def figures(models, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(LABELS))
    for offset, key, label, color in [
        (-0.20, "main", "Main: all 28 relations", "#2878b5"),
        (
            0,
            "main_at_least_four",
            "Main: at least four tests (25 relations)",
            "#d07828",
        ),
        (0.20, "shared_split", "Shared: 25 eligible relations", "#298465"),
    ]:
        x = [output["comparison"][name][key]["rho"] for name in LABELS]
        ax.scatter(x, y + offset, label=label, color=color, s=42)
    ax.set_yticks(y, list(LABELS.values()))
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 0.9)
    ax.set_xlabel(
        "Spearman correlation: original differential lens vs LRE faithfulness"
    )
    ax.axvline(0, color="gray", lw=0.7)
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Protocol and cohort explain much of the apparent OLMo gap")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(DEST / f"protocol_comparison.{extension}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    names = ["olmo3_7b", "mistral7b"]
    relations = sorted(models[names[0]]["main"])
    small = np.array([models[names[0]]["main"][r]["n_test"] < 4 for r in relations])
    for i, yn in enumerate(names):
        for j, xn in enumerate(names):
            ax = axes[i, j]
            x = np.array([models[xn]["main"][r]["x"] for r in relations])
            y = np.array([models[yn]["main"][r]["y"] for r in relations])
            ax.scatter(x[~small], y[~small], s=28, color="#2878b5")
            ax.scatter(
                x[small],
                y[small],
                s=50,
                marker="x",
                color="#c66625",
                label="Three-test relations",
            )
            ax.set_title(f"rho={corr(x, y)['rho']:.3f}")
            ax.grid(alpha=0.15)
            if j == 0:
                ax.set_ylabel(f"{LABELS[yn]} LRE faithfulness")
            if i == 1:
                ax.set_xlabel(f"{LABELS[xn]} lens score")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Axis substitution diagnostic: all 28 main relations\nOff-diagonal panels combine two models; they are not model evaluations",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for extension in ["png", "pdf"]:
        fig.savefig(DEST / f"axis_substitution.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retain", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    models = records()
    output = analyze(models)
    (DEST / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    report(output)
    figures(models, output)
    if args.retain:
        (DEST / "provenance").mkdir(exist_ok=True)
        if (RUN / "tests.log").exists():
            shutil.copy2(RUN / "tests.log", DEST / "provenance/tests.txt")
        for folder in [
            "results",
            "targets",
            "olmo_confirmation_main",
            "olmo_confirmation_shared_split",
        ]:
            for path in (RUN / folder).rglob("*.json"):
                target = DEST / "raw" / path.relative_to(RUN)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for path in RUN.glob("*.json"):
            target = DEST / "provenance" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    print(
        json.dumps(
            dict(
                reproduction=output["olmo_reproduction"]["protocols"],
                fresh_split=output["fresh_split"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
