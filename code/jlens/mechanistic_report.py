"""CPU-only paired summaries and figures for the mechanistic measurements."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def attach_probe_diagnostics(data, directory):
    """Post-run diagnostic; train centroids on training subjects only.

    Score only held-out values with replicated training examples. This does
    not select directions, layers, or interventions, and changes no GPU result.
    """
    import torch

    for item in data:
        tensors = torch.load(
            directory / f"{item['relation']}.pt", map_location="cpu", weights_only=True
        )
        item["additional_geometry"] = {}
        for layer, arrays in tensors.items():
            ar = (
                (arrays["relation_affine"] - arrays["relation_base"])
                .numpy()
                .astype(float)
            )
            an = (
                (arrays["neutral_affine"] - arrays["neutral_base"])
                .numpy()
                .astype(float)
            )

            def cosine(a, b):
                return (a * b).sum(-1) / np.maximum(
                    np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-16
                )

            item["additional_geometry"][str(layer)] = {
                "taylor_differential_cosine": float(
                    cosine(ar, an)[item["nval"] :].mean()
                ),
                "taylor_differential_shuffled_cosine": float(
                    cosine(ar, np.roll(an, 1, axis=0))[item["nval"] :].mean()
                ),
            }
        train_labels = np.array([x["object"] for x in item["plan"]["parts"]["train"]])
        test_labels = np.array([x["object"] for x in item["plan"]["parts"]["test"]])
        counts = Counter(train_labels)
        classes = np.array(sorted(label for label, n in counts.items() if n >= 2))
        keep = np.isin(test_labels, classes)
        item["probe_diagnostics"] = {}
        if len(classes) < 2 or not keep.any():
            continue
        majority = max(classes, key=lambda label: counts[label])
        for layer, arrays in tensors.items():
            q = arrays["attribute_basis"].numpy().astype(float)
            if not q.shape[1]:
                continue
            result = {
                "probe_coverage": float(keep.mean()),
                "probe_majority": float(np.mean(test_labels[keep] == majority)),
            }
            for kind in ["neutral", "relation"]:
                train = arrays[f"train_{kind}"].numpy().astype(float)
                test = arrays[f"subject_{kind}"][item["nval"] :].numpy().astype(float)
                # Same training mean for fit and prediction; never fit on test.
                center = train.mean(0)
                ztrain, ztest = (train - center) @ q, (test - center) @ q
                centroids = np.stack(
                    [ztrain[train_labels == label].mean(0) for label in classes]
                )
                pred = classes[
                    ((ztest[:, None] - centroids[None]) ** 2).sum(-1).argmin(-1)
                ]
                result[f"probe_{kind}"] = float(
                    np.mean(pred[keep] == test_labels[keep])
                )
                result[f"probe_{kind}_gain"] = (
                    result[f"probe_{kind}"] - result["probe_majority"]
                )
            item["probe_diagnostics"][str(layer)] = result


def validate_result(data, plan, layers):
    """Reject incomplete/mismatched scientific records before aggregation."""
    expected = plan["parts"]["validation"] + plan["parts"]["test"]
    assert data["subjects"] == expected, data["relation"]
    assert data["nval"] == len(plan["parts"]["validation"])
    assert set(data["layers"]) == {str(layer) for layer in layers}
    subjects = [x["subject"] for part in plan["parts"].values() for x in part]
    assert len(subjects) == len(set(subjects))
    for layer in data["layers"].values():
        ntest = len(expected) - data["nval"]
        for name in ["clean", "restored", "attention_rescue", "mlp_rescue"]:
            assert layer["full_model"][name]["hit"] == [1] * ntest, (
                data["relation"],
                name,
            )
        for grid_name in [
            "relation_grids",
            "neutral_grids",
            "shared_grids",
            "calibrated_relation_grids",
        ]:
            for scores in layer[grid_name].values():
                assert all(len(values) == len(expected) for values in scores.values())
        assert (
            layer["shared_grids"]["0.0"]["top"]
            == layer["calibrated_relation_grids"]["0.0"]["top"]
        )
        assert (
            layer["comparison"]["relation_beta1"]["top"]
            == layer["relation_grids"]["1.0"]["top"]
        )
        if not layer["attribute_rank"]:
            baseline = layer["relation_grids"][str(layer["beta"])]["top"]
            assert layer["erasure_lre"]["targeted"]["top"] == baseline
            for ri in range(3):
                assert layer["erasure_lre"][f"random_{ri}"]["top"] == baseline


def relation_rows(data, layer):
    """Reduce each relation to one paired observation; no test-set selection."""
    rows = []
    for d in data:
        nv = d["nval"]
        if layer == "selected":
            chosen = max(
                d["layers"],
                key=lambda layer_id: (
                    np.mean(
                        d["layers"][layer_id]["relation_grids"][
                            str(d["layers"][layer_id]["beta"])
                        ]["hit"][:nv]
                    ),
                    np.mean(
                        d["layers"][layer_id]["relation_grids"][
                            str(d["layers"][layer_id]["beta"])
                        ]["logp"][:nv]
                    ),
                    -int(layer_id),
                ),
            )
        else:
            chosen = str(layer)
        if chosen not in d["layers"]:
            continue
        x = d["layers"][chosen]
        c, full, erasure, ctrl = (
            x["comparison"],
            x["full_model"],
            x["erasure_lre"],
            x["control_model"],
        )

        def avg(values, split=True):
            return float(np.mean(values[nv:] if split else values))

        def grid(name, beta_key, metric="hit"):
            return avg(x[name][str(x[beta_key])][metric])

        r = {
            "relation": d["relation"],
            "layer": int(chosen),
            "n_test": len(d["subjects"]) - nv,
            "attribute_rank": x["attribute_rank"],
            "beta": x["beta"],
            "shared_beta": x["shared_beta"],
            "model_correct": avg(d["model_correct"]),
            "faith": grid("relation_grids", "beta"),
            "original_bias_faith": avg(x["relation_grids"]["0.0"]["hit"]),
            "uses_subject_term": float(x["beta"] > 0),
            "neutral_faith": grid("neutral_grids", "neutral_beta"),
            "calibrated_faith": grid("calibrated_relation_grids", "calibrated_beta"),
            "shared_faith": grid("shared_grids", "shared_beta"),
            "matched_neutral_faith": grid(
                "matched_neutral_grids", "matched_neutral_beta"
            ),
            "bias_faith": avg(x["shared_grids"]["0.0"]["hit"]),
            "transplant_lre": avg(x["transplant_lre"]["hit"]),
            "transplant_full": avg(full["neutral_transplant"]["hit"], False),
            "transplant_logp_change": avg(
                np.array(full["neutral_transplant"]["logp"]) - full["clean"]["logp"],
                False,
            ),
            "original_beta1_faith": avg(c["relation_beta1"]["hit"]),
            "neutral_beta1_faith": avg(c["neutral_native_beta1"]["hit"]),
            "map_token_agreement": avg(
                (
                    np.array(c["relation_beta1"]["top"])
                    == c["neutral_native_beta1"]["top"]
                ).astype(float)
            ),
            "raw_cosine": avg(c["affine_cosine_beta1"]),
            "raw_shuffled_cosine": avg(c["affine_shuffled_cosine_beta1"]),
            "native_centered_cosine": avg(c["centered_native_cosine"]),
            "same_input_cosine": avg(c["centered_same_input_cosine"]),
            "centered_shuffled_cosine": avg(c["centered_shuffled_cosine"]),
            "operator_relative_error": avg(c["operator_relative_error"]),
            "neutral_rank_score": -avg(
                np.log10(
                    np.array(x["neutral_grids"][str(x["neutral_beta"])]["rank"]) + 1
                )
            ),
            "restoration_max_error": x["restoration_max_error"],
            "clean_full": avg(full["clean"]["hit"], False),
            "restored_full": avg(full["restored"]["hit"], False),
        }
        r["transplant_lre_change"] = r["transplant_lre"] - r["faith"]
        r["subject_term_gain"] = r["faith"] - r["original_bias_faith"]
        r.update(d.get("probe_diagnostics", {}).get(chosen, {}))
        r.update(d.get("additional_geometry", {}).get(chosen, {}))
        if "differential" in d:
            diff = d["differential"]["layers"][chosen]
            r["differential_faith"] = avg(diff["hit"])
            r["differential_log_rank"] = avg(np.log10(np.array(diff["rank"]) + 1))
        if "neutral_wrong_subject" in full:
            r["wrong_subject_full"] = avg(full["neutral_wrong_subject"]["hit"], False)
            r["neutral_subject_full_advantage"] = (
                r["transplant_full"] - r["wrong_subject_full"]
            )
            wrong_top = np.roll(np.array(x["transplant_lre"]["top"][nv:]), -1)
            r["wrong_subject_lre"] = float(np.mean(wrong_top == d["targets"][nv:]))
            r["neutral_subject_lre_advantage"] = (
                r["transplant_lre"] - r["wrong_subject_lre"]
            )
            target_ids = np.array(d["targets"][nv:])
            different = target_ids != np.roll(target_ids, -1)
            if different.any():
                correct_donor = np.array(full["neutral_transplant"]["hit"])
                wrong_donor = np.array(full["neutral_wrong_subject"]["hit"])
                r["neutral_subject_advantage_different_answer"] = float(
                    np.mean((correct_donor - wrong_donor)[different])
                )
        r["shared_change"] = r["shared_faith"] - r["calibrated_faith"]
        r["shared_gain_over_bias"] = r["shared_faith"] - r["bias_faith"]
        if "gain_control" in d:
            gains = d["gain_control"]["layers"][chosen]
            for kind, label in [
                ("relation", "calibrated"),
                ("shared", "shared"),
                ("matched_neutral", "matched_neutral"),
            ]:
                r[f"wide_{label}_faith"] = avg(gains[kind]["scores"]["hit"])
                r[f"wide_{label}_beta"] = gains[kind]["beta"]
            r["wide_shared_change"] = (
                r["wide_shared_faith"] - r["wide_calibrated_faith"]
            )
            r["wide_shared_gain_over_bias"] = r["wide_shared_faith"] - r["bias_faith"]
            r["wide_shared_above_five"] = float(gains["shared"]["beta"] > 5)
            r["shared_norm_ratio"] = gains["shared"]["mean_product_norm"] / max(
                gains["relation"]["mean_product_norm"], 1e-12
            )
        r["raw_cosine_subject_advantage"] = r["raw_cosine"] - r["raw_shuffled_cosine"]
        r["centered_cosine_subject_advantage"] = (
            r["native_centered_cosine"] - r["centered_shuffled_cosine"]
        )
        for kind in ["attention", "mlp"]:
            r[kind + "_full"] = avg(full[kind]["hit"], False)
            r[kind + "_rescue"] = avg(full[kind + "_rescue"]["hit"], False)
            r[kind + "_logp_drop"] = avg(
                np.array(full["clean"]["logp"]) - full[kind]["logp"], False
            )
            r[kind + "_lre"] = avg(x["component_lre"][kind]["hit"], False)
            r[kind + "_lre_change"] = r[kind + "_lre"] - r["faith"]
        if x["attribute_rank"] > 0:
            r["erasure_fraction"] = avg(x["erasure_norm_fraction"])
            for method, scores, baseline, cut in [
                ("lre", erasure, x["relation_grids"][str(x["beta"])], True),
                ("full", full, full["clean"], False),
                ("control", ctrl, ctrl.get("clean"), False),
            ]:
                if not scores:
                    continue
                for metric in ["hit", "logp", "margin"]:
                    target = np.array(scores["targeted"][metric])
                    rand = np.mean(
                        [scores[f"random_{i}"][metric] for i in range(3)], axis=0
                    )
                    base = np.array(baseline[metric])
                    r[f"erase_{method}_{metric}"] = avg(target, cut)
                    r[f"random_{method}_{metric}"] = avg(rand, cut)
                    r[f"erase_{method}_{metric}_drop"] = avg(base - target, cut)
                    r[f"random_{method}_{metric}_drop"] = avg(base - rand, cut)
                    r[f"selectivity_{method}_{metric}"] = avg(rand - target, cut)
            if "selectivity_control_logp" in r:
                r["attribute_specificity_logp"] = (
                    r["selectivity_full_logp"] - r["selectivity_control_logp"]
                )
        # Conditional analysis only where the original LM matches the gold token.
        correct = np.array(d["model_correct"][nv:], dtype=bool)
        r["n_model_correct"] = int(correct.sum())
        if correct.any():
            for key, values in [
                ("correct_faith", x["relation_grids"][str(x["beta"])]["hit"][nv:]),
                (
                    "correct_shared_faith",
                    x["shared_grids"][str(x["shared_beta"])]["hit"][nv:],
                ),
                (
                    "correct_calibrated_faith",
                    x["calibrated_relation_grids"][str(x["calibrated_beta"])]["hit"][
                        nv:
                    ],
                ),
                ("correct_transplant_lre", x["transplant_lre"]["hit"][nv:]),
                ("correct_transplant_full", full["neutral_transplant"]["hit"]),
            ]:
                r[key] = float(np.mean(np.array(values)[correct]))
            if x["attribute_rank"] > 0:
                rand = np.mean([full[f"random_{i}"]["logp"] for i in range(3)], axis=0)
                target = np.array(full["targeted"]["logp"])
                r["correct_selectivity_full_logp"] = float(
                    np.mean((rand - target)[correct])
                )
        rows.append(r)
    return rows


def summaries(rows):
    rng = np.random.default_rng(1909)
    result = {}
    keys = sorted(set().union(*(r.keys() for r in rows)) - {"relation"})
    for key in keys:
        selected = [r for r in rows if key in r]
        values = np.array([r[key] for r in selected], dtype=float)
        boots = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(
            axis=1
        )
        result[key] = {
            "mean": float(values.mean()),
            "ci95": np.quantile(boots, [0.025, 0.975]).tolist(),
            "n_relations": len(values),
            "n_test": sum(r["n_test"] for r in selected),
            "subject_weighted_mean": float(
                np.average(values, weights=[r["n_test"] for r in selected])
            ),
        }
    return result


def fmt(s, key, percent=False):
    if key not in s:
        return "not estimable"
    v = s[key]
    scale, suffix = (100, "%") if percent else (1, "")
    return f"{v['mean'] * scale:.3f}{suffix} [{v['ci95'][0] * scale:.3f}, {v['ci95'][1] * scale:.3f}]"


def figures(data, out):
    rows = relation_rows(data, 17)
    if not rows:
        return
    s = summaries(rows)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [
        (
            [
                "raw_cosine",
                "raw_shuffled_cosine",
                "native_centered_cosine",
                "centered_shuffled_cosine",
                "same_input_cosine",
            ],
            [
                "Affine",
                "Affine shuffled",
                "Centered",
                "Centered shuffled",
                "Same input",
            ],
            "Transformation similarity",
            "Cosine similarity",
        ),
        (
            [
                "faith",
                "transplant_lre",
                "wrong_subject_lre",
                "transplant_full",
                "wrong_subject_full",
            ],
            [
                "Original LRE",
                "Neutral → LRE",
                "Wrong subject → LRE",
                "Neutral → LM",
                "Wrong subject → LM",
            ],
            "Neutral subject transplantation",
            "Agreement with original LM",
        ),
        (
            ["calibrated_faith", "matched_neutral_faith", "shared_faith", "bias_faith"],
            ["Relation matrix", "Matched neutral", "Global neutral", "Bias only"],
            "Replace the matrix (gain ≤ 5)",
            "Agreement with original LM",
        ),
    ]
    for ax, (keys, labels, title, ylabel) in zip(axes, specs):
        means = np.array([s[k]["mean"] for k in keys])
        ci = np.array([s[k]["ci95"] for k in keys])
        ax.bar(
            np.arange(len(keys)),
            means,
            color=["#315a7d", "#5b8e8a", "#dd9860", "#919497", "#806f9e"][: len(keys)],
        )
        ax.errorbar(
            np.arange(len(keys)),
            means,
            yerr=np.maximum(0, np.array([means - ci[:, 0], ci[:, 1] - means])),
            fmt="none",
            color="black",
            capsize=3,
        )
        ax.set_xticks(np.arange(len(keys)), labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylabel.startswith("Agreement"):
            ax.set_ylim(0, 1)
    fig.suptitle(
        f"OLMo-3-7B · layer 17 · {len(rows)} relations · held-out tests", y=1.02
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            out / f"comparison_and_transfer.{ext}", dpi=180, bbox_inches="tight"
        )
    plt.close(fig)
    if all("wide_shared_faith" in row for row in rows):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        keys = [
            "calibrated_faith",
            "shared_faith",
            "wide_calibrated_faith",
            "wide_shared_faith",
            "bias_faith",
        ]
        labels = [
            "Relation, gain ≤5",
            "Global, gain ≤5",
            "Relation, wide gain",
            "Global, wide gain",
            "Bias only",
        ]
        means = np.array([s[key]["mean"] for key in keys])
        ci = np.array([s[key]["ci95"] for key in keys])
        axes[0].bar(
            range(len(keys)),
            means,
            color=["#315a7d", "#5b8e8a", "#315a7d", "#5b8e8a", "#919497"],
        )
        axes[0].errorbar(
            range(len(keys)),
            means,
            yerr=np.maximum(0, [means - ci[:, 0], ci[:, 1] - means]),
            fmt="none",
            color="black",
            capsize=3,
        )
        axes[0].set_xticks(range(len(keys)), labels, rotation=35, ha="right")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Agreement with original LM")
        axes[0].set_title("Does a wider gain range explain transfer loss?")
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].scatter(
            [r["wide_calibrated_faith"] for r in rows],
            [r["wide_shared_faith"] for r in rows],
            s=[r["n_test"] * 3 for r in rows],
            color="#5b8e8a",
            alpha=0.7,
            edgecolors="black",
            linewidth=0.4,
        )
        axes[1].set_xlabel("Relation matrix: held-out faithfulness")
        axes[1].set_ylabel("Global neutral matrix: held-out faithfulness")
        axes[1].set_title("Each point is one relation; wider gain range")
        axes[1].set_xlim(-0.03, 1.03)
        axes[1].set_ylim(-0.03, 1.03)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(out / f"gain_control.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    layers = [5, 11, 17, 23]
    by_layer = {
        layer_id: summaries(relation_rows(data, layer_id))
        for layer_id in layers
        if relation_rows(data, layer_id)
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, label in [
        ("faith", "LRE"),
        ("differential_faith", "Neutral differential"),
        ("shared_faith", "Shared matrix (calibrated)"),
    ]:
        if not all(key in stats for stats in by_layer.values()):
            continue
        axes[0].plot(
            list(by_layer),
            [s[key]["mean"] for s in by_layer.values()],
            "o-",
            label=label,
        )
        axes[0].fill_between(
            list(by_layer),
            [s[key]["ci95"][0] for s in by_layer.values()],
            [s[key]["ci95"][1] for s in by_layer.values()],
            alpha=0.12,
        )
    axes[0].set_ylabel("Agreement with original LM")
    axes[0].set_title("Readouts across layers")
    axes[0].legend(fontsize=8)
    for key, label in [
        ("mlp_logp_drop", "MLP update"),
        ("attention_logp_drop", "Attention update"),
    ]:
        axes[1].plot(
            list(by_layer),
            [s[key]["mean"] for s in by_layer.values()],
            "o-",
            label=label,
        )
        axes[1].fill_between(
            list(by_layer),
            [s[key]["ci95"][0] for s in by_layer.values()],
            [s[key]["ci95"][1] for s in by_layer.values()],
            alpha=0.12,
        )
    axes[1].set_ylabel("Original-answer log-probability drop (nats)")
    axes[1].set_title("Replace subject-token block updates")
    axes[1].legend(fontsize=8)
    for key, label in [
        ("selectivity_full_logp", "Full LM"),
        ("selectivity_lre_logp", "LRE"),
        ("selectivity_control_logp", "Alternate relation"),
    ]:
        available = [layer_id for layer_id in by_layer if key in by_layer[layer_id]]
        axes[2].plot(
            available,
            [by_layer[layer_id][key]["mean"] for layer_id in available],
            "o-",
            label=label,
        )
        axes[2].fill_between(
            available,
            [by_layer[layer_id][key]["ci95"][0] for layer_id in available],
            [by_layer[layer_id][key]["ci95"][1] for layer_id in available],
            alpha=0.12,
        )
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Targeted minus random damage (nats)")
    axes[2].set_title("Attribute erasure beyond matched noise")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Hidden-state layer")
        ax.set_xticks(layers)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"erasure_and_layers.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if "differential_log_rank" in s:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, key, label in [
            (axes[0], "faith", "LRE faithfulness"),
            (axes[1], "differential_log_rank", "Neutral differential: object rank"),
        ]:
            values = np.array([v[key]["mean"] for v in by_layer.values()])
            low = np.array([v[key]["ci95"][0] for v in by_layer.values()])
            high = np.array([v[key]["ci95"][1] for v in by_layer.values()])
            if key.endswith("log_rank"):
                values, low, high = 10**values, 10**low, 10**high
                ax.set_yscale("log")
                ax.set_ylabel("Geometric mean answer rank (lower is better)")
            else:
                ax.set_ylabel("Agreement with original LM")
                ax.set_ylim(0, 1)
            ax.plot(list(by_layer), values, "o-")
            ax.fill_between(list(by_layer), low, high, alpha=0.15)
            ax.set_xticks(list(by_layer))
            ax.set_xlabel("Hidden-state layer")
            ax.set_title(label)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(
                out / f"neutral_layer_access.{ext}", dpi=180, bbox_inches="tight"
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    names = [n for n, p in manifest["relations"].items() if "parts" in p]
    data, missing = [], []
    for name in names:
        path = args.directory / f"{name}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("complete"):
                validate_result(value, manifest["relations"][name], manifest["layers"])
                data.append(value)
                continue
        missing.append(name)
    if missing and not args.allow_partial:
        raise SystemExit(f"Incomplete relations: {missing}")
    if not data:
        raise SystemExit("No completed relation yet")
    calibration_missing = [
        name
        for name in names
        if not (args.directory / "calibration" / f"{name}.json").exists()
    ]
    differential_missing = [
        name
        for name in names
        if not (args.directory / "differential" / f"{name}.json").exists()
    ]
    if calibration_missing and not args.allow_partial:
        raise SystemExit(f"Incomplete wider-gain controls: {calibration_missing}")
    for item in data:
        gain_path = args.directory / "calibration" / f"{item['relation']}.json"
        if gain_path.exists():
            item["gain_control"] = json.loads(gain_path.read_text())
            assert item["gain_control"]["complete"]
        diff_path = args.directory / "differential" / f"{item['relation']}.json"
        if diff_path.exists():
            item["differential"] = json.loads(diff_path.read_text())
        elif not args.allow_partial:
            raise SystemExit(f"Missing differential readout for {item['relation']}")
    attach_probe_diagnostics(data, args.directory)
    out = args.directory / "analysis"
    out.mkdir(exist_ok=True)
    summaries_all = {}
    all_rows = []
    for layer in [5, 11, 17, 23, "selected"]:
        rows = relation_rows(data, layer)
        summaries_all[str(layer)] = summaries(rows)
        for row in rows:
            all_rows.append({"selection": layer, **row})
    (out / "summary.json").write_text(
        json.dumps(
            {
                "complete": not missing
                and not calibration_missing
                and not differential_missing,
                "missing": missing,
                "calibration_missing": calibration_missing,
                "differential_missing": differential_missing,
                "summaries": summaries_all,
            },
            indent=2,
        )
        + "\n"
    )
    with (out / "by_relation.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted(set().union(*(r.keys() for r in all_rows)))
        )
        writer.writeheader()
        writer.writerows(all_rows)
    s = summaries_all["17"]
    lines = [
        "# Mechanistic experiment measurements",
        "",
        f"Status: {'COMPLETE' if not missing and not calibration_missing and not differential_missing else 'PARTIAL'}. {len(data)}/{len(names)} eligible relations, {sum(len(d['subjects']) - d['nval'] for d in data)} held-out relation–subject pairs. Wider-gain controls: {len(names) - len(calibration_missing)}/{len(names)}. Differential readouts: {len(names) - len(differential_missing)}/{len(names)}.",
        "",
        "OLMo-3-1025-7B, float32, seed 1909. Primary results use fixed layer 17. Means weight relations equally; brackets are 95% paired relation-cluster bootstrap intervals (10,000 resamples). Intervals do not capture training-split or model variation. All answers are first tokens; agreement means matching the original full model, not necessarily factual accuracy.",
        "",
    ]
    groups = [
        (
            "1. User-proposed affine transformation comparison",
            [
                ("Raw affine cosine (beta 1)", "raw_cosine", False),
                ("Raw affine cosine, shuffled subjects", "raw_shuffled_cosine", False),
                (
                    "Raw cosine advantage over shuffled subjects",
                    "raw_cosine_subject_advantage",
                    False,
                ),
                ("Centered native-input cosine", "native_centered_cosine", False),
                (
                    "Reference-centred differential cosine",
                    "taylor_differential_cosine",
                    False,
                ),
                (
                    "Reference-centred differential shuffled cosine",
                    "taylor_differential_shuffled_cosine",
                    False,
                ),
                ("Centered shuffled-subject cosine", "centered_shuffled_cosine", False),
                ("Same-input operator-response cosine", "same_input_cosine", False),
                ("Same-input relative error", "operator_relative_error", False),
                (
                    "Neutral/relation affine top-token agreement (beta 1)",
                    "map_token_agreement",
                    True,
                ),
                ("Relation affine faithfulness (beta 1)", "original_beta1_faith", True),
                ("Neutral affine faithfulness (beta 1)", "neutral_beta1_faith", True),
            ],
        ),
        (
            "2. Neutral subject transplantation",
            [
                ("Original LRE faithfulness", "faith", True),
                ("Original LRE bias-only faithfulness", "original_bias_faith", True),
                ("Gain from the subject-dependent term", "subject_term_gain", True),
                ("LRE with neutral subject state", "transplant_lre", True),
                ("Paired LRE change", "transplant_lre_change", True),
                ("Full model retains original answer", "transplant_full", True),
                ("Full model with wrong neutral subject", "wrong_subject_full", True),
                (
                    "Correct-minus-wrong donor advantage, full model",
                    "neutral_subject_full_advantage",
                    True,
                ),
                ("LRE with wrong neutral subject", "wrong_subject_lre", True),
                (
                    "Donor advantage where original answers differ",
                    "neutral_subject_advantage_different_answer",
                    True,
                ),
                (
                    "Full model target log-probability change",
                    "transplant_logp_change",
                    False,
                ),
            ],
        ),
        (
            "3. Targeted attribute erasure",
            [
                ("Attribute probe coverage of held-out values", "probe_coverage", True),
                (
                    "Training-majority prediction on covered values",
                    "probe_majority",
                    True,
                ),
                ("Neutral-state attribute classification", "probe_neutral", True),
                ("Relation-state attribute classification", "probe_relation", True),
                ("Neutral probe gain over majority", "probe_neutral_gain", True),
                ("Targeted LRE faithfulness drop", "erase_lre_hit_drop", True),
                ("Matched-random LRE faithfulness drop", "random_lre_hit_drop", True),
                (
                    "Targeted full-model answer-agreement drop",
                    "erase_full_hit_drop",
                    True,
                ),
                (
                    "Matched-random full-model answer-agreement drop",
                    "random_full_hit_drop",
                    True,
                ),
                (
                    "Extra targeted damage, LRE log probability (nats)",
                    "selectivity_lre_logp",
                    False,
                ),
                (
                    "Extra targeted damage, full-model log probability (nats)",
                    "selectivity_full_logp",
                    False,
                ),
                (
                    "Extra targeted damage, alternate relation (nats)",
                    "selectivity_control_logp",
                    False,
                ),
                (
                    "Target minus alternate-relation extra damage (nats)",
                    "attribute_specificity_logp",
                    False,
                ),
            ],
        ),
        (
            "4. Shared neutral matrix",
            [
                ("Calibrated relation matrix faithfulness", "calibrated_faith", True),
                ("Matched-reference neutral matrix", "matched_neutral_faith", True),
                ("One global neutral matrix", "shared_faith", True),
                ("Paired global-minus-relation change", "shared_change", True),
                ("Relation-specific bias only", "bias_faith", True),
                ("Global matrix gain over bias", "shared_gain_over_bias", True),
                (
                    "Wider gain range: calibrated relation matrix",
                    "wide_calibrated_faith",
                    True,
                ),
                (
                    "Wider gain range: matched-reference neutral matrix",
                    "wide_matched_neutral_faith",
                    True,
                ),
                ("Wider gain range: global neutral matrix", "wide_shared_faith", True),
                (
                    "Wider gain range: global-minus-relation change",
                    "wide_shared_change",
                    True,
                ),
                (
                    "Wider gain range: global gain over bias",
                    "wide_shared_gain_over_bias",
                    True,
                ),
                (
                    "Relations selecting global gain above 5",
                    "wide_shared_above_five",
                    True,
                ),
                ("Global/relation mean product norm ratio", "shared_norm_ratio", False),
            ],
        ),
        (
            "5. Layer/component interventions",
            [
                (
                    "Full-model answer retained after MLP-update replacement",
                    "mlp_full",
                    True,
                ),
                ("After attention-update replacement", "attention_full", True),
                (
                    "LRE faithfulness change after MLP replacement",
                    "mlp_lre_change",
                    True,
                ),
                (
                    "LRE faithfulness change after attention replacement",
                    "attention_lre_change",
                    True,
                ),
                ("MLP replacement plus subject-state rescue", "mlp_rescue", True),
                (
                    "Attention replacement plus subject-state rescue",
                    "attention_rescue",
                    True,
                ),
            ],
        ),
        (
            "Gold-correct subset",
            [
                ("Original full-model gold correctness", "model_correct", True),
                ("LRE faithfulness on gold-correct cases", "correct_faith", True),
                (
                    "Neutral transplant LRE on gold-correct cases",
                    "correct_transplant_lre",
                    True,
                ),
                (
                    "Neutral transplant full model on gold-correct cases",
                    "correct_transplant_full",
                    True,
                ),
                (
                    "Calibrated relation matrix on gold-correct cases",
                    "correct_calibrated_faith",
                    True,
                ),
                (
                    "Global neutral matrix on gold-correct cases",
                    "correct_shared_faith",
                    True,
                ),
                (
                    "Extra targeted full-model damage on gold-correct cases",
                    "correct_selectivity_full_logp",
                    False,
                ),
            ],
        ),
    ]
    for title, metrics in groups:
        lines += [
            f"## {title}",
            "",
            "| Measurement | Mean [95% CI] | Relations |",
            "|---|---:|---:|",
        ]
        for label, key, percent in metrics:
            lines.append(
                f"| {label} | {fmt(s, key, percent)} | {s[key]['n_relations'] if key in s else 0} |"
            )
        lines.append("")
    lines += [
        "## Layer scan",
        "",
        "Neutral differential ranks are geometric mean ranks over the full vocabulary; lower is better. This readout has no relation-selected gain or restored output prior.",
        "",
        "| Layer | LRE faithfulness | Neutral differential rank | Shared matrix (calibrated) | MLP log-p drop | Attention log-p drop |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for layer in [5, 11, 17, 23]:
        sl = summaries_all[str(layer)]
        rank = sl.get("differential_log_rank")
        rank_text = (
            f"{10 ** rank['mean']:.1f} [{10 ** rank['ci95'][0]:.1f}, {10 ** rank['ci95'][1]:.1f}]"
            if rank
            else "pending"
        )
        lines.append(
            f"| {layer} | {fmt(sl, 'faith', True)} | {rank_text} | {fmt(sl, 'shared_faith', True)} | {fmt(sl, 'mlp_logp_drop')} | {fmt(sl, 'attention_logp_drop')} |"
        )
    lines += ["", "## Validation-selected layer (secondary)", ""]
    for key in [
        "faith",
        "transplant_lre",
        "transplant_full",
        "calibrated_faith",
        "shared_faith",
    ]:
        lines.append(f"- {key}: {fmt(summaries_all['selected'], key, True)}")
    lines += [
        "",
        "## Checks and limitations",
        "",
        f"- Maximum erase/restore final-state error: {max(r['restoration_max_error'] for r in all_rows):.8g}.",
        "- Numerical restoration is a sanity check; adding the removed vector back is not independent evidence.",
        "- Component ablations affect only the subject position within one block. Restoring that block's complete subject state also restores the entire changed pathway by construction; rescue alone is not independent mediation evidence.",
        "- Related relations can share subjects and attributes. The relation bootstrap is descriptive and does not resolve every dependency.",
        "- Operator similarity is measured on held-out subject directions, not all possible hidden states.",
        "- Shared-matrix comparison uses training-calibrated centering and intercepts for both matrices; other experiments use the exact reference-centred Taylor form.",
        "- Erasure analyses exclude relations without at least two replicated attribute classes in the training split. Up to eight directions and three norm-matched random controls are used.",
        "- Alternate-relation attributes may be correlated; they are not guaranteed independent controls.",
        "- A neutral donor still supplies subject identity; successful transplantation alone does not prove that all factual content was already stored in it.",
        "- Four individual blocks are tested; null effects cannot exclude distributed or redundant pathways.",
        "- One model and one data split; small country datasets have only 4–8 held-out cases. This is an exploratory study, not proof of subject dominance.",
        "",
        "## Per-relation primary results",
        "",
        "| Relation | Test n | LRE | Neutral-state LRE | Neutral-state LM | Calibrated LRE | Global matrix | Attribute rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in relation_rows(data, 17):
        lines.append(
            f"| {r['relation']} | {r['n_test']} | {r['faith']:.3f} | {r['transplant_lre']:.3f} | {r['transplant_full']:.3f} | {r['calibrated_faith']:.3f} | {r['shared_faith']:.3f} | {r['attribute_rank']} |"
        )
    lines += ["", "## Excluded datasets", ""]
    for name, plan in manifest["relations"].items():
        if "skipped" in plan:
            lines.append(f"- {name}: {plan['skipped']} (n={plan['n']}).")
    if missing:
        lines += ["", "Pending: " + ", ".join(missing)]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    figures(data, out)
    print(f"Wrote {out / 'REPORT.md'}; completed {len(data)}/{len(names)} relations")


if __name__ == "__main__":
    main()
