"""Separate layer/prompt effects from small-cohort and target-cardinality effects."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from jlens import grid_statistics as stats
from jlens import model_sweep as sweep
from jlens import splits

ROOT = Path(__file__).resolve().parents[1] / "generated/mistral_full_grid"


def main():
    rows = [
        json.loads(path.read_text()) for path in sorted((ROOT / "main").glob("*.json"))
    ]
    assert len(rows) == 28
    analysis = json.loads((ROOT / "analysis/main.json").read_text())
    coarse = analysis["analyses"]["coarse_refs8"]["selected_faith"]
    continuous = analysis["analyses"]["continuous_refs8"]["selected_faith"]
    base = dict(
        prompt="known_for",
        layer=17,
        references=8,
        readout="diff",
        beta=None,
        metric="median_logrank",
    )
    x = np.array([stats.lens_value(row, base) for row in rows])
    y = np.array([coarse[row["relation"]]["score"] for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.3", local_files_only=True
    )
    diagnostics = []
    for row, score in zip(rows, x):
        counts = Counter(row["faith"]["targets"])
        probabilities = np.asarray(list(counts.values())) / row["faith"]["n"]
        gold = [
            splits.first_object_token(value, tokenizer, row["trailing_space"])
            for value in row["faith"]["gold"]
        ]
        prior = float(np.mean(row["faith"]["grid"]["0"]["8"]["0.0"]["hits"]))
        diagnostics.append(
            dict(
                relation=row["relation"],
                n_test=row["faith"]["n"],
                n_distinct_model_targets=len(counts),
                model_target_tokens=[tokenizer.decode([value]) for value in counts],
                model_target_entropy_bits=float(
                    -np.sum(probabilities * np.log2(probabilities))
                ),
                n_distinct_gold_first_tokens=len(set(gold)),
                lm_accuracy=float(
                    np.mean(np.asarray(gold) == np.asarray(row["faith"]["targets"]))
                ),
                prior_only_faith=prior,
                coarse_faith=coarse[row["relation"]]["score"],
                continuous_faith=continuous[row["relation"]]["score"],
                original_lens_score=float(score),
                selected_continuous_layer=continuous[row["relation"]]["layer"],
                selected_continuous_beta=continuous[row["relation"]]["beta"],
            )
        )
    thresholds = {}
    for minimum in [1, 4, 10, 20, 50, 100]:
        indices = [i for i, row in enumerate(rows) if row["faith"]["n"] >= minimum]
        thresholds[str(minimum)] = dict(
            **stats.correlation(x[indices], y[indices]),
            relations=[rows[i]["relation"] for i in indices],
        )
    configs = list(stats.candidate_configs(rows[0]))
    matrix = np.asarray(
        [[stats.lens_value(row, config) for row in rows] for config in configs]
    )
    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    configs = [config for config, keep in zip(configs, valid) if keep]
    scores = stats.all_correlations(matrix, y)
    predicates = {
        "original_fixed": lambda c: c == base,
        "layer_only_original_differential": lambda c: (
            c["prompt"] == "known_for"
            and c["references"] == 8
            and c["readout"] == "diff"
        ),
        "affine_beta_at_original_layer": lambda c: (
            c["prompt"] == "known_for"
            and c["references"] == 8
            and c["layer"] == 17
            and c["readout"] == "affine"
        ),
        "layer_and_affine_beta_original_prompt": lambda c: (
            c["prompt"] == "known_for"
            and c["references"] == 8
            and c["readout"] == "affine"
        ),
        "bare_subject_at_original_layer": lambda c: (
            c["prompt"] == "bare_subject"
            and c["layer"] == 17
            and c["references"] == 8
            and c["readout"] == "diff"
        ),
        "bare_subject_differential_layer_search": lambda c: (
            c["prompt"] == "bare_subject"
            and c["references"] == 8
            and c["readout"] == "diff"
        ),
        "bare_subject_layer_and_affine_beta": lambda c: (
            c["prompt"] == "bare_subject"
            and c["references"] == 8
            and c["readout"] == "affine"
        ),
        "full_search_four_references": lambda c: c["references"] == 4,
    }
    ablations = {}
    for name, predicate in predicates.items():
        eligible = [
            i for i, c in enumerate(configs) if predicate(c) and np.isfinite(scores[i])
        ]
        best = max(eligible, key=lambda i: scores[i])
        ablations[name] = dict(
            config=configs[best],
            **stats.correlation(matrix[best], y),
            n_searched=len(eligible),
        )
    loo = {
        row["relation"]: stats.correlation(np.delete(x, i), np.delete(y, i))
        for i, row in enumerate(rows)
    }
    prior_indices = [i for i, d in enumerate(diagnostics) if d["prior_only_faith"] < 1]
    excess = {
        mode: stats.correlation(
            x[prior_indices],
            [
                (diagnostics[i][mode + "_faith"] - diagnostics[i]["prior_only_faith"])
                / (1 - diagnostics[i]["prior_only_faith"])
                for i in prior_indices
            ],
        )
        for mode in ["coarse", "continuous"]
    }
    faith_ablations = {}
    refined = {
        path.stem: json.loads(path.read_text())
        for path in (ROOT / "continuous/main").glob("*.json")
    }
    for name, expanded_layers, expanded_betas in [
        ("original_layers_original_betas", False, False),
        ("original_layers_dense_betas", False, True),
        ("all_states_original_betas", True, False),
        ("all_states_dense_betas", True, True),
    ]:
        outcomes = []
        settings = {}
        for row in rows:
            layers = row["layers"] if expanded_layers else stats.COARSE_LAYERS
            betas = (
                [b for b in row["betas"] if b > 0]
                if expanded_betas
                else stats.COARSE_BETAS
            )
            cells = [
                (
                    layer,
                    beta,
                    float(
                        np.mean(
                            row["faith"]["grid"][str(layer)]["8"][str(beta)]["hits"]
                        )
                    ),
                )
                for layer in layers
                for beta in betas
                if refined[row["relation"]]["grid"][str(layer)]["8"]["input_dependent"]
            ]
            layer, beta, score = max(cells, key=lambda cell: cell[2])
            outcomes.append(score)
            settings[row["relation"]] = dict(layer=layer, beta=beta, faith=score)
        faith_ablations[name] = dict(
            mean_faith=float(np.mean(outcomes)),
            original_lens=stats.correlation(x, outcomes),
            settings=settings,
        )
    result = dict(
        diagnostics=diagnostics,
        min_test_count_sensitivity=thresholds,
        one_factor_ablations=ablations,
        lre_fit_ablations=faith_ablations,
        leave_one_relation_out=loo,
        normalized_excess_over_prior=dict(
            correlations=excess,
            excluded=[d["relation"] for d in diagnostics if d["prior_only_faith"] == 1],
            note="Exploratory alternative outcome (faith-prior)/(1-prior); not the original raw faithfulness metric.",
        ),
    )
    sweep.write(ROOT / "analysis/diagnostics_main.json", result)
    print(
        json.dumps(
            dict(
                thresholds={
                    k: {a: b for a, b in v.items() if a != "relations"}
                    for k, v in thresholds.items()
                },
                ablations=ablations,
                excess=result["normalized_excess_over_prior"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
