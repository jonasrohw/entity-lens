"""Analyze complete expanded grids and queue frozen-config fresh-split checks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jlens import grid_statistics as stats
from jlens import model_sweep as sweep

ROOT = Path(__file__).resolve().parents[1]


def report_protocol(root, protocol):
    rows = [
        json.loads(path.read_text())
        for path in sorted((root / protocol).glob("*.json"))
    ]
    rows = [row for row in rows if "faith" in row and "prompts" in row.get("lens", {})]
    refined = {
        path.stem: json.loads(path.read_text())
        for path in (root / "continuous" / protocol).glob("*.json")
    }
    folds = json.loads((root / "relation_folds.json").read_text())
    analyses = {}
    for scenario, refs in [
        ("coarse", 8),
        ("dense", 8),
        ("continuous", 8),
        ("continuous", 4),
        ("calibrated", 8),
    ]:
        analyses[f"{scenario}_refs{refs}"] = stats.search(
            rows,
            refined,
            folds,
            scenario,
            refs,
            permutations=2000
            if scenario in ("coarse", "continuous") and refs == 8
            else 0,
        )
    sensitivity = {
        metric: stats.search(rows, refined, folds, "continuous", 8, metric)
        for metric in stats.METRICS
        if metric != "median_logrank"
    }
    ys = [
        analyses["continuous_refs8"]["selected_faith"][row["relation"]]["score"]
        for row in rows
    ]
    prior_scores = [
        -row["lens"]["prompts"]["known_for"]["17"]["8"]["affine"]["0.0"][
            "median_logrank"
        ]
        for row in rows
    ]
    prior_faith = [
        float(np.mean(row["faith"]["grid"]["0"]["8"]["0.0"]["hits"])) for row in rows
    ]
    controls = dict(
        matched_predictions=stats.matched_control(rows, refined),
        zero_beta_neutral=stats.correlation(prior_scores, ys),
        mean_prior_only_faithfulness=float(np.mean(prior_faith)),
        mean_continuous_faithfulness=float(np.mean(ys)),
    )
    calibrated_relations = analyses["calibrated_refs8"]["selected_faith"]
    cohort = [row for row in rows if row["relation"] in calibrated_relations]
    original_config = dict(
        prompt="known_for",
        layer=17,
        references=8,
        readout="diff",
        beta=None,
        metric="median_logrank",
    )
    controls["calibration_cohort"] = dict(
        n=len(cohort),
        excluded=sorted(
            set(row["relation"] for row in rows) - set(calibrated_relations)
        ),
        original_lens_original_lre=stats.correlation(
            [stats.lens_value(row, original_config) for row in cohort],
            [
                analyses["coarse_refs8"]["selected_faith"][row["relation"]]["score"]
                for row in cohort
            ],
        ),
        original_lens_continuous_lre=stats.correlation(
            [stats.lens_value(row, original_config) for row in cohort],
            [
                analyses["continuous_refs8"]["selected_faith"][row["relation"]]["score"]
                for row in cohort
            ],
        ),
    )
    verification = [
        (row["relation"], layer, ref, mode, fit["verified_optimum"])
        for row in rows
        for layer, refs in refined[row["relation"]]["grid"].items()
        for ref, cell in refs.items()
        for mode, fit in [
            ("all_subjects", cell["all_subjects"]),
            ("calibration", cell["calibration"]),
        ]
        if fit is not None
    ]
    result = dict(
        protocol=protocol,
        n_relations=len(rows),
        analyses=analyses,
        metric_sensitivity=sensitivity,
        controls=controls,
        unverified_interval_optima=[v[:-1] for v in verification if not v[-1]],
        caveat="Exploratory maxima; nominal p-values at selected settings do not account for search. Model choice used earlier full-relation results.",
    )
    sweep.write(root / "analysis" / f"{protocol}.json", result)
    if protocol == "main":
        baseline = dict(
            prompt="known_for",
            layer=17,
            references=8,
            readout="diff",
            beta=None,
            metric="median_logrank",
        )
        selected = analyses["continuous_refs8"]
        queue = dict(
            seed=1,
            protocol="main",
            lens_configs=dict(
                baseline=baseline,
                exploratory_winner=selected["exploratory_top10"][0]["config"],
                original_target_winner=analyses["coarse_refs8"]["exploratory_top10"][0][
                    "config"
                ],
                calibration_target_winner=analyses["calibrated_refs8"][
                    "exploratory_top10"
                ][0]["config"],
                relation_selection_winner=selected["heldout_relations"]["config"],
                calibration_relation_selection_winner=analyses["calibrated_refs8"][
                    "heldout_relations"
                ]["config"],
            ),
            faith_configs={
                row["relation"]: dict(
                    coarse=analyses["coarse_refs8"]["selected_faith"][row["relation"]],
                    continuous=analyses["continuous_refs8"]["selected_faith"][
                        row["relation"]
                    ],
                    **(
                        {"calibrated_continuous": calibrated_relations[row["relation"]]}
                        if row["relation"] in calibrated_relations
                        else {}
                    ),
                )
                for row in rows
            },
            note="Hyperparameters frozen from seed 0. Seed 1 re-draws demonstrations/references and refits the operators; subjects may overlap because the dataset is reused.",
        )
        sweep.write(root / "confirmation_config.json", queue)
    print(
        "ANALYZED",
        protocol,
        "best",
        analyses["continuous_refs8"]["exploratory_top10"][0],
        flush=True,
    )
    return result


def write_table(root):
    lines = [
        "# Mistral expanded-grid results",
        "",
        "All-relation maxima below are exploratory. The held-out column selects the lens configuration using only the predefined selection relations.",
        "",
        "| Protocol | LRE search | Fixed original lens rho | Best exploratory rho | Held-out relation rho |",
        "| --- | --- | --- | --- | --- |",
    ]
    for protocol in ["main", "shared_split"]:
        path = root / "analysis" / f"{protocol}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for name, data in result["analyses"].items():
            best = data["exploratory_top10"][0]["rho"]
            held = data["heldout_relations"]["evaluation"]["rho"]
            lines.append(
                f"| {protocol} | {name} | {data['baseline_lens']['rho']:.3f} | {best:.3f} | {held:.3f} |"
            )
    lines += [
        "",
        "These checks do not undo the earlier model selection. A fresh-split robustness check uses frozen settings, without searching its outcomes.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "generated/mistral_full_grid"
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    expected = len(list(sweep.DATA.glob("*.json")))
    analyzed = set()
    while True:
        for protocol in ["main", "shared_split"]:
            if protocol in analyzed:
                continue
            existing = (args.root / "analysis" / f"{protocol}.json").exists()
            ready = (
                len(list((args.root / protocol).glob("*.json"))) == expected
                and len(list((args.root / "continuous" / protocol).glob("*.json")))
                == expected
            )
            if existing and not args.force:
                analyzed.add(protocol)
            elif ready:
                report_protocol(args.root, protocol)
                analyzed.add(protocol)
                write_table(args.root)
        if not args.watch or len(analyzed) == 2:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
