"""Summarize the frozen-setting fresh-split experiment without retuning."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jlens import grid_search as grid
from jlens import model_sweep as sweep
from jlens import splits
from jlens.grid_statistics import correlation

ROOT = Path(__file__).resolve().parents[1] / "generated/mistral_full_grid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    while len(list((root / "confirmation").glob("*.json"))) < 28:
        if not args.watch:
            return
        time.sleep(5)
    config = json.loads((root / "confirmation_config.json").read_text())
    folds = json.loads((root / "relation_folds.json").read_text())
    rows = [
        json.loads(path.read_text())
        for path in sorted((root / "confirmation").glob("*.json"))
    ]
    excluded = [row["relation"] for row in rows if "faith" not in row]
    rows = [row for row in rows if "faith" in row]
    results = {}
    for faith_name in ["coarse", "continuous", "calibrated_continuous"]:
        cohort = [row for row in rows if faith_name in row["faith"]]
        for row in cohort:
            selected = config["faith_configs"][row["relation"]][faith_name]
            assert row["faith"][faith_name]["layer"] == selected["layer"]
            assert row["faith"][faith_name]["beta"] == selected["beta"]
        results[faith_name] = {}
        for lens_name in config["lens_configs"]:
            x = np.asarray(
                [-row["lens"][lens_name]["result"]["median_logrank"] for row in cohort]
            )
            y = np.asarray([row["faith"][faith_name]["score"] for row in cohort])
            heldout = [
                i
                for i, row in enumerate(cohort)
                if row["relation"] in folds["evaluation"]
            ]
            prior = [i for i, row in enumerate(cohort) if row["prior_only_faith"] < 1]
            normalized = [
                (y[i] - cohort[i]["prior_only_faith"])
                / (1 - cohort[i]["prior_only_faith"])
                for i in prior
            ]
            results[faith_name][lens_name] = dict(
                all_relations=correlation(x, y),
                predefined_evaluation_relations=correlation(x[heldout], y[heldout]),
                normalized_excess_diagnostic=correlation(x[prior], normalized),
                normalized_excess_excluded=[
                    row["relation"] for row in cohort if row["prior_only_faith"] == 1
                ],
                mean_faith=float(y.mean()),
            )
    overlaps = []
    for row in rows:
        original_root = Path(config.get("source_grid_root", str(ROOT)))
        protocol = config.get("protocol", "main")
        original = json.loads(
            (original_root / protocol / f"{row['relation']}.json").read_text()
        )
        subjects = row.get("faith_subjects")
        if subjects is None:
            assert protocol == "main", (
                "Shared-split confirmation must record its subjects"
            )
            samples = json.loads((sweep.DATA / f"{row['relation']}.json").read_text())[
                "samples"
            ]
            partition = grid.partition(samples, "main", seed=config["seed"])
            assert (
                splits.split_fingerprint(*partition) == row["faith_split_fingerprint"]
            )
            subjects = [sample["subject"] for sample in partition[2]]
        overlap = len(set(original["faith"]["subjects"]) & set(subjects))
        overlaps.append(
            dict(relation=row["relation"], n_test=row["n_faith"], overlap=overlap)
        )
    result = dict(
        seed=config["seed"],
        protocol=config.get("protocol", "main"),
        subject_policy=config.get("subject_policy", "original_rows"),
        input_policy_note=config.get("note"),
        excluded_relations=excluded,
        results=results,
        subject_overlap=overlaps,
        note="No hyperparameters selected on this draw. Operators were refitted with fresh demonstrations/references. This is a same-dataset robustness check, not new external data.",
    )
    sweep.write(root / "analysis/confirmation.json", result)
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
