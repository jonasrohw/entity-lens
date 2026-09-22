"""Compare overlapping pretrained measurements and audit continuous optima."""

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
GRID = BASE / "generated/mistral_full_grid"
ORIGINAL = BASE / "generated/five_model_sweep/results/mistral7b"


def main():
    report = dict(
        original_faith_cells_checked=0,
        individual_ranks_checked=0,
        faith_differences=[],
        individual_rank_differences=[],
        split_differences=[],
        continuous_optima_checked=0,
        continuous_unverified=[],
        max_readout_error=0.0,
        protocols={},
    )
    for protocol in ["main", "shared_split"]:
        count = 0
        for path in sorted((GRID / protocol).glob("*.json")):
            row = json.loads(path.read_text())
            old = json.loads((ORIGINAL / protocol / path.name).read_text())
            if "faith" not in row:
                assert "skipped" in old
                continue
            count += 1
            for layer, cells in old["faithfulness"]["grid"].items():
                for beta, (score, _) in cells.items():
                    actual = np.mean(row["faith"]["grid"][layer]["8"][beta]["hits"])
                    report["original_faith_cells_checked"] += 1
                    if actual != score:
                        report["faith_differences"].append(
                            [protocol, path.stem, layer, beta, score, actual]
                        )
            if (
                row["faith"]["split_fingerprint"]
                != old["faithfulness"]["split_fingerprint"]
            ):
                report["split_differences"].append([protocol, path.stem, "faith"])
            if row["lens"].get("n", 0) < 2:
                assert old["lens"].get("n", 0) < 2
                continue
            if row["lens"]["subjects"] != [r["subject"] for r in old["lens"]["rows"]]:
                report["split_differences"].append([protocol, path.stem, "lens"])
            lens = row["lens"]["prompts"]["known_for"]["17"]["8"]
            for kind, new in [("diff", lens["diff"]), ("lre", lens["affine"]["1.0"])]:
                before = [r["rank_" + kind] for r in old["lens"]["rows"]]
                assert len(before) == len(new["ranks"])
                for i, (a, b) in enumerate(zip(before, new["ranks"])):
                    report["individual_ranks_checked"] += 1
                    if a != b:
                        report["individual_rank_differences"].append(
                            [protocol, path.stem, kind, i, b - a]
                        )
        report["protocols"][protocol] = dict(relations_compared=count)
        for path in (GRID / "continuous" / protocol).glob("*.json"):
            row = json.loads(path.read_text())
            report["max_readout_error"] = max(
                report["max_readout_error"], row.get("readout_max_abs_error", 0)
            )
            for layer, refs in row.get("grid", {}).items():
                for reference, modes in refs.items():
                    for mode in ["all_subjects", "calibration"]:
                        if not modes.get(mode):
                            continue
                        report["continuous_optima_checked"] += 1
                        if not modes[mode]["verified_optimum"]:
                            report["continuous_unverified"].append(
                                [protocol, path.stem, layer, reference, mode]
                            )
    (GRID / "regression_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert not report["faith_differences"]
    assert not report["split_differences"]
    assert not report["continuous_unverified"]
    assert all(abs(row[-1]) <= 1 for row in report["individual_rank_differences"])


if __name__ == "__main__":
    main()
