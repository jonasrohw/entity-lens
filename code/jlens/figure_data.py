"""Reconstruct the paper's 28-row figure input from retained measurements."""

import argparse
from pathlib import Path

from jlens.config import OUTPUT_DIR, ROOT, YEAR_RELATIONS
from jlens.io import read_json, relation_data, write_json


def faithfulness_scores(root: Path) -> dict[str, float]:
    """Prefer corrected trailing-space runs, independent of directory order."""
    runs = {}
    for path in sorted((root / "results/main").glob("faithfulness_*.json")):
        result = read_json(path)
        if isinstance(result.get("best_faith"), (int, float)):
            variant = "corrected" if path.stem.endswith("_ts") else "original"
            runs.setdefault(result["relation"], {})[variant] = result["best_faith"]
    return {
        relation: values["corrected"] if "corrected" in values else values["original"]
        for relation, values in runs.items()
    }


def build_figure_data(root: Path = ROOT) -> dict:
    """Return the original five-field schema without rounding measurements."""
    faithfulness = faithfulness_scores(root)
    lens = read_json(root / "results/jacobian_lens.json")
    years = read_json(root / "results/jacobian_lens_years.json")
    prior = read_json(root / "results/answer_prior.json")
    data = relation_data(root)
    relations = sorted(
        relation
        for relation in lens
        if not relation.startswith("_") and relation in faithfulness
    )
    return {
        "rels": relations,
        "faith": [faithfulness[relation] for relation in relations],
        "lens": [
            (years if relation in YEAR_RELATIONS else lens)[relation]["agg"]["diff"][
                "median_lograng"
            ]
            for relation in relations
        ],
        "k": [
            len(data[relation]["samples"])
            / len({sample["object"] for sample in data[relation]["samples"]})
            for relation in relations
        ],
        "prior": [prior[relation]["prior_modeltop1"] for relation in relations],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "paper_figure_data.json",
        help="destination JSON; relative paths are resolved under --root",
    )
    args = parser.parse_args()
    data = build_figure_data()
    path = ROOT / args.output
    write_json(path, data)
    print(f"wrote {path} ({len(data['rels'])} relations)")


if __name__ == "__main__":
    main()
