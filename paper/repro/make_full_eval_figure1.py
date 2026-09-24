"""Update only Figure 1, preserving the established paper plotting style.

Run from the repository root:
    code/.venv/bin/python paper/repro/make_full_eval_figure1.py

All 28 original OLMo relations and all 9,747 rows contribute. The saved
measurements preserve labels and duplicates, use gender alternatives and
original continent references, and retain OLMo's audited year relations.
No measurements or manuscript text are changed.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr

from jlens import plot_relation_agreement as plot


def main():
    results = ROOT / "code/results/paper_full_evaluation"
    with (results / "relation_scores.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["model"] == "olmo3_7b"]
    assert len(rows) == len({r["relation"] for r in rows}) == 28
    assert sum(int(r["n_rows"]) for r in rows) == 9747
    for row in rows:
        source = json.loads((ROOT / "code/data/relations" / (row["relation"] + ".json")).read_text())
        assert int(row["n_rows"]) == len(source["samples"])
        assert row["primary_scored"] == "True"
    data = {
        "rels": [r["relation"] for r in rows],
        "faith": [float(r["faithfulness"]) for r in rows],
        "lens": [-float(r["differential_score"]) for r in rows],
    }
    expected = json.loads((results / "paper_statistics.json").read_text())["models"]["olmo3_7b"]["primary"]["rho"]
    assert abs(spearmanr(-np.array(data["lens"]), data["faith"]).statistic - expected) < 1e-12
    labels = {
        "presidents_birth_year": ("presidents_birth_year", -4.55, 0.84, "left"),
        "presidents_election_year": ("presidents_election_year", -3.85, 0.925, "left"),
        "landmark_on_continent": ("landmark_on_continent", -3.10, 0.85, "left"),
        "person_native_language": ("person_native_language", -2.05, 1.045, "left"),
        "country_capital_city": ("country_capital", -0.48, 0.72, "right"),
        "name_gender": ("name_gender", -1.65, 0.43, "left"),
        "person_father": ("person_father", -3.25, 0.23, "left"),
        "person_mother": ("person_mother", -4.22, 0.15, "left"),
        "company_ceo": ("company_ceo", -4.56, 0.23, "left"),
    }
    plot.OUTPUT_DIR = ROOT / "paper/build/figure1_full_evaluation"
    fig = plot.main(data=data, labels=labels)
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    boxes = [(t.get_text(), Text.get_window_extent(t, renderer)) for t in ax.texts]
    for i, (name, box) in enumerate(boxes):
        for other, other_box in boxes[i + 1:]:
            assert not box.overlaps(other_box), ("overlapping labels", name, other)
    for label in [t for t in ax.texts if isinstance(t, Annotation)]:
        box = Text.get_window_extent(label, renderer)
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1), label.get_text()
        assert not box.overlaps(ax.get_legend().get_window_extent(renderer)), label.get_text()
        for rel, x, y in zip(data["rels"], -np.array(data["lens"]), data["faith"], strict=True):
            px, py = ax.transData.transform((x, y))
            radius = (np.sqrt(30 if "year" in rel else 20) + 1) * fig.dpi / 144
            assert not box.overlaps(Bbox.from_extents(px-radius, py-radius, px+radius, py+radius)), ("label covers point", label.get_text(), rel)
    (ROOT / "paper/fig_iclr_agreement.pdf").write_bytes((plot.OUTPUT_DIR / "relation_agreement.pdf").read_bytes())
    print(f"Verified all 28 relations, 9,747 rows, rho={expected:.12f}, and non-overlapping labels.")


if __name__ == "__main__":
    main()
