"""The ablation baselines must score exactly the retained lens's subjects."""

import numpy as np
import pytest

from jlens import matched_baseline
from jlens.config import ROOT


def test_references_cover_the_paper_relations_with_corrected_years():
    references = matched_baseline.load_references(ROOT)
    assert len(references) == 28
    assert all(reference["layer"] == 17 for reference in references.values())
    # The year relations come from the corrected run, whose targets are content
    # tokens rather than the bare space the uncorrected run recorded.
    assert all(
        row["model_top1"].isdigit()
        for row in references["presidents_birth_year"]["rows"]
    )


@pytest.mark.parametrize(
    "relation", ["country_capital_city", "person_father", "presidents_birth_year"]
)
def test_rebuilt_split_matches_the_retained_lens_run(relation):
    references = matched_baseline.load_references(ROOT)
    reference = references[relation]
    template, context, subjects, anchors = matched_baseline.matched_split(
        relation, reference, 17, ROOT
    )
    assert template == reference["template"]
    assert subjects == [row["subject"] for row in reference["rows"]]
    assert anchors == subjects[: matched_baseline.N_REFERENCES]
    assert context.count("\n") == matched_baseline.N_ICL
    assert not set(anchors) & {
        row["subject"] for row in reference["rows"][matched_baseline.N_REFERENCES :]
    }


def test_split_rejects_a_drifted_reference():
    reference = matched_baseline.load_references(ROOT)["country_capital_city"]
    with pytest.raises(ValueError, match="layer differs"):
        matched_baseline.matched_split("country_capital_city", reference, 20, ROOT)
    with pytest.raises(ValueError, match="template differs"):
        matched_baseline.matched_split(
            "country_capital_city", {**reference, "template": "{} ?"}, 17, ROOT
        )
    with pytest.raises(ValueError, match="subjects differ"):
        matched_baseline.matched_split(
            "country_capital_city",
            {**reference, "rows": reference["rows"][:-1]},
            17,
            ROOT,
        )


def test_aggregate_reproduces_the_retained_summary_statistics():
    reference = matched_baseline.load_references(ROOT)["country_capital_city"]
    recomputed = matched_baseline.aggregate(
        [row["rank_diff"] for row in reference["rows"]]
    )
    for key, value in reference["agg"]["diff"].items():
        assert np.isclose(recomputed[key], value), key
