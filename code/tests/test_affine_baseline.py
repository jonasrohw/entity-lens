"""The affine baseline uses the same corrected targets as the differential lens."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jlens import paper_statistics


def test_affine_scores_preserve_order_and_use_corrected_years(monkeypatch):
    def entry(score):
        return {"agg": {"lre": {"median_lograng": score}}}

    files = {
        "results/jacobian_lens.json": {
            "country_capital_city": entry(2),
            "presidents_birth_year": entry(99),
            "presidents_election_year": entry(98),
        },
        "results/jacobian_lens_years.json": {
            "presidents_birth_year": entry(3),
            "presidents_election_year": entry(4),
        },
    }
    monkeypatch.setattr(paper_statistics, "load", files.__getitem__)
    scores = paper_statistics.affine_lens_scores(
        ["presidents_election_year", "country_capital_city", "presidents_birth_year"]
    )
    np.testing.assert_array_equal(scores, [-4, -2, -3])


@pytest.mark.parametrize(
    "model,rho,p,delta",
    [("olmo", "0.420", "2.59e-02", "-0.351"), ("gptj", "0.263", "1.77e-01", "-0.081")],
)
def test_saved_model_statistics_report_affine_baseline(model, rho, p, delta):
    root = Path(__file__).resolve().parents[1] / "results" / model
    result = subprocess.run(
        [sys.executable, "-m", "jlens", "--root", str(root), "statistics"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
    )
    label = "Affine Jacobian-lens baseline (+zeroth-order term)"
    assert (
        f"{label} (28 relations): rho={rho}, p={p}, delta_rho={delta}" in result.stdout
    )
    assert f"Shared split, {label}" in result.stdout
