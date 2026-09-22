"""Beta selection and cached-vector replay must match the measured readout."""

import argparse
import sys

import pytest

from jlens.affine_beta import parse_betas, summarize, tuned_score
from jlens.io import read_json


@pytest.mark.parametrize("value", ["0,1", "nan,1", "inf,1", "-1,1", "2,3", "x"])
def test_invalid_beta_grid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_betas(value)


def test_selection_uses_top1_and_smallest_beta_for_ties():
    betas = parse_betas("5,1,0.5,1")
    rows = [
        {"rank_lre_beta": {"0.5": 0, "1.0": 1, "5.0": 0}},
        {"rank_lre_beta": {"0.5": 99, "1.0": 1, "5.0": 2}},
        {"rank_lre_beta": {"0.5": 99, "1.0": 1, "5.0": 2}},
    ]
    assert summarize(rows, betas, "top1")["best_beta"] == 0.5
    assert summarize(rows, betas, "median-logrank")["best_beta"] == 1.0


def test_saved_sweep_supports_both_selection_rules():
    """A run records one rule; the retained per-beta aggregates allow the other."""
    betas = parse_betas("0.5,1,5")
    rows = [
        {"rank_lre_beta": {"0.5": 0, "1.0": 1, "5.0": 0}},
        {"rank_lre_beta": {"0.5": 99, "1.0": 1, "5.0": 2}},
        {"rank_lre_beta": {"0.5": 99, "1.0": 1, "5.0": 2}},
    ]
    sweep = summarize(rows, betas, "top1")
    assert tuned_score(sweep, "top1")[0] == sweep["best_beta"] == 0.5
    assert tuned_score(sweep, "median-logrank")[0] == 1.0
    assert tuned_score(sweep, "median-logrank")[1] == pytest.approx(
        -sweep["agg"]["1.0"]["median_lograng"]
    )
    with pytest.raises(ValueError):
        tuned_score(sweep, "mean-rank")


def test_sweep_matches_cached_vectors(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from test_baseline_measurements import LAYER, toy_repository

    from jlens import measure_jacobian_lens
    from jlens.model import final_logits

    model, tokenizer, subjects = toy_repository(tmp_path)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
    )
    output = tmp_path / "sweep.json"
    vectors = tmp_path / "vectors"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lens",
            "--data-dir",
            str(tmp_path / "data/relations"),
            "--out",
            str(output),
            "--layer",
            str(LAYER),
            "--device",
            "cpu",
            "--legacy-split",
            "--vector-dir",
            str(vectors),
        ],
    )
    measure_jacobian_lens.main()
    result = read_json(output)["toy"]
    saved = torch.load(vectors / "toy.pt", weights_only=True)
    assert [row["subject"] for row in result["rows"]] == subjects
    assert result["beta_sweep"]["agg"]["1.0"] == result["agg"]["lre"]
    for row, vector in zip(result["rows"], saved["rows"]):
        for beta in result["beta_sweep"]["betas"]:
            hidden = (
                vector["affine_beta1"]
                if beta == 1.0
                else vector["reference_mean"] + beta * vector["differential"]
            )
            logits = final_logits(hidden, model.model.norm, model.lm_head.weight)
            rank = int((logits > logits[vector["target_token_id"]]).sum())
            assert row["rank_lre_beta"][str(beta)] == rank
        assert row["rank_lre_beta"]["1.0"] == row["rank_lre"]
