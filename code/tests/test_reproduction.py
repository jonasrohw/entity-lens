"""Regression checks against the retained scientific results."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from jlens.cli import COMMANDS
from jlens.config import ROOT
from jlens.figure_data import build_figure_data, faithfulness_scores
from jlens.io import read_json, write_json
from jlens.paper_statistics import fact_points


def run_cli(*arguments, cwd=None):
    environment = {**os.environ, "JLENS_ROOT": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "jlens", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_figure_data_matches_every_retained_value():
    assert build_figure_data(ROOT) == read_json(ROOT / "results/paper_figure_data.json")


def test_year_correction_without_original_run(tmp_path):
    write_json(
        tmp_path / "results/main/faithfulness_year_ts.json",
        {"relation": "year", "best_faith": 0.25},
    )
    assert faithfulness_scores(tmp_path) == {"year": 0.25}


def test_paper_statistics_match_original_output():
    expected = Path(__file__).with_name("expected_statistics.txt").read_text()
    assert run_cli("statistics").stdout == expected


@pytest.mark.parametrize(
    "relation,count,misses,rho",
    [
        ("product_by_company", 99, 32, 0.407),
        ("superhero_archnemesis", 55, 37, 0.080),
        ("person_occupation", 100, 34, -0.066),
        ("person_university", 29, 19, -0.319),
    ],
)
def test_fact_rank_correlations(relation, count, misses, rho):
    points = fact_points(relation, "shared_split/jacobian_lens.json", "shared_split")
    assert len(points) == count
    assert np.count_nonzero(points[:, 1]) == misses
    assert spearmanr(*points.T).statistic == pytest.approx(rho, abs=0.0005)


def test_fact_join_rejects_mismatched_splits(monkeypatch):
    from jlens import paper_statistics

    results = {"relation": {"split_fingerprint": "a", "rows": []}}
    monkeypatch.setattr(
        paper_statistics,
        "load",
        lambda path: (
            results
            if path.endswith("lens.json")
            else {"split_fingerprint": "b", "ranks": {}}
        ),
    )
    with pytest.raises(ValueError, match="split fingerprints"):
        fact_points("relation", "lens.json", "ranks")


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_without_model_loading(command, tmp_path):
    result = run_cli(command, "--help", cwd=tmp_path)
    assert "usage:" in result.stdout


def test_commands_work_outside_checkout(tmp_path):
    output = tmp_path / "figures"
    run_cli("--output-dir", str(output), "build-figure-data", cwd=tmp_path)
    assert read_json(output / "paper_figure_data.json") == build_figure_data(ROOT)


def test_imports_do_not_execute_analyses(tmp_path):
    # No data exists under this root; importing modules must not try to read it.
    modules = sorted({module for module, _ in COMMANDS.values()})
    code = "import importlib; " + "; ".join(
        f"importlib.import_module('jlens.{module}')" for module in modules
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "JLENS_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    assert list(tmp_path.iterdir()) == []


def test_retained_assets_match_manifest():
    manifest = read_json(ROOT / "docs/asset_manifest.json")
    actual_paths = {
        str(path.relative_to(ROOT))
        for folder in (ROOT / "data", ROOT / "results")
        for path in folder.rglob("*.json")
    }
    assert actual_paths == set(manifest)
    for relative_path, digest in manifest.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == digest
