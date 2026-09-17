"""Validate the GPU command plan without running the expensive experiments."""

from test_reproduction import run_cli

from jlens.config import ROOT, YEAR_RELATIONS
from jlens.workflow import measurement_plan


def test_plan_keeps_paper_protocols_and_year_corrections(tmp_path):
    relations = sorted(path.stem for path in (ROOT / "data/relations").glob("*.json"))
    stages = measurement_plan(tmp_path, relations)
    assert len(stages) == 96
    assert len({path for path, _, _ in stages}) == len(stages)
    for artifact, command, arguments in stages:
        assert artifact.is_relative_to(tmp_path)
        if command == "measure-faithfulness":
            if artifact.parent.name == "main":
                assert "--legacy-split" in arguments
                assert "--distinct-first-token" not in arguments
            else:
                assert "--distinct-first-token" in arguments
                assert "--legacy-split" not in arguments
    corrected = {
        arguments[arguments.index("--relation") + 1]
        for artifact, command, arguments in stages
        if command == "measure-faithfulness" and artifact.name.endswith("_ts.json")
    }
    assert corrected == YEAR_RELATIONS


def test_dry_run_has_no_side_effects(tmp_path):
    destination = tmp_path / "new-run"
    output = run_cli("rerun", "--run-dir", str(destination), "--dry-run").stdout
    assert "--legacy-split" in output
    assert "--trailing-space" in output
    assert "reproduce --supplementary" in output
    assert not destination.exists()
