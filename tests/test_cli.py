from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from entity_lens import cli, scheduler

runner = CliRunner()


def test_help_lists_every_stage() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "download-data",
        "download-models",
        "evaluate-all",
        "experiments",
        "corpus-download",
        "corpus-scan",
        "wikipedia",
        "attribute-rates",
        "report",
        "manifest",
        "verify-artifacts",
    ):
        assert command in result.output


def test_manifest_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTITY_LENS_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("ENTITY_LENS_DATA_DIR", str(tmp_path / "absent-data"))
    (tmp_path / "result.json").write_text("{}")
    assert runner.invoke(cli.app, ["manifest"]).exit_code == 0
    assert runner.invoke(cli.app, ["verify-artifacts"]).exit_code == 0
    (tmp_path / "result.json").write_text("[]")
    assert runner.invoke(cli.app, ["verify-artifacts"]).exit_code != 0


def test_evaluate_all_schedules_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[scheduler.Job] = []
    monkeypatch.setattr(scheduler, "available_devices", lambda memory: [0, 1, 2, 3])
    monkeypatch.setattr(scheduler, "run_jobs", lambda jobs, log_dir, devices: captured.extend(jobs))
    assert runner.invoke(cli.app, ["evaluate-all"]).exit_code == 0
    assert len(captured) == 7
    assert {job.gpus for job in captured} == {1, 2}


def test_experiments_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[scheduler.Job] = []
    monkeypatch.setattr(scheduler, "available_devices", lambda memory: [0, 1, 2, 3])
    monkeypatch.setattr(scheduler, "run_jobs", lambda jobs, log_dir, devices: captured.extend(jobs))
    monkeypatch.setattr(cli.prominence, "evaluate", lambda project: None)
    monkeypatch.setattr(cli, "write_record", lambda path, record: None)
    assert runner.invoke(cli.app, ["experiments"]).exit_code == 0
    names = {job.name for job in captured}
    assert {"numeric_objects", "prompt_variants", "reference_sets", "swap", "prominence_0"} <= names
