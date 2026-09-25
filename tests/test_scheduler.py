from __future__ import annotations

import sys
from pathlib import Path

import pytest

from entity_lens import scheduler


def job(name: str, code: str, gpus: int = 1) -> scheduler.Job:
    return scheduler.Job(name=name, command=(sys.executable, "-c", code), gpus=gpus)


def test_jobs_receive_distinct_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)
    code = "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
    jobs = [job("wide", code, gpus=2), job("a", code), job("b", code)]
    scheduler.run_jobs(jobs, tmp_path, [4, 5])
    assert (tmp_path / "wide.log").read_text().strip() == "4,5"
    assert (tmp_path / "a.log").read_text().strip() in {"4", "5"}


def test_failures_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="bad"):
        scheduler.run_jobs([job("bad", "raise SystemExit(3)"), job("good", "pass")], tmp_path, [0])


def test_oversized_job_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="more GPUs"):
        scheduler.run_jobs([job("wide", "pass", gpus=3)], tmp_path, [0])


def test_available_devices_filters_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        stdout = "0, 10\n1, 50000\n2, 0\n"

    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: Result())
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert scheduler.available_devices(1024) == [0, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    assert scheduler.available_devices(1024) == [2]


def test_cli_command_uses_module() -> None:
    assert scheduler.cli_command("report")[1:] == ("-m", "entity_lens", "report")
