from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "reproduce.sh"
    shutil.copy2(ROOT / "reproduce.sh", script)
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--version" ]]; then echo "uv ${TEST_UV_VERSION:-0.9.27}"; exit; fi\n'
        'echo "$*" >> "$TEST_LOG"\n'
        'echo "${TEST_SETTING:-}" >> "$TEST_VALUES"\n'
        'if [[ "$*" == *"${TEST_FAIL_STAGE:-never-fail}"* ]]; then exit 1; fi\n'
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_LOG", str(tmp_path / "calls"))
    monkeypatch.setenv("TEST_VALUES", str(tmp_path / "values"))
    monkeypatch.delenv("ENV_FILE", raising=False)
    return script


def run(script: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script), mode], cwd=script.parent, capture_output=True, text=True, check=False)


@pytest.mark.parametrize(("mode", "code"), [("--help", 0), ("invalid", 2)])
def test_usage_does_not_install_or_load_environment(launcher: Path, mode: str, code: int) -> None:
    (launcher.parent / ".env").write_text("invalid environment syntax\n")
    assert run(launcher, mode).returncode == code
    assert not (launcher.parent / "calls").exists()


def test_analysis_stops_at_failed_integrity_check(launcher: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_FAIL_STAGE", "verify-artifacts")
    assert run(launcher, "analysis").returncode != 0
    calls = (launcher.parent / "calls").read_text()
    assert "verify-artifacts" in calls
    assert "download-data" not in calls
    assert "report" not in calls
    assert "manifest" not in calls


def test_analysis_checks_before_and_after_report(launcher: Path) -> None:
    assert run(launcher, "analysis").returncode == 0
    assert (launcher.parent / "calls").read_text().splitlines() == [
        "sync --locked",
        "run --locked entity-lens verify-artifacts",
        "run --locked entity-lens download-data",
        "run --locked entity-lens report",
        "run --locked entity-lens verify-artifacts",
    ]


def test_environment_is_literal_and_accepts_comments(launcher: Path) -> None:
    value = "$(touch should-not-exist) with spaces"
    (launcher.parent / ".env").write_bytes(f'# local settings\r\n\r\nTEST_SETTING="{value}"'.encode())
    assert run(launcher, "analysis").returncode == 0
    assert set((launcher.parent / "values").read_text().splitlines()) == {value}
    assert not (launcher.parent / "should-not-exist").exists()


def test_exported_environment_takes_precedence(launcher: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_SETTING", "exported")
    (launcher.parent / ".env").write_text("TEST_SETTING=local\n")
    assert run(launcher, "analysis").returncode == 0
    assert set((launcher.parent / "values").read_text().splitlines()) == {"exported"}


def test_missing_explicit_environment_fails(launcher: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_FILE", str(launcher.parent / "absent"))
    assert run(launcher, "analysis").returncode != 0
    assert not (launcher.parent / "calls").exists()


def test_wrong_uv_version_fails(launcher: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_UV_VERSION", "0.0.1")
    assert run(launcher, "check").returncode != 0
    assert not (launcher.parent / "calls").exists()
