from __future__ import annotations

from pathlib import Path

import pytest

from entity_lens import artifacts
from entity_lens.records import read, sha256, write
from entity_lens.statistics import Correlation


def test_gzip_records_are_deterministic(tmp_path: Path) -> None:
    record = Correlation(rho=0.5, p=0.01, n=10)
    first, second = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    write(first, record)
    write(second, record)
    assert first.read_bytes() == second.read_bytes()
    assert read(first, Correlation) == record
    plain = tmp_path / "nested" / "c.json"
    write(plain, record)
    assert read(plain, Correlation) == record
    assert len(sha256(plain)) == 64
    assert not list(tmp_path.rglob("*.tmp"))


def test_manifest_round_trip(tmp_path: Path) -> None:
    (tmp_path / "measurements").mkdir()
    (tmp_path / "measurements" / "x.json").write_text("{}")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "run.log").write_text("ignored")
    manifest = artifacts.write_manifest(tmp_path)
    assert "measurements/x.json" in manifest.read_text()
    assert "run.log" not in manifest.read_text()
    artifacts.verify_manifest(tmp_path)
    (tmp_path / "measurements" / "x.json").write_text("{ }")
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"extra\.json") as error:
        artifacts.verify_manifest(tmp_path)
    assert "changed" in str(error.value)
