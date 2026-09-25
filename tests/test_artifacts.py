from __future__ import annotations

from pathlib import Path

import pytest

from entity_lens import artifacts


@pytest.mark.parametrize("change", ["missing", "unexpected", "changed"])
def test_manifest_detects_changes(tmp_path: Path, change: str) -> None:
    target = tmp_path / "result.json"
    target.write_text("{}")
    artifacts.write_manifest(tmp_path)
    if change == "missing":
        target.unlink()
    elif change == "unexpected":
        (tmp_path / "extra.json").write_text("{}")
    else:
        target.write_text("[]")
    with pytest.raises(RuntimeError, match=change):
        artifacts.verify_manifest(tmp_path)


@pytest.mark.parametrize("entry", ["", "bad  result.json", "{hash}  ../result.json", "{hash}  /result.json"])
def test_manifest_rejects_invalid_entries(tmp_path: Path, entry: str) -> None:
    (tmp_path / artifacts.MANIFEST).write_text(entry.format(hash="0" * 64))
    with pytest.raises(ValueError, match=r"Invalid|must contain"):
        artifacts.read_manifest(tmp_path)


def test_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    entry = f"{'0' * 64}  result.json\n"
    (tmp_path / artifacts.MANIFEST).write_text(entry * 2)
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.read_manifest(tmp_path)
