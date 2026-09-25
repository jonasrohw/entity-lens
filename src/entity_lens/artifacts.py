from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MANIFEST = "SHA256SUMS"
EXCLUDED = frozenset({"logs", MANIFEST})


def files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] not in EXCLUDED and not path.name.endswith(".tmp")
    )


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_manifest(root: Path) -> Path:
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in files(root)]
    target = root / MANIFEST
    target.write_text("\n".join(lines) + "\n")
    return target


def read_manifest(root: Path) -> dict[str, str]:
    entries = {}
    for line in (root / MANIFEST).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            message = "Invalid SHA256SUMS entry"
            raise ValueError(message)
        checksum, name = match.groups()
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name in entries:
            message = f"Invalid or duplicate artifact path: {name}"
            raise ValueError(message)
        entries[name] = checksum
    if not entries:
        message = "SHA256SUMS must contain at least one artifact"
        raise ValueError(message)
    return entries


def verify_manifest(root: Path) -> None:
    expected = read_manifest(root)
    actual = {path.relative_to(root).as_posix(): digest(path) for path in files(root)}
    problems = {
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "changed": sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name]),
    }
    if any(problems.values()):
        message = json.dumps(problems, indent=1)
        raise RuntimeError(message)
