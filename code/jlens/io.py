"""UTF-8 JSON helpers shared by measurements and analyses."""

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read a JSON document without leaving an open file handle."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Create parent directories and atomically replace a JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def relation_data(root: Path) -> dict[str, dict]:
    """Load the 28 relation files, preserving each file's sample order."""
    return {
        path.stem: read_json(path)
        for path in sorted((root / "data/relations").glob("*.json"))
    }
