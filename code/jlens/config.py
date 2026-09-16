"""Portable paths and shared experiment constants.

The CLI sets these environment variables before importing analysis modules.
The retained datasets are repository assets, not part of the Python wheel.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("JLENS_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUT_DIR = Path(os.environ.get("JLENS_OUTPUT_DIR", ROOT / "generated")).resolve()
MODEL_ID = "allenai/Olmo-3-1025-7B"
NEUTRAL_PROMPT = "{} is known for"
YEAR_RELATIONS = frozenset({"presidents_birth_year", "presidents_election_year"})
