"""Portable paths and shared experiment constants.

The CLI sets these environment variables before importing analysis modules.
The retained datasets are repository assets, not part of the Python wheel.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("JLENS_ROOT", Path(__file__).resolve().parents[1])).resolve()
OUTPUT_DIR = Path(os.environ.get("JLENS_OUTPUT_DIR", ROOT / "generated")).resolve()
MODEL_ID = os.environ.get("JLENS_MODEL", "allenai/Olmo-3-1025-7B")
NEUTRAL_PROMPT = "{} is known for"
YEAR_RELATIONS = frozenset({"presidents_birth_year", "presidents_election_year"})

# Layer choices below were tuned to OLMo-3-1025-7B's 32-layer depth. A model
# with a different depth needs its own choices, not these literals; override
# via JLENS_LENS_LAYER / JLENS_LRE_LAYERS rather than editing in place.
LENS_LAYER = int(os.environ.get("JLENS_LENS_LAYER", 17))
LRE_LAYERS = os.environ.get("JLENS_LRE_LAYERS", "5,11,14,17,20,23")
