"""The backend's settings, from environment variables (read once, at import)."""

from __future__ import annotations

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# The model: the tensors folder built by scripts/build_tensors.py. Unset: the
# API starts without it (/api/health: model_loaded false, /api/predict: 503).
DATA_DIR = os.environ.get("ALPHAML_DATA_DIR")

# The built frontend, served at / when the folder exists
FRONTEND_DIST = Path(
    os.environ.get("FRONTEND_DIST", Path(__file__).parents[2] / "frontend" / "dist")
)

# Molecules of more than PARALLEL_ATOMS atoms are predicted on THREADS threads
# (about 2x faster with 4, but each thread holds a chunk of atoms: more memory)
THREADS = _int("ALPHAML_THREADS", 1)
PARALLEL_ATOMS = _int("ALPHAML_PARALLEL_ATOMS", 256)

# Working memory of a prediction: atoms per chunk (each atom's prediction
# depends on its own neighbours only), pairs per sum, and array elements per
# step. The defaults: about 380 MB peak for the largest molecules (2000 atoms),
# 280 MB with ALPHAML_ATOM_CHUNK=64 (5% slower).
ATOM_CHUNK = _int("ALPHAML_ATOM_CHUNK", 128)
PAIR_CHUNK = _int("ALPHAML_PAIR_CHUNK", 20000)
ELEMENT_CHUNK = _int("ALPHAML_ELEMENT_CHUNK", 4_000_000)
