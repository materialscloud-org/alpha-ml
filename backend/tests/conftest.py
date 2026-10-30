import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "tests" / "golden" / "legacy"
DATA_DIR = os.environ.get("ALPHAML_DATA_DIR")


class FakeModel:
    """Deterministic stand-in so the API tests run without the model data."""

    def predict(self, symbols, positions, model_name):
        n = len(symbols)
        return np.tile(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)
        )


@pytest.fixture(scope="session")
def model_data():
    if not DATA_DIR:
        pytest.skip("ALPHAML_DATA_DIR is not set")
    from alphaml_api.core import ModelData

    return ModelData(DATA_DIR)


@pytest.fixture
def golden_files():
    if not DATA_DIR:
        pytest.skip("ALPHAML_DATA_DIR is not set")
    files = sorted(GOLDEN.glob("*.json"))
    if not files:
        pytest.skip(f"no golden files in {GOLDEN}")
    return files
