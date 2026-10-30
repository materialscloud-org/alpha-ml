import json

import numpy as np
import pytest

from alphaml_api.core import MODELS
from alphaml_api.service import compute
from .conftest import GOLDEN

FILES = sorted(GOLDEN.glob("*.json"))


def _model_name(path):
    for name in MODELS:
        if path.stem.endswith("_" + name):
            return name
    raise ValueError(f"Cannot determine the model from {path.name}")


def _assert_close(actual, expected):
    """Relative to the size of the tensors: entries near zero differ from the
    legacy code by round-off (~1e-9 a.u.), not by 1e-6 of themselves."""
    expected = np.asarray(expected)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=1e-8 * np.abs(expected).max()
    )


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_golden(path, model_data):
    golden = json.loads(path.read_text())
    result = compute(golden["symbols"], golden["positions"], _model_name(path), model_data)

    _assert_close(result.atomic_polarizabilities, golden["atomic_polarizabilities"])
    _assert_close(result.molecular_polarizability, golden["molecular_polarizability"])
