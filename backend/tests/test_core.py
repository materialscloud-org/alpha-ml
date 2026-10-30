import json

import numpy as np

from alphaml_api import config, core
from .conftest import GOLDEN


def test_isolated_atom_is_isotropic(model_data):
    """An atom without neighbours (here a Cl 20 Å from a water molecule) has
    no lambda = 2 power spectrum: an isotropic tensor, not NaN."""
    symbols = ["O", "H", "H", "Cl"]
    positions = [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [20, 0, 0]]
    for model in core.MODELS:
        pols = model_data.predict(symbols, positions, model)
        assert np.isfinite(pols).all()
        cl = pols[3].reshape(3, 3)
        np.testing.assert_allclose(cl, cl[0, 0] * np.eye(3), atol=1e-12)


def test_threads_predict_the_same(model_data, monkeypatch):
    """Large molecules are split in chunks of atoms on several threads: the
    same result as in one go (here caffeine, in chunks of 4 atoms)."""
    golden = json.loads((GOLDEN / "Caffeine_ccsd.json").read_text())
    serial = model_data.predict(golden["symbols"], golden["positions"], "ccsd")
    monkeypatch.setattr(config, "PARALLEL_ATOMS", 8)
    monkeypatch.setattr(config, "ATOM_CHUNK", 4)
    monkeypatch.setattr(config, "THREADS", 3)
    threaded = model_data.predict(golden["symbols"], golden["positions"], "ccsd")
    np.testing.assert_allclose(threaded, serial, rtol=0, atol=1e-12 * np.abs(serial).max())
