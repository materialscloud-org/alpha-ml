"""The /predict pipeline: the legacy input checks and the model."""

from __future__ import annotations

import time

import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from scipy.spatial.distance import pdist

from .schemas import PredictResponse

MAX_NUMBER_OF_ATOMS = 2000
ALLOWED_NUMBERS = {1, 6, 7, 8, 16, 17}  # H, C, N, O, S, Cl
MIN_DISTANCE = 0.5  # Å


class InputError(ValueError):
    """The input is rejected; the message is shown to the user."""


def _numbers(symbols):
    numbers = []
    for symbol in symbols:
        try:
            numbers.append(atomic_numbers[symbol])
        except KeyError as exc:
            raise InputError(f"Unknown element symbol '{symbol}'") from exc
    return numbers


def validate(symbols, positions):
    """The checks of the legacy ``test_frame``, in the same order/messages."""
    natoms = len(symbols)
    if natoms > MAX_NUMBER_OF_ATOMS:
        raise InputError(
            "Sorry, this online app is limited to {} atoms "
            "in the input cell, while your structure has {} atoms."
            "".format(MAX_NUMBER_OF_ATOMS, natoms)
        )

    numbers = _numbers(symbols)
    if set(numbers) - ALLOWED_NUMBERS:
        species = np.unique(
            [chemical_symbols[z] for z in np.unique(numbers).tolist()]
        )
        raise InputError(
            "Sorry, AlphaML is limited to the predictions of "
            " polarizabilities of molecules containing H, C, N, O, S, Cl."
            "The atomic species you provided are {}"
            "".format(species)
        )

    if natoms > 1:
        dist = pdist(np.asarray(positions, dtype=float))
        if dist[dist < MIN_DISTANCE].size > 0:
            raise InputError(
                "There are some overlapping atoms. Please check your input file."
            )


def compute(symbols, positions, model_name, model) -> PredictResponse:
    start = time.time()
    validate(symbols, positions)

    pols = np.asarray(model.predict(symbols, positions, model_name), dtype=float)
    molecular = pols.sum(axis=0)

    return PredictResponse(
        atomic_polarizabilities=pols.tolist(),
        molecular_polarizability=molecular.tolist(),
        compute_time=time.time() - start,
    )
