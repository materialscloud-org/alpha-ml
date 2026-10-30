# alpha-ml backend

FastAPI service for machine-learned molecular polarizabilities. It is a
Python 3 / NumPy port of the legacy alpha-ml (`legacy/alpha-ml/webservice/`).

## Run

```bash
uv sync
uv run python scripts/build_tensors.py ../data ../tensors   # once: the model's tensors (~90 MB, 2 s)
export ALPHAML_DATA_DIR=../tensors
uv run uvicorn alphaml_api.main:app --reload            # http://localhost:8000
```

`ALPHAML_DATA_DIR` is the tensors folder, read once at startup (FastAPI
lifespan), built from the training data (`../data`, the unpacked data.tar.gz:
`natoms.npy`, `PStrain_*.npy`, `SPARSE_*.npy`, `*-weights/`) by
`scripts/build_tensors.py`. If it is unset the API still starts;
`/api/health` reports `model_loaded: false` and `/api/predict` returns 503.
Neither is in the repository.

Molecules of more than 256 atoms are predicted in chunks of atoms on
`ALPHAML_THREADS` threads (default: up to 4; about 2x faster with 4).

## The model as tensors

SA-GPR predicts with kernels between the molecule's power spectra and the
15,000 (λ = 0) and 1,500 (λ = 2) training environments, weighted. Both kernels
are low-order polynomials of the molecule's sparsified power spectrum, so the
sum over the training set collapses into two fixed tensors per model (an
800 × 800 quadratic form for λ = 0, 600 × 600 for `ccsd`, and an 800 × 800
complex bilinear form for λ = 2, which also takes in the sparsification
of its power spectrum); see `scripts/build_tensors.py`. They replace the
training arrays: about 90 MB instead of 411 MB, and the kernels no longer
cost anything. Checked against the training-set kernels on 729 molecules (the
MC3D structures with these elements as molecules, and displaced and rotated
copies of the examples), five models each: largest difference 1e-10 of the
largest polarizability entry.

## API

- `GET /api/health`
- `POST /api/predict`

```json
{
  "symbols": ["C", "H", "H", "H", "H"],
  "positions": [[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]],
  "model": "ccsd"
}
```

`model` is one of `ccsd`, `b3lyp`, `scan0`, `delta-b3lyp-ccsd`,
`delta-scan0-ccsd`. The response contains the 9 Cartesian components per atom
(row-major, a.u., in the input atom order), the 9-component molecular
polarizability (their sum) and `compute_time`. Downloads (xyz, pdb, ...) are
written by the frontend from the structure, with matsci-parse.

Rejected inputs return HTTP 422 with the legacy message: more than 2000 atoms,
elements other than H/C/N/O/S/Cl, or atoms closer than 0.5 Å.

## Tests

```bash
uv run pytest                          # API tests, with a stand-in model (no data needed)
ALPHAML_DATA_DIR=../tensors uv run pytest     # plus the model: legacy golden files
```

Golden files are read from `tests/golden/legacy/<Example>_<model>.json` and
are skipped when the folder is empty or `ALPHAML_DATA_DIR` is unset. They are
the outputs of the legacy app (its Celery task called directly, in the legacy
Docker image) for the six example molecules with each of the five models.

## What was ported

All in `alphaml_api/core.py`:

- `sagpr.py`: power spectra for λ = 0 and λ = 2, spherical harmonics, radial
  integrals, Wigner-3j coupling, sparsification matrices, both kernels,
  model weights and the spherical-to-Cartesian conversion.
- `test.pyx`: the `k0`/`k2` kernels, collapsed with the weights into the
  tensors of `scripts/build_tensors.py` (no Cython, no numba).
- `seekpath_app.py`: `predict_polarizabilities` for the five models (radial
  scaling for the λ = 0 part of `ccsd` only, as before).

The input limits and the overlapping-atom check of `seekpath_web_module.py`
are in `alphaml_api/service.py`, with the original messages.

The species order of the power-spectrum features is the Python 2 set order the
training data was built with (`C, Cl, H, O, N, S`), hard-coded in `core.py`.
The predictions agree with the legacy code (`tests/golden/legacy/`, the six
example molecules with each model) to 1e-6 relative to the largest tensor
entry.
