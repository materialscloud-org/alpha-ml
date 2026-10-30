"""The model as fixed tensors, from the training data: run once, the backend
loads the folder it writes (ALPHAML_DATA_DIR).

    uv run python scripts/build_tensors.py <training data> <tensors folder>

where the training data is data.tar.gz unpacked (natoms.npy, PStrain_*.npy,
SPARSE_*.npy, *-weights/, 411 MB) and the tensors folder gets one .npy per
array (about 90 MB).

SA-GPR predicts with kernels between the molecule's power spectra and the
training set's, weighted. Both kernels are low-order polynomials in the
molecule's (sparsified) power spectrum x, so the sum over the training set
collapses into one tensor per model:

- lambda = 0: out0 = sum_i w_i / n_i (Re(p_i . conj x))^2 + mean = u^T M u + mean,
  u = Re x, M = sum_i (w_i / n_i) a_i a_i^T, a_i = Re p_i (the training
  spectra are real: imaginary parts below 1e-9 of the real ones);
- lambda = 2: out2[q] = Re sum_n CT[n, q] conj(x0)^T G conj(x2[n]),
  G = sum_i p0_i (sum_m u_i[m] p2_i[m])^T, u_i = (w2_i / n_i) @ conj(CR).

The sparsification x = PS A (A: SPARSE_Amat*, PS: the power spectrum's
features of SPARSE_fps*) is linear, so that of lambda = 2 (5 rows per atom)
is folded in: G <- G conj(A2)^T, and the model takes its features directly.
Not those of lambda = 0, kept as matrices: their entries reach 1e8, and
A M A^T loses 1e-6 of the result to rounding (A2: 3e3, no loss).

Checked against the training-set kernels (the legacy formulation) on 729
molecules, five models each: largest difference 1e-10 of the largest
polarizability entry.
"""

import sys
import time
from pathlib import Path

import numpy as np

from alphaml_api.core import MODELS, TENSORS_FORMAT, complex_to_real_transformation

# The features of the power spectra the model takes: (lambda, radial scaling)
FPS = {"fps0": "SPARSE_fps0.npy", "fps0_rs": "SPARSE_fps0_RS.npy", "fps2": "SPARSE_fps2.npy"}
# The sparsification of lambda = 0 (with and without radial scaling)
AMAT = {"amat0": "SPARSE_Amat0.npy", "amat0_rs": "SPARSE_Amat0_RS.npy"}


def build(data_dir):
    """The tensors of every model, and the arrays of the power spectra, from
    the training data in `data_dir`."""
    data = Path(data_dir)
    read = lambda name: np.asarray(np.load(data / name, mmap_mode="r"))  # noqa: E731
    arrays = {key: read(name) for key, name in {**FPS, **AMAT}.items()}
    arrays["format"] = np.asarray(TENSORS_FORMAT)
    natoms = np.load(data / "natoms.npy").astype(float)
    cc = np.conj(complex_to_real_transformation(2))
    amat2c = np.conj(read("SPARSE_Amat2.npy"))

    p0z = read("PStrain_lam0_zeta2_lam2.npy")[:, 0, :]
    p2 = read("PStrain_lam2.npy")[:, 0]
    for model in MODELS:
        suffix = "_RS" if model == "ccsd" else ""  # radial scaling (lambda = 0 only)
        weights = data / f"{model}-weights"
        a = read(f"PStrain_lam0{suffix}.npy")[:, 0, :].real
        w0 = np.load(weights / f"weights_lm0{suffix}.npy")[:, 0] / natoms[: len(a)]
        w2 = np.load(weights / "weights_lm2.npy").reshape(-1, 5) / natoms[: len(p0z), None]
        u = w2 @ cc
        arrays[f"{model}_M"] = (a.T * w0) @ a
        arrays[f"{model}_G"] = p0z.T @ np.einsum("im,imd->id", u, p2) @ amat2c.T
        arrays[f"{model}_mean"] = np.asarray(
            float(np.load(weights / f"weights_lm0_mean{suffix}.npy"))
        )
    return arrays


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/build_tensors.py <training data> <tensors folder>")
    start = time.time()
    arrays = build(sys.argv[1])
    folder = Path(sys.argv[2])
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        np.save(folder / f"{key}.npy", value)
    size = sum(v.nbytes for v in arrays.values()) / 1e6
    print(f"{len(arrays)} arrays, {size:.0f} MB, in {folder} ({time.time() - start:.1f} s)")
