"""Python 3 / NumPy port of the legacy SA-GPR model.

Ported from ``legacy/alpha-ml/webservice/sagpr.py`` and ``test.pyx`` (the
Cython kernels are replaced by vectorised NumPy).  The algorithm, constants
and model choices are kept unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import special
from sympy.physics.wigner import wigner_3j

ATOM_VALENCE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30,
}
ATOM_SYMBOLS = {value: key for key, value in ATOM_VALENCE.items()}

# The legacy code obtained these orders from Python 2 sets:
#   set([6, 17, 1, 7, 8, 16])           -> [1, 6, 7, 8, 16, 17]  (centres)
#   set(["H", "C", "N", "O", "S", "Cl"]) -> ["C", "Cl", "H", "O", "N", "S"]
# The second order fixes the (s, d) feature layout of the training data.
CENTER_SYMBOLS = ["H", "C", "N", "O", "S", "Cl"]
SPECIES_SYMBOLS = ["C", "Cl", "H", "O", "N", "S"]

NSMAX = 26
MODELS = ("ccsd", "b3lyp", "scan0", "delta-b3lyp-ccsd", "delta-scan0-ccsd")


def _sph_harm(m, n, azimuth, polar):
    """Legacy ``scipy.special.sph_harm`` argument order on old/new SciPy."""
    if hasattr(special, "sph_harm"):
        return special.sph_harm(m, n, azimuth, polar)
    # SciPy >= 1.15: sph_harm_y(n, m, theta, phi), theta is the polar angle.
    return special.sph_harm_y(n, m, polar, azimuth)


def complex_to_real_transformation(lval):
    """Transformation matrix from complex to real spherical harmonics."""
    if lval == 0:
        return 1.0
    if lval == 1:
        return np.array(
            [[1.0j, 0.0, 1.0j],
             [0.0, np.sqrt(2.0), 0.0],
             [1.0, 0.0, -1.0]],
            dtype=complex,
        ) / np.sqrt(2.0)
    if lval == 2:
        return np.array(
            [[1.0j, 0.0, 0.0, 0.0, -1.0j],
             [0.0, 1.0j, 0.0, 1.0j, 0.0],
             [0.0, 0.0, np.sqrt(2.0), 0.0, 0.0],
             [0.0, 1.0, 0.0, -1.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=complex,
        ) / np.sqrt(2.0)
    raise ValueError(f"Unsupported l value {lval}")


def radialscaling(r, c, r0, m):
    if m == 0:
        return 1.0
    if c == 0.0:
        return 1.0 / (r / r0) ** m
    return c / (c + (r / r0) ** m)


def setup_orthomatrix(nmax, rc, rk):
    sigma = np.zeros(nmax, dtype=float)
    dsigma = 0.2 * rc / float(nmax)
    sigma[0] = dsigma
    for i in range(1, nmax):
        sigma[i] = sigma[i - 1] + dsigma

    overlap = np.zeros((nmax, nmax), dtype=float)
    for n1 in range(nmax):
        for n2 in range(nmax):
            overlap[n1, n2] = (
                (0.5 / sigma[n1] ** 2 + 0.5 / sigma[n2] ** 2) ** (-0.5 * (3.0 + n1 + n2))
                / (sigma[n1] ** n1 * sigma[n2] ** n2)
                * special.gamma(0.5 * (3.0 + n1 + n2))
                / (
                    (sigma[n1] * sigma[n2]) ** 1.5
                    * np.sqrt(special.gamma(1.5 + n1) * special.gamma(1.5 + n2))
                )
            )

    eigenvalues, unitary = np.linalg.eig(overlap)
    diagoverlap = np.diag(np.sqrt(eigenvalues))
    newoverlap = np.dot(np.conj(unitary), np.dot(diagoverlap, unitary.T))
    return [np.linalg.inv(newoverlap), sigma]


@lru_cache(maxsize=None)
def get_w3j(lam, lmax):
    w3j = np.zeros((2 * lam + 1, lmax + 1, lmax + 1, 2 * lmax + 1), dtype=float)
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for m in range(2 * l1 + 1):
                for mu in range(2 * lam + 1):
                    w3j[mu, l1, l2, m] = float(
                        wigner_3j(lam, l2, l1, mu - lam, m - l1 - mu + lam, -m + l1)
                    ) * (-1.0) ** (m - l1)
    return w3j


def _species_layout(symbols):
    all_species = [ATOM_VALENCE[name] for name in SPECIES_SYMBOLS]
    atom_indexes = [[] for _ in range(NSMAX)]
    for ii, name in enumerate(symbols):
        for j in all_species:
            if name == ATOM_SYMBOLS[j]:
                atom_indexes[j].append(ii)
    centers = [ATOM_VALENCE[name] for name in CENTER_SYMBOLS]
    return all_species, atom_indexes, centers


def fill_arrays(
    nspecies, nnmax, lmax, nmax, centers, atom_indexes, all_species, coords,
    rc, rk, cw, sigma, sg, orthomatrix, rad_scal,
):
    alpha = 1.0 / (2 * sg**2)
    sg2 = sg**2

    nat = 0
    for k in centers:
        nat += len(atom_indexes[k])

    length = np.zeros((nat, nspecies, nnmax), dtype=float)
    efact = np.zeros((nat, nspecies, nnmax), dtype=float)
    nneigh = np.zeros((nat, nspecies), dtype=int)

    omega = np.zeros((nat, nspecies, nmax, lmax + 1, 2 * lmax + 1), dtype=complex)
    harmonic = np.zeros((nat, nspecies, lmax + 1, 2 * lmax + 1, nnmax), dtype=complex)
    orthoradint = np.zeros((nat, nspecies, lmax + 1, nmax, nnmax), dtype=float)
    radint = np.zeros((nat, nspecies, nnmax, lmax + 1, nmax), dtype=float)
    nat = 0

    def switch(r):
        if r <= rc - rk:
            return 1.0
        if rc - rk < r < rc:
            return (1.0 + np.cos(np.pi * (r - (rc - rk)) / rk)) / 2.0
        return 0.0

    iat = 0
    for k in centers:
        nat += len(atom_indexes[k])
        for l in atom_indexes[k]:
            ispe = 0
            for ix in all_species:
                n = 0
                for m in atom_indexes[ix]:
                    rr = coords[m] - coords[l]
                    if np.dot(rr, rr) <= rc**2:
                        if m == l:
                            length[iat, ispe, n] = 0.0
                            efact[iat, ispe, n] = cw
                            harmonic[iat, ispe, 0, 0, n] = _sph_harm(0, 0, 0, 0)
                        else:
                            length[iat, ispe, n] = np.linalg.norm(rr)
                            theta = np.arccos(rr[2] / length[iat, ispe, n])
                            phi = np.arctan2(rr[1], rr[0])
                            efact[iat, ispe, n] = (
                                np.exp(-alpha * length[iat, ispe, n] ** 2)
                                * radialscaling(
                                    length[iat, ispe, n],
                                    rad_scal[0], rad_scal[1], rad_scal[2],
                                )
                            )
                            efact[iat, ispe, n] *= switch(length[iat, ispe, n])
                            for lval in range(lmax + 1):
                                mxrange = np.arange(-lval, lval + 1)
                                harmonic[iat, ispe, lval, : 2 * lval + 1, n] = np.conj(
                                    _sph_harm(mxrange, lval, phi, theta)
                                )
                        nneigh[iat, ispe] += 1
                        n += 1
                ispe += 1
            iat += 1

    for n in range(nmax):
        normfact = np.sqrt(
            2.0 / (special.gamma(1.5 + n) * sigma[n] ** (3.0 + 2.0 * n))
        )
        sigmafact = (sg2**2 + sg2 * sigma[n] ** 2) / sigma[n] ** 2
        for l in range(lmax + 1):
            radint[:, :, :, l, n] = (
                efact[:, :, :]
                * 2.0 ** (-0.5 * (1.0 + l - n))
                * (1.0 / sg2 + 1.0 / sigma[n] ** 2) ** (-0.5 * (3.0 + l + n))
                * special.gamma(0.5 * (3.0 + l + n)) / special.gamma(1.5 + l)
                * (length[:, :, :] / sg2) ** l
                * special.hyp1f1(
                    0.5 * (3.0 + l + n),
                    1.5 + l,
                    0.5 * length[:, :, :] ** 2 / sigmafact,
                )
            )
        radint[:, :, :, :, n] *= normfact

    for iat in range(nat):
        for ispe in range(nspecies):
            for neigh in range(nneigh[iat, ispe]):
                for l in range(lmax + 1):
                    orthoradint[iat, ispe, l, :, neigh] = np.dot(
                        orthomatrix, radint[iat, ispe, neigh, l]
                    ).real

    for iat in range(nat):
        for ispe in range(nspecies):
            omega[iat, ispe] = np.einsum(
                "lnh,lmh->nlm", orthoradint[iat, ispe], harmonic[iat, ispe]
            )

    return [nat, omega, harmonic, orthoradint]


def get_power_spectrum(lam, symbols, coords, rad_scal, use_radial_scaling, data):
    all_species, atom_indexes, centers = _species_layout(symbols)
    nspecies = len(SPECIES_SYMBOLS)
    natmax = len(symbols)

    sparse_options = data.sparse_options(lam, use_radial_scaling)

    params = dict(lmax=6, nmax=8, sg=0.3, cw=1.0, rk=1e-10, rc=4.0, nnmax=35)
    params_rc = dict(lmax=6, nmax=8, sg=0.3, cw=1.0, rk=1e-10, rc=5.0, nnmax=35)
    p = params_rc if use_radial_scaling else params
    lmax, nmax, sg = p["lmax"], p["nmax"], p["sg"]
    cw, rk, rc, nnmax = p["cw"], p["rk"], p["rc"], p["nnmax"]

    llmax = 0
    lvalues = {}
    if lam > 0:
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                if abs(l2 - lam) <= l1 <= (l2 + lam):
                    lvalues[llmax] = [l1, l2]
                    llmax += 1

    orthomatrix, sigma = setup_orthomatrix(nmax, rc, rk)

    if lam == 0:
        featsize = nspecies * nspecies * nmax**2 * (lmax + 1)
        do_list = np.array(sparse_options[1], dtype=int)
        PS = np.zeros((1, natmax, len(do_list)), dtype=complex)

        nat, omega, harmonic, orthoradint = fill_arrays(
            nspecies, nnmax, lmax, nmax, centers, atom_indexes, all_species,
            coords, rc, rk, cw, sigma, sg, orthomatrix, rad_scal,
        )

        omegatrue = np.zeros((nat, nspecies, nmax, lmax + 1, 2 * lmax + 1), complex)
        omegaconj = np.zeros((nat, nspecies, nmax, lmax + 1, 2 * lmax + 1), complex)
        for l in range(lmax + 1):
            scale = np.sqrt(np.sqrt(2 * l + 1))
            omegatrue[:, :, :, l, :] = omega[:, :, :, l, :] / scale
            omegaconj[:, :, :, l, :] = np.conj(omega[:, :, :, l, :]) / scale

        power = np.einsum("asblm,adnlm->asdbnl", omegatrue, omegaconj)
        power = power.reshape(nat, featsize)
        for iat in range(nat):
            inner = np.real(np.dot(power[iat], np.conj(power[iat])))
            power[iat] /= np.sqrt(inner)

        PS[0, :nat] = power[:, do_list]

    else:
        w3j = get_w3j(lam, lmax)

        featsize = nspecies * nspecies * nmax**2 * llmax
        do_list = np.array(sparse_options[1], dtype=int)
        PS = np.zeros((1, natmax, 2 * lam + 1, len(do_list)), dtype=complex)

        nat, omega, harmonic, orthoradint = fill_arrays(
            nspecies, nnmax, lmax, nmax, centers, atom_indexes, all_species,
            coords, rc, rk, cw, sigma, sg, orthomatrix, rad_scal,
        )
        power = np.zeros(
            (nat, 2 * lam + 1, nspecies, nspecies, nmax, nmax, lmax + 1, lmax + 1),
            complex,
        )
        p2 = np.zeros((nat, 2 * lam + 1, nspecies, nspecies, nmax, nmax, llmax), complex)
        omegaconj = np.zeros(
            (nat, nspecies, 2 * lam + 1, nmax, lmax + 1, lmax + 1, 2 * lmax + 1),
            complex,
        )
        harmconj = np.zeros(
            (nat, nspecies, lmax + 1, lmax + 1, 2 * lmax + 1, 2 * lam + 1, nnmax),
            dtype=complex,
        )

        for lval in range(lmax + 1):
            for im in range(2 * lval + 1):
                for lval2 in range(lmax + 1):
                    for mu in range(2 * lam + 1):
                        if abs(im - lval - mu + lam) <= lval2:
                            harmconj[:, :, lval2, lval, im, mu, :] = np.conj(
                                harmonic[:, :, lval2, lval2 + im - lval - mu + lam, :]
                            )
        for iat in range(nat):
            for ispe in range(nspecies):
                omegaconj[iat, ispe] = np.einsum(
                    "lnh,lkmvh->vnklm", orthoradint[iat, ispe], harmconj[iat, ispe]
                )
        for iat in range(nat):
            for ia in range(nspecies):
                for ib in range(nspecies):
                    power[iat, :, ia, ib] = np.einsum(
                        "nlv,xmlkv,xlkv->xnmlk",
                        omega[iat, ia],
                        omegaconj[iat, ib],
                        w3j,
                    )

        for l in range(llmax):
            p2[:, :, :, :, :, :, l] = power[:, :, :, :, :, :, lvalues[l][0], lvalues[l][1]]
        p2 = p2.reshape(nat, 2 * lam + 1, featsize)
        for iat in range(nat):
            inner = np.zeros((2 * lam + 1, 2 * lam + 1), complex)
            for mu in range(2 * lam + 1):
                for nu in range(2 * lam + 1):
                    inner[mu, nu] = np.dot(p2[iat, mu], np.conj(p2[iat, nu]))
            p2[iat] /= np.sqrt(np.real(np.linalg.norm(inner)))

        PS[0, :nat] = p2[:, :, do_list]

    PS = np.dot(PS, sparse_options[0])
    return [PS, featsize]


def split_power_spectra(PS, natom):
    ps0_new = np.zeros((natom, 1, len(PS[0][0, 0])), dtype=complex)
    ps02_new = np.zeros((natom, 1, len(PS[1][0, 0])), dtype=complex)
    ps2_new = np.zeros((natom, 1, 5, len(PS[2][0, 0, 0])), dtype=complex)
    for k in range(natom):
        ps0_new[k, 0, :] = PS[0][0, k, :]
        ps02_new[k, 0, :] = PS[1][0, k, :]
        ps2_new[k, 0, :, :] = PS[2][0, k, :, :]
    return [ps0_new, ps02_new, ps2_new]


def get_kernel(lam, PS, natom, PS0, use_radial_scaling, data):
    nat1 = data.natoms
    if lam == 0:
        pstrain = data.pstrain(0, use_radial_scaling)
        ntrain = len(pstrain)
        kreal = (pstrain[:, 0, :] @ PS[:, 0, :].conj().T).real**2
        return kreal / nat1[:ntrain, None]

    pstrain0 = data.pstrain_zeta(use_radial_scaling)
    pstrain = data.pstrain(2, use_radial_scaling)
    ntrain = len(pstrain0)

    klocal0 = pstrain0[:, 0, :] @ PS0[:, 0, :].conj().T
    b = np.einsum(
        "imk,jnk->mnij", pstrain[:, 0, :, :], PS[:, 0, :, :].conj()
    )

    CR = complex_to_real_transformation(lam)
    CC = np.conj(CR)
    CT = np.transpose(CR)
    kreal = np.einsum("pm,mnij,nq->ijpq", CC, b, CT) * klocal0[:, :, None, None]
    return (np.real(kreal) / nat1[:ntrain, None, None, None]).astype(float)


def get_pol(kernels, type_str, use_radial_scaling, data):
    weights0 = data.weights(type_str, "lm0")
    mean0 = data.weights(type_str, "lm0_mean")
    weights2 = data.weights(type_str, "lm2")

    output0 = (np.dot(weights0.T, kernels[0]) + mean0)[0]
    output2 = np.dot(
        weights2.T,
        kernels[1].transpose(0, 2, 1, 3).reshape(
            len(kernels[1]) * len(kernels[1][0, 0, 0]),
            len(kernels[1][0]) * len(kernels[1][0, 0]),
        ),
    )

    outvec = [
        [np.array(output0[i]), output2[i * 5 : (i + 1) * 5]]
        for i in range(len(kernels[1][0]))
    ]
    return [spherical_to_cartesian(vec) for vec in outvec]


def spherical_to_cartesian(vector):
    mask1 = [1.0, 2.0**0.5, 2.0**0.5, 1.0, 2.0**0.5, 1.0]
    mask2 = [[0], [1, 3], [2, 6], [4], [5, 7], [8]]

    sizes = [1, 5]
    cs = get_CS_matrix(2)
    for i in range(len(mask1)):
        cs[i] = cs[i] * mask1[i]
    cr = [complex_to_real_transformation(sz) for sz in [0, 2]]

    outvecs = [vector[j].reshape((1, sizes[j])) for j in range(len(sizes))]
    sphe = np.concatenate(
        [np.dot(np.conj(cr[j]).T, outvecs[j][0]) for j in range(len(sizes))]
    )
    cart = np.real(np.dot(sphe, np.conj(cs).T))

    crt = np.zeros(len(np.concatenate(mask2)), dtype=float)
    for j in range(len(mask2)):
        for k in mask2[j]:
            crt[k] = cart[j] / mask1[j]
    return crt


def get_CS_matrix(rank):
    if rank == 0:
        return np.array([1.0], dtype=complex)
    if rank == 1:
        return np.array(
            [[1.0, 0.0, -1.0], [-1.0j, 0.0, -1.0j], [0.0, np.sqrt(2.0), 0.0]],
            dtype=complex,
        ) / np.sqrt(2.0)
    if rank == 2:
        return np.array(
            [[-1.0 / np.sqrt(3.0), 0.5, 0.0, -1.0 / np.sqrt(6.0), 0.0, 0.5],
             [0.0, -0.5j, 0.0, 0.0, 0.0, 0.5j],
             [0.0, 0.0, 0.5, 0.0, -0.5, 0.0],
             [-1.0 / np.sqrt(3.0), -0.5, 0.0, -1.0 / np.sqrt(6.0), 0.0, -0.5],
             [0.0, 0.0, -0.5j, 0.0, -0.5j, 0.0],
             [-1.0 / np.sqrt(3.0), 0.0, 0.0, 2.0 / np.sqrt(6.0), 0.0, 0.0]],
            dtype=complex,
        )
    raise ValueError(f"Unsupported rank {rank}")


def reorder_polarizabilities(pols, atom_indexes):
    reorder_data = np.concatenate(atom_indexes).astype(int)
    pols2 = [None for _ in range(len(pols))]
    for i in range(len(pols)):
        pols2[reorder_data[i]] = pols[i]
    return pols2


class ModelData:
    """Model arrays loaded once from ``ALPHAML_DATA_DIR``."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.natoms = np.load(self.dir / "natoms.npy")

        self._sparse = {
            (0, False): (self._load("SPARSE_Amat0.npy"), self._load("SPARSE_fps0.npy")),
            (0, True): (self._load("SPARSE_Amat0_RS.npy"), self._load("SPARSE_fps0_RS.npy")),
            (2, False): (self._load("SPARSE_Amat2.npy"), self._load("SPARSE_fps2.npy")),
        }
        self._pstrain = {
            (0, False): self._load("PStrain_lam0.npy"),
            (0, True): self._load("PStrain_lam0_RS.npy"),
            (2, False): self._load("PStrain_lam2.npy"),
        }
        self._pstrain_zeta = {
            False: self._load("PStrain_lam0_zeta2_lam2.npy"),
        }

        self._weights = {}
        for model_name in MODELS:
            suffix = "_RS" if model_name == "ccsd" else ""
            folder = f"{model_name}-weights"
            self._weights[(model_name, "lm0")] = self._load(
                f"{folder}/weights_lm0{suffix}.npy"
            )
            self._weights[(model_name, "lm0_mean")] = self._load(
                f"{folder}/weights_lm0_mean{suffix}.npy"
            )
            self._weights[(model_name, "lm2")] = self._load(
                f"{folder}/weights_lm2.npy"
            )

        get_w3j(2, 6)

    def _load(self, name):
        return np.load(self.dir / name, mmap_mode="r")

    def sparse_options(self, lam, use_radial_scaling):
        return self._sparse[(lam, bool(use_radial_scaling))]

    def pstrain(self, lam, use_radial_scaling):
        return self._pstrain[(lam, bool(use_radial_scaling))]

    def pstrain_zeta(self, use_radial_scaling):
        return self._pstrain_zeta[bool(use_radial_scaling)]

    def weights(self, model_name, kind):
        return self._weights[(model_name, kind)]

    def predict(self, symbols, positions, model_name):
        coords = np.asarray(positions, dtype=float)
        if model_name == "ccsd":
            rad_scal = [[1.0, 3.5, 8], [1.0, 0.0, 0], [1.0, 0.0, 0]]
            use_radial_scaling = [True, False, False]
        else:
            rad_scal = [[1.0, 0.0, 0]] * 3
            use_radial_scaling = [False, False, False]

        lm = [0, 0, 2]
        all_PS = [
            np.array(
                get_power_spectrum(
                    lm[i], symbols, coords, rad_scal[i], use_radial_scaling[i], self
                )[0]
            )
            for i in range(3)
        ]
        all_PS = split_power_spectra(all_PS, len(symbols))
        all_ker = [
            get_kernel(lm[0], all_PS[0], len(symbols), all_PS[0], use_radial_scaling[0], self),
            get_kernel(lm[2], all_PS[2], len(symbols), all_PS[1], use_radial_scaling[2], self),
        ]
        pols = np.array(get_pol(all_ker, model_name, use_radial_scaling, self))
        _, atom_indexes, _ = _species_layout(symbols)
        return np.array(reorder_polarizabilities(pols, atom_indexes))
