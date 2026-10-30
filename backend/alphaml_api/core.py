"""Python 3 / NumPy port of the legacy SA-GPR model.

Ported from ``legacy/alpha-ml/webservice/sagpr.py`` and ``test.pyx`` (the
Cython kernels are replaced by vectorised NumPy).  The algorithm, constants
and model choices are kept unchanged.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from math import factorial, sqrt
from pathlib import Path

import numpy as np
from scipy import special

MODELS = ("ccsd", "b3lyp", "scan0", "delta-b3lyp-ccsd", "delta-scan0-ccsd")

# The format of the tensors folder (scripts/build_tensors.py): older ones are refused
TENSORS_FORMAT = 2

# Molecules of more atoms are predicted in chunks of atoms on several threads
# (each atom's prediction depends on its own neighbours only)
PARALLEL_ATOMS = 256
# Working memory: atoms per chunk, pairs per sum, array elements per step
ATOM_CHUNK = 128
PAIR_CHUNK = 20000
ELEMENT_CHUNK = 4_000_000
THREADS = int(os.environ.get("ALPHAML_THREADS", min(4, os.cpu_count() or 1)))

ATOM_VALENCE = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
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
            [[1.0j, 0.0, 1.0j], [0.0, np.sqrt(2.0), 0.0], [1.0, 0.0, -1.0]],
            dtype=complex,
        ) / np.sqrt(2.0)
    if lval == 2:
        return np.array(
            [
                [1.0j, 0.0, 0.0, 0.0, -1.0j],
                [0.0, 1.0j, 0.0, 1.0j, 0.0],
                [0.0, 0.0, np.sqrt(2.0), 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=complex,
        ) / np.sqrt(2.0)
    raise ValueError(f"Unsupported l value {lval}")


def radialscaling(r, c, r0, m):
    if m == 0:
        return 1.0
    if c == 0.0:
        return 1.0 / (r / r0) ** m
    return c / (c + (r / r0) ** m)


@lru_cache(maxsize=None)
def setup_orthomatrix(nmax, rc, rk):
    """Orthogonalization of the radial basis (8 x 8) and its Gaussian widths.
    Cached: it depends on the cutoff only (two in use). Read-only arrays, as
    they are shared between calls."""
    sigma = np.zeros(nmax, dtype=float)
    dsigma = 0.2 * rc / float(nmax)
    sigma[0] = dsigma
    for i in range(1, nmax):
        sigma[i] = sigma[i - 1] + dsigma

    overlap = np.zeros((nmax, nmax), dtype=float)
    for n1 in range(nmax):
        for n2 in range(nmax):
            overlap[n1, n2] = (
                (0.5 / sigma[n1] ** 2 + 0.5 / sigma[n2] ** 2)
                ** (-0.5 * (3.0 + n1 + n2))
                / (sigma[n1] ** n1 * sigma[n2] ** n2)
                * special.gamma(0.5 * (3.0 + n1 + n2))
                / (
                    (sigma[n1] * sigma[n2]) ** 1.5
                    * np.sqrt(special.gamma(1.5 + n1) * special.gamma(1.5 + n2))
                )
            )

    # Symmetric: eigh, real eigenvectors (eig gave a complex matrix with zero
    # imaginary part, and complex arithmetic downstream)
    eigenvalues, unitary = np.linalg.eigh(overlap)
    diagoverlap = np.diag(np.sqrt(eigenvalues))
    newoverlap = np.dot(np.conj(unitary), np.dot(diagoverlap, unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)
    orthomatrix.flags.writeable = sigma.flags.writeable = False
    return [orthomatrix, sigma]


def wigner_3j(j1, j2, j3, m1, m2, m3):
    """The Wigner 3j symbol of integer angular momenta (Racah's formula), as
    sympy.physics.wigner.wigner_3j (as a float)."""
    if m1 + m2 + m3 != 0 or not abs(j1 - j2) <= j3 <= j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    f = factorial
    triangle = f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
    norm = f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3)
    total = 0
    for k in range(max(0, j2 - j3 - m1, j1 - j3 + m2), min(j1 + j2 - j3, j1 - m1, j2 + m2) + 1):
        total += (-1) ** k / (
            f(k) * f(j3 - j2 + k + m1) * f(j3 - j1 + k - m2)
            * f(j1 + j2 - j3 - k) * f(j1 - k - m1) * f(j2 - k + m2)
        )
    return (-1) ** (j1 - j2 - m3) * sqrt(triangle * norm) * total


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


@lru_cache(maxsize=None)
def _lambda_tables(lam, lmax):
    """The (l1, l2) pairs of the power spectrum of order lam, and the sum
    over m of each feature as a sparse table (only the non-zero 3j).

    For pair p = (lval, lval2), im and mu: the m of the second harmonic is
    m_index[p, im, mu] = lval2 + im - lval - mu + lam, for im < 2 lval + 1 and
    |im - lval - mu + lam| <= lval2 (else w[p, im, mu] = 0, the 3j included).

    The normalisation, sum over all features of p2[mu] conj(p2[mu']), is
    inner[mu, mu'] = sum_p sum_{im, im'} w[p, im, mu] w[p, im', mu']
        D[lval, im, im'] conj(D[lval2, m_index[p, im, mu], m_index[p, im', mu']]),
    D[l, m, m'] = sum_k omega[k, l, m] conj(omega[k, l, m']): `norm_terms` are
    its non-zero terms for mu <= mu' (inner is Hermitian), (flat index of D,
    flat index of conj D, coefficient) by (mu, mu') (as reduceat starts), and
    the weight of each (mu, mu') in |inner|^2 (2 off the diagonal).
    Read-only, as shared between calls."""
    w3j = get_w3j(lam, lmax)
    pairs = [(l1, l2) for l1 in range(lmax + 1) for l2 in range(lmax + 1) if abs(l2 - lam) <= l1 <= l2 + lam]
    L1, L2 = (np.array(v) for v in zip(*pairs))
    nm, nmu = 2 * lmax + 1, 2 * lam + 1
    im = np.arange(nm)[None, :, None]
    mu = np.arange(nmu)[None, None, :]
    shift = im - L1[:, None, None] - mu + lam
    valid = (np.abs(shift) <= L2[:, None, None]) & (im < 2 * L1[:, None, None] + 1)
    m_index = np.where(valid, L2[:, None, None] + shift, 0)
    w = w3j[:, L1, L2, :].transpose(1, 2, 0) * valid  # (pairs, im, mu)

    first, second, coef, out = [], [], [], []
    for p in range(len(pairs)):
        v, x = np.nonzero(w[p])
        a, b = np.meshgrid(np.arange(len(v)), np.arange(len(v)), indexing="ij")
        a, b = a.ravel(), b.ravel()
        a, b = a[x[a] <= x[b]], b[x[a] <= x[b]]
        first.append((L1[p] * nm + v[a]) * nm + v[b])
        second.append((L2[p] * nm + m_index[p, v[a], x[a]]) * nm + m_index[p, v[b], x[b]])
        coef.append(w[p, v[a], x[a]] * w[p, v[b], x[b]])
        out.append(x[a] * nmu + x[b])
    out = np.concatenate(out)
    order = np.argsort(out, kind="stable")
    slots, starts = np.unique(out[order], return_index=True)
    weight = np.where(slots // nmu == slots % nmu, 1.0, 2.0)
    norm_terms = tuple(np.concatenate(t)[order] for t in (first, second, coef)) + (starts, weight)
    for array in (L1, L2, m_index, w, *norm_terms):
        array.flags.writeable = False
    return L1, L2, m_index, w, norm_terms


def _species_layout(symbols):
    all_species = [ATOM_VALENCE[name] for name in SPECIES_SYMBOLS]
    atom_indexes = [[] for _ in range(NSMAX)]
    for ii, name in enumerate(symbols):
        for j in all_species:
            if name == ATOM_SYMBOLS[j]:
                atom_indexes[j].append(ii)
    centers = [ATOM_VALENCE[name] for name in CENTER_SYMBOLS]
    return all_species, atom_indexes, centers


def _radial_integrals(r, e, lmax, nmax, sigma, sg2):
    """radint[pair, l, n] of neighbours at distances r with weights e."""
    radint = np.empty((len(r), lmax + 1, nmax))
    for n in range(nmax):
        normfact = np.sqrt(2.0 / (special.gamma(1.5 + n) * sigma[n] ** (3.0 + 2.0 * n)))
        sigmafact = (sg2**2 + sg2 * sigma[n] ** 2) / sigma[n] ** 2
        for l in range(lmax + 1):
            radint[:, l, n] = (
                e
                * 2.0 ** (-0.5 * (1.0 + l - n))
                * (1.0 / sg2 + 1.0 / sigma[n] ** 2) ** (-0.5 * (3.0 + l + n))
                * special.gamma(0.5 * (3.0 + l + n))
                / special.gamma(1.5 + l)
                * (r / sg2) ** l
                * special.hyp1f1(0.5 * (3.0 + l + n), 1.5 + l, 0.5 * r**2 / sigmafact)
            ) * normfact
    return radint


def fill_arrays(
    nspecies,
    lmax,
    nmax,
    center_atoms,
    atom_indexes,
    all_species,
    coords,
    rc,
    rk,
    cw,
    sigma,
    sg,
    orthomatrix,
    rad_scal,
):
    """omega[centre, species, n, l, m]: the neighbour density of each centre
    (the atoms center_atoms), by species, on the orthogonalised radial
    basis and the spherical harmonics. Built pair by pair (only the neighbours
    within rc, not the legacy fixed slots, 35 per (centre, species), so no
    limit on the neighbours)."""
    alpha = 1.0 / (2 * sg**2)
    sg2 = sg**2

    center_atoms = np.asarray(center_atoms, dtype=int)
    nat = len(center_atoms)
    species = np.zeros(len(coords), dtype=int)
    for ispe, ix in enumerate(all_species):
        species[atom_indexes[ix]] = ispe

    # The pairs (centre iat, neighbour j) within rc, grouped by (centre,
    # species). The centre itself is its own neighbour at distance 0.
    rr = coords[None, :, :] - coords[center_atoms][:, None, :]
    iat, j = np.nonzero(np.einsum("ijk,ijk->ij", rr, rr) <= rc**2)
    group = iat * nspecies + species[j]
    order = np.argsort(group, kind="stable")
    iat, j, group = iat[order], j[order], group[order]
    groups, starts, counts = np.unique(group, return_index=True, return_counts=True)
    vec = rr[iat, j]
    is_self = j == center_atoms[iat]
    o = ~is_self

    # efact (the weight) and the spherical harmonics of each pair
    r = np.linalg.norm(vec, axis=1)
    efact = np.full(len(r), cw)
    harmonic = np.zeros((len(r), lmax + 1, 2 * lmax + 1), dtype=complex)
    harmonic[is_self, 0, 0] = _sph_harm(0, 0, 0, 0)
    ro = r[o]
    theta = np.arccos(vec[o, 2] / ro)
    phi = np.arctan2(vec[o, 1], vec[o, 0])
    switch = np.where(
        ro <= rc - rk,
        1.0,
        np.where(ro < rc, (1.0 + np.cos(np.pi * (ro - (rc - rk)) / rk)) / 2.0, 0.0),
    )
    efact[o] = (
        np.exp(-alpha * ro**2)
        * radialscaling(ro, rad_scal[0], rad_scal[1], rad_scal[2])
        * switch
    )
    for lval in range(lmax + 1):
        mxrange = np.arange(-lval, lval + 1)[:, None]
        harmonic[o, lval, : 2 * lval + 1] = np.conj(
            _sph_harm(mxrange, lval, phi[None, :], theta[None, :])
        ).T

    # Radial integrals (orthogonalised) once per distance: (i, j) and (j, i)
    # are the same, and efact depends on the distance only
    distances, first, inverse = np.unique(r, return_index=True, return_inverse=True)
    radint = _radial_integrals(distances, efact[first], lmax, nmax, sigma, sg2)
    orthoradint = (radint @ orthomatrix.T)[inverse]  # (pair, l, n)

    # omega: the sum over each group's pairs, in chunks of centres (memory)
    omega = np.zeros((nat * nspecies, nmax, lmax + 1, 2 * lmax + 1), dtype=complex)
    bounds = np.append(starts, len(r))
    chunk = max(1, PAIR_CHUNK // max(1, int(counts.mean())))
    for g0 in range(0, len(groups), chunk):
        g1 = min(g0 + chunk, len(groups))
        pairs = slice(bounds[g0], bounds[g1])
        terms = np.einsum("pln,plm->pnlm", orthoradint[pairs], harmonic[pairs])
        omega[groups[g0:g1]] = np.add.reduceat(
            terms, starts[g0:g1] - bounds[g0], axis=0
        )

    return [nat, omega.reshape(nat, nspecies, nmax, lmax + 1, 2 * lmax + 1)]


def get_power_spectrum(
    lam, symbols, coords, rad_scal, use_radial_scaling, data, center_atoms, cache=None
):
    """The power spectrum of order lam of the atoms center_atoms of the
    molecule. `cache` (a dict) keeps the neighbour arrays of fill_arrays,
    which do not depend on lam, for the other calls of the same prediction."""
    all_species, atom_indexes, _ = _species_layout(symbols)
    nspecies = len(SPECIES_SYMBOLS)
    natmax = len(center_atoms)

    do_list = data.features(lam, use_radial_scaling)

    params = dict(lmax=6, nmax=8, sg=0.3, cw=1.0, rk=1e-10, rc=4.0)
    params_rc = dict(lmax=6, nmax=8, sg=0.3, cw=1.0, rk=1e-10, rc=5.0)
    p = params_rc if use_radial_scaling else params
    lmax, nmax, sg = p["lmax"], p["nmax"], p["sg"]
    cw, rk, rc = p["cw"], p["rk"], p["rc"]

    llmax = len(_lambda_tables(lam, lmax)[0]) if lam > 0 else 0

    orthomatrix, sigma = setup_orthomatrix(nmax, rc, rk)

    def filled():
        key = (bool(use_radial_scaling), tuple(rad_scal))
        if cache is not None and key in cache:
            return cache[key]
        arrays = fill_arrays(
            nspecies,
            lmax,
            nmax,
            center_atoms,
            atom_indexes,
            all_species,
            coords,
            rc,
            rk,
            cw,
            sigma,
            sg,
            orthomatrix,
            rad_scal,
        )
        if cache is not None:
            cache[key] = arrays
        return arrays

    nat, omega = filled()
    # Only the species of the molecule: the others have no neighbours, so
    # every feature with them is zero
    present = [i for i, z in enumerate(all_species) if atom_indexes[z]]
    position = np.full(nspecies, -1)
    position[present] = np.arange(len(present))
    om = omega[:, present]  # (nat, P, nmax, lmax + 1, 2 lmax + 1)
    # D[a, l, m, m'] = sum_k om[a, k, l, m] conj(om[a, k, l, m']), k = (species, n):
    # the norms of the power spectra over all their features, without them
    flat = om.transpose(0, 3, 1, 2, 4).reshape(nat, lmax + 1, -1, 2 * lmax + 1)
    D = np.swapaxes(flat, 2, 3) @ flat.conj()

    if lam == 0:
        featsize = nspecies * nspecies * nmax**2 * (lmax + 1)
        PS = np.zeros((1, natmax, len(do_list)), dtype=complex)

        # power[a, s, d, b, n, l] = sum_m om[a, s, b, l, m] conj(om[a, d, n, l, m]) / sqrt(2 l + 1),
        # normalised over all (s, d, b, n, l): sum_l |D[a, l]|^2 / (2 l + 1)
        degeneracy = 2 * np.arange(lmax + 1) + 1
        norm = np.sqrt(np.einsum("almk,almk,l->a", D, D.conj(), 1.0 / degeneracy).real)

        # Only the features of do_list (the others are thrown away)
        shape = (nspecies, nspecies, nmax, nmax, lmax + 1)
        fs, fd, fb, fn, fl = np.unravel_index(do_list, shape)
        kept = (position[fs] >= 0) & (position[fd] >= 0)
        fs, fd, fb, fn, fl = (f[kept] for f in (fs, fd, fb, fn, fl))
        first = om[:, position[fs], fb, fl] / np.sqrt(degeneracy[fl])[:, None]
        second = om[:, position[fd], fn, fl]
        PS[0, :nat][:, kept] = np.einsum("afm,afm->af", first, second.conj()) / norm[:, None]

    else:
        featsize = nspecies * nspecies * nmax**2 * llmax
        PS = np.zeros((1, natmax, 2 * lam + 1, len(do_list)), dtype=complex)
        L1, L2, m_index, w, (d_first, d_second, d_coef, d_starts, d_weight) = _lambda_tables(lam, lmax)
        nmu = 2 * lam + 1

        # The power spectrum: for each pair p = (lval, lval2) and mu,
        #   p2[a, mu, s, d, n, n2, p] = sum_im w[p, im, mu] om[a, s, n, lval, im]
        #                               conj(om[a, d, n2, lval2, m_index[p, im, mu]])
        # (the orthogonalised radial integrals are real: the sum over the
        # neighbours of conj(harmonic) is conj(omega)), only for the features
        # of do_list, as a sparse sum: its non-zero terms by (mu, feature).
        shape = (nspecies, nspecies, nmax, nmax, llmax)
        fa, fb, fn, fm, fp = np.unravel_index(do_list, shape)
        kept = (position[fa] >= 0) & (position[fb] >= 0)
        fa, fb, fn, fm, fp = (f[kept] for f in (fa, fb, fn, fm, fp))
        f, v, x = np.nonzero(w[fp])
        order = np.argsort(x * len(fp) + f, kind="stable")
        f, v, x = f[order], v[order], x[order]
        slot = x * len(fp) + f
        f_starts = np.flatnonzero(np.r_[True, slot[1:] != slot[:-1]])
        om_shape = om.shape[1:]
        f_first = np.ravel_multi_index((position[fa][f], fn[f], L1[fp][f], v), om_shape)
        f_second = np.ravel_multi_index(
            (position[fb][f], fm[f], L2[fp][f], m_index[fp[f], v, x]), om_shape
        )
        f_coef = w[fp[f], v, x]

        # Atoms in chunks (memory)
        chunk = max(1, ELEMENT_CHUNK // max(len(d_coef), len(f_coef)))
        flat_om = om.reshape(nat, -1)
        flat_D = D.reshape(nat, -1)
        for a0 in range(0, nat, chunk):
            atoms = slice(a0, min(a0 + chunk, nat))
            terms = flat_D[atoms, d_first] * flat_D[atoms, d_second].conj() * d_coef
            inner = np.add.reduceat(terms, d_starts, axis=1)  # (a, mu <= mu')
            norm = np.sqrt(np.sqrt((np.abs(inner) ** 2) @ d_weight))
            terms = flat_om[atoms, f_first] * flat_om[atoms, f_second].conj() * f_coef
            selected = np.zeros((terms.shape[0], nmu * len(fp)), dtype=complex)
            selected[:, slot[f_starts]] = np.add.reduceat(terms, f_starts, axis=1)
            selected = selected.reshape(-1, nmu, len(fp))
            # An atom without neighbours: no anisotropy (p2 = 0), not 0 / 0
            PS[0, atoms][:, :, kept] = np.divide(
                selected,
                norm[:, None, None],
                out=np.zeros_like(selected),
                where=norm[:, None, None] > 0,
            )

    if lam == 0:  # sparsified here (for lambda = 2, in the model's tensors)
        PS = PS[0] @ data.arrays["amat0_rs" if use_radial_scaling else "amat0"]
        PS = PS[None]
    return [PS, featsize]


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
            [
                [-1.0 / np.sqrt(3.0), 0.5, 0.0, -1.0 / np.sqrt(6.0), 0.0, 0.5],
                [0.0, -0.5j, 0.0, 0.0, 0.0, 0.5j],
                [0.0, 0.0, 0.5, 0.0, -0.5, 0.0],
                [-1.0 / np.sqrt(3.0), -0.5, 0.0, -1.0 / np.sqrt(6.0), 0.0, -0.5],
                [0.0, 0.0, -0.5j, 0.0, -0.5j, 0.0],
                [-1.0 / np.sqrt(3.0), 0.0, 0.0, 2.0 / np.sqrt(6.0), 0.0, 0.0],
            ],
            dtype=complex,
        )
    raise ValueError(f"Unsupported rank {rank}")


class ModelData:
    """The model, loaded once from ``ALPHAML_DATA_DIR``: the folder of tensors
    built from the training data by ``scripts/build_tensors.py`` (memory-mapped,
    so shared between workers)."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.arrays = {p.stem: np.load(p, mmap_mode="r") for p in self.dir.glob("*.npy")}
        if "format" not in self.arrays or int(self.arrays["format"]) != TENSORS_FORMAT:
            raise ValueError(
                f"{self.dir} is not a folder of tensors of this version: build it with "
                "scripts/build_tensors.py <training data> <tensors folder>"
            )
        _lambda_tables(2, 6)

    def features(self, lam, use_radial_scaling):
        """The features of the power spectrum the model takes (its do_list)."""
        suffix = "_rs" if use_radial_scaling else ""
        return np.asarray(self.arrays[f"fps{lam}{suffix}"], dtype=int)

    def predict(self, symbols, positions, model_name):
        """The polarizability (9 Cartesian components, a.u.) of each atom."""
        coords = np.asarray(positions, dtype=float)
        _, atom_indexes, centers = _species_layout(symbols)
        order = np.array([c for k in centers for c in atom_indexes[k]], dtype=int)
        chunks = np.array_split(order, -(-len(order) // ATOM_CHUNK))
        if THREADS == 1 or len(order) <= PARALLEL_ATOMS:
            pols = np.concatenate([self._predict(symbols, coords, model_name, atoms) for atoms in chunks])
        else:
            with ThreadPoolExecutor(THREADS) as pool:
                parts = pool.map(lambda atoms: self._predict(symbols, coords, model_name, atoms), chunks)
                pols = np.concatenate(list(parts))
        result = np.empty_like(pols)
        result[order] = pols
        return result

    def _predict(self, symbols, coords, model_name, center_atoms):
        """The polarizabilities of the atoms center_atoms (in that order)."""
        if model_name == "ccsd":
            rad_scal = [[1.0, 3.5, 8], [1.0, 0.0, 0], [1.0, 0.0, 0]]
            use_radial_scaling = [True, False, False]
        else:
            rad_scal = [[1.0, 0.0, 0]] * 3
            use_radial_scaling = [False, False, False]

        # The power spectra: lambda = 0 with the model's radial scaling, and
        # lambda = 0 and 2 without. The neighbour arrays and equal spectra are
        # computed once.
        arrays, spectra = {}, {}

        def spectrum(lam, i):
            key = (lam, use_radial_scaling[i], tuple(rad_scal[i]))
            if key not in spectra:
                spectra[key] = get_power_spectrum(
                    lam,
                    symbols,
                    coords,
                    rad_scal[i],
                    use_radial_scaling[i],
                    self,
                    center_atoms,
                    arrays,
                )[0][0]
            return spectra[key]

        x0, x0z, x2 = spectrum(0, 0), spectrum(0, 1), spectrum(2, 2)

        # The model's tensors, the sparsification of lambda = 2 folded in
        # (see scripts/build_tensors.py)
        M = self.arrays[f"{model_name}_M"]
        G = self.arrays[f"{model_name}_G"]
        out0 = (
            np.sum((x0.real @ M) * x0.real, axis=1) + self.arrays[f"{model_name}_mean"]
        )
        CT = complex_to_real_transformation(2).T
        out2 = (np.einsum("jb,jnb->jn", np.conj(x0z) @ G, np.conj(x2)) @ CT).real

        return np.array(
            [spherical_to_cartesian([np.array(out0[j]), out2[j]]) for j in range(len(center_atoms))]
        )
