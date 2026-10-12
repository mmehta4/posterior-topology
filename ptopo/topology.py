"""Betti numbers of grid superlevel sets, persistence, and the symmetry lift.

Conventions. A grid region is the union of its closed cells (pixels / voxels). Two cells
sharing only a corner are therefore connected (8-connectivity in 2D, 26 in 3D) and the
complement uses the dual connectivity (4 / 6). This is exactly the cubical complex GUDHI
builds from top-dimensional cells, so both methods must agree (tests/test_topology.py).

The lift (theory/reduction.md, Lemma R). Let C be a connected compact region in the
reduced half-space {rho >= 0}, A = C ∩ {rho = 0} with m components (each assumed
contractible), and X ⊂ R^d the set swept out by rotating C about the axis plane with
spheres S^n, n = d - k. Then, writing b~ for reduced Betti numbers,
    m = 0 :  b_q(X) = b_q(C) + b_(q-n)(C)                       (X ≃ C × S^n)
    m >= 1:  b~_q(X) = b~_q(C) + b~_(q-n)(C) + (m - 1)[q = n+1]
For n = 0 the "rotation" is the reflection rho -> -rho and the formula still holds.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_S8 = np.ones((3, 3), bool)
_S4 = ndimage.generate_binary_structure(2, 1)
_S26 = np.ones((3, 3, 3), bool)
_S6 = ndimage.generate_binary_structure(3, 1)


# ------------------------------------------------------------------ plain Betti numbers

def betti_1d(mask: np.ndarray):
    _, n = ndimage.label(mask)
    return [n]


def betti_2d(mask: np.ndarray):
    if not mask.any():
        return [0, 0]
    _, b0 = ndimage.label(mask, structure=_S8)
    comp = np.pad(~mask, 1, constant_values=True)
    _, nc = ndimage.label(comp, structure=_S4)
    return [b0, nc - 1]


def euler_characteristic(mask: np.ndarray) -> int:
    """Euler characteristic of the union of closed cells (any dimension 1..3)."""
    m = np.pad(mask, 1).astype(bool)
    D = m.ndim
    chi = 0
    # A face of the cubical complex is indexed by which axes it is "thin" in. A face of
    # dimension j spanning axes S is present iff any of the 2^(D-j) cells around it is in.
    import itertools
    for thin in itertools.product([0, 1], repeat=D):  # 1 = cell spans the axis (thick)
        dim = sum(thin)
        present = np.zeros(tuple(s - 1 + t for s, t in zip(m.shape, thin)), bool)
        # cells adjacent to a face: along thin axes (t=0) the face sits between two cells
        offsets = [range(1) if t else range(2) for t in thin]
        for off in itertools.product(*offsets):
            sl = tuple(slice(o, o + present.shape[i]) for i, o in enumerate(off))
            present |= m[sl]
        chi += (-1) ** dim * int(present.sum())
    return chi


def betti_3d(mask: np.ndarray):
    if not mask.any():
        return [0, 0, 0]
    _, b0 = ndimage.label(mask, structure=_S26)
    comp = np.pad(~mask, 1, constant_values=True)
    _, nc = ndimage.label(comp, structure=_S6)
    b2 = nc - 1
    b1 = b0 + b2 - euler_characteristic(mask)
    return [b0, b1, b2]


def betti(mask: np.ndarray):
    return {1: betti_1d, 2: betti_2d, 3: betti_3d}[mask.ndim](mask)


def betti_gudhi(mask: np.ndarray):
    """Independent check through GUDHI's cubical homology."""
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=np.where(mask, 0.0, 1.0))
    cc.compute_persistence(homology_coeff_field=2, min_persistence=-1)
    return [len([1 for b, d in cc.persistence_intervals_in_dimension(q) if b <= 0 < d])
            for q in range(mask.ndim)]


# ------------------------------------------------------------------ persistence

def superlevel_persistence(logp: np.ndarray):
    """Persistent homology of the superlevel filtration {log p >= c}, c decreasing.

    Returns {q: array of (birth_level, death_level)} in log-density units, with
    birth >= death (a feature is born when the cut drops to `birth`, dies at `death`;
    death = -inf for essential features).
    """
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=-logp)
    cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
    out = {}
    for q in range(logp.ndim):
        iv = cc.persistence_intervals_in_dimension(q)
        out[q] = -np.asarray(iv).reshape(-1, 2) if len(iv) else np.zeros((0, 2))
    return out


def betti_at_level(diagram, level: float):
    """Betti numbers of {log p >= level} read off a superlevel diagram."""
    return [int(((iv[:, 0] >= level) & (iv[:, 1] < level)).sum()) for q, iv in sorted(diagram.items())]


# ------------------------------------------------------------------ the symmetry lift

def lift_betti(C_betti, m: int, n: int, d: int):
    """Betti numbers (length d+1) of the S^n-sweep of one connected reduced component."""
    out = np.zeros(d + 1, int)
    red = list(C_betti) + [0] * (d + 1)
    red_tilde = red.copy()
    red_tilde[0] -= 1  # reduced Betti numbers of C (C connected)
    for q in range(d + 1):
        if m == 0:
            out[q] = red[q] + (red[q - n] if q - n >= 0 else 0)
        else:
            out[q] = red_tilde[q] + (red_tilde[q - n] if q - n >= 0 else 0) + ((m - 1) if q == n + 1 else 0)
    out[0] += 0 if m == 0 else 1  # back from reduced to ordinary b_0
    return out


def reduced_betti(mask: np.ndarray, n: int, d: int):
    """Betti numbers in R^d of the region whose reduced picture is `mask`.

    `mask` is 1D (rho only, k = 1) or 2D (z, rho, k = 2) with the rho axis last and
    rho-index 0 touching the symmetry axis. Returns (betti (length d+1), components) where
    components lists (C_betti, m) per reduced component.
    """
    if mask.ndim == 1:
        lab, nc = ndimage.label(mask)
        comps = [([1], int(mask[0] and lab[0] == c)) for c in range(1, nc + 1)]
    elif mask.ndim == 2:
        lab, nc = ndimage.label(mask, structure=_S8)
        comps = []
        for c in range(1, nc + 1):
            Cm = lab == c
            sl = ndimage.find_objects(Cm.astype(int))[0]
            sub = Cm[sl]
            # holes inside the half-plane: pad all round with empty (incl. below the axis)
            b1 = betti_2d(sub)[1]
            axis_row = Cm[:, 0]
            _, m = ndimage.label(axis_row)
            comps.append(([1, b1], int(m)))
    else:
        raise NotImplementedError("lift implemented for reduced dimension 1 and 2")
    total = np.zeros(d + 1, int)
    for Cb, m in comps:
        total += lift_betti(Cb, m, n, d)
    return total, comps
