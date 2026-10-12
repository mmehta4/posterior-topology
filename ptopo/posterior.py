"""Range-measurement posterior, on full grids and on symmetry-reduced grids.

Model:  r_i = ||x - a_i|| + eps_i,  eps_i ~ N(0, sigma^2) iid,  flat prior on x in R^d.
        log p(x | r) = -sum_i (||x - a_i|| - r_i)^2 / (2 sigma^2) + const.

Symmetry reduction (the trick that unlocks high dimensions)
-----------------------------------------------------------
For k <= d anchors in general position, the anchors span an affine (k-1)-plane P.
The posterior is invariant under rotations about P, so x is determined up to symmetry by
    y   = coordinates of the projection of x onto P        (in R^(k-1))
    rho = distance from x to P                              (>= 0)
The *density* at x depends only on (y, rho), while the *volume element* is
    dV  =  |S^(n)| * rho^n  dy drho,      n = d - k.
So the superlevel sets of the density live in a fixed k-dimensional half-space, independent
of d. The dimension enters only through the weight rho^n, which moves the HPD cut-off, and
through the S^n "lift" back to R^d (topology.lift_betti).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def log_lik_full(X: np.ndarray, anchors: np.ndarray, ranges: np.ndarray, sigma: float) -> np.ndarray:
    """Unnormalised log posterior at points X (..., d)."""
    out = np.zeros(X.shape[:-1])
    for a, r in zip(anchors, ranges):
        dist = np.sqrt(((X - a) ** 2).sum(axis=-1))
        out -= (dist - r) ** 2
    return out / (2.0 * sigma ** 2)


def log_lik_reduced(Y: np.ndarray, rho: np.ndarray, B: np.ndarray, ranges, sigma: float) -> np.ndarray:
    """Unnormalised log posterior in reduced coordinates.

    Y   : (..., k-1) coordinates in the anchor plane (may have last dim 0 when k = 1)
    rho : (...)      distance to the anchor plane
    B   : (k, k-1)   anchor coordinates in the plane
    """
    out = np.zeros(rho.shape)
    for b, r in zip(B, ranges):
        sq = rho ** 2
        if B.shape[1]:
            sq = sq + ((Y - b) ** 2).sum(axis=-1)
        out -= (np.sqrt(sq) - r) ** 2
    return out / (2.0 * sigma ** 2)


def reduce_anchors(anchors: np.ndarray):
    """Coordinates of k anchors in their own affine hull: returns (B (k, k-1), origin, basis (k-1, d))."""
    anchors = np.asarray(anchors, float)
    k, d = anchors.shape
    origin = anchors[0]
    if k == 1:
        return np.zeros((1, 0)), origin, np.zeros((0, d))
    Q, R = np.linalg.qr((anchors[1:] - origin).T)  # columns span the hull directions
    if np.min(np.abs(np.diag(R))) < 1e-12:
        raise ValueError("anchors are not affinely independent")
    basis = Q.T[: k - 1]
    B = (anchors - origin) @ basis.T
    return B, origin, basis


def to_reduced(X: np.ndarray, origin, basis):
    Z = X - origin
    Y = Z @ basis.T
    rho = np.sqrt(np.maximum((Z ** 2).sum(-1) - (Y ** 2).sum(-1), 0.0))
    return Y, rho


# ------------------------------------------------------------------------- grids

@dataclass
class Grid:
    """Cell-centred tensor grid. `axes[j]` are cell centres; `h[j]` the spacings."""
    axes: list
    h: np.ndarray
    log_weight: np.ndarray | None = None  # per-cell log volume weight (reduced grids)
    reduced: bool = False
    n: int = 0                             # sphere dimension of the lift (reduced grids)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(len(a) for a in self.axes)

    def mesh(self):
        return np.meshgrid(*self.axes, indexing="ij")


def full_grid(lo, hi, cells) -> Grid:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cells = np.broadcast_to(np.asarray(cells), lo.shape)
    h = (hi - lo) / cells
    axes = [lo[j] + h[j] * (np.arange(cells[j]) + 0.5) for j in range(len(lo))]
    return Grid(axes=axes, h=h)


def reduced_grid(y_lo, y_hi, rho_hi, cells_y, cells_rho, n: int) -> Grid:
    """Grid on R^(k-1) x [0, rho_hi]; the last axis is rho (row 0 touches the axis rho = 0)."""
    y_lo, y_hi = np.atleast_1d(np.asarray(y_lo, float)), np.atleast_1d(np.asarray(y_hi, float))
    cells_y = np.broadcast_to(np.asarray(cells_y), y_lo.shape)
    hy = (y_hi - y_lo) / cells_y if y_lo.size else np.zeros(0)
    hr = rho_hi / cells_rho
    axes = [y_lo[j] + hy[j] * (np.arange(cells_y[j]) + 0.5) for j in range(y_lo.size)]
    axes.append(hr * (np.arange(cells_rho) + 0.5))
    g = Grid(axes=axes, h=np.append(hy, hr), reduced=True, n=n)
    rho = np.meshgrid(*axes, indexing="ij")[-1]
    # exact cell integral of rho^n over [rho_lo, rho_hi] (matters for n large near the axis)
    # computed in log space: (n+1) log r1 + log(1 - (r0/r1)^(n+1)) - log((n+1) hr)
    r0, r1 = rho - hr / 2, rho + hr / 2
    g.log_weight = ((n + 1) * np.log(r1) + np.log1p(-(r0 / r1) ** (n + 1)) - np.log((n + 1) * hr))
    return g


def evaluate(grid: Grid, B_or_anchors, ranges, sigma) -> np.ndarray:
    """Log posterior on a grid (reduced grids take reduced anchor coordinates B)."""
    M = grid.mesh()
    if grid.reduced:
        Y = np.stack(M[:-1], axis=-1) if len(M) > 1 else np.zeros(M[-1].shape + (0,))
        return log_lik_reduced(Y, M[-1], np.asarray(B_or_anchors, float), ranges, sigma)
    X = np.stack(M, axis=-1)
    return log_lik_full(X, np.asarray(B_or_anchors, float), ranges, sigma)
