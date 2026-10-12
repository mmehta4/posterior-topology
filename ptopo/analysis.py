"""One call from (geometry, noise) to the Betti numbers of the 95% HPD region in R^d."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hpd, posterior, topology


@dataclass
class Result:
    betti: np.ndarray          # Betti numbers of C_alpha in R^d (length d+1)
    log_level: float           # log c_alpha (unnormalised)
    log_max: float             # max log density on the grid
    mask: np.ndarray           # region on the (reduced or full) grid
    logp: np.ndarray
    grid: posterior.Grid
    info: dict

    @property
    def depth(self):
        """HPD depth q_alpha = log(p_max / c_alpha)."""
        return self.log_max - self.log_level


def _auto_reduced_grid(B, ranges, sigma, n, cells):
    B = np.asarray(B, float)
    rmax = float(np.max(ranges))
    # the rho^n weight pushes mass outward by roughly n sigma^2 / r
    push = n * sigma ** 2 / max(rmax, sigma)
    pad = rmax + 9 * sigma + 1.5 * push
    if B.shape[1] == 0:
        return posterior.reduced_grid([], [], pad, [], cells, n)
    lo, hi = B.min(0) - pad, B.max(0) + pad
    return posterior.reduced_grid(lo, hi, pad, cells, cells // 2, n)


def analyze_reduced(d, B, ranges, sigma, alpha=0.05, cells=800, grid=None, max_expand=4):
    """Topology in R^d via the k-dimensional symmetry-reduced grid (k = len(ranges) <= 2)."""
    k = len(ranges)
    n = d - k
    assert n >= 0 and k <= 2
    for attempt in range(max_expand + 1):
        g = grid or _auto_reduced_grid(B, ranges, sigma * (1.6 ** attempt), n, cells)
        logp = posterior.evaluate(g, B, ranges, sigma)
        level, mask, info = hpd.hpd_level(logp, g.log_weight, alpha, reduced=True)
        if info["boundary_mass"] < 1e-7 or grid is not None:
            break
    b, comps = topology.reduced_betti(mask, n, d)
    info["components"] = comps
    return Result(b, level, float(logp.max()), mask, logp, g, info)


def analyze_full(anchors, ranges, sigma, alpha=0.05, cells=400, lo=None, hi=None):
    """Topology on a full d-dimensional grid (d = 2 or 3). Used for k = d+1 and cross-checks."""
    anchors = np.asarray(anchors, float)
    rmax = float(np.max(ranges))
    if lo is None:
        lo = anchors.min(0) - rmax - 9 * sigma
        hi = anchors.max(0) + rmax + 9 * sigma
    g = posterior.full_grid(lo, hi, cells)
    logp = posterior.evaluate(g, anchors, ranges, sigma)
    level, mask, info = hpd.hpd_level(logp, None, alpha, reduced=False)
    b = np.array(topology.betti(mask))
    return Result(b, level, float(logp.max()), mask, logp, g, info)


def noise_free_betti(d, k):
    """The staircase: Betti numbers of the noise-free solution set S^(d-k) (k <= d+1)."""
    b = np.zeros(d + 1, int)
    if k == d + 1:
        b[0] = 1
    elif k == d:
        b[0] = 2
    else:
        b[0] = 1
        b[d - k] += 1
    return b


def symmetric_pair(theta):
    """Two anchors at z = -1, +1 and a true point at (0, h) seen under crossing angle theta.

    theta is the angle a1 x* a2, i.e. the angle at which the two range circles cross.
    Returns (B, ranges) for exact (noise-free) data.
    """
    h = 1.0 / np.tan(theta / 2.0)
    r = float(np.hypot(1.0, h))
    return np.array([[-1.0], [1.0]]), np.array([r, r])


def analyze_mirror_beta0(anchors, ranges, sigma, alpha=0.05, cells=72, pad_widths=10.0, max_expand=5):
    """beta_0 of C_alpha for k = d anchors (mirror symmetry through the anchor hyperplane).

    Works on the reduced half-space {rho >= 0} (n = 0, so no volume weight) with an adaptive
    box: all low local minima of S are located by multistart optimisation, the box covers them
    plus `pad_widths` local posterior widths (sigma / smallest singular value of the range
    Jacobian), and is widened until the boundary mass is negligible. Each reduced component
    not touching rho = 0 lifts to two pieces; one touching it lifts to one piece.
    Returns (beta0, info).
    """
    from scipy import ndimage, optimize

    A = np.asarray(anchors, float)
    k, d = A.shape
    assert k == d, "mirror case needs k = d"
    B, _, _ = posterior.reduce_anchors(A)
    r = np.asarray(ranges, float)

    def S(v):
        y, rho = v[:-1], abs(v[-1])
        return float(sum((np.sqrt(((y - b) ** 2).sum() + rho ** 2) - ri) ** 2 for b, ri in zip(B, r)))

    rng = np.random.default_rng(0)
    c0, scale = B.mean(0), max(r.max(), 1e-6)
    starts = [np.append(c0 + scale * rng.normal(size=d - 1) * .7, abs(rng.normal()) * scale) for _ in range(16)]
    mins = [optimize.minimize(S, s0, method="Nelder-Mead",
                              options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000)).x for s0 in starts]
    vals = np.array([S(m) for m in mins])
    keep = [np.append(m[:-1], abs(m[-1])) for m, v in zip(mins, vals) if v <= vals.min() + 40 * sigma ** 2]
    P = np.array(keep)

    def width_at(p):
        U = np.array([np.append(p[:-1] - b, p[-1]) / max(np.sqrt(((p[:-1] - b) ** 2).sum() + p[-1] ** 2), 1e-12)
                      for b in B])
        return sigma / max(np.linalg.svd(U, compute_uv=False).min(), 0.05)

    w = max(width_at(p) for p in P)
    pad = pad_widths * w
    for attempt in range(max_expand + 1):
        lo, hi = P[:, :-1].min(0) - pad, P[:, :-1].max(0) + pad
        g = posterior.reduced_grid(lo, hi, P[:, -1].max() + pad, cells, cells, 0)
        logp = posterior.evaluate(g, B, r, sigma)
        level, mask, info = hpd.hpd_level(logp, None, alpha, reduced=True)
        if info["boundary_mass"] < 1e-6:
            break
        pad *= 1.6
    lab, nc = ndimage.label(mask, structure=np.ones((3,) * d, bool))
    touching = set(np.unique(lab[..., 0])) - {0}
    beta0 = sum(1 if c in touching else 2 for c in range(1, nc + 1))
    info.update(pad=pad, local_width=w, n_minima=len(P), S_min=float(vals.min()))
    return beta0, info
