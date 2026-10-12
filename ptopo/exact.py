"""Exact (quadrature) computation for two ranges, any dimension, in residual coordinates.

In residual coordinates e = (w1 - r1, w2 - r2) the posterior on the half-strip F has density proportional to
exp(-|e|^2 / 2 sigma^2) J(e), J = w1 w2 h^(d-3) (paper, Lemma 4.2), and every HPD region is the disc
|e| <= R = sqrt(2 Delta_alpha) cut by the walls of F. In polar coordinates e = rho (cos t, sin t),

    Pi(Delta <= R^2/2) = int_0^R exp(-rho^2 / 2 sigma^2) rho L(rho) d rho  /  (same integral over all rho),
    L(rho) = integral of J over the arcs of the circle of radius rho inside F.

The disc boundary is exact, so no grid and no topology computation is involved.

Accuracy. By Heron, h^2 = sqrt(2) s' l_* l_1 l_2 / D^2 with l_k the distances to the walls (paper, Lemma 5.2), so J
is a product of powers of distances to the walls. Wall distances are evaluated without cancellation near an arc end
(where the wall distance vanishes) by a product formula, and every endpoint power-law factor is removed by the
substitution t = t_end +/- H v^2 (angles) and rho = rho_end +/- H v^2 (radii). The resulting integrands are smooth
and are integrated by composite Gauss-Legendre rules. Logarithms throughout, so large d is safe. Node doubling
changes the computed thresholds by less than 1e-9 relative for d >= 3 and 5e-6 for d = 2 (tests/test_exact.py).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import optimize
from scipy.special import logsumexp

SQ2 = np.sqrt(2.0)


def _gl(n):
    x, w = leggauss(n)
    return (x + 1) / 2, w / 2                          # nodes and weights on [0, 1]


class TwoRange:
    """Two ranges with observed (D, r1, r2) satisfying the strict triangle inequality, in R^d."""

    def __init__(self, D, r1, r2, d, sigma_max=3.0, n=24, pieces=6):
        self.D, self.r1, self.r2, self.d = float(D), float(r1), float(r2), int(d)
        s = (self.D + self.r1 + self.r2) / 2
        self.s, self.ts, self.t1, self.t2 = s, s - self.D, s - self.r2, s - self.r1
        if min(self.ts, self.t1, self.t2) <= 0:
            raise ValueError("observed ranges violate the strict triangle inequality")
        # walls a_k . e < b_k, k = 0 (base, t*), 1 (beyond a2, t2), 2 (beyond a1, t1); |a_k| = sqrt 2
        self.A = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        self.b = np.array([2 * self.ts, 2 * self.t2, 2 * self.t1])
        self.phi = np.arctan2(self.A[:, 1], self.A[:, 0])
        self.n, self.pieces = n, pieces
        self.v, self.wv = _gl(n)
        walls = SQ2 * np.array([self.ts, self.t1, self.t2])
        corners = np.sqrt(2 * np.array([self.ts ** 2 + self.t1 ** 2, self.ts ** 2 + self.t2 ** 2]))
        self.rho_max = corners.max() + sigma_max * (3 * np.sqrt(self.d) + 14)
        self.brk = np.unique(np.concatenate([[0.0], walls, corners, [self.rho_max]]))
        rho, lw = [], []
        for lo, hi in zip(self.brk[:-1], self.brk[1:]):
            x, w = self._radial_nodes(lo, hi); rho.append(x); lw.append(np.log(w))
        self.rho = np.concatenate(rho); self.logw_rho = np.concatenate(lw)
        self.logL = np.array([self._logL(r) for r in self.rho])

    # ---------------------------------------------------------------- quadrature nodes
    def _sym_v2(self, lo, hi):
        """Nodes/weights on [lo, hi] with t = lo + H v^2 on the left half and hi - H v^2 on the right half."""
        H = (hi - lo) / 2; xs, ws = [], []
        for k in range(self.pieces):                        # composite in v on [0, 1]
            a, c = k / self.pieces, (k + 1) / self.pieces
            v = a + (c - a) * self.v; w = (c - a) * self.wv
            xs += [lo + H * v ** 2, hi - H * v ** 2]; ws += [2 * H * v * w, 2 * H * v * w]
        return np.concatenate(xs), np.concatenate(ws)

    def _radial_nodes(self, lo, hi):
        return self._sym_v2(lo, hi)

    # ---------------------------------------------------------------- geometry
    def arcs(self, rho):
        """Arcs of the circle |e| = rho inside F, as (start, end, wall vanishing at start, wall vanishing at end);
        -1 means no wall (full circle)."""
        removed = []
        for k in range(3):
            c = self.b[k] / (rho * SQ2)
            if c < 1:
                half = np.arccos(c)
                removed.append((self.phi[k] - half, 2 * half, k))          # (start, width, wall)
        if not removed:
            return [(0.0, 2 * np.pi, -1, -1)]
        base = removed[0][0] + removed[0][1]                              # end of the first removed arc
        iv = []
        for lo, width, k in removed:
            st = (lo - base) % (2 * np.pi)
            if st + width > 2 * np.pi + 1e-15:
                iv += [(st, 2 * np.pi, k), (0.0, st + width - 2 * np.pi, k)]
            else:
                iv.append((st, st + width, k))
        iv.sort()
        free, cur, cur_wall = [], 0.0, removed[0][2]
        for st, en, k in iv:
            if st > cur + 1e-14:
                free.append((cur + base, st + base, cur_wall, k))
            if en > cur:
                cur, cur_wall = en, k
        return free

    def _log_wall_dist(self, rho, t, wall_end, t_end, dt):
        """log l_k (distance from e = rho u(t) to wall k) without cancellation.
        l_k = (b_k - rho a_k.u)/sqrt2 = rho [cos(h_k) - cos(t - phi_k)] = 2 rho sin((t-phi_k+h_k)/2) sin((t-phi_k-h_k)/2)
        when the circle meets wall k (cos h_k = b_k / (rho sqrt2) < 1); for the wall vanishing at the arc end t_end the
        exact offset dt = t - t_end is used."""
        out = []
        for k in range(3):
            c = self.b[k] / (rho * SQ2)
            if k == wall_end:
                val = 2 * rho * np.sin((t + t_end) / 2 - self.phi[k]) * np.sin(dt / 2)
            elif c < 1:
                h = np.arccos(c)
                val = 2 * rho * np.sin((t - self.phi[k] + h) / 2) * np.sin((t - self.phi[k] - h) / 2)
            else:
                val = rho * (c - np.cos(t - self.phi[k]))
            out.append(np.log(np.abs(val)))
        return out

    def _logJ_arc(self, rho, t, wall_end, t_end, dt):
        e1, e2 = rho * np.cos(t), rho * np.sin(t)
        w1, w2 = self.r1 + e1, self.r2 + e2
        sp = self.s + (e1 + e2) / 2
        lls = self._log_wall_dist(rho, t, wall_end, t_end, dt)
        logh = 0.5 * (np.log(SQ2) + np.log(sp) + sum(lls) - 2 * np.log(self.D))
        return np.log(w1) + np.log(w2) + (self.d - 3) * logh

    def _logL(self, rho):
        if rho <= 0:
            return -np.inf
        parts = []
        for lo, hi, ka, kb in self.arcs(rho):
            H = (hi - lo) / 2
            for k in range(self.pieces):
                a, c = k / self.pieces, (k + 1) / self.pieces
                v = a + (c - a) * self.v; w = (c - a) * self.wv
                dt = H * v ** 2
                for t, wall, t_end, off in ((lo + dt, ka, lo, dt), (hi - dt, kb, hi, -dt)):
                    lj = self._logJ_arc(rho, t, wall, t_end, off)
                    parts.append(logsumexp(lj + np.log(2 * H * v * w)))
        return logsumexp(parts)

    # ---------------------------------------------------------------- probabilities
    def _logterms(self, sigma):
        return self.logw_rho + np.log(self.rho) + self.logL - self.rho ** 2 / (2 * sigma ** 2)

    def prob_inside(self, delta, sigma):
        """Posterior probability of {Delta <= delta}; the cut at R = sqrt(2 delta) is exact."""
        lt = self._logterms(sigma); R = np.sqrt(2 * delta)
        if R >= self.rho_max:
            return 1.0
        near = np.isclose(self.brk, R, rtol=1e-12, atol=0)
        if near.any():                                   # snap to a breakpoint (event depths are breakpoints)
            R = float(self.brk[np.argmax(near)])
        j = np.searchsorted(self.brk, R, side="right") - 1
        lo = self.brk[j]
        below = lt[self.rho < lo]
        if np.isclose(R, lo, rtol=1e-13, atol=0):
            inside = below
        else:
            x, w = self._sym_v2(lo, R)
            logL = np.array([self._logL(r) for r in x])
            inside = np.concatenate([below, np.log(w) + np.log(x) + logL - x ** 2 / (2 * sigma ** 2)])
        return 0.0 if inside.size == 0 else float(np.exp(logsumexp(inside) - logsumexp(lt)))

    def hpd_depth(self, sigma, alpha=0.05):
        """Delta_alpha(sigma)."""
        f = lambda R: self.prob_inside(R * R / 2, sigma) - (1 - alpha)
        return optimize.brentq(f, 1e-9, self.rho_max * 0.999, xtol=1e-11) ** 2 / 2

    def sigma_at_depth(self, delta, alpha=0.05, lo=1e-3, hi=None):
        """Noise level where Delta_alpha(sigma) = delta (unique by Theorem 2.1)."""
        hi = hi or self.rho_max / (3 * np.sqrt(self.d) + 14)
        f = lambda sg: self.prob_inside(delta, sg) - (1 - alpha)
        grid = np.geomspace(lo, hi, 60); vals = [f(g) for g in grid]
        for a, b, fa, fb in zip(grid, grid[1:], vals, vals[1:]):
            if fa > 0 >= fb:
                return optimize.brentq(f, a, b, xtol=1e-13, rtol=1e-12)
        return np.nan

    def events(self):
        ts, t1, t2 = self.ts, self.t1, self.t2
        return {"inner contact": ts ** 2, "contact beyond a1": t1 ** 2, "contact beyond a2": t2 ** 2,
                "anchor a1": ts ** 2 + t1 ** 2, "anchor a2": ts ** 2 + t2 ** 2}
