"""Critical points on the symmetry axis, in closed form, and Morse-predicted thresholds.

Key facts (theory/problem2.md):

1. Noise does not move critical points. log p = -S(x) / (2 sigma^2) with
   S(x) = sum_i (||x - a_i|| - r_i)^2, so the critical points of p are those of S, fixed by
   the geometry alone. Noise only rescales the critical *values*.

2. On the axis S is piecewise quadratic. With anchors at b_i on the z-axis and
   s_i = sign(z - b_i), S(z, 0) = sum_i (s_i (z - b_i) - r_i)^2, so on each sign pattern the
   unique stationary point is
        z_c = mean_i (b_i + s_i r_i),
   valid when its own signs match s. The anchors b_i themselves are non-smooth candidates.

3. Transverse curvature at an axis point (w_i = |z - b_i|):
        d^2 S / d rho^2 = 2 sum_i (1 - r_i / w_i).

4. Morse threshold. A feature tied to critical point c changes exactly when
        q_alpha(sigma) = Delta_c / sigma^2,   Delta_c = (S(c) - min S) / 2,
   where q_alpha = log(p_max / c_alpha) is the HPD depth. In the small-noise limit q_alpha
   tends to a chi-square quantile; at finite noise it is computed.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import optimize

from . import analysis


def S_axis(z, b, r):
    b, r = np.asarray(b, float), np.asarray(r, float)
    return float(np.sum((np.abs(z - b) - r) ** 2))


def axis_critical_points(b, r):
    """All critical points of S restricted to the axis (smooth + kink candidates).

    Returns a list of dicts: z, S, kind_axis ('max'/'min' of p along the axis),
    transverse ('max'/'min' of p across the axis), smooth (bool).
    """
    b, r = np.asarray(b, float).ravel(), np.asarray(r, float).ravel()
    pts = []
    for s in itertools.product([-1, 1], repeat=len(b)):
        s = np.array(s)
        zc = float(np.mean(b + s * r))
        if np.all(np.sign(zc - b) == s):
            pts.append((zc, True))
    for bi in b:
        pts.append((float(bi), False))
    out = []
    for z, smooth in sorted(set(pts)):
        w = np.abs(z - b)
        S = S_axis(z, b, r)
        # along the axis: S is convex on each piece -> smooth stationary points are minima of S
        if smooth:
            kind_axis = "max"  # max of p along the axis
        else:
            eps = 1e-7
            kind_axis = "min" if (S_axis(z - eps, b, r) < S and S_axis(z + eps, b, r) < S) else "other"
        with np.errstate(divide="ignore"):
            trans = 2 * np.sum(1 - r / w) if np.all(w > 0) else -np.inf  # at an anchor: cone point
        out.append(dict(z=z, S=S, smooth=smooth, kind_axis=kind_axis,
                        transverse=("min of p" if trans < 0 else "max of p"), d2S_drho2=float(trans)))
    return out


def morse_sigma(Delta, depth_fn, lo=1e-3, hi=50.0):
    """Solve depth(sigma) * sigma^2 = Delta for sigma by bracketing."""
    f = lambda s: depth_fn(s) * s * s - Delta
    grid = np.geomspace(lo, hi, 60)
    vals = [f(s) for s in grid]
    for a, c, fa, fc in zip(grid, grid[1:], vals, vals[1:]):
        if fa < 0 <= fc:
            return optimize.brentq(f, a, c, xtol=1e-6)
    return np.nan


def topology_transitions(d, B, r, sigmas, alpha=0.05, cells=800, refine=True, tol=1e-4):
    """Sweep sigma, record every change in Betti numbers, refine each change by bisection.

    Returns (sweep, transitions) where transitions are dicts
    {sigma, before, after}. Bisection assumes a single change inside each bracket; the
    sweep spacing controls that and is reported.
    """
    run = lambda s: analysis.analyze_reduced(d, B, r, s, alpha, cells).betti
    sweep = [(float(s), run(s)) for s in sigmas]
    trans = []
    for (s0, b0), (s1, b1) in zip(sweep, sweep[1:]):
        if np.array_equal(b0, b1):
            continue
        lo, hi = s0, s1
        while refine and hi - lo > tol * hi:
            mid = 0.5 * (lo + hi)
            if np.array_equal(run(mid), b0):
                lo = mid
            else:
                hi = mid
        trans.append(dict(sigma=0.5 * (lo + hi), before=b0.tolist(), after=run(hi).tolist(),
                          bracket=[lo, hi]))
    return sweep, trans


def depth_fn(d, B, r, alpha=0.05, cells=800):
    return lambda s: analysis.analyze_reduced(d, B, r, s, alpha, cells).depth


def morse_ladder(d, B, r, alpha=0.05, cells=800, eps=0.01, sigma_max=20.0):
    """Predict every topological transition from the axis critical values, then verify.

    For each axis critical point c (Delta_c = S(c)/2, exact data so min S = 0) solve
    sigma_c^2 q_alpha(sigma_c) = Delta_c, and compute the Betti numbers just below and
    just above sigma_c. Points with equal Delta (mirror pairs) are merged into one rung.
    Returns a list of rungs sorted by sigma.
    """
    q = depth_fn(d, B, r, alpha, cells)
    pts = axis_critical_points(B, r)
    rungs = {}
    for c in pts:
        key = round(c["S"] / 2, 10)
        rungs.setdefault(key, []).append(c)
    out = []
    for Delta, cs in sorted(rungs.items()):
        s = morse_sigma(Delta, q, hi=sigma_max)
        if not np.isfinite(s):
            out.append(dict(Delta=Delta, sigma=np.nan, points=cs))
            continue
        below = analysis.analyze_reduced(d, B, r, s * (1 - eps), alpha, cells).betti.tolist()
        above = analysis.analyze_reduced(d, B, r, s * (1 + eps), alpha, cells).betti.tolist()
        out.append(dict(Delta=Delta, sigma=float(s), before=below, after=above,
                        changes=below != above, points=[(p["z"], p["smooth"]) for p in cs]))
    return out
