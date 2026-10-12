"""Highest-posterior-density cut-off on a grid.

C_alpha = {x : p(x) >= c_alpha} with posterior mass 1 - alpha. On a grid: sort cells by
density, accumulate mass (density x cell volume x reduced-coordinate weight) until it
reaches 1 - alpha. Everything is done in log space to avoid underflow.
"""
from __future__ import annotations

import numpy as np


def hpd_level(logp: np.ndarray, log_weight: np.ndarray | None = None, alpha: float = 0.05,
              reduced: bool | None = None):
    """Return (log c_alpha, mask, info).

    `log_weight` is the per-cell log volume weight (None = uniform). Uniform cell volume
    cancels in the normalisation, so it is never needed. `reduced` (default: True iff a
    weight is given) marks the last axis as rho, whose index-0 face is the symmetry axis.
    """
    lw = logp if log_weight is None else logp + log_weight
    shift = lw.max()
    mass = np.exp(lw - shift).ravel()
    order = np.argsort(logp.ravel(), kind="stable")[::-1]
    cum = np.cumsum(mass[order])
    total = cum[-1]
    idx = int(np.searchsorted(cum, (1 - alpha) * total))
    idx = min(idx, order.size - 1)
    level = logp.ravel()[order[idx]]
    mask = logp >= level
    info = {
        "boundary_mass": _boundary_mass(mass.reshape(logp.shape),
                                        log_weight is not None if reduced is None else reduced) / total,
        "cells_in_region": int(mask.sum()),
        "mass_captured": float(cum[idx] / total),
    }
    return float(level), mask, info


def _boundary_mass(m: np.ndarray, reduced: bool) -> float:
    """Mass on the outer faces of the grid — must be negligible for the grid to be trusted.
    For reduced grids the rho = 0 face is the symmetry axis, not a boundary, and is skipped."""
    s = 0.0
    for ax in range(m.ndim):
        sl_hi = [slice(None)] * m.ndim
        sl_hi[ax] = -1
        s += m[tuple(sl_hi)].sum()
        if not (reduced and ax == m.ndim - 1):
            sl_lo = [slice(None)] * m.ndim
            sl_lo[ax] = 0
            s += m[tuple(sl_lo)].sum()
    return float(s)


def hpd_depth(logp, log_weight=None, alpha=0.05) -> float:
    """q_alpha = log(p_max / c_alpha) — how far below the peak the HPD cut sits."""
    level, _, _ = hpd_level(logp, log_weight, alpha)
    return float(logp.max() - level)
