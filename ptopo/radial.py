"""Problem 0 — one range measurement in R^d.

With one anchor a and observed range r, the flat-prior posterior depends on x only
through s = ||x - a||. In units of noise (u = s/sigma, kappa = r/sigma) the radial
mass density is proportional to

    g_{d,kappa}(u) = u^(d-1) * exp(-(u - kappa)^2 / 2),   u > 0.

The HPD region is the shell {|u - kappa| <= t}. It has a hole (does not contain the
anchor) iff t < kappa, i.e. iff

    M_d(kappa) = P(U <= 2 kappa) > 1 - alpha,   U ~ g_{d,kappa}.

Theorem 0 (theory/problem0.tex): M_d is strictly increasing in kappa, M_d(0+) = 0,
M_d(inf) = 1, and M_{d+1} < M_d pointwise; hence the critical ratio kappa_d is unique
and strictly increasing in d, with kappa_d = sqrt(d/2) + z_{1-alpha} sqrt(3/8) + o(1).

Two evaluators are provided:
  * closed form (exact up to floating point), via truncated-normal moments, evaluated
    in mpmath at high precision — used for the certified table;
  * a fast float version for sweeps and figures.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy import integrate, optimize, stats


# ---------------------------------------------------------------- closed form (mpmath)

def _gauss_moments(j_max: int, lo, hi):
    """J_i = int_lo^hi v^i exp(-v^2/2) dv for i = 0..j_max (mpmath, lo may be -inf? no: finite or +inf)."""
    J = [mp.mpf(0)] * (j_max + 1)
    sqrt2pi = mp.sqrt(2 * mp.pi)
    J[0] = sqrt2pi * (mp.ncdf(hi) - mp.ncdf(lo))
    e = lambda v: mp.mpf(0) if v == mp.inf else mp.exp(-v * v / 2)
    if j_max >= 1:
        J[1] = e(lo) - e(hi)
    for i in range(2, j_max + 1):
        # int v^i e^{-v^2/2} = [-v^{i-1} e^{-v^2/2}] + (i-1) J_{i-2}
        top = mp.mpf(0) if hi == mp.inf else hi ** (i - 1) * e(hi)
        bot = lo ** (i - 1) * e(lo)
        J[i] = -(top - bot) + (i - 1) * J[i - 2]
    return J


def _radial_integral(d: int, kappa, a, b):
    """int_a^b u^(d-1) exp(-(u-kappa)^2/2) du, exactly, for 0 <= a < b <= inf."""
    j = d - 1
    lo, hi = a - kappa, (mp.inf if b == mp.inf else b - kappa)
    J = _gauss_moments(j, lo, hi)
    return mp.fsum(mp.binomial(j, i) * kappa ** (j - i) * J[i] for i in range(j + 1))


def M_exact(d: int, kappa, dps: int = 60):
    """M_d(kappa) via the closed form, at `dps` decimal digits."""
    with mp.workdps(dps):
        kappa = mp.mpf(kappa)
        num = _radial_integral(d, kappa, mp.mpf(0), 2 * kappa)
        tail = _radial_integral(d, kappa, 2 * kappa, mp.inf)
        return num / (num + tail)


def M_quad(d: int, kappa, dps: int = 30):
    """M_d(kappa) by adaptive quadrature — independent cross-check of M_exact."""
    with mp.workdps(dps):
        kappa = mp.mpf(kappa)
        f = lambda u: mp.exp((d - 1) * mp.log(u) - (u - kappa) ** 2 / 2) if u > 0 else mp.mpf(0)
        num = mp.quad(f, [0, kappa, 2 * kappa])
        tail = mp.quad(f, [2 * kappa, 2 * kappa + 10, mp.inf])
        return num / (num + tail)


def kappa_exact(d: int, alpha: float = 0.05, dps: int = 60, tol_digits: int = 15):
    """Critical ratio kappa_d solving M_d(kappa) = 1 - alpha, by bisection on the closed form.

    Bisection (not Newton) because M_d is proved monotone: every evaluation certifies a
    bracket [lo, hi] with M(lo) < 1-alpha < M(hi). Returns (kappa, lo, hi) as mpf.
    """
    with mp.workdps(dps):
        target = 1 - mp.mpf(alpha)
        lo, hi = mp.mpf("0.01"), mp.sqrt(mp.mpf(d) / 2) + 4
        assert M_exact(d, lo, dps) < target < M_exact(d, hi, dps)
        tol = mp.mpf(10) ** (-tol_digits)
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if M_exact(d, mid, dps) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2, lo, hi


def kappa_1_closed_form(alpha: float = 0.05) -> float:
    """d = 1: M_1(kappa) = (2 Phi(kappa) - 1) / Phi(kappa)  =>  kappa_1 = Phi^{-1}(1/(1+alpha))."""
    return float(stats.norm.ppf(1.0 / (1.0 + alpha)))


def kappa_asymptotic(d, alpha: float = 0.05):
    """Leading two terms: kappa_d = sqrt(d/2) + z_{1-alpha} * sqrt(3/8) + o(1)."""
    z = stats.norm.ppf(1 - alpha)
    return np.sqrt(np.asarray(d, dtype=float) / 2.0) + z * math.sqrt(3.0 / 8.0)


# ---------------------------------------------------------------- fast float versions

def _log_g(u, d, kappa):
    return (d - 1) * np.log(u) - 0.5 * (u - kappa) ** 2


def _mode(d, kappa):
    # maximiser of log g: (d-1)/u = u - kappa
    return 0.5 * (kappa + math.sqrt(kappa * kappa + 4 * (d - 1)))


def shell_mass(d: int, kappa: float, t: float) -> float:
    """Posterior mass of the shell {|u - kappa| <= t} (float, normalised)."""
    m = _mode(d, kappa)
    shift = _log_g(max(m, 1e-300), d, kappa) if d > 1 else 0.0
    f = lambda u: math.exp(_log_g(u, d, kappa) - shift) if u > 0 else 0.0
    hi_all = max(kappa, m) + 40
    tot, _ = integrate.quad(f, 0, hi_all, points=[m, kappa], limit=400)
    a, b = max(0.0, kappa - t), kappa + t
    part, _ = integrate.quad(f, a, b, points=[x for x in (m, kappa) if a < x < b] or None, limit=400)
    return part / tot


def M(d: int, kappa: float) -> float:
    return shell_mass(d, kappa, kappa)


def kappa_d(d: int, alpha: float = 0.05) -> float:
    return optimize.brentq(lambda k: M(d, k) - (1 - alpha), 1e-3, math.sqrt(d / 2) + 6, xtol=1e-12)


def hpd_halfwidth(d: int, kappa: float, alpha: float = 0.05) -> float:
    """Half-width t (in noise units) of the 1-alpha HPD shell {|u-kappa| <= t}."""
    f = lambda t: shell_mass(d, kappa, t) - (1 - alpha)
    return optimize.brentq(f, 1e-9, kappa + 40, xtol=1e-12)


def has_hole(d: int, kappa: float, alpha: float = 0.05) -> bool:
    return M(d, kappa) > 1 - alpha


@lru_cache(maxsize=None)
def kappa_table(alpha: float = 0.05, d_max: int = 10):
    return {d: kappa_d(d, alpha) for d in range(1, d_max + 1)}
