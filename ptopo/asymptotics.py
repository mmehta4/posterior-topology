"""Large-dimension thresholds (theory/problem3_random.md, Section "Large-d Morse rule").

Rescale lengths by sqrt(d): ranges r_i = R_i sqrt(d), S = d * S~. In reduced coordinates the
weighted log-mass is  d * F(y, eta) + O(log d)  with
        F = -S~(y, eta) / (2 sigma^2) + log eta .
As d -> inf the 1-alpha HPD cut converges (at rate d^{-1/2}) to the density at the maximiser
m*(sigma) of F, independent of alpha. Hence the large-d Morse rule:

    a critical point c is crossed at the sigma where  S~(m*(sigma)) = S~(c).

Closed forms (R = 1):
    one range:                       sigma_inf = sqrt(2)
    two ranges, crossing theta>=60:  sigma_inf^2 = 8 (1 - a)^2 / (2 - a),   a = sin(theta/2)
"""
from __future__ import annotations

import numpy as np
from scipy import optimize


def sigma_inf_one_range(R=1.0):
    return np.sqrt(2.0) * R


def sigma_inf_two_ranges(theta, R=1.0):
    a = np.sin(np.asarray(theta) / 2.0)
    return R * np.sqrt(8.0 * (1.0 - a) ** 2 / (2.0 - a))


def sigma_inf_numeric_two_ranges(theta, R=1.0):
    """Same quantity by direct maximisation of F — check of the closed form."""
    a = np.sin(theta / 2.0)
    S_c = 2 * (a - 1) ** 2  # inner (incircle) critical point at the origin

    def S_on_line(eta):
        return 2 * (np.sqrt(a * a + eta * eta) - 1) ** 2

    def gap(sig):
        eta = optimize.minimize_scalar(lambda e: S_on_line(e) / (2 * sig ** 2) - np.log(e),
                                       bounds=(1e-6, 10), method="bounded").x
        return S_on_line(eta) - S_c

    return R * optimize.brentq(gap, 1e-3, 5.0)
