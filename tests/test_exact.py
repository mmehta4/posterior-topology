"""Tests for the exact quadrature engine (ptopo.exact)."""
import numpy as np
import pytest

from ptopo.exact import TwoRange

ASYM = (2.0, float(np.hypot(1.2, 1.1)), float(np.hypot(0.8, 1.1)))


def test_arcs_partition_the_circle():
    T = TwoRange(*ASYM, d=3)
    for rho in (0.3, 1.0, 1.3, 1.6, 3.0):
        free = sum(hi - lo for lo, hi, *_ in T.arcs(rho))
        # each free arc's midpoint lies inside F
        for lo, hi, *_ in T.arcs(rho):
            m = (lo + hi) / 2; e = rho * np.array([np.cos(m), np.sin(m)])
            assert np.all(T.A @ e < T.b)
        assert 0 < free <= 2 * np.pi + 1e-12


@pytest.mark.parametrize("d,tol", [(2, 5e-6), (3, 1e-9), (10, 1e-9)])
def test_node_doubling(d, tol):
    coarse = TwoRange(*ASYM, d=d, n=24, pieces=6)
    fine = TwoRange(*ASYM, d=d, n=40, pieces=10)
    for v in sorted(coarse.events().values()):
        a, b = coarse.sigma_at_depth(v, hi=3.0), fine.sigma_at_depth(v, hi=3.0)
        assert abs(a - b) <= tol * b


def test_known_values():
    # asymmetric example in the plane (paper, Section 4.4) and the symmetric right angle in d = 3
    T = TwoRange(*ASYM, d=2)
    got = [T.sigma_at_depth(v, hi=3.0) for v in sorted(T.events().values())]
    assert np.allclose(got, [0.282088, 0.517158, 0.608692, 0.708132, 0.789810], atol=2e-6)
    a = np.sin(np.radians(45)); S = TwoRange(2.0, 1 / a, 1 / a, d=3)
    assert abs(S.sigma_at_depth(S.ts ** 2) - 0.241804) < 2e-6


def test_probability_is_monotone_in_depth_and_noise():
    T = TwoRange(*ASYM, d=3)
    p = [T.prob_inside(dl, 0.5) for dl in (0.1, 0.3, 0.6, 1.0)]
    assert all(x < y for x, y in zip(p, p[1:]))
    q = [T.prob_inside(0.5, s) for s in (0.3, 0.5, 0.8)]
    assert all(x > y for x, y in zip(q, q[1:]))
