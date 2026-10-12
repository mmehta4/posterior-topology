"""Correctness tests. Run:  ~/.venvs/posterior-topology/bin/python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, critical, hpd, posterior, radial, topology  # noqa: E402


# ------------------------------------------------------------------ Problem 0
def test_kappa1_closed_form():
    assert abs(radial.kappa_d(1) - radial.kappa_1_closed_form()) < 1e-9


@pytest.mark.parametrize("d", [1, 2, 3, 7])
def test_closed_form_matches_quadrature(d):
    for k in (0.5, 1.7, 3.2):
        assert abs(float(radial.M_exact(d, k)) - float(radial.M_quad(d, k))) < 1e-20


def test_kappa_known_values():
    assert abs(radial.kappa_d(2) - 1.99615275773) < 1e-9
    assert abs(radial.kappa_d(3) - 2.230604565176) < 1e-9


def test_kappa_increasing_in_d():
    ks = [radial.kappa_d(d) for d in range(1, 11)]
    assert all(b > a for a, b in zip(ks, ks[1:]))


# ------------------------------------------------------------------ topology engine
def _disk(n=121):
    x = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.hypot(X, Y)


def test_betti_2d_shapes():
    R = _disk()
    assert topology.betti_2d(abs(R - 1) < .2) == [1, 1]
    assert topology.betti_2d(R < .5) == [1, 0]


def test_betti_3d_matches_gudhi():
    g = np.linspace(-2, 2, 49)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    shapes = [abs(np.sqrt(X**2 + Y**2 + Z**2) - 1.3) < .3,
              np.hypot(np.hypot(X, Y) - 1.2, Z) < .35]
    for m in shapes:
        assert topology.betti_3d(m) == topology.betti_gudhi(m)


@pytest.mark.parametrize("C,m,n,d,expected", [
    ([1, 0], 0, 1, 3, [1, 1, 0, 0]),   # disk off axis, rotated in 3D -> solid torus
    ([1, 0], 1, 1, 3, [1, 0, 0, 0]),   # touches axis once -> ball
    ([1, 0], 2, 1, 3, [1, 0, 1, 0]),   # touches twice -> hollow shell
    ([1, 0], 0, 0, 2, [2, 0, 0]),      # reflection: two blobs
    ([1, 0], 2, 0, 2, [1, 1, 0]),      # reflection, two touches: ring
    ([1, 1], 0, 1, 3, [1, 2, 1, 0]),   # annulus x S^1 = hollow torus
])
def test_lift(C, m, n, d, expected):
    assert topology.lift_betti(C, m, n, d).tolist() == expected


def test_reduced_vs_full_3d():
    """Symmetry reduction must agree with a brute-force 3D grid."""
    A = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    x = np.array([0.0, 0.8, 0.5])
    r = np.linalg.norm(A - x, axis=1)
    B, _, _ = posterior.reduce_anchors(A)
    for s in (0.1, 0.5):
        red = analysis.analyze_reduced(3, B, r, s, cells=500).betti
        full = analysis.analyze_full(A, r, s, cells=110).betti
        assert red[:3].tolist() == full.tolist() and red[3] == 0


def test_hpd_mass():
    x = np.linspace(-8, 8, 2001)
    logp = -x**2 / 2
    level, mask, info = hpd.hpd_level(logp)
    half = x[mask].max()
    assert abs(half - 1.95996) < 0.01


# ------------------------------------------------------------------ Problem 2 structure
def test_incircle_critical_values():
    rng = np.random.default_rng(0)
    for _ in range(200):
        D = rng.uniform(.5, 3); x = rng.normal(size=2) * 2
        r1, r2 = np.hypot(x[0] + D / 2, x[1]), np.hypot(x[0] - D / 2, x[1])
        s = (D + r1 + r2) / 2
        cps = critical.axis_critical_points([[-D / 2], [D / 2]], [r1, r2])
        smooth = sorted(c["S"] / 2 for c in cps if c["smooth"])
        assert np.allclose(smooth, sorted([(s - D) ** 2, (s - r1) ** 2, (s - r2) ** 2]))


def test_kappa_threshold_in_reduced_pipeline():
    k2 = radial.kappa_d(3)
    below = analysis.analyze_reduced(3, np.zeros((1, 0)), [1.0], 1 / (k2 * 0.98), cells=3000).betti
    above = analysis.analyze_reduced(3, np.zeros((1, 0)), [1.0], 1 / (k2 * 1.02), cells=3000).betti
    assert below.tolist() == [1, 0, 0, 0] and above.tolist() == [1, 0, 1, 0]


def test_M2_closed_form():
    from scipy.stats import norm
    for k in (0.3, 1.0, 1.99615275773, 3.5):
        closed = k * (2 * norm.cdf(k) - 1) / (k * norm.cdf(k) + norm.pdf(k))
        assert abs(closed - float(radial.M_exact(2, k))) < 1e-12


def test_large_d_closed_form():
    from ptopo import asymptotics
    for th in (1.2, 1.6, 2.2):
        assert abs(asymptotics.sigma_inf_two_ranges(th) - asymptotics.sigma_inf_numeric_two_ranges(th)) < 1e-5
