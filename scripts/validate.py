"""Validation suite — guards against grid artefacts masquerading as topology.

  V1  Conjecture Q evidence: sigma^2 q_alpha(sigma) strictly increasing, random geometries, k = 1, 2
  V2  resolution convergence of Morse thresholds (cells 400 -> 1600)
  V3  reduced (symmetry) pipeline vs brute-force full grids, d = 2, 3
  V4  sample-based cross-check: posterior samples inside C_alpha + alpha complex vs grid Betti numbers
Writes results/data/validation.json
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, critical, posterior  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def v1_job(seed):
    rng = np.random.default_rng(seed)
    k = 1 + seed % 2
    d = int(rng.choice([2, 3, 5, 10]))
    if k == 1:
        B, r = np.zeros((1, 0)), [1.0]
    else:
        th = rng.uniform(np.radians(20), np.radians(160))
        z = rng.uniform(-0.8, 0.8)
        h = 1 / np.tan(th / 2)
        B = np.array([[-1.0], [1.0]])
        r = [np.hypot(z + 1, h), np.hypot(z - 1, h)]
    s = np.geomspace(0.02, 4, 40)
    psi = np.array([x * x * analysis.analyze_reduced(d, B, r, x, cells=500).depth for x in s])
    return dict(k=k, d=d, monotone=bool(np.all(np.diff(psi) > 0)), min_ratio=float(np.min(psi[1:] / psi[:-1])))


def v2():
    out = []
    for th, d in [(30, 2), (90, 3), (120, 5)]:
        B, r = analysis.symmetric_pair(np.radians(th))
        row = dict(theta=th, d=d)
        for cells in (400, 800, 1600):
            row[cells] = critical.morse_sigma((min((r[0] - 1) ** 2, 1.0)), critical.depth_fn(d, B, r, cells=cells))
        out.append(row)
    return out


def v3():
    out = []
    rng = np.random.default_rng(5)
    for _ in range(12):
        d = int(rng.choice([2, 3]))
        A = np.zeros((2, d)); A[0, 0], A[1, 0] = -1, 1
        x = rng.normal(size=d) * .8; x[1] = abs(x[1]) + .3
        r = np.linalg.norm(A - x, axis=1)
        s = float(rng.choice([0.08, 0.2, 0.4, 0.8]))
        B, _, _ = posterior.reduce_anchors(A)
        red = analysis.analyze_reduced(d, B, r, s, cells=700).betti[:d].tolist()
        full = analysis.analyze_full(A, r, s, cells=500 if d == 2 else 120).betti.tolist()
        out.append(dict(d=d, sigma=s, reduced=red, full=full, agree=red == full))
    return out


def v4():
    import gudhi
    rng = np.random.default_rng(11)
    cases = [("ring", np.zeros((1, 2)), [1.0], 0.2), ("filled", np.zeros((1, 2)), [1.0], 0.6),
             ("two blobs", np.array([[-1.0, 0], [1, 0]]), [1.3, 1.3], 0.12),
             ("ring from shallow pair", np.array([[-1.0, 0], [1, 0]]), [3.864, 3.864], 0.8)]
    out = []
    for name, A, r, s in cases:
        res = analysis.analyze_full(A, r, s, cells=500)
        g = res.grid
        P = np.exp(res.logp - res.logp.max()).ravel(); P /= P.sum()
        idx = rng.choice(P.size, 6000, p=P)
        X = np.stack([ax.ravel() for ax in g.mesh()], 1)[idx] + (rng.random((6000, 2)) - .5) * g.h
        lp = posterior.log_lik_full(X, A, r, s)
        inside = X[lp >= res.log_level]
        ac = gudhi.AlphaComplex(points=inside).create_simplex_tree()
        ac.compute_persistence()
        eps = 2.5 * np.sqrt(np.prod(res.grid.h)) * 6  # ball radius ~ local sample spacing
        b = ac.persistent_betti_numbers(eps ** 2, eps ** 2)
        out.append(dict(case=name, grid=res.betti.tolist(), samples_alpha=[int(v) for v in b[:2]],
                        agree=res.betti.tolist() == [int(v) for v in b[:2]]))
    return out


def main():
    with Pool(8) as pool:
        v1 = pool.map(v1_job, range(48))
    res = dict(
        V1_conjecture_Q=dict(cases=len(v1), all_monotone=all(r["monotone"] for r in v1), detail=v1),
        V2_resolution=v2(), V3_reduced_vs_full=v3(), V4_samples_alpha=v4())
    print("V1 monotone in", sum(r["monotone"] for r in v1), "/", len(v1))
    print("V2", res["V2_resolution"])
    print("V3 agree", sum(r["agree"] for r in res["V3_reduced_vs_full"]), "/", len(res["V3_reduced_vs_full"]))
    print("V4", res["V4_samples_alpha"])
    (OUT / "data/validation.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
