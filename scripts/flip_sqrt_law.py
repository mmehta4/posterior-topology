"""Theorem 7.3 and Conjecture 7.4 (two ranges in the plane): the failure probability by the exact criterion.

Anchors a1, a2 ~ N(0, I_2), truth 0, r_i = |a_i| + sigma eps_i. The 95% region is exactly two pieces iff the observed
ranges satisfy the strict triangle inequality (Remark 4.6) and the posterior mass outside the disc |e| < sqrt2 t_min
is below alpha (Proposition 5.1), computed by exact quadrature (ptopo.exact); when sigma is below the certified level
of Theorem 5.3(i) the answer is two pieces without computation. Common random numbers across sigma.
Fits log P(fail) = c + gamma log sigma by weighted least squares, with a bootstrap interval for gamma.
Writes results/data/flip_sqrt_law.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptopo.exact import TwoRange  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from stage4_check import sigma1 as certified_sigma1  # noqa: E402  (Theorem 5.3(i))

ALPHA, N, SEED = 0.05, 4000, 20261006
SIGMAS = np.geomspace(0.004, 0.08, 7)


def fails(args):
    i, sigma = args
    rng = np.random.default_rng([SEED, i])
    a1, a2 = rng.normal(size=2), rng.normal(size=2); eps = rng.normal(size=2)
    r1, r2 = np.linalg.norm(a1) + sigma * eps[0], np.linalg.norm(a2) + sigma * eps[1]
    D = np.linalg.norm(a1 - a2)
    if not (r1 + r2 > D and abs(r1 - r2) < D):
        return 1, 1                                   # inconsistent data: never two pieces
    if sigma <= certified_sigma1(D, r1, r2, 2, ALPHA):
        return 0, 0                                   # two pieces, certified by Theorem 5.3(i)
    T = TwoRange(D, r1, r2, 2, sigma_max=sigma, n=16, pieces=4)
    tmin2 = min(T.ts, T.t1, T.t2) ** 2
    return int(T.prob_inside(tmin2, sigma) <= 1 - ALPHA), 0


if __name__ == "__main__":
    rows = []
    with Pool(8) as pool:
        for s in SIGMAS:
            res = pool.map(fails, [(i, float(s)) for i in range(N)], chunksize=50)
            f = np.array([a for a, _ in res]); inc = np.array([b for _, b in res])
            rows.append(dict(sigma=float(s), n=N, fail=int(f.sum()), inconsistent=int(inc.sum())))
            print(f"sigma={s:.4f}: P(not two pieces) = {f.mean():.4f} (inconsistent data {inc.mean():.4f})", flush=True)
    sg = np.array([r["sigma"] for r in rows]); p = np.array([r["fail"] / N for r in rows])
    X = np.c_[np.ones_like(sg), np.log(sg)]; w = N * p / (1 - p)          # inverse variance of log p-hat
    coef = np.linalg.lstsq(X * np.sqrt(w)[:, None], np.log(p) * np.sqrt(w), rcond=None)[0]
    rng = np.random.default_rng(1); boots = []
    for _ in range(2000):
        pb = rng.binomial(N, p) / N
        wb = N * pb / (1 - pb)
        boots.append(np.linalg.lstsq(X * np.sqrt(wb)[:, None], np.log(pb) * np.sqrt(wb), rcond=None)[0][1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = dict(alpha=ALPHA, n_per_sigma=N, rows=rows, gamma=float(coef[1]), gamma_ci95=[float(lo), float(hi)])
    (ROOT / "results/data/flip_sqrt_law.json").write_text(json.dumps(out, indent=1))
    print(f"fitted exponent gamma = {coef[1]:.3f}, 95% bootstrap interval [{lo:.3f}, {hi:.3f}]")
