"""What the topology does to inference: frequentist coverage and area of 95% regions, two ranges in the plane.

Truth x* = (0.2, 1.1), anchors (+-1, 0), the asymmetric example of Section 4. For each noise level, N data sets
r_i = |x* - a_i| + sigma eps_i. For each data set (flat prior):
  HPD      the 95% highest-posterior-density region (grid, 600 x 600, HPD level from sorted cell masses);
  Laplace  the 95% ellipse of the Gaussian at a posterior mode with covariance the inverse Hessian of -log p there
           (what a Laplace approximation, or an extended Kalman filter linearised at the mode, reports). When the
           posterior has two mirror modes they have equal height, so the mode is chosen at random between them;
  Mixture  the 95% HPD region of the equal-weight mixture of the Laplace Gaussians at the two mirror modes (one
           Gaussian when the mode is on the anchor line);
  Moments  the 95% ellipse of the Gaussian with the posterior's mean and covariance.
Coverage = fraction of data sets whose region contains x*. Writes results/data/coverage.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
A = np.array([[-1.0, 0.0], [1.0, 0.0]]); X = np.array([0.2, 1.1]); R0 = np.linalg.norm(A - X, axis=1)
ALPHA, N, SIGMAS, NG = 0.05, 2000, [0.15, 0.40, 0.56, 0.66, 0.75, 1.00], 600
Q = stats.chi2.ppf(1 - ALPHA, 2)


def S(x, r):
    return sum((np.hypot(x[..., 0] - a[0], x[..., 1] - a[1]) - ri) ** 2 for a, ri in zip(A, r))


def one(args):
    sigma, seed = args
    rng = np.random.default_rng(seed)
    r = R0 + sigma * rng.normal(size=2)
    lim = r.max() + 1.2 + 5 * sigma
    g = np.linspace(-lim, lim, NG); h = g[1] - g[0]
    Xg = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    lp = -S(Xg, r) / (2 * sigma ** 2); lp -= lp.max(); p = np.exp(lp); p /= p.sum()
    order = np.sort(p.ravel())[::-1]; cut = order[np.searchsorted(np.cumsum(order), 1 - ALPHA)]
    hpd_cover = np.exp(-S(X, r) / (2 * sigma ** 2) - (-S(Xg, r) / (2 * sigma ** 2)).max()) / np.exp(lp).sum() >= cut
    hpd_area = (p >= cut).sum() * h * h
    # Laplace at a mode: start from the grid maximum, mirror at random (the two mirror modes have equal height)
    i = np.unravel_index(np.argmax(lp), lp.shape); x0 = Xg[i].copy()
    if rng.random() < 0.5:
        x0[1] = -x0[1]
    m = optimize.minimize(lambda x: S(x, r), x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-14)).x
    eps = 1e-5 * max(1.0, np.abs(m).max()); Hs = np.zeros((2, 2))
    for a_ in range(2):
        for b_ in range(2):
            ea, eb = np.eye(2)[a_] * eps, np.eye(2)[b_] * eps
            Hs[a_, b_] = (S(m + ea + eb, r) - S(m + ea - eb, r) - S(m - ea + eb, r) + S(m - ea - eb, r)) / (4 * eps * eps)
    Hl = Hs / (2 * sigma ** 2)
    ev = np.linalg.eigvalsh(Hl)
    if ev.min() > 0:
        lap_cover = (X - m) @ Hl @ (X - m) <= Q; lap_area = np.pi * Q / np.sqrt(np.linalg.det(Hl))
    else:                                                  # degenerate mode: the ellipse is unbounded
        lap_cover, lap_area = True, np.inf
    if ev.min() > 0:                                       # Laplace mixture at the two mirror modes, weights 1/2
        Hm = np.diag([1.0, -1.0]) @ Hl @ np.diag([1.0, -1.0]); mm = m * np.array([1.0, -1.0])
        comps = [(m, Hl)] if abs(m[1]) < 1e-9 else [(m, Hl), (mm, Hm)]
        def gdens(Z, c, H):
            dZ = Z - c; return np.sqrt(np.linalg.det(H)) / (2 * np.pi) * np.exp(-0.5 * np.einsum("...a,ab,...b->...", dZ, H, dZ))
        q = sum(gdens(Xg, c, H) for c, H in comps) / len(comps); qs = q / q.sum()
        oq = np.sort(qs.ravel())[::-1]; cq = oq[np.searchsorted(np.cumsum(oq), 1 - ALPHA)]
        qx = sum(gdens(X, c, H) for c, H in comps) / len(comps) / q.sum()
        mix_cover, mix_area = bool(qx >= cq), float((qs >= cq).sum() * h * h)
    else:
        mix_cover, mix_area = True, np.inf
    mu = (p[..., None] * Xg).sum((0, 1)); dX = Xg - mu
    C = np.einsum("ij,ija,ijb->ab", p, dX, dX); Ci = np.linalg.inv(C)
    mom_cover = (X - mu) @ Ci @ (X - mu) <= Q; mom_area = np.pi * Q * np.sqrt(np.linalg.det(C))
    mean_in_hpd = np.exp(-S(mu, r) / (2 * sigma ** 2) - (-S(Xg, r) / (2 * sigma ** 2)).max()) / np.exp(lp).sum() >= cut
    return dict(hpd=bool(hpd_cover), hpd_area=float(hpd_area), lap=bool(lap_cover), lap_area=float(lap_area),
                mix=mix_cover, mix_area=mix_area,
                mom=bool(mom_cover), mom_area=float(mom_area), mean_in_hpd=bool(mean_in_hpd))


if __name__ == "__main__":
    rows = []
    with Pool(8) as pool:
        for k, s in enumerate(SIGMAS):
            res = pool.map(one, [(s, [20261007, k, i]) for i in range(N)], chunksize=25)
            f = lambda key: float(np.mean([x[key] for x in res]))
            med = lambda key: float(np.median([x[key] for x in res]))
            row = dict(sigma=s, n=N, hpd=f("hpd"), laplace=f("lap"), mixture=f("mix"), moments=f("mom"),
                       mean_in_hpd=f("mean_in_hpd"), hpd_area=med("hpd_area"), laplace_area=med("lap_area"),
                       mixture_area=med("mix_area"), moments_area=med("mom_area"),
                       degenerate=int(sum(np.isinf(x["lap_area"]) for x in res)))
            rows.append(row)
            print(f"sigma={s:.2f}  coverage HPD {row['hpd']:.3f}  Laplace {row['laplace']:.3f}  mixture {row['mixture']:.3f}  moments {row['moments']:.3f}"
                  f"  | median area HPD {row['hpd_area']:.3f}  Laplace {row['laplace_area']:.3f}  mixture {row['mixture_area']:.3f}  moments {row['moments_area']:.3f}"
                  f"  | mean inside HPD {row['mean_in_hpd']:.3f}", flush=True)
    (ROOT / "results/data/coverage.json").write_text(json.dumps(dict(alpha=ALPHA, truth=X.tolist(), rows=rows), indent=1))
