"""Every two-range noise threshold in the paper, by exact quadrature in residual coordinates (ptopo.exact).

Writes results/data/exact_thresholds.json with
  ladders      symmetric geometry (anchors at +-1), theta = 15..165 step 5, d in {2,3,4,6,10}: the noise level of
               every transition (Theorem 4.3 depths; equal depths merged), alpha = 0.05
  asym         the example D = 2, x* = (0.2, 1.1), d in {2, 3}
  table2       exact first thresholds, the proved sigma_1 of Theorem 5.3, their ratio, and the exact posterior mass
               outside the disc |e| < R at the proved sigma_1 (must be < alpha)
  large_d      first thresholds for d in {10, 40, 160, 640, 2560}, in units where the ranges are sqrt(d)
Quadrature: n = 24 Gauss-Legendre nodes, 6 pieces (node doubling changes thresholds by < 1e-6; tests/test_exact.py).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ptopo.exact import TwoRange  # noqa: E402
from stage4_check import sigma1 as proved_sigma1  # noqa: E402

ALPHA = 0.05


def sym(theta_deg):
    a = np.sin(np.radians(theta_deg) / 2); return 2.0, 1 / a, 1 / a


def ladder(job):
    th, d = job
    D, r1, r2 = sym(th)
    T = TwoRange(D, r1, r2, d, sigma_max=6.0)
    depths = {}
    for name, v in T.events().items():
        depths.setdefault(round(v, 12), []).append(name)
    rungs = [dict(depth=v, events=names, sigma=T.sigma_at_depth(v, lo=1e-3, hi=6.0)) for v, names in sorted(depths.items())]
    return dict(theta=th, d=d, r=r1, rungs=rungs)


def table2(job):
    tag, d = job
    if tag == "asym":
        D = 2.0; r1, r2 = np.hypot(1.2, 1.1), np.hypot(0.8, 1.1)
    else:
        D, r1, r2 = sym(tag)
    T = TwoRange(D, r1, r2, d, sigma_max=3.0)
    tmin2 = min(T.ts, T.t1, T.t2) ** 2
    ex = T.sigma_at_depth(tmin2, lo=1e-3, hi=3.0)
    pr = proved_sigma1(D, r1, r2, d)
    esc = 1 - T.prob_inside(tmin2, pr)
    return dict(case=tag, d=d, exact=ex, proved=pr, ratio=pr / ex, escaping_mass_at_proved=esc)


def large_d(job):
    th, d = job
    D, r1, r2 = sym(th); a = np.sin(np.radians(th) / 2)
    smax = 2.0 / (np.sqrt(d) * a)                 # scaled sigma <= 2
    T = TwoRange(D, r1, r2, d, sigma_max=smax)
    t2min = min(T.ts, T.t1, T.t2) ** 2
    s = T.sigma_at_depth(t2min, lo=smax * 1e-3, hi=smax)
    return dict(theta=th, d=d, sigma_scaled=s * np.sqrt(d) * a)


if __name__ == "__main__":
    with Pool(8) as pool:
        L = pool.map(ladder, [(t, d) for t in range(15, 170, 5) for d in (2, 3, 4, 6, 10)])
        T2 = pool.map(table2, [(t, d) for t in (30, 60, 90, 120, 150) for d in (2, 3, 4, 6, 10)] + [("asym", 2), ("asym", 3)])
        LD = pool.map(large_d, [(t, d) for t in (65, 75, 90, 105, 120, 135, 150) for d in (10, 40, 160, 640, 2560)])
    asym = []
    for d in (2, 3):
        T = TwoRange(2.0, np.hypot(1.2, 1.1), np.hypot(0.8, 1.1), d, sigma_max=3.0)
        asym.append(dict(d=d, rungs=[dict(depth=v, event=k, sigma=T.sigma_at_depth(v, hi=3.0))
                                     for k, v in sorted(T.events().items(), key=lambda kv: kv[1])]))
    out = dict(alpha=ALPHA, method="exact quadrature in residual coordinates (ptopo.exact)", ladders=L, asym=asym,
               table2=T2, large_d=LD)
    (ROOT / "results/data/exact_thresholds.json").write_text(json.dumps(out, indent=1, default=float))
    for r in T2:
        print(r["case"], r["d"], "exact %.5f proved %.5f ratio %.3f escaping %.2e" % (r["exact"], r["proved"], r["ratio"], r["escaping_mass_at_proved"]))
    for a in asym:
        print("asym d", a["d"], [round(g["sigma"], 4) for g in a["rungs"]])
    for r in LD:
        print("large d", r["theta"], r["d"], round(r["sigma_scaled"], 5))
