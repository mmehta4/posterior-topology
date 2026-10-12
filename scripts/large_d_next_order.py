"""Stage 05 reference values: the large-d rule for two ranges and its 1/sqrt(d) correction.

Symmetric geometry in units where the ranges are sqrt(d) (scaled ranges 1, half-separation a = sin(theta/2)).
For each theta: sigma_inf from the 2D maximiser of phi = log h - |e|^2/(2 sigma^2) in scaled residual coordinates,
the correction coefficient C = sigma^3/(2g), g^2 = e*^T H^{-1} e*, both against the closed forms, and the
two-term prediction sigma_inf - z C/sqrt(d) against the finite-d thresholds in results/data/large_d.json.
Writes results/data/large_d_next_order.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
Z = stats.norm.ppf(0.95)


def numeric(th):
    a = np.sin(np.radians(th) / 2); D = 2 * a

    def logh(e):
        w1, w2 = 1 + e[0], 1 + e[1]; sp = (D + w1 + w2) / 2
        A2 = sp * (sp - D) * (sp - w1) * (sp - w2)
        return 0.5 * np.log(4 * A2 / D ** 2) if A2 > 0 else -np.inf

    def estar(sig):
        f = lambda e: -(logh(e) - e @ e / (2 * sig ** 2))
        return optimize.minimize(f, np.array([0.1, 0.1]), method="Nelder-Mead",
                                 options=dict(xatol=1e-12, fatol=1e-15, maxiter=20000)).x

    target = 2 * ((1 - a) ** 2 if th >= 60 else a ** 2)   # |e|^2 at the first event
    sig = optimize.brentq(lambda s: estar(s) @ estar(s) - target, 0.05, 3, xtol=1e-12)
    e = estar(sig); h = 1e-4; phi = lambda x: logh(x) - x @ x / (2 * sig ** 2); H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            H[i, j] = -(phi(e + ei + ej) - phi(e + ei - ej) - phi(e - ei + ej) + phi(e - ei - ej)) / (4 * h * h)
    return a, sig, sig ** 3 / (2 * np.sqrt(e @ np.linalg.solve(H, e)))


def main():
    finite = json.loads((ROOT / "results/data/large_d.json").read_text())["two_ranges"]
    rows = []
    for th in (30, 45, 59, 65, 75, 90, 105, 120, 135, 150):
        a, s_num, C_num = numeric(th)
        if th >= 60:
            s_cf = np.sqrt(8 * (1 - a) ** 2 / (2 - a)); C_cf = 2 * (1 - a) * np.sqrt(a * a - 4 * a + 6) / (2 - a) ** 1.5
        else:
            s_cf = np.sqrt(2 * a * (1 + 2 * a) / (1 + a)); C_cf = None
        fd = {r["d"]: r["sigma"] for r in finite if r["theta"] == th}
        rows.append(dict(theta=th, a=a, sigma_inf_numeric=s_num, sigma_inf_closed=s_cf, C_numeric=C_num, C_closed=C_cf,
                         finite_d={d: dict(exact=v, two_term=s_num - Z * C_num / np.sqrt(d)) for d, v in fd.items()}))
        print(th, f"sigma_inf {s_num:.6f}/{s_cf:.6f}  C {C_num:.5f}/{C_cf if C_cf is None else round(C_cf, 5)}",
              " ".join(f"{d}:{v['exact']:.4f}/{v['two_term']:.4f}" for d, v in rows[-1]["finite_d"].items()))
    (ROOT / "results/data/large_d_next_order.json").write_text(json.dumps(dict(alpha=0.05, units="ranges = sqrt(d)",
                                                                              rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main()
