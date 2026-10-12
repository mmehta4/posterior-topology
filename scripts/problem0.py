"""Problem 0 — one measurement, every dimension.

Produces
  results/data/problem0_kappa_table.{json,csv}   certified kappa_d, d = 1..10 (+ large d)
  results/data/problem0_checks.json              monotonicity checks (Theorem 0 a, b)
  results/figures/fig03_threshold_curve.{pdf,png}
"""
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import radial  # noqa: E402
from ptopo.plotting import save, style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
ALPHA = 0.05


def main():
    rows = []
    for d in list(range(1, 11)) + [15, 20, 30, 50, 100, 200, 400, 1000, 2000]:
        k, lo, hi = radial.kappa_exact(d, ALPHA)
        quad = radial.M_quad(d, k)  # independent cross-check
        rows.append(dict(
            d=d, kappa=float(k), kappa_str=mp.nstr(k, 13), bracket=[mp.nstr(lo, 17), mp.nstr(hi, 17)],
            M_at_kappa_quad=float(quad), asymptotic=float(radial.kappa_asymptotic(d, ALPHA)),
            excess=float(k - mp.sqrt(mp.mpf(d) / 2)),
        ))
        print(f"d={d:5d}  kappa_d={rows[-1]['kappa_str']}  excess={rows[-1]['excess']:.6f}  quad M={float(quad):.12f}")
    assert abs(rows[0]["kappa"] - radial.kappa_1_closed_form(ALPHA)) < 1e-12

    # Theorem 0 numerical sanity: M_d increasing in kappa; M_{d+1} < M_d.
    # (exact arithmetic: in floats M saturates at 1.0 for large kappa and diffs read as 0)
    ks = [mp.mpf(i) / 25 for i in range(1, 201)]
    checks = {}
    for d in range(1, 11):
        Ms = [radial.M_exact(d, k, 40) for k in ks]
        Ms1 = [radial.M_exact(d + 1, k, 40) for k in ks]
        checks[d] = dict(increasing_in_kappa=all(b > a for a, b in zip(Ms, Ms[1:])),
                         decreasing_in_d=all(b < a for a, b in zip(Ms, Ms1)),
                         kappa_range=[0.04, 8.0], points=len(ks))
    print(checks)

    (OUT / "data").mkdir(parents=True, exist_ok=True)
    meta = dict(alpha=ALPHA, z=float(radial.stats.norm.ppf(1 - ALPHA)),
                asymptotic_constant=float(radial.stats.norm.ppf(1 - ALPHA) * np.sqrt(3 / 8)),
                kappa_1_closed_form="Phi^{-1}(1/(1+alpha))",
                method="closed-form truncated-normal moments at 60 digits, bisection to 1e-15; "
                       "cross-checked by adaptive quadrature")
    (OUT / "data/problem0_kappa_table.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    with open(OUT / "data/problem0_kappa_table.csv", "w") as f:
        f.write("d,kappa_d,sqrt(d/2)+z*sqrt(3/8),excess\n")
        for r in rows:
            f.write(f"{r['d']},{r['kappa_str']},{r['asymptotic']:.6f},{r['excess']:.6f}\n")
    (OUT / "data/problem0_checks.json").write_text(json.dumps(checks, indent=1))

    # Figure 3 — threshold curve
    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    small = [r for r in rows if r["d"] <= 10]
    dd = np.linspace(1, 10, 200)
    a1.plot(dd, radial.kappa_asymptotic(dd), "--", color="0.55", lw=1, label=r"$\sqrt{d/2}+z_{0.95}\sqrt{3/8}$")
    a1.plot([r["d"] for r in small], [r["kappa"] for r in small], "o-", color="C0", label=r"$\kappa_d$ (exact)")
    a1.set_xlabel("dimension $d$"); a1.set_ylabel(r"critical ratio $\kappa_d = r/\sigma$")
    a1.set_title("Hole exists iff $r/\\sigma > \\kappa_d$"); a1.legend(frameon=False)
    a2.semilogx([r["d"] for r in rows], [r["excess"] for r in rows], "o-", color="C0")
    a2.axhline(meta["asymptotic_constant"], ls="--", color="0.55", lw=1)
    a2.text(1.2, meta["asymptotic_constant"] - 0.012, r"$z_{0.95}\sqrt{3/8}=1.0073$", color="0.4", fontsize=8)
    a2.set_xlabel("dimension $d$"); a2.set_ylabel(r"$\kappa_d-\sqrt{d/2}$"); a2.set_title("Second-order term")
    save(fig, "fig03_threshold_curve")


if __name__ == "__main__":
    main()
