"""Table 1 of the Floor note: kappa_d(alpha) for d = 1..10, alpha = 0.05, 0.01.

Method (Appendix B): closed form of M_d via truncated-normal moments at 60 significant digits,
bisection on M_d(kappa) - (1 - alpha) to a bracket of width 1e-15 (valid since M_d is increasing),
cross-checked by mpmath adaptive quadrature at 30 digits. Writes results/data/table1.json.
"""
import json
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import radial  # noqa: E402

rows = []
for a in (0.05, 0.01):
    for d in range(1, 11):
        k, lo, hi = radial.kappa_exact(d, a)
        resid = radial.M_quad(d, k) - (1 - mp.mpf(a))
        assert abs(resid) < 1e-12, (a, d, resid)
        rows.append(dict(alpha=a, d=d, kappa=mp.nstr(k, 15), table=f"{float(k):.6f}",
                         quadrature_residual=mp.nstr(resid, 3)))
        print(a, d, rows[-1]["table"], rows[-1]["quadrature_residual"])
out = Path(__file__).resolve().parents[1] / "results/data/table1.json"
out.write_text(json.dumps(rows, indent=1))
print("wrote", out)
