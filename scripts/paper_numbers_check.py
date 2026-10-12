"""Recompute every closed-form number quoted in the paper's text (alpha = 0.05) and report OK/FAIL."""
import numpy as np
from scipy import optimize, stats

z = stats.norm.ppf(0.95); ok = []


def chk(name, got, want, tol=0):
    good = bool(got) if isinstance(want, bool) else abs(float(got) - want) <= tol
    ok.append((name, good, want))


chk("z*sqrt(3/8)", z * np.sqrt(3 / 8), 1.00726, 5e-6)
chk("c_alpha", (13 * z * z - 28) / (72 * np.sqrt(2)), 0.0704, 5e-5)
chk("kappa_1", stats.norm.ppf(1 / 1.05), 1.6684, 5e-5)
f = lambda k: k * (2 * stats.norm.cdf(k) - 1) - 0.95 * (k * stats.norm.cdf(k) + stats.norm.pdf(k))
chk("kappa_2", optimize.brentq(f, 1, 3), 1.9962, 5e-5)
chk("Laplace kappa_1", np.log(10.5), 2.351, 5e-4)
chk("Laplace kappa_1 in sd", np.log(10.5) / np.sqrt(2), 1.663, 5e-4)
chk("sqrt2 scaling limit", 1 - stats.norm.cdf((np.sqrt(2) * z * np.sqrt(3 / 8)) / np.sqrt(2.5)), 0.1838, 5e-5)
chk("2 delta_0", 2 * z * np.sqrt(3 / 8), 2.015, 5e-4)
a = 1 / np.sqrt(2)
chk("sigma_inf(90)", np.sqrt(8 * (1 - a) ** 2 / (2 - a)), 0.7286, 5e-5)
chk("z C(90)", z * 2 * (1 - a) * np.sqrt(a * a - 4 * a + 6) / (2 - a) ** 1.5, 1.256, 5e-4)
chk("outer contacts at 90: sigma^2", 2 * a * (1 + 2 * a) / (1 + a), 2.0, 1e-12)
u = np.sqrt(2 - np.sqrt(2)); w = 1 + u
chk("anchors at 90", np.sqrt(2 * u * (w * w - a * a) / w), 1.5062, 5e-5)
chk("ln(48 sqrt 2) > 4.2", np.log(48 * np.sqrt(2)) > 4.2, True)
mu = np.sqrt(2)
chk("G(4.2) < 2.36 mu", (2 * mu + (mu + 0.5) / 4.2) / (1 - np.exp(-4.2)) / mu < 2.36, True)
r1, r2 = np.hypot(1.2, 1.1), np.hypot(0.8, 1.1); s = (2 + r1 + r2) / 2
for nm, v, want in (("t*^2", (s - 2) ** 2, 0.244), ("t2^2", (s - r1) ** 2, 0.750), ("t1^2", (s - r2) ** 2, 1.286),
                    ("t*^2+t2^2", (s - 2) ** 2 + (s - r1) ** 2, 0.994), ("t*^2+t1^2", (s - 2) ** 2 + (s - r2) ** 2, 1.530)):
    chk(nm, v, want, 5e-4)
chk("Chernoff rate", 0.5 - 1 + np.log(2), 0.193, 5e-4)
chk("closed forms agree at 60 deg", abs(8 * 0.25 / 1.5 - 2 * 0.5 * 2 / 1.5) < 1e-12, True)
for n_, g, want in ok:
    print("OK  " if g else "FAIL", n_, want)
print("ALL OK:", all(g for _, g, _ in ok), f"({len(ok)} checks)")
