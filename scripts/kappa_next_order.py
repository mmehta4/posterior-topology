"""sqrt(d) * (kappa_d - sqrt(d/2) - z sqrt(3/8)) at alpha = 0.05, converging to c = (13 z^2 - 28)/(72 sqrt 2).
d <= 2000: 60-digit closed form (radial.kappa_exact); d >= 1e4: 30-digit adaptive quadrature + bisection.
Writes results/data/kappa_next_order.json (used by Figure 3b)."""
import json, sys
from pathlib import Path
import mpmath as mp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import radial  # noqa: E402
mp.mp.dps = 30
z = mp.sqrt(2) * mp.erfinv(mp.mpf("0.9")); c0 = z * mp.sqrt(mp.mpf(3) / 8)
def M(d, k):
    k = mp.mpf(k); u0 = (k + mp.sqrt(k * k + 4 * (d - 1))) / 2
    phi = lambda u: (d - 1) * mp.log(u) - (u - k) ** 2 / 2; f0 = phi(u0); g = lambda u: mp.exp(phi(u) - f0)
    s = 1 / mp.sqrt(1 + (d - 1) / u0 ** 2); pts = sorted(set(u0 + j * s for j in range(-14, 15)))
    lo = [p for p in pts if p < 2 * k] + [2 * k]; hi = [2 * k] + [p for p in pts if p > 2 * k] + [mp.inf]
    a, b = mp.quad(g, lo), mp.quad(g, hi); return a / (a + b)
def kap_quad(d):
    lo = mp.sqrt(mp.mpf(d) / 2) + 0.9; hi = lo + 0.3
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if M(d, m) < 0.95 else (lo, m)
    return (lo + hi) / 2
rows = []
for d in [10, 15, 20, 30, 50, 100, 200, 400, 1000, 2000, 10**4, 10**5, 10**6]:
    k = radial.kappa_exact(d)[0] if d <= 2000 else kap_quad(d)
    v = mp.sqrt(d) * (k - mp.sqrt(mp.mpf(d) / 2) - c0)
    rows.append(dict(d=d, value=float(v))); print(d, mp.nstr(v, 6))
out = Path(__file__).resolve().parents[1] / "results/data/kappa_next_order.json"
out.write_text(json.dumps(dict(c_limit=float((13 * z**2 - 28) / (72 * mp.sqrt(2))), rows=rows), indent=1))
