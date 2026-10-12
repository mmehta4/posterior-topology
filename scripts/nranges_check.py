"""Check the general criterion for n = d = 3: region is two pieces iff Delta_alpha < Delta_A (min depth on anchor plane)."""
import numpy as np
from scipy import optimize, ndimage
A = np.array([[1.0, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0]])
X = np.array([0.2, 0.1, 0.5]); R = np.linalg.norm(A - X, axis=1)
def S(x): return sum((np.linalg.norm(x - a, axis=-1) - r) ** 2 for a, r in zip(A, R))
# min depth on the plane z = 0 (multistart)
best = min((optimize.minimize(lambda p: S(np.array([p[0], p[1], 0.0])), p0, method="Nelder-Mead",
            options=dict(xatol=1e-12, fatol=1e-15)) for p0 in np.random.default_rng(0).uniform(-2, 2, (60, 2))),
           key=lambda o: o.fun)
DA = best.fun / 2
print("r =", R.round(4), " Delta_A =", round(DA, 6), " at", best.x.round(4))
N = 281; L = 1.1
g = np.linspace(-L, L, N); h = g[1] - g[0]
XX = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1); SS = S(XX); Dg = SS / 2
def level(sig, alpha=0.05):
    lp = -Dg / sig**2; p = np.exp(lp - lp.max()); p /= p.sum()
    o = np.sort(p.ravel())[::-1]; c = o[np.searchsorted(np.cumsum(o), 1 - alpha)]
    return Dg[p >= c].max()           # Delta_alpha on the grid
def pieces(sig, alpha=0.05):
    lp = -Dg / sig**2; p = np.exp(lp - lp.max()); p /= p.sum()
    o = np.sort(p.ravel())[::-1]; c = o[np.searchsorted(np.cumsum(o), 1 - alpha)]
    lab, k = ndimage.label(p >= c, structure=np.ones((3, 3, 3)))
    return k
lo, hi = 0.01, 2.0
for _ in range(40):
    m = (lo + hi) / 2
    lo, hi = (m, hi) if level(m) < DA else (lo, m)
sstar = (lo + hi) / 2
print("sigma* (Delta_alpha = Delta_A) ~", round(sstar, 4))
for f in [0.5, 0.8, 0.95, 1.05, 1.2, 1.6]:
    print(f"  sigma = {f:.2f} sigma*: Delta_alpha/Delta_A = {level(f*sstar)/DA:.3f}, components = {pieces(f*sstar)}")
