"""Remark 4.7: at the depths between the last anchor and the smooth maximum, the 95% region has one hole, around no anchor."""
import numpy as np
from scipy import ndimage
A = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]]); X = np.array([0, -8.0])
r = np.linalg.norm(A - X, axis=1); M = np.array([0.0, 0.2214])
def S(x): return sum((np.linalg.norm(x - a, axis=-1) - ri) ** 2 for a, ri in zip(A, r))
th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
for e in [1e-3, 1e-2, 5e-2]:
    ring = M + e * np.c_[np.cos(th), np.sin(th)]; print("circle", e, "S(M) - max S(ring) =", S(M) - S(ring).max())
N = 2001; L = 45.0
g = np.linspace(-L, L, N); XX = np.stack(np.meshgrid(g, g, indexing="ij"), -1); SS = S(XX); D = (SS - SS.min()) / 2
def state(s):
    p = np.exp(-D / s**2); p /= p.sum(); o = np.sort(p.ravel())[::-1]; c = o[np.searchsorted(np.cumsum(o), 0.95)]
    R = p >= c; lev = D[R].max(); lab, k = ndimage.label(~R)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    holes = [j for j in range(1, k + 1) if j not in border]
    idx = lambda x: tuple(np.round((np.asarray(x) + L) / (2 * L) * (N - 1)).astype(int))
    contents = [(("M" if lab[idx(M)] == j else "") + "".join(f"a{i+1}" for i, a in enumerate(A) if lab[idx(a)] == j)) for j in holes]
    return lev, ndimage.label(R)[1], contents
lo, hi = 1.0, 30.0
for target in [74.6, 74.9]:
    a, b = lo, hi
    for _ in range(50):
        m = (a + b) / 2; a, b = (m, b) if state(m)[0] < target else (a, m)
    print("target depth", target, "sigma", round((a + b) / 2, 4), state((a + b) / 2))
