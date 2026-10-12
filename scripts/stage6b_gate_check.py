"""Stage 06B gate checks: weighted Fermat point (Q1), spurious minimum and false ambiguity (Q2), anchor emission on a
115-degree triangle (Q3), four anchors with two smooth maxima (Q5)."""
import numpy as np
from scipy.optimize import root, minimize
from scipy import ndimage
def gh(x, A, r, w):
    g = np.zeros(2); H = np.zeros((2, 2))
    for a, ri, wi in zip(A, r, w):
        d = x - a; dist = np.linalg.norm(d); u = d / dist; e = dist - ri
        g += 2 * wi * e * u; H += 2 * wi * (np.outer(u, u) + e / dist * (np.eye(2) - np.outer(u, u)))
    return g, H
def crits(A, r, w, box, n=1500, seed=0):
    out = []
    for x0 in np.random.default_rng(seed).uniform(box[0], box[1], (n, 2)):
        sol = root(lambda x: gh(x, A, r, w)[0], x0, jac=lambda x: gh(x, A, r, w)[1])
        if not sol.success or np.min(np.linalg.norm(A - sol.x, axis=1)) < 1e-7: continue
        g, H = gh(sol.x, A, r, w)
        if np.linalg.norm(g) > 1e-7 * max(1, np.abs(sol.x).max()) or any(np.linalg.norm(sol.x - c[0]) < 1e-4 * max(1, np.abs(sol.x).max()) for c in out): continue
        ev = np.linalg.eigvalsh(H); out.append((sol.x, "min" if ev.min() > 0 else ("max" if ev.max() < 0 else "saddle"), sum(wi * (np.linalg.norm(sol.x - a) - ri) ** 2 for a, ri, wi in zip(A, r, w)) / 2))
    return out
E = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])
print("Q1 weighted Fermat, R = 300 (theta = -90)")
X = np.array([0, -300.0]); r = np.linalg.norm(E - X, axis=1)
for s in [(1, 1, 3), (1, 1.83, 1.83)]:
    w = 1 / np.array(s, float) ** 2
    mx = [c for c in crits(E, r, w, (-1.5, 1.5), n=400) if c[1] == "max"]
    fw = minimize(lambda x: sum(wi * np.linalg.norm(x - a) for a, wi in zip(E, w)), [0.01, 0.02], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14)).x
    cen = (w[:, None] * E).sum(0) / w.sum()
    v = sum(wj * (np.linalg.norm(E[0] - b) - rj) * (E[0] - b) / np.linalg.norm(E[0] - b) for b, rj, wj in zip(E[1:], r[1:], w[1:]))
    print(f"   s={s}: smooth max {[m[0].round(4).tolist() for m in mx]}, weighted Fermat {fw.round(4)}, weighted centroid {cen.round(4)}, a1 weighted test margin/R {(w[0]*r[0]-np.linalg.norm(v))/300:.3f}")
print("Q2 spurious minimum, anchors (+-2,0),(0,1.5), x* = (0,-20)")
A = np.array([[2.0, 0], [-2.0, 0], [0, 1.5]]); X = np.array([0, -20.0]); r = np.linalg.norm(A - X, axis=1); w = np.ones(3)
print("   ", [(c[1], c[0].round(3).tolist(), round(c[2], 3)) for c in sorted(crits(A, r, w, (-25, 25), n=3000), key=lambda c: c[2])])
L, N = 40.0, 2001; g = np.linspace(-L, L, N); XX = np.stack(np.meshgrid(g, g, indexing="xy"), -1)
D = sum((np.linalg.norm(XX - a, axis=-1) - ri) ** 2 for a, ri in zip(A, r)) / 2
for sig in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    p = np.exp(-(D - D.min()) / sig ** 2); p /= p.sum(); o = np.sort(p.ravel())[::-1]; Rg = p >= o[np.searchsorted(np.cumsum(o), 0.95)]
    lab, k = ndimage.label(~Rg); border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    print(f"   sigma={sig}: pieces {ndimage.label(Rg)[1]}, holes {k - len(border)}")
print("Q3 emission on a 115-degree isosceles triangle, apex a1")
h = 1 / np.tan(np.radians(57.5)); T = np.array([[0, h], [1.0, 0], [-1.0, 0]])
def test1(R):
    X = np.array([0, -R]); r = np.linalg.norm(T - X, axis=1)
    v = sum((np.linalg.norm(T[0] - b) - rj) * (T[0] - b) / np.linalg.norm(T[0] - b) for b, rj in zip(T[1:], r[1:]))
    return r[0] - np.linalg.norm(v)
from scipy.optimize import brentq
Rc = brentq(test1, 5, 200); print("   apex test fails at R =", round(Rc, 3))
X = np.array([0, -46.0]); r = np.linalg.norm(T - X, axis=1)
print("   R=46 critical points near the cluster:", [(c[1], c[0].round(4).tolist(), "dist to apex %.4f" % np.linalg.norm(c[0] - T[0])) for c in crits(T, r, np.ones(3), (-1.2, 1.2), n=800)])
print("Q5 four anchors, two smooth maxima")
A4 = np.array([[2.4153, -0.1525], [2.4712, 0.0281], [-3.478, 1.7811], [-3.2678, 1.3587]]); X4 = np.array([-1.0069, -1.0872]); r4 = np.linalg.norm(A4 - X4, axis=1)
cs = crits(A4, r4, np.ones(4), (-5, 5), n=4000)
print("   ", [(c[1], c[0].round(4).tolist(), round(c[2], 2)) for c in sorted(cs, key=lambda c: c[2])])
