"""Stage 06B checks: Fermat limit, far critical points (astroid), obtuse case, inside-triangle counterexample."""
import numpy as np
from scipy.optimize import root, minimize
from three_range_smooth_max_search import grad_hess
def crits(A, r, box, n=2500, seed=0):
    out = []
    for x0 in np.random.default_rng(seed).uniform(box[0], box[1], (n, 2)):
        sol = root(lambda x: grad_hess(x, A, r)[0], x0, jac=lambda x: grad_hess(x, A, r)[1])
        if not sol.success or np.min(np.linalg.norm(A - sol.x, axis=1)) < 1e-6: continue
        g, H = grad_hess(sol.x, A, r)
        if np.linalg.norm(g) > 1e-7 * max(1, np.abs(sol.x).max()) or any(np.allclose(sol.x, c[0], atol=1e-5) for c in out): continue
        ev = np.linalg.eigvalsh(H); k = "min" if ev.min() > 0 else ("max" if ev.max() < 0 else "saddle")
        out.append((sol.x, k, ev, sum((np.linalg.norm(sol.x - a) - ri) ** 2 for a, ri in zip(A, r)) / 2))
    return out
def anchor_tests(A, r):
    res = []
    for i, a in enumerate(A):
        v = sum((np.linalg.norm(a - b) - rj) * (a - b) / np.linalg.norm(a - b) for j, (b, rj) in enumerate(zip(A, r)) if j != i)
        res.append(round(r[i] - np.linalg.norm(v), 3))
    return res
A = np.array([[2.0, 0], [-2.0, 0], [0, 1.5]])
print("A. Fermat limit, anchors (+-2,0),(0,1.5); x_F =", 2 / np.sqrt(3))
for R in [60, 600, 6000]:
    for th in [-90, -30]:
        X = R * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]); r = np.linalg.norm(A - X, axis=1)
        mx = [c for c in crits(A, r, (-3, 3), n=300) if c[1] == "max"]
        print(f"   R={R} theta={th}: smooth max", [m[0].round(4).tolist() for m in mx])
print("B. far critical points, R = 60")
for th in [-90, -60, 0]:
    X = 60 * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]); r = np.linalg.norm(A - X, axis=1)
    cs = crits(A, r, (-75, 75), n=4000)
    print(f"   theta={th}:", [(c[1], c[0].round(4).tolist(), round(c[3], 3)) for c in sorted(cs, key=lambda c: c[3])])
print("F. obtuse apex (+-2,0),(0,0.5), R=60")
A2 = np.array([[2.0, 0], [-2.0, 0], [0, 0.5]])
for th in [-90, -30, 40]:
    X = 60 * np.array([np.cos(np.radians(th)), np.sin(np.radians(th))]); r = np.linalg.norm(A2 - X, axis=1)
    mx = [c for c in crits(A2, r, (-3, 3), n=400) if c[1] == "max"]
    print(f"   theta={th}: smooth maxima {len(mx)}; anchor tests {anchor_tests(A2, r)}")
print("G. inside-triangle counterexample")
A3 = np.array([[2.2308, -0.2168], [-1.2836, -3.3176], [2.0261, 0.0052]]); X3 = np.array([1.0874, -1.1301]); r3 = np.linalg.norm(A3 - X3, axis=1)
def inside(p, T):
    d = lambda p1, p2, p3: (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    s = [d(p, T[0], T[1]), d(p, T[1], T[2]), d(p, T[2], T[0])]; return all(x > 0 for x in s) or all(x < 0 for x in s)
print("   x* inside:", inside(X3, A3))
for c in crits(A3, r3, (-5, 5), n=3000): 
    if c[1] == "max": print("   smooth max", c[0].round(4), c[2].round(2), round(c[3], 4), "sum r/w", round(sum(r3 / np.linalg.norm(A3 - c[0], axis=1)), 2))
