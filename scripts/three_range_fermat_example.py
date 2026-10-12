"""Remark 4.7: smooth local maxima of S for exact data, against the weighted Fermat point (equilateral anchors)."""
import numpy as np
from scipy.optimize import root, minimize
from three_range_smooth_max_search import grad_hess
A = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])
def S(x, r): return sum((np.linalg.norm(x - a) - ri) ** 2 for a, ri in zip(A, r))
for X in [(0, -2.5), (0, -4.0), (0, -8.0), (3.0, 4.0), (0.3, 1.4)]:
    X = np.array(X, float); r = np.linalg.norm(A - X, axis=1)
    found = []
    for x0 in np.random.default_rng(0).uniform(-2, 2, (400, 2)):
        sol = root(lambda x: grad_hess(x, A, r)[0], x0, jac=lambda x: grad_hess(x, A, r)[1])
        if not sol.success or np.min(np.linalg.norm(A - sol.x, axis=1)) < 1e-6: continue
        g, H = grad_hess(sol.x, A, r); ev = np.linalg.eigvalsh(H)
        if np.linalg.norm(g) < 1e-9 and ev.max() < 0 and not any(np.allclose(sol.x, f[0], atol=1e-7) for f in found):
            found.append((sol.x, ev))
    # weighted Fermat point of sum r_i w_i
    F = minimize(lambda x: sum(ri * np.linalg.norm(x - a) for a, ri in zip(A, r)), [0.01, 0.02], method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14)).x
    print("x*", X, "r", r.round(3), "| smooth maxima:", [(f[0].round(4).tolist(), f[1].round(2).tolist(), round(S(f[0], r) / 2, 4)) for f in found], "| Fermat pt", F.round(4),
          "| anchor depths", [round(S(a, r) / 2, 3) for a in A])
