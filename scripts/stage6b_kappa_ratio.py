"""Stage 06B gate Q4: sigma at which the hole closes (Delta_alpha = Delta(x_R)) against sqrt(2 Delta(x_R)) / kappa_2."""
import numpy as np
from scipy.optimize import root
from three_range_smooth_max_search import grad_hess
A = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]]); K2 = 1.996153
for R in [8, 16, 32]:
    X = np.array([0, -R]); r = np.linalg.norm(A - X, axis=1)
    xr = root(lambda x: grad_hess(x, A, r)[0], [0, 0.05], jac=lambda x: grad_hess(x, A, r)[1]).x
    DR = sum((np.linalg.norm(xr - a) - ri) ** 2 for a, ri in zip(A, r)) / 2
    approx = np.sqrt(2 * DR) / K2
    L = R + 6 * approx; N = 2401
    g = np.linspace(-L, L, N); XX = np.stack(np.meshgrid(g, g, indexing="xy"), -1)
    D = sum((np.linalg.norm(XX - a, axis=-1) - ri) ** 2 for a, ri in zip(A, r)) / 2
    def lev(s):
        p = np.exp(-(D - D.min()) / s ** 2); p /= p.sum(); o = np.sort(p.ravel())[::-1]
        return D[p >= o[np.searchsorted(np.cumsum(o), 0.95)]].max()
    lo, hi = 0.8 * approx, 1.2 * approx
    for _ in range(30):
        m = (lo + hi) / 2; lo, hi = (m, hi) if lev(m) < DR else (lo, m)
    s = (lo + hi) / 2; print(f"R={R}: x_R={xr.round(4)}, sigma_close={s:.4f}, approx={approx:.4f}, ratio={s/approx:.4f}")
