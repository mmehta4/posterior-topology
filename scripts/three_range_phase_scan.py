"""Remark 4.7 / Stage 06B: for which noise-free points x* does S have a smooth local maximum? Equilateral anchors."""
import numpy as np
from scipy.optimize import root
from multiprocessing import Pool
from three_range_smooth_max_search import grad_hess
A = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])
def has_max(X):
    r = np.linalg.norm(A - X, axis=1); rng = np.random.default_rng(0); n = 0
    for x0 in rng.uniform(-3, 3, (150, 2)):
        sol = root(lambda x: grad_hess(x, A, r)[0], x0, jac=lambda x: grad_hess(x, A, r)[1])
        if not sol.success or np.min(np.linalg.norm(A - sol.x, axis=1)) < 1e-6: continue
        g, H = grad_hess(sol.x, A, r)
        if np.linalg.norm(g) < 1e-8 and np.linalg.eigvalsh(H).max() < 0: return 1
    return 0
if __name__ == "__main__":
    g = np.linspace(-9, 9, 73); pts = [np.array([x, y]) for x in g for y in g if np.min(np.linalg.norm(A - np.array([x, y]), axis=1)) > 1e-3]
    with Pool(8) as P: res = P.map(has_max, pts, chunksize=20)
    pts = np.array(pts); res = np.array(res); d = np.linalg.norm(pts, axis=1)
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 13)]:
        m = (d >= lo) & (d < hi); print(f"|x*| in [{lo},{hi}): smooth max in {res[m].mean():.2f} of {m.sum()}")
    np.savez("../results/data/three_range_phase_scan.npz", pts=pts, res=res)
