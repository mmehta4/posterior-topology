"""Search for smooth local maxima of S = sum (|x - a_i| - r_i)^2, three anchors in the plane."""
import numpy as np
from scipy.optimize import root
from multiprocessing import Pool
def grad_hess(x, A, r):
    g = np.zeros(2); H = np.zeros((2, 2))
    for a, ri in zip(A, r):
        d = x - a; w = np.linalg.norm(d); u = d / w; e = w - ri
        g += 2 * e * u
        H += 2 * np.outer(u, u) + 2 * e / w * (np.eye(2) - np.outer(u, u))
    return g, H
def trial(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 2)); mode = seed % 3
    if mode == 0: r = rng.uniform(0.01, 4, 3)                       # arbitrary ranges
    else:
        X = rng.normal(size=2) * (1 if mode == 1 else 3); r = np.linalg.norm(A - X, axis=1) + rng.normal(size=3) * rng.choice([0.01, 0.3, 1.0])
    out = []
    lo, hi = A.min(0) - 5, A.max(0) + 5
    for x0 in rng.uniform(lo, hi, (60, 2)):
        sol = root(lambda x: grad_hess(x, A, r)[0], x0, jac=lambda x: grad_hess(x, A, r)[1], method="hybr")
        if not sol.success: continue
        x = sol.x
        if np.min(np.linalg.norm(A - x, axis=1)) < 1e-6: continue
        g, H = grad_hess(x, A, r)
        if np.linalg.norm(g) > 1e-9: continue
        ev = np.linalg.eigvalsh(H)
        if ev.max() < -1e-9: out.append((x.tolist(), ev.tolist(), (r / np.linalg.norm(A - x, axis=1)).sum()))
    return seed, out, A.tolist(), r.tolist()
if __name__ == "__main__":
    found = []; N = 60000
    with Pool(8) as P:
        for seed, out, A, r in P.imap_unordered(trial, range(N), chunksize=200):
            if out: found.append((seed, out, A, r))
    print("trials", N, "with a smooth local max:", len(found))
    for f in found[:5]: print(f)
