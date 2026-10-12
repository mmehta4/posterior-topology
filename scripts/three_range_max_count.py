"""Stage 06B Task 4(iii): how many distinct smooth local maxima can S have at once? Random anchors, exact and noisy data."""
import numpy as np
from scipy.optimize import root
from multiprocessing import Pool
from three_range_smooth_max_search import grad_hess
def trial(seed):
    rng = np.random.default_rng([7, seed]); A = rng.normal(size=(3, 2))
    X = rng.normal(size=2) * rng.choice([1, 3, 8]); r = np.linalg.norm(A - X, axis=1) + rng.normal(size=3) * rng.choice([0, 0.05, 0.3])
    lo, hi = A.min(0) - 3, A.max(0) + 3; maxima = []
    for x0 in rng.uniform(lo, hi, (80, 2)):
        sol = root(lambda x: grad_hess(x, A, r)[0], x0, jac=lambda x: grad_hess(x, A, r)[1])
        if not sol.success or np.min(np.linalg.norm(A - sol.x, axis=1)) < 1e-6: continue
        g, H = grad_hess(sol.x, A, r)
        if np.linalg.norm(g) < 1e-9 and np.linalg.eigvalsh(H).max() < 0 and not any(np.linalg.norm(sol.x - m) < 1e-5 for m in maxima):
            maxima.append(sol.x)
    return len(maxima)
if __name__ == "__main__":
    with Pool(8) as P: counts = np.array(P.map(trial, range(30000), chunksize=100))
    print({k: int((counts == k).sum()) for k in sorted(set(counts))})
