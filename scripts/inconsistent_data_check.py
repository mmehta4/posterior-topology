"""Lemma 4.6 (inconsistent data): brute-force Betti numbers of the 95% region over 25 noise levels, 90 random cases."""
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
def betti(D, r1, r2, s, N=700):
    L = max(abs(r1), abs(r2)) + D + 4 * s + 1
    g = np.linspace(-L, L, N); X, Y = np.meshgrid(g, g, indexing="ij")
    S = (np.hypot(X + D / 2, Y) - r1) ** 2 + (np.hypot(X - D / 2, Y) - r2) ** 2
    p = np.exp(-(S - S.min()) / (2 * s * s)); p /= p.sum()
    o = np.sort(p.ravel())[::-1]; c = o[np.searchsorted(np.cumsum(o), 0.95)]
    R = p >= c
    b0 = ndimage.label(R)[1]
    lab, k = ndimage.label(~R)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    return b0, k - len(border)
def case(i):
    rng = np.random.default_rng(i); D = 2.0
    kind = i % 3
    if kind == 0:   # separated
        r1 = rng.uniform(0.05, 1.9); r2 = rng.uniform(0.0, D - r1 - 0.02)
    elif kind == 1: # nested r1 >= r2 + D, r1 + r2 > D
        r2 = rng.uniform(0.05, 2.0); r1 = r2 + D + rng.uniform(0.0, 1.5)
    else:           # one negative range
        r1 = rng.uniform(-0.5, -0.01); r2 = rng.uniform(0.0, 4.0)
    seq = [betti(D, r1, r2, s) for s in np.geomspace(0.03, 3, 25)]
    ok_conn = all(b0 == 1 for b0, _ in seq)
    holes = [b1 for _, b1 in seq]
    pred_sep = (r1 + r2 <= D)
    ok = ok_conn and (max(holes) == 0 if pred_sep else max(holes) <= 1)
    return i, kind, ok, ok_conn, max(holes), (r1, r2)
if __name__ == "__main__":
    with Pool(8) as P: res = P.map(case, range(90))
    bad = [r for r in res if not r[2]]
    print("cases", len(res), "violations", len(bad)); print(bad[:10])
    print("nested cases with a hole:", sum(1 for r in res if r[1] == 1 and r[4] == 1), "of", sum(1 for r in res if r[1] == 1))
