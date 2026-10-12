"""Lemma 4.6: the nested cases whose ring phase is narrow, checked at the noise level midway between the predicted depths."""
import numpy as np
from scipy import ndimage
from inconsistent_data_check import case
def level_and_betti(D, r1, r2, s, N=1400):
    L = max(abs(r1), abs(r2)) + D + 4 * s + 1
    g = np.linspace(-L, L, N); X, Y = np.meshgrid(g, g, indexing="ij")
    S = (np.hypot(X + D / 2, Y) - r1) ** 2 + (np.hypot(X - D / 2, Y) - r2) ** 2
    p = np.exp(-(S - S.min()) / (2 * s * s)); p /= p.sum()
    o = np.sort(p.ravel())[::-1]; c = o[np.searchsorted(np.cumsum(o), 0.95)]
    R = p >= c; lev = S[R].max()
    lab, k = ndimage.label(~R)
    border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    return lev, ndimage.label(R)[1], k - len(border)
from multiprocessing import Pool
def run(i):
    _, kind, ok, _, mh, (r1, r2) = case(i)
    if kind != 1 or mh == 1: return None
    D = 2.0; lo_c = (r1 - r2 + D) ** 2 / 2; hi_c = r1 ** 2 + (D - r2) ** 2; target = (lo_c + hi_c) / 2
    a, b = 0.01, 5.0
    for _ in range(40):
        m = (a + b) / 2; lev = level_and_betti(D, r1, r2, m, N=500)[0]
        a, b = (m, b) if lev < target else (a, m)
    lev, b0, b1 = level_and_betti(D, r1, r2, (a + b) / 2)
    return i, (round(r1, 3), round(r2, 3)), round(lo_c, 3), round(hi_c, 3), round(lev, 3), b0, b1
if __name__ == "__main__":
    with Pool(8) as P:
        for r in P.map(run, range(90)):
            if r: print(r)
