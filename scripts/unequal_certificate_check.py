"""Stage 06A Task 3: unequal-noise certificate sigma_1 (Meet's formulas) vs brute-force escaping mass."""
import numpy as np
from scipy.optimize import brentq
ALPHA = 0.05

def geom(x, D=2.0):
    r1 = np.hypot(x[0] + D / 2, x[1]); r2 = np.hypot(x[0] - D / 2, x[1]); s = (D + r1 + r2) / 2
    return D, r1, r2, s, s - D, s - r2, s - r1          # D, r1, r2, s, t*, t1, t2

def sigma1(x, s1, s2, d, alpha=ALPHA):
    D, r1, r2, s, ts, t1, t2 = geom(x); tmin = min(ts, t1, t2); R = np.sqrt(2) * tmin
    sb = np.sqrt((s1**2 + s2**2) / 2); RW = R / sb; Ri = [s1 / sb * R, s2 / sb * R]; r = [r1, r2]; sv = [s1, s2]
    if d == 2:
        best = 0
        for i in (0, 1):
            j = 1 - i; rho = r[i] / (sv[j] * RW); q = sv[i] / sv[j]
            G = lambda y: (2 * rho + (rho + q / 2) / y) / (1 - np.exp(-y))
            L = np.log(G(np.log(2 * rho / alpha)) / alpha)
            best = max(best, RW / np.sqrt(2 * L + 2 * np.log(2 * L)))
        return best
    beta = 1 / np.sqrt(d); C = np.sqrt(2) * np.array([s, ts, t1, t2])
    Th = lambda u: np.prod(1 + u / C)
    K = (r1 + Ri[0]) * (r2 + Ri[1]) / ((r1 - beta * Ri[0]) * (r2 - beta * Ri[1])) * (Th(R) / Th(-beta * R)) ** ((d - 3) / 2)
    g = Ri[0] / (r1 + Ri[0]) + Ri[1] / (r2 + Ri[1]) + (d - 3) / 2 * np.sum(R / (C + R))
    Phi = lambda y: 2 * y / (2 * y - g) * (1 + 1 / (2 * y - g)) / (1 - np.exp(-beta**2 * y))
    y0 = np.log(K / alpha); assert 2 * y0 > g
    return RW / np.sqrt(2 * np.log(K * Phi(y0) / alpha))

def escaping(x, s1, s2, d, sig, N=1600):
    """Posterior mass with S_W = sum e_i^2/s_i^2 >= R_W^2, half-plane grid with weight rho^(d-2)."""
    D, r1, r2, s, ts, t1, t2 = geom(x); tmin = min(ts, t1, t2)
    sb = np.sqrt((s1**2 + s2**2) / 2); RW2 = 2 * tmin**2 / sb**2
    ext = max(r1, r2) + 1 + 10 * sig * max(s1, s2)
    xi = np.linspace(-ext, ext, N); rh = np.linspace(0, ext, N // 2 + 1)[1:] - ext / N
    Xi, Rh = np.meshgrid(xi, rh, indexing="ij")
    e1 = np.hypot(Xi + D / 2, Rh) - r1; e2 = np.hypot(Xi - D / 2, Rh) - r2
    SW = e1**2 / s1**2 + e2**2 / s2**2
    w = np.exp(-(SW - SW.min()) / (2 * sig**2)) * Rh ** (d - 2)
    return w[SW >= RW2].sum() / w.sum()

if __name__ == "__main__":
    import sys
    rows = []
    for x in [(0.2, 1.1), (0.0, 1.0), (0.3, 1.4), (0.0, 3.7)]:
        for (s1, s2) in [(1, 1), (2, 1), (1, 4)]:
            line = []
            for d in (2, 3, 4):
                sg = sigma1(x, s1, s2, d); m = escaping(x, s1, s2, d, sg)
                st = brentq(lambda z: escaping(x, s1, s2, d, z) - ALPHA, sg, 20 * sg, xtol=1e-5)
                line.append(f"d={d}: sig1={sg:.4f} mass={m:.2e} ratio={sg/st:.3f}")
                rows.append(m)
            print(x, (s1, s2), " | ".join(line), flush=True)
    print("mass range", min(rows), max(rows))
