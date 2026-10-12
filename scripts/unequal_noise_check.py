"""Unequal noise levels sigma_i = sigma*s_i, d = 2: brute-force Betti numbers vs the ellipse-in-half-strip prediction."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ptopo import topology

def geometry(x, D=2.0):
    a1, a2 = np.array([-D/2, 0]), np.array([D/2, 0])
    r1, r2 = np.linalg.norm(x - a1), np.linalg.norm(x - a2); s = (D + r1 + r2)/2
    return a1, a2, r1, r2, s - D, s - r2, s - r1     # t*, t1, t2

def predicted_m(delta, ts, t1, t2, s1, s2):
    """m = #edges met - #corners met, ellipse {e1^2/s1^2 + e2^2/s2^2 <= 2 delta} against the half-strip F."""
    W = np.diag([1/s1**2, 1/s2**2]); q = lambda e: e @ W @ e / 2      # weighted depth
    c1 = np.array([-ts - t1, t1 - ts]); c2 = np.array([t2 - ts, -t2 - ts])  # corners a1, a2
    def edge_min(P, dvec, lo, hi):          # min of q on P + u dvec, u in [lo, hi]
        u = -(dvec @ W @ P)/(dvec @ W @ dvec); u = min(max(u, lo), hi); return q(P + u*dvec)
    base = edge_min(c1, c2 - c1, 0, 1)
    left = edge_min(c1, np.array([1.0, 1.0]), 0, np.inf)    # ray beyond a1: from corner a1 in direction (1,1)
    right = edge_min(c2, np.array([1.0, 1.0]), 0, np.inf)
    edges = sum(v <= delta for v in (base, left, right)); corners = sum(q(c) <= delta for c in (c1, c2))
    return edges - corners, dict(base=base, left=left, right=right, a1=q(c1), a2=q(c2))

def brute(x, s1, s2, sigma, n=1100, alpha=0.05):
    a1, a2, r1, r2, *_ = geometry(x)
    lim = max(r1, r2) + 1 + 5*sigma*max(s1, s2)
    g = np.linspace(-lim, lim, n); X, Y = np.meshgrid(g, g, indexing="ij")
    e1 = np.hypot(X - a1[0], Y) - r1; e2 = np.hypot(X - a2[0], Y) - r2
    dep = (e1**2/s1**2 + e2**2/s2**2)/2                       # depth in units sigma^2
    lp = -dep/sigma**2; m = np.exp(lp - lp.max()); order = np.sort(m.ravel())[::-1]
    c = np.cumsum(order)/order.sum(); level = order[np.searchsorted(c, 1 - alpha)]
    mask = m >= level
    assert not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()), 'region hits grid edge'
    return topology.betti_2d(mask), dep[mask].max()

def m_to_betti2(m): return [2, 0] if m == 0 else [1, m - 1]

if __name__ == "__main__":
    for x, s1, s2 in (((0.2, 1.1), 1, 1), ((-1.5, 0.6), 1, 1), ((-1.5, 0.6), 2, 1), ((0.4, 0.9), 1, 2.5), ((0.3, 1.4), 3, 1)):
        x = np.array(x); a1, a2, r1, r2, ts, t1, t2 = geometry(x)
        _, crit = predicted_m(0, ts, t1, t2, s1, s2)
        print(f"x*={tuple(x)} s=({s1},{s2})  t*={ts:.3f} t1={t1:.3f} t2={t2:.3f}  weighted event depths:",
              {k: round(v, 4) for k, v in crit.items()})
        bad = 0; seq = []
        for sig in np.geomspace(0.05, 1.2, 40):
            b, delta = brute(x, s1, s2, sig)
            if min(abs(delta/v - 1) for v in crit.values()) < 0.03: continue
            mp, _ = predicted_m(delta, ts, t1, t2, s1, s2)
            ok = list(b) == m_to_betti2(mp); bad += (not ok)
            if not seq or seq[-1] != list(b): seq.append(list(b))
        print("   observed sequence of Betti numbers:", seq, " mismatches:", bad)
