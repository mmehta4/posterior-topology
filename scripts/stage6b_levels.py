"""Stage 06B: noise levels at which the 95% region (x* = (0,-8), unit equilateral anchors, d = 2) forms its ring
(depth of the far saddle) and closes its hole (depth of the smooth maximum)."""
import numpy as np
A = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]]); X = np.array([0, -8.0]); r = np.linalg.norm(A - X, axis=1)
def level(s, L, N):
    g = np.linspace(-L, L, N); XX = np.stack(np.meshgrid(g, g, indexing="ij"), -1)
    D = sum((np.linalg.norm(XX - a, axis=-1) - ri) ** 2 for a, ri in zip(A, r)) / 2
    p = np.exp(-(D - D.min()) / s**2); p /= p.sum(); o = np.sort(p.ravel())[::-1]
    return D[p >= o[np.searchsorted(np.cumsum(o), 0.95)]].max()
for target, L, N, lo, hi in [(2.9883, 13.0, 2601, 0.5, 2.0), (75.0469, 45.0, 2001, 5.0, 8.0)]:
    a, b = lo, hi
    for _ in range(35):
        m = (a + b) / 2; a, b = (m, b) if level(m, L, N) < target else (a, m)
    print(f"Delta_alpha = {target}: sigma = {(a + b) / 2:.4f}")
