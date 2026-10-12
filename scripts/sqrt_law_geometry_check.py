import numpy as np
rng = np.random.default_rng(7); M = 2_000_000
a1, a2 = rng.normal(size=(M, 2)), rng.normal(size=(M, 2))
r1, r2 = np.linalg.norm(a1, axis=1), np.linalg.norm(a2, axis=1); D = np.linalg.norm(a1 - a2, axis=1)
u = a1 - a2; v = (a1 + a2) / 2; n = np.c_[-u[:, 1], u[:, 0]] / D[:, None]
ell = np.abs((v * n).sum(1)); w = (v * u).sum(1) / D
s = (D + r1 + r2) / 2; tmin = np.minimum.reduce([s - D, s - r1, s - r2])
c = D * np.minimum(D / (4 * s**3), 1 / (2 * s**2))
print("E ell", ell.mean(), "1/sqrt(pi)", 1 / np.sqrt(np.pi), "corr(ell,D)", np.corrcoef(ell, D)[0, 1], "corr(ell,w)", np.corrcoef(ell, w)[0, 1])
print("lower-bound violations", int((tmin < c * ell**2 * (1 - 1e-12)).sum()), "of", M)
T0 = (D >= 1) & (D <= 3) & (np.abs(w) <= 1) & (np.abs(np.abs(w) - D / 2) >= 0.25)
print("P(T0)", T0.mean())
# on T0 with ell <= 0.05: replace ell by values in (0, 0.05] keeping (D, w)
for e in [0.05, 0.02, 0.005]:
    rr1 = np.sqrt((w + D / 2) ** 2 + e**2); rr2 = np.sqrt((w - D / 2) ** 2 + e**2)
    ss = (D + rr1 + rr2) / 2; tm = np.minimum.reduce([ss - D, ss - rr1, ss - rr2])
    print(f"ell={e}: max t_min/ell^2 on T0 = {(tm[T0] / e**2).max():.3f}  (claimed bound 72, numerically 1.31)")
