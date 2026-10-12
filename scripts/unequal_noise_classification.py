"""Unequal noise: randomised checks of the closed-form weighted depths, the annihilation classification, and brute force."""
import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
from unequal_noise_check import geometry, predicted_m, brute, m_to_betti2
rng = np.random.default_rng(3)
# (A) analytic: closed forms vs numerical edge minima, and the classification, on 20000 random cases
bad_cf = bad_cls = first_at_anchor = 0; n_ann = {0: 0, 1: 0, 2: 0}
for _ in range(20000):
    x = np.array([rng.uniform(-3, 3), rng.uniform(0.05, 3)]); s1, s2 = np.exp(rng.uniform(-1.5, 1.5, 2))
    _, _, r1, r2, ts, t1, t2 = geometry(x)
    _, c = predicted_m(0, ts, t1, t2, s1, s2); S = s1**2 + s2**2; rho = (s1**2 - s2**2)/S
    a1 = ((ts + t1)**2/s1**2 + (t1 - ts)**2/s2**2)/2; a2 = ((t2 - ts)**2/s1**2 + (t2 + ts)**2/s2**2)/2
    inside = dict(base=(t1 > rho*ts) and (t2 > -rho*ts), left=ts > rho*t1, right=ts > -rho*t2)
    cf = dict(base=2*ts**2/S if inside['base'] else (a1 if t1 <= rho*ts else a2),
              left=2*t1**2/S if inside['left'] else a1, right=2*t2**2/S if inside['right'] else a2)
    bad_cf += any(abs(cf[k] - c[k]) > 1e-9*(1 + c[k]) for k in cf) or abs(a1 - c['a1']) > 1e-9 or abs(a2 - c['a2']) > 1e-9
    k = sum(not v for v in inside.values()); n_ann[k] += 1
    noisier = 'a1' if s1 > s2 else 'a2'
    if k == 1:
        e = [kk for kk, v in inside.items() if not v][0]
        corner = {'left': 'a1', 'right': 'a2'}.get(e, 'a1' if t1 <= rho*ts else 'a2')
        bad_cls += corner != noisier
    first_at_anchor += min(c['base'], c['left'], c['right']) >= min(c['a1'], c['a2']) - 1e-12 and \
        min(c['base'], c['left'], c['right']) > min(v for kk, v in c.items() if kk in ('base','left','right') and inside[kk]) if any(inside.values()) else 0
print("closed-form depth mismatches:", bad_cf, "| annihilations per case:", n_ann, "| annihilation not at noisier anchor:", bad_cls)
# first event never at an anchor: global min over the boundary attained at a foot inside its edge
fa = 0
for _ in range(20000):
    x = np.array([rng.uniform(-3, 3), rng.uniform(0.05, 3)]); s1, s2 = np.exp(rng.uniform(-1.5, 1.5, 2))
    _, _, r1, r2, ts, t1, t2 = geometry(x); S = s1**2 + s2**2; rho = (s1**2 - s2**2)/S
    feet = [2*ts**2/S if (t1 > rho*ts and t2 > -rho*ts) else np.inf, 2*t1**2/S if ts > rho*t1 else np.inf,
            2*t2**2/S if ts > -rho*t2 else np.inf]
    _, c = predicted_m(0, ts, t1, t2, s1, s2)
    fa += min(feet) > min(c['base'], c['left'], c['right']) + 1e-12
print("cases where the first contact is not at a foot:", fa)
# (B) brute force on 10 random geometries with unequal noise
tot = mism = 0
for _ in range(10):
    x = np.array([rng.uniform(-2, 2), rng.uniform(0.4, 2)]); s1, s2 = np.exp(rng.uniform(-1, 1, 2))
    _, _, r1, r2, ts, t1, t2 = geometry(x); _, crit = predicted_m(0, ts, t1, t2, s1, s2)
    hi = 2.2*np.sqrt(max(crit.values())/np.log(20))
    for sig in np.geomspace(0.04, hi, 14):
        b, delta = brute(x, s1, s2, sig, n=900)
        if min(abs(delta/v - 1) for v in crit.values()) < 0.03: continue
        tot += 1; mism += list(b) != m_to_betti2(predicted_m(delta, ts, t1, t2, s1, s2)[0])
print("brute force: states", tot, "mismatches", mism)
