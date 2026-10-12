"""Stage 04 verification: Meet's explicit staircase bound sigma1 against the exact first threshold.
Writes results/data/stage4_bound_quality.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, critical  # noqa: E402

def sigma1(D, r1, r2, d, alpha=0.05, beta=None):
    s = (D + r1 + r2)/2; ts, t1, t2 = s - D, s - r2, s - r1
    R = np.sqrt(2)*min(ts, t1, t2)
    if d == 2:
        L = np.log(2.4*min(r1, r2)/(alpha*R))
        return R/np.sqrt(2*L + 2*np.log(2*L))
    beta = 1/np.sqrt(d) if beta is None else beta
    cs = [np.sqrt(2)*s, np.sqrt(2)*ts, np.sqrt(2)*t1, np.sqrt(2)*t2]
    H = lambda u: np.prod([1 + u/c for c in cs])
    K = (r1 + R)*(r2 + R)/((r1 - beta*R)*(r2 - beta*R)) * (H(R)/H(-beta*R))**((d - 3)/2)
    kR = R/(r1 + R) + R/(r2 + R) + (d - 3)/2*sum(R/(c + R) for c in cs)
    Phi = lambda y: (2*y/(2*y - kR))*(1 + 1/(2*y - kR))/(1 - np.exp(-beta**2*y))
    y0 = np.log(K/alpha); assert y0 > kR/2
    return R/np.sqrt(2*np.log(K*Phi(y0)/alpha))

def escaping_mass(d, B, r, sigma, R):
    res = analysis.analyze_reduced(d, B, np.asarray(r), sigma, cells=1400)
    S = -2*sigma**2*res.logp
    m = np.exp(res.logp + res.grid.log_weight - (res.logp + res.grid.log_weight).max())
    return float(m[S >= R**2].sum()/m.sum())

if __name__ == "__main__":
    exact = {(r["theta_deg"], r["d"]): r["sigma1"] for r in json.load(open("results/data/stage4_exact_sigma1.json"))["rows"]}
    rows = []
    for th in (30, 60, 90, 120, 150):
        B, r = analysis.symmetric_pair(np.radians(th))
        for d in (2, 3, 4, 6, 10):
            s1 = sigma1(2.0, r[0], r[1], d); ex = exact[(th, d)]
            s = (2 + 2*r[0])/2; R = np.sqrt(2)*min(s - 2, s - r[0])
            esc = escaping_mass(d, B, r, s1, R)
            rows.append(dict(theta=th, d=d, proved=s1, exact=ex, ratio=s1/ex, escaping_mass_at_proved=esc))
    # asymmetric example
    x = np.array([0.2, 1.1]); r1, r2 = np.hypot(x[0]+1, x[1]), np.hypot(x[0]-1, x[1])
    for d, ex in ((2, 0.2821), (3, 0.2880)):
        rows.append(dict(theta="asym", d=d, proved=sigma1(2.0, r1, r2, d), exact=ex, ratio=sigma1(2.0, r1, r2, d)/ex))
    Path("results/data/stage4_bound_quality.json").write_text(json.dumps(rows, indent=1, default=float))
    for t in (30, 60, 90, 120, 150):
        print(t, "  ".join(f"{q['ratio']:.2f}" for q in rows if q["theta"] == t))
    print("asym", [round(q["ratio"], 2) for q in rows if q["theta"] == "asym"])
    esc = [q["escaping_mass_at_proved"] for q in rows if "escaping_mass_at_proved" in q]
    print("escaping mass at proved sigma1: min %.2e max %.2e (allowed 5e-2)" % (min(esc), max(esc)))
