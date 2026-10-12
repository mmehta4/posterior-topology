"""Problem 3 — the random version.

Anchors a_i ~ N(0, I_d) iid, truth x* = 0, data r_i = ||a_i|| + sigma * eps_i.

k = 1 (exact, no simulation needed):
    P_hole(sigma, d) = P(R + sigma Z > kappa_d sigma),  R ~ chi_d, Z ~ N(0,1)
                     = E[ Phi(R/sigma - kappa_d) ].
  Theorem 3 (theory/problem3.md): P_hole -> 1{sigma < sqrt(2)} as d -> inf, and at
  sigma = sqrt(2) + tau/sqrt(d):  P_hole -> 1 - Phi((sqrt(2) c + tau/sqrt(2)) / sqrt(5/2)),
  c = z_{0.95} sqrt(3/8).
k = 2: Monte Carlo on the symmetry-reduced grid, any d.
k = d = 2, 3: Monte Carlo on an adaptive mirror-reduced grid (the flip-ambiguity case).

Produces results/data/problem3_*.json and results/figures/fig07_random_curves.{pdf,png}
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, posterior, radial  # noqa: E402
from ptopo.plotting import TYPE_COLORS, save, style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
C = stats.norm.ppf(0.95) * np.sqrt(3 / 8)


def p_hole_k1(sigma, d, kd):
    f = lambda R: stats.chi.pdf(R, d) * stats.norm.cdf(R / sigma - kd)
    lo, hi = max(0, np.sqrt(d) - 12), np.sqrt(d) + 12
    return integrate.quad(f, lo, hi, limit=200)[0]


def scaling_limit(tau):
    return 1 - stats.norm.cdf((np.sqrt(2) * C + tau / np.sqrt(2)) / np.sqrt(2.5))


def mc_k2(args):
    d, sigma, n, seed = args
    rng = np.random.default_rng(seed)
    out = {"contractible": 0, "staircase": 0, "other": 0, "split_or_hole": 0}
    target = analysis.noise_free_betti(d, 2)
    for _ in range(n):
        A = rng.normal(size=(2, d))
        r = np.linalg.norm(A, axis=1) + sigma * rng.normal(size=2)
        B, _, _ = posterior.reduce_anchors(A)
        b = analysis.analyze_reduced(d, B, r, sigma, cells=320).betti
        if b[0] == 1 and b[1:].sum() == 0:
            out["contractible"] += 1
        else:
            out["split_or_hole"] += 1
            out["staircase" if np.array_equal(b, target) else "other"] += 1
    return dict(d=d, sigma=sigma, n=n, **out)


def mc_mirror(args):
    """k = d (the flip-ambiguity case): P(two pieces), adaptive mirror-reduced grid."""
    d, sigma, n, seed = args
    rng = np.random.default_rng(seed)
    split = 0
    for _ in range(n):
        A = rng.normal(size=(d, d))
        r = np.linalg.norm(A, axis=1) + sigma * rng.normal(size=d)
        b0, _ = analysis.analyze_mirror_beta0(A, r, sigma, cells=300 if d == 2 else 64)
        split += int(b0 >= 2)
    return dict(d=d, sigma=sigma, n=n, split=split)


def main(parts=("k1", "k2", "mirror")):
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    if "k1" in parts:
        run_k1()
    if "k2" in parts:
        run_k2()
    if "mirror" in parts:
        run_mirror()
    figure()


def run_k1():
    # ---------------- k = 1, exact
    sig = np.linspace(0.2, 3.0, 141)
    k1 = {}
    for d in [2, 3, 5, 10, 20, 50, 100, 1000]:
        kd = float(radial.kappa_exact(d)[0])
        k1[d] = [p_hole_k1(s, d, kd) for s in sig]
    # Monte Carlo spot check of the exact formula
    rng = np.random.default_rng(1)
    spot = []
    for d, s in [(3, 1.0), (10, 1.3), (50, 1.4)]:
        kd = float(radial.kappa_exact(d)[0])
        R = np.linalg.norm(rng.normal(size=(200000, d)), axis=1)
        mc = np.mean(R + s * rng.normal(size=R.size) > kd * s)
        spot.append(dict(d=d, sigma=s, exact=p_hole_k1(s, d, kd), mc=float(mc)))
    print("k=1 MC spot checks:", spot)
    # finite-size scaling check
    fss = []
    for d in [100, 1000, 10000]:
        kd = float(radial.kappa_exact(d)[0]) if d <= 2000 else float(radial.kappa_asymptotic(d))
        for tau in [-4, -2, 0, 2]:
            s = np.sqrt(2) + tau / np.sqrt(d)
            fss.append(dict(d=d, tau=tau, exact=p_hole_k1(s, d, kd), limit=float(scaling_limit(tau))))
    (OUT / "data/problem3_k1.json").write_text(json.dumps(dict(
        sigma=sig.tolist(), p_hole={str(k): v for k, v in k1.items()}, mc_spot=spot, finite_size=fss), indent=1))


def run_k2():
    sig2 = np.round(np.geomspace(0.1, 4.0, 22), 4)
    jobs = [(d, float(s), 240, 1000 * d + i) for d in [2, 3, 5, 10, 20] for i, s in enumerate(sig2)]
    with Pool(8) as pool:
        k2 = pool.map(mc_k2, jobs)
    (OUT / "data/problem3_k2.json").write_text(json.dumps(k2, indent=1))


def run_mirror():
    sig3 = np.round(np.geomspace(0.02, 2.0, 16), 4)
    jobs = [(d, float(s), 200, 77 * d + i) for d in (2, 3) for i, s in enumerate(sig3)]
    with Pool(8) as pool:
        res = pool.map(mc_mirror, jobs)
    (OUT / "data/problem3_mirror.json").write_text(json.dumps(res, indent=1))


def figure():
    J1 = json.loads((OUT / "data/problem3_k1.json").read_text())
    sig = np.array(J1["sigma"]); k1 = {int(k): v for k, v in J1["p_hole"].items()}
    k2 = json.loads((OUT / "data/problem3_k2.json").read_text())
    sig2 = sorted({r["sigma"] for r in k2})
    km = json.loads((OUT / "data/problem3_mirror.json").read_text())
    plt = style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, (d, p) in enumerate(k1.items()):
        axs[0].plot(sig, p, color=plt.cm.viridis(i / (len(k1) - 1)), label=f"d={d}")
    axs[0].axvline(np.sqrt(2), color="k", ls="--", lw=.8)
    axs[0].text(np.sqrt(2) + .04, .9, r"$\sqrt{2}$", fontsize=8)
    axs[0].set_title("one range: P(95% region has a hole) — exact")
    axs[0].set_xlabel(r"$\sigma$"); axs[0].legend(fontsize=6.5, frameon=False)
    for i, d in enumerate([2, 3, 5, 10, 20]):
        rows = [r for r in k2 if r["d"] == d]
        p = np.array([r["split_or_hole"] / r["n"] for r in rows])
        se = np.sqrt(p * (1 - p) / rows[0]["n"])
        axs[1].errorbar(sig2, p, yerr=1.96 * se, fmt="-o", ms=2.5, lw=1, capsize=0,
                        color=plt.cm.viridis(i / 4), label=f"d={d}")
    axs[1].set_xscale("log"); axs[1].set_xlabel(r"$\sigma$")
    axs[1].set_title("two ranges: P(not one hole-free piece) — MC")
    axs[1].legend(fontsize=6.5, frameon=False)
    for i, d in enumerate((2, 3)):
        rows = [r for r in km if r["d"] == d]
        p3 = np.array([r["split"] / r["n"] for r in rows])
        axs[2].errorbar([r["sigma"] for r in rows], p3, yerr=1.96 * np.sqrt(p3 * (1 - p3) / rows[0]["n"]),
                        fmt="-o", ms=3, color=TYPE_COLORS[i], label=f"k=d={d}")
    axs[2].set_xscale("log"); axs[2].set_xlabel(r"$\sigma$"); axs[2].legend(fontsize=7, frameon=False)
    axs[2].set_title("k = d ranges: P(two mirror pieces) — MC")
    for a in axs:
        a.set_ylim(-.03, 1.03)
    save(fig, "fig07_random_curves")


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("k1", "k2", "mirror"))
