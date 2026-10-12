"""Problem 2 — critical noise for two range measurements, every dimension.

Produces
  results/data/problem2_ladder.json        Morse ladder (sigma of every transition, verified)
                                           for symmetric geometries, theta x d
  results/data/problem2_phase_d{2,3}.npz   brute-force phase diagrams (theta x sigma)
  results/figures/fig04_phase_diagram.{pdf,png}
  results/figures/fig04b_first_event_map.{pdf,png}
  results/figures/fig09_ladder_vs_d.{pdf,png}
Units: anchors at z = -1, +1 (half-separation = 1).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, critical  # noqa: E402
from ptopo.plotting import TYPE_COLORS, betti_label, save, style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
THETAS_DEG = list(range(15, 170, 5))
DIMS = [2, 3, 4, 6, 10]
CELLS = 700


def ladder_job(args):
    theta_deg, d = args
    B, r = analysis.symmetric_pair(np.radians(theta_deg))
    rungs = critical.morse_ladder(d, B, r, cells=CELLS)
    for g in rungs:
        g["points"] = [[float(z), bool(sm)] for z, sm in g["points"]]
    return dict(theta_deg=theta_deg, d=d, r=float(r[0]), rungs=rungs)


def phase_job(args):
    theta_deg, d, sigmas = args
    B, r = analysis.symmetric_pair(np.radians(theta_deg))
    return [analysis.analyze_reduced(d, B, r, s, cells=400).betti.tolist() for s in sigmas]


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        ladders = pool.map(ladder_job, [(t, d) for t in THETAS_DEG for d in DIMS])
    chi2 = stats.chi2.ppf(0.95, 2)
    for L in ladders:
        for g in L["rungs"]:
            g["small_noise_approx"] = float(np.sqrt(2 * g["Delta"] / chi2))
    n_bad = sum(1 for L in ladders for g in L["rungs"] if np.isfinite(g["sigma"]) and not g["changes"])
    print("rungs that did NOT change topology:", n_bad)
    (OUT / "data/problem2_ladder.json").write_text(json.dumps(dict(
        units="anchor half-separation = 1", alpha=0.05, cells=CELLS,
        note="sigma solves sigma^2 q_alpha(sigma) = Delta; before/after = Betti numbers at sigma(1 -/+ 1%)",
        ladders=ladders), indent=1))

    # brute-force phase diagrams
    sig = np.geomspace(0.02, 3.0, 90)
    th = np.arange(15, 166, 2.5)
    phases = {}
    with Pool(8) as pool:
        for d in (2, 3):
            res = pool.map(phase_job, [(t, d, sig) for t in th])
            phases[d] = np.array(res)  # (theta, sigma, d+1)
            np.savez(OUT / f"data/problem2_phase_d{d}.npz", theta_deg=th, sigma=sig, betti=phases[d])

    plt = style()
    # ---- Fig 4: phase diagrams with Morse curves overlaid
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, d in zip(axes, (2, 3)):
        B = phases[d]
        types = sorted({tuple(b) for b in B.reshape(-1, d + 1)}, key=lambda b: (-b[0], b))
        idx = np.array([[types.index(tuple(b)) for b in row] for row in B])
        cmap = plt.matplotlib.colors.ListedColormap(TYPE_COLORS[: len(types)])
        ax.pcolormesh(th, sig, idx.T, cmap=cmap, shading="nearest", vmin=-.5, vmax=len(types) - .5)
        for key, ls in (("inner", "-"), ("outer", "--"), ("anchor", ":")):
            xs, ys = [], []
            for L in ladders:
                if L["d"] != d:
                    continue
                r = L["r"]
                D_in, D_out = (r - 1) ** 2, 1.0
                target = {"inner": D_in, "outer": D_out, "anchor": D_in + D_out}[key]
                for g in L["rungs"]:
                    if abs(g["Delta"] - target) < 1e-8 and np.isfinite(g["sigma"]):
                        xs.append(L["theta_deg"]); ys.append(g["sigma"])
            ax.plot(xs, ys, ls, color="k", lw=1.1, label=f"Morse: {key}")
        ax.axvline(60, color="w", lw=0.8, ls="-.")
        ax.set_yscale("log"); ax.set_xlabel(r"crossing angle $\theta$ (deg)")
        ax.set_title(f"two ranges, $d={d}$")
        handles = [plt.matplotlib.patches.Patch(color=TYPE_COLORS[i], label=betti_label(t)) for i, t in enumerate(types)]
        ax.legend(handles=handles + ax.get_lines()[:3], fontsize=7, loc="lower left", frameon=True)
    axes[0].set_ylabel(r"noise $\sigma$ (units of half-separation)")
    save(fig, "fig04_phase_diagram")

    # ---- Fig 4b: which event comes first, over all true positions x* (geometry only)
    zs, hs = np.linspace(-3, 3, 601), np.linspace(0.01, 3, 300)
    Z, H = np.meshgrid(zs, hs)
    r1, r2 = np.hypot(Z + 1, H), np.hypot(Z - 1, H)
    s = (2 + r1 + r2) / 2
    D_in, D_o1, D_o2 = (s - 2) ** 2, (s - r1) ** 2, (s - r2) ** 2
    first = np.where(D_in < np.minimum(D_o1, D_o2), 0, 1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.contourf(Z, H, first, levels=[-.5, .5, 1.5], colors=[TYPE_COLORS[0], TYPE_COLORS[1]], alpha=.85)
    ax.contour(Z, H, np.sqrt(np.minimum(D_in, np.minimum(D_o1, D_o2))), levels=8, colors="k", linewidths=.4)
    ax.plot([-1, 1], [0, 0], "k^", ms=7)
    ax.set_aspect("equal"); ax.set_xlabel("$z^*$"); ax.set_ylabel("$h^*$")
    ax.set_title("First transition: blue = pieces merge / hole fills,  orange = ring (shell) forms\n"
                 "boundary: anchor separation is the longest side of the triangle", fontsize=8)
    save(fig, "fig04b_first_event_map")

    # ---- Fig 9: sigma* of the first transition vs theta, one curve per d
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, d in enumerate(DIMS):
        xs = [L["theta_deg"] for L in ladders if L["d"] == d]
        ys = [min(g["sigma"] for g in L["rungs"] if np.isfinite(g["sigma"])) for L in ladders if L["d"] == d]
        ax.plot(xs, ys, "-o", ms=2.5, color=TYPE_COLORS[i], label=f"d={d}")
    xs = [L["theta_deg"] for L in ladders if L["d"] == 2]
    ax.plot(xs, [min(g["small_noise_approx"] for g in L["rungs"]) for L in ladders if L["d"] == 2],
            "k--", lw=.8, label=r"$\sqrt{2\Delta_{\min}/\chi^2_{2,0.95}}$")
    ax.axvline(60, color="0.6", lw=.8, ls="-.")
    ax.set_xlabel(r"crossing angle $\theta$ (deg)"); ax.set_ylabel(r"first critical noise $\sigma^*$")
    ax.set_yscale("log"); ax.legend(fontsize=7, frameon=False)
    save(fig, "fig09_sigma_star_vs_theta")


if __name__ == "__main__":
    main()
