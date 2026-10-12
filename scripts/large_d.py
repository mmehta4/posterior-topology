"""Large-dimension thresholds: finite-d Morse thresholds converge to the closed form.

Produces results/data/large_d.json and results/figures/fig11_large_d.{pdf,png}.
Units: ranges = sqrt(d) (R = 1).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, asymptotics, critical, radial  # noqa: E402
from ptopo.plotting import save, style  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
DS = [10, 40, 160, 640, 2560]
THETAS = [65, 75, 90, 105, 120, 135, 150]


def job(args):
    th, d = args
    t = np.radians(th)
    B, r = analysis.symmetric_pair(t)
    s = critical.morse_sigma((r[0] - 1) ** 2, critical.depth_fn(d, B, r, cells=1200), lo=1e-5)
    return dict(theta=th, d=d, sigma=float(s * np.sqrt(d) * np.sin(t / 2)))


def main():
    with Pool(8) as pool:
        rows = pool.map(job, [(t, d) for t in THETAS for d in DS])
    k1 = [dict(d=d, sigma=float(np.sqrt(d) / radial.kappa_d(d))) for d in [10, 40, 160, 640, 2560]]
    (OUT / "data/large_d.json").write_text(json.dumps(dict(
        units="ranges = sqrt(d)", two_ranges=rows, one_range=k1,
        closed_form={"one_range": float(np.sqrt(2)),
                     "two_ranges": {th: float(asymptotics.sigma_inf_two_ranges(np.radians(th))) for th in THETAS}}),
        indent=1))
    plt = style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    tt = np.linspace(60, 165, 200)
    a1.plot(tt, asymptotics.sigma_inf_two_ranges(np.radians(tt)), "k-", lw=1.4,
            label=r"$d\to\infty$: $\sigma_\infty^2=8(1-a)^2/(2-a)$")
    for i, d in enumerate(DS):
        xs = [r["theta"] for r in rows if r["d"] == d]
        a1.plot(xs, [r["sigma"] for r in rows if r["d"] == d], "o--", ms=3, lw=.8,
                color=plt.cm.viridis(i / (len(DS) - 1)), label=f"d={d}")
    a1.set_xlabel(r"crossing angle $\theta$ (deg)"); a1.set_ylabel(r"first critical $\sigma$ (ranges $=\sqrt{d}$)")
    a1.set_title("Two ranges: finite-$d$ thresholds converge to the closed form"); a1.legend(fontsize=7, frameon=False)
    for th in (75, 90, 120):
        ys = [r["sigma"] for r in rows if r["theta"] == th]
        a2.plot(np.array(DS) ** -0.5, ys, "o-", ms=3, label=fr"$\theta={th}^\circ$")
        a2.plot([0], [asymptotics.sigma_inf_two_ranges(np.radians(th))], "k*", ms=7)
    a2.plot(np.array([r["d"] for r in k1]) ** -0.5, [r["sigma"] for r in k1], "s-", ms=3, label="one range")
    a2.plot([0], [np.sqrt(2)], "k*", ms=7)
    a2.set_xlabel(r"$d^{-1/2}$"); a2.set_ylabel(r"$\sigma^*$"); a2.set_title(r"Convergence at rate $d^{-1/2}$ (stars: limits)")
    a2.legend(fontsize=7, frameon=False)
    save(fig, "fig11_large_d")


if __name__ == "__main__":
    main()
