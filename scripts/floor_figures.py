"""Figures 2–5 of the Floor note (Figure 1 is atlas fig01). HPD regions are drawn from the exact
shell formula (Lemma 2.1), not from grids. Output: results/figures/floor/fig{2..5}.pdf"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import radial  # noqa: E402
import ptopo.plotting as P  # noqa: E402

P.FIG = Path(__file__).resolve().parents[1] / "results" / "figures" / "floor"
plt = P.style()
BLUE, ORANGE, PINK = P.TYPE_COLORS[0], P.TYPE_COLORS[1], P.TYPE_COLORS[3]
ALPHA = 0.05
K2 = float(radial.kappa_exact(2)[0])


def density_panel(ax, kappa, lim, extra=None):
    """One-range posterior in the plane, r = 1, sigma = 1/kappa; exact HPD shell."""
    s = 1.0 / kappa
    g = np.linspace(-lim, lim, 600)
    X, Y = np.meshgrid(g, g)
    R = np.hypot(X, Y)
    p = np.exp(-(R - 1) ** 2 / (2 * s * s))
    ax.imshow(p, origin="lower", extent=[-lim, lim, -lim, lim], cmap="Greys", alpha=.5)
    t = radial.hpd_halfwidth(2, kappa, ALPHA)
    inner, outer = max(0.0, (kappa - t) * s), (kappa + t) * s
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(outer * np.cos(th), outer * np.sin(th), color=BLUE, alpha=.45, lw=0)
    if inner > 0:
        ax.fill(inner * np.cos(th), inner * np.sin(th), color="white", lw=0)
        ax.plot(inner * np.cos(th), inner * np.sin(th), color=BLUE, lw=1)
    ax.plot(outer * np.cos(th), outer * np.sin(th), color=BLUE, lw=1)
    ax.plot([0], [0], "k^", ms=6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return inner, outer


def fig2():
    ks = [3.0, 2.5, K2, 1.5]
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
    for k, ax in zip(ks, axs):
        density_panel(ax, k, 2.6)
        lab = r"$r/\sigma=\kappa_2=1.996$" if abs(k - K2) < 1e-9 else fr"$r/\sigma={k:g}$"
        ax.set_title(lab, fontsize=9)
    P.save(fig, "fig2_hole_filling")


def laplace_kappa(d):
    mp.mp.dps = 25
    def M(k):
        k = mp.mpf(k)
        f = lambda u: mp.exp((d - 1) * mp.log(u) - abs(u - k)) if u > 0 else mp.mpf(0)
        a = mp.quad(f, [0, k, 2 * k]); b = mp.quad(f, [2 * k, 2 * k + d + 40 * mp.sqrt(d), mp.inf])
        return a / (a + b)
    return float(mp.findroot(lambda k: M(k) - mp.mpf(1 - ALPHA), d / 2 + 1))


def fig3():
    ds = np.arange(1, 101)
    kd = np.array([radial.kappa_d(int(d), ALPHA) for d in ds])
    dd = np.linspace(1, 100, 400)
    c0, c1 = 1.007263, 0.070436
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.loglog(ds, kd, "o", ms=3, color=BLUE, label=r"$\kappa_d$ (certified)", zorder=3)
    ax.loglog(dd, np.sqrt(dd / 2) + c0, "-", color="k", lw=1.1, label=r"$\sqrt{d/2}+1.0073$")
    ax.loglog(dd, np.sqrt(dd / 2) + c0 + c1 / np.sqrt(dd), ":", color="k", lw=1.1,
              label=r"$\sqrt{d/2}+1.0073+0.0704/\sqrt{d}$")
    ax.loglog(dd, np.sqrt(dd / 2), "--", color="0.55", lw=1, label=r"$\sqrt{d/2}$")
    dl = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    ax.loglog(dl, [laplace_kappa(d) / np.sqrt(2) for d in dl], "x", ms=6, color=ORANGE, mew=1.4,
              label=r"Laplace noise, $\kappa^{L}_d/\sqrt{2}$ (sd units)")
    ax.set_xlabel("dimension $d$"); ax.set_ylabel(r"critical ratio $r/\sigma$ ($\sigma$ = noise standard deviation)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    P.save(fig, "fig3_threshold")


def fig4():
    kappa, d = 4.0, 2
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    density_panel(ax, kappa, 2.25)
    s = 1 / kappa
    f = lambda u, j: u ** j * np.exp(-(u - kappa) ** 2 / 2)
    EU2 = integrate.quad(lambda u: f(u, 3), 0, kappa + 30)[0] / integrate.quad(lambda u: f(u, 1), 0, kappa + 30)[0]
    v = s * s * EU2 / d
    rad = np.sqrt(v * stats.chi2.ppf(1 - ALPHA, d))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(rad * np.cos(th), rad * np.sin(th), "--", color="k", lw=1.2)
    ax.plot([0], [0], "o", color=PINK, ms=7, mec="k", mew=.6, zorder=4)
    ax.annotate("posterior mean (= anchor)", (0, 0), xytext=(-.62, -.42), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=.6))
    ax.text(-2.2, 2.2, "solid: 95% HPD ring\ndashed: matched Gaussian 95% region", fontsize=7.5, va="top")
    P.save(fig, "fig4_mean_matched_gaussian")


def p_hole(sigma, d, kd):
    f = lambda R: stats.chi.pdf(R, d) * stats.norm.cdf(R / sigma - kd)
    lo, hi = max(0, np.sqrt(d) - 14), np.sqrt(d) + 14
    return integrate.quad(f, lo, hi, limit=300)[0]


def fig5():
    c = 1.007263
    kd = {d: float(radial.kappa_exact(d, ALPHA)[0]) for d in (2, 10, 100, 1000)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    sig = np.linspace(0.2, 3, 160)
    for i, d in enumerate((2, 10, 100, 1000)):
        col = plt.cm.viridis(i / 3)
        a1.plot(sig, [p_hole(s, d, kd[d]) for s in sig], color=col, label=f"d = {d}")
        tau = np.linspace(max(-8, np.sqrt(d) * (0.3 - np.sqrt(2))), 5, 120)
        if d >= 10:
            a2.plot(tau, [p_hole(np.sqrt(2) + t / np.sqrt(d), d, kd[d]) for t in tau], color=col, label=f"d = {d}")
    a1.axvline(np.sqrt(2), color="k", ls="--", lw=.8)
    a1.set_xlabel(r"noise $\sigma$"); a1.set_ylabel("P(95% region has a hole)"); a1.legend(fontsize=7.5, frameon=False)
    tau = np.linspace(-8, 5, 200)
    a2.plot(tau, 1 - stats.norm.cdf((np.sqrt(2) * c + tau / np.sqrt(2)) / np.sqrt(2.5)), "k-", lw=1.6, label="limit")
    a2.set_xlabel(r"$\tau=\sqrt{d}\,(\sigma-\sqrt{2})$"); a2.legend(fontsize=7.5, frameon=False)
    P.save(fig, "fig5_sqrt2_transition")


if __name__ == "__main__":
    for f in (fig2, fig3, fig4, fig5) if len(sys.argv) == 1 else [globals()[n] for n in sys.argv[1:]]:
        f()
