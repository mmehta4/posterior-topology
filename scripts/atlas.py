"""The atlas of posterior shapes — figures 1, 2, 5, 6, 8 (3, 4, 7, 9 come from problem scripts).

  fig01  2D staircase: one, two, three ranges at low noise (ring, two blobs, one blob)
  fig02  the hole filling in: one range, noise rising, kappa_2 marked
  fig05  3D staircase: shell, ring-shaped tube, two blobs (isosurfaces of the 95% region)
  fig06  persistence diagrams with the 95% cut drawn as a line
  fig08  the mean in the hole
  fig10  the two-range ladder in 2D: blobs -> ring -> ... for a shallow crossing
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis, hpd, posterior, radial, topology  # noqa: E402
from ptopo.plotting import TYPE_COLORS, betti_label, save, style  # noqa: E402

plt = style()
A2 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.3, -1.4]])
X_TRUE = np.array([0.2, 1.1])


def ranges(anchors, x=X_TRUE):
    return np.linalg.norm(anchors - x, axis=1)


def panel(ax, anchors, r, sigma, lim=3.2, cells=500, show_mean=False, title=None):
    res = analysis.analyze_full(anchors, r, sigma, cells=cells, lo=[-lim, -lim], hi=[lim, lim])
    g = res.grid
    ext = [g.axes[0][0], g.axes[0][-1], g.axes[1][0], g.axes[1][-1]]
    p = np.exp(res.logp - res.logp.max())
    ax.imshow(p.T, origin="lower", extent=ext, cmap="Greys", alpha=.55)
    ax.contourf(g.axes[0], g.axes[1], res.mask.T.astype(float), levels=[.5, 1.5], colors=[TYPE_COLORS[0]], alpha=.55)
    ax.contour(g.axes[0], g.axes[1], res.mask.T.astype(float), levels=[.5], colors=[TYPE_COLORS[0]], linewidths=1)
    ax.plot(*anchors.T, "k^", ms=5)
    if show_mean:
        P = p / p.sum()
        X, Y = g.mesh()
        m = np.array([(P * X).sum(), (P * Y).sum()])
        ax.plot(*m, "o", color=TYPE_COLORS[3], ms=6, mec="k", mew=.5)
        ax.annotate("posterior mean", m, xytext=(m[0] + .5, m[1] - 1.1), fontsize=7,
                    arrowprops=dict(arrowstyle="-", lw=.6))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or betti_label(res.betti.tolist()), fontsize=8)
    return res


def fig01():
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    s = 0.08
    for k, ax in zip((1, 2, 3), axs):
        a = A2[:k]
        res = panel(ax, a, ranges(a), s)
        ax.set_title(f"{k} range{'s' if k > 1 else ''}:  " + betti_label(res.betti.tolist()), fontsize=8)
        ax.plot(*X_TRUE, "x", color="k", ms=5)
    fig.suptitle(r"The 2D staircase ($\sigma=0.08$): ring $\to$ two pieces $\to$ one piece", fontsize=10)
    save(fig, "fig01_staircase_2d")


def fig02():
    k2 = radial.kappa_d(2)
    kappas = [3.5, 2.6, 2.15, k2 * 1.002, k2 * 0.998, 1.8, 1.3]
    fig, axs = plt.subplots(1, len(kappas), figsize=(13, 2.3))
    for kap, ax in zip(kappas, axs):
        res = panel(ax, np.zeros((1, 2)), [1.0], 1 / kap, lim=1 + 3.2 / kap, cells=400)
        ax.set_title(fr"$\kappa={kap:.3f}$" + "\n" + betti_label(res.betti.tolist()), fontsize=7.5)
        if abs(kap - k2) < .01:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(TYPE_COLORS[1]); sp.set_linewidth(1.5)
    fig.suptitle(fr"The hole filling in — one range, noise rising; closes at $\kappa_2={k2:.4f}$", fontsize=10)
    save(fig, "fig02_hole_filling")


def fig05():
    from skimage import measure
    A3 = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0.2, 1.3, 0.1]])
    x = np.array([0.1, 0.4, 0.9])
    fig = plt.figure(figsize=(11, 3.8))
    for i, k in enumerate((1, 2, 3)):
        a = A3[:k]
        r = np.linalg.norm(a - x, axis=1)
        lim = 2.4
        res = analysis.analyze_full(a, r, 0.07, cells=150, lo=[-lim] * 3, hi=[lim] * 3)
        g = res.grid
        verts, faces, _, _ = measure.marching_cubes(res.mask.astype(float), .5, spacing=tuple(g.h))
        verts += np.array([ax[0] for ax in g.axes])
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=TYPE_COLORS[i], lw=0, alpha=.9,
                        shade=True)
        ax.scatter(*a.T, color="k", marker="^", s=18)
        ax.set_title(f"{k} range{'s' if k > 1 else ''}: " + betti_label(res.betti.tolist()), fontsize=8)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    fig.suptitle("The 3D staircase: shell → ring-shaped tube → two blobs", fontsize=10)
    save(fig, "fig05_staircase_3d")


def fig06():
    cases = [("one range, κ=3", np.zeros((1, 2)), [1.0], 1 / 3.0),
             ("one range, κ=1.6", np.zeros((1, 2)), [1.0], 1 / 1.6),
             ("two ranges, σ=0.15", A2[:2], ranges(A2[:2]), 0.15),
             ("two ranges, shallow angle, σ=0.7", np.array([[-1.0, 0], [1, 0]]), [3.864, 3.864], 0.7)]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
    for (name, a, r, s), ax in zip(cases, axs):
        rm = max(r)
        res = analysis.analyze_full(a, r, s, cells=400, lo=a.min(0) - rm - 7 * s, hi=a.max(0) + rm + 7 * s)
        dg = topology.superlevel_persistence(res.logp)
        top = res.logp.max()
        lv = res.log_level - top
        for q, mk in ((0, "o"), (1, "s")):
            iv = dg[q] - top
            fin = np.isfinite(iv[:, 1])
            ax.scatter(-iv[fin, 0], -iv[fin, 1], marker=mk, s=16, color=TYPE_COLORS[q], label=f"H{q}", zorder=3)
            ax.scatter(-iv[~fin, 0], np.full((~fin).sum(), -lv * 1.9 + 1), marker=mk, s=30,
                       facecolors="none", edgecolors=TYPE_COLORS[q], zorder=3)
        m = -lv * 2 + 1
        ax.plot([0, m], [0, m], color="0.6", lw=.6)
        ax.axvline(-lv, color=TYPE_COLORS[3], lw=1); ax.axhline(-lv, color=TYPE_COLORS[3], lw=1)
        ax.set_xlim(-.2, m); ax.set_ylim(-.2, m)
        ax.set_xlabel("birth depth  log(p_max/p)"); ax.set_ylabel("death depth")
        ax.set_title(name + "\n95% region: " + betti_label(res.betti.tolist()), fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Persistence of the superlevel filtration. A feature is in the 95% region iff "
                 "it is born left of the red line and dies above it.", fontsize=9)
    save(fig, "fig06_persistence")


def fig08():
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    panel(ax, np.zeros((1, 2)), [1.0], 0.2, lim=1.9, show_mean=True,
          title="The mean in the hole (one range, κ = 5)")
    save(fig, "fig08_mean_in_hole")


def fig10():
    a = np.array([[-1.0, 0], [1, 0]])
    r = [3.864, 3.864]  # theta = 30 deg
    sig = [0.3, 0.7, 1.95, 2.08, 2.5]
    fig, axs = plt.subplots(1, len(sig), figsize=(13, 3.0))
    for s, ax in zip(sig, axs):
        res = panel(ax, a, r, s, lim=4.2 + 2 * s, cells=500)
        ax.set_title(fr"$\sigma={s}$" + "\n" + betti_label(res.betti.tolist()), fontsize=8)
    fig.suptitle(r"Two ranges crossing at $\theta=30^\circ$: two pieces $\to$ ring $\to$ two holes "
                 r"$\to$ one piece — the staircase runs backwards", fontsize=10)
    save(fig, "fig10_ladder_2d")


if __name__ == "__main__":
    for f in (fig01, fig02, fig05, fig06, fig08, fig10):
        f()
