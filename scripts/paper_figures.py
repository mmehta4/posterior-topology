"""Publication figures for the paper, in one visual system (Source Serif / IBM Plex, house palette).

    python scripts/paper_figures.py            # all figures
    python scripts/paper_figures.py hero kappa  # selected

Output: paper/full/figs/*.pdf (vector) and results/figures/paper/*.png (previews).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LightSource, LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptopo import analysis, posterior, radial  # noqa: E402

# ------------------------------------------------------------------ visual system
FONTS = Path.home() / ".venvs/posterior-topology/fonts"
for f in FONTS.glob("*.otf"):
    font_manager.fontManager.addfont(str(f))
INK, MUTED, RULE = "#151A21", "#5F6B7A", "#D9DFE6"
ACCENT, ACCENT_SOFT, AMBER, WARN = "#27548A", "#A9C1DE", "#B0702A", "#8C3A2E"
REGION = LinearSegmentedColormap.from_list("region", ["#FFFFFF", "#DCE6F2", "#7FA2CC", "#27548A"])
DENS = LinearSegmentedColormap.from_list("dens", ["#FFFFFF", "#E9EDF3", "#B9C3D0", "#5F6B7A"])
plt.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": 8.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Source Serif Pro",
    "mathtext.it": "Source Serif Pro:italic", "mathtext.bf": "Source Serif Pro:bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 8.5, "axes.titleweight": "medium", "axes.titlecolor": INK, "legend.frameon": False,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.1,
})
OUT_PDF = ROOT / "paper/full/figs"
OUT_PNG = ROOT / "results/figures/paper"
W1, W2 = 6.3, 3.1  # full text width, half width (inches)
ALPHA = 0.05


def save(fig, name):
    OUT_PDF.mkdir(parents=True, exist_ok=True); OUT_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF / f"{name}.pdf"); fig.savefig(OUT_PNG / f"{name}.png")
    plt.close(fig); print("saved", name)


def tag(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="semibold", color=INK, va="bottom", ha="left")


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def anchor(ax, x, y, s=26):
    ax.scatter([x], [y], marker="^", s=s, color=INK, zorder=5, linewidths=0)


def region_panel(ax, A, r, sigma, lim, cells=520, truth=None):
    """HPD region drawn from the continuous log-density (smooth contour at the exact HPD level)."""
    res = analysis.analyze_full(np.asarray(A, float), r, sigma, cells=cells, lo=[-lim, -lim], hi=[lim, lim])
    g = res.grid
    p = np.exp(res.logp - res.logp.max())
    ext = [g.axes[0][0], g.axes[0][-1], g.axes[1][0], g.axes[1][-1]]
    ax.imshow(np.where(res.mask, np.nan, p).T, origin="lower", extent=ext, cmap=DENS, vmin=0, vmax=1.15)
    ax.imshow(np.where(res.mask, 0.35 + 0.65 * np.sqrt(p), np.nan).T, origin="lower", extent=ext, cmap=REGION,
              vmin=0, vmax=1)
    ax.contour(g.axes[0], g.axes[1], res.logp.T, levels=[res.log_level], colors=[ACCENT], linewidths=0.9, linestyles='solid')
    for a in np.atleast_2d(A):
        anchor(ax, a[0], a[1])
    if truth is not None:
        ax.scatter([truth[0]], [truth[1]], marker="x", s=18, color=INK, linewidths=1.0, zorder=6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); bare(ax)
    return res


# ------------------------------------------------------------------ 3D rendering: lit surface + amber cut face
CUT = np.array([0xD9, 0x9A, 0x4E]) / 255          # cut face (amber, lighter than AMBER for contrast)
SURF_DARK, SURF_LIGHT = np.array([0x1F, 0x3F, 0x6B]) / 255, np.array([0xB9, 0xCF, 0xE8]) / 255


def mesh3d(ax, A, r, sigma, lim, n=150, clip=True, elev=20, azim=58, centre=None, half=None, truth=None):
    """Draw the 3D HPD region as a shaded surface. clip=True removes y > 0 and paints the cut face amber, so
    cavities show as holes in the cut."""
    from skimage import measure
    from matplotlib.colors import LightSource  # noqa: F401
    A = np.asarray(A, float); lo = [-lim] * 3; hi = [lim] * 3
    res = analysis.analyze_full(A, r, sigma, cells=n, lo=lo, hi=hi)
    g = res.grid; field = res.logp.copy(); lev = res.log_level
    if clip:
        Y = g.axes[1][None, :, None] * np.ones_like(field)
        field = np.where(Y > 0, lev - 60.0, field)
    field = np.pad(field, 1, constant_values=lev - 60.0)
    verts, faces, normals, _ = measure.marching_cubes(field, lev, spacing=tuple(g.h))
    verts += np.array([ax_[0] - h for ax_, h in zip(g.axes, g.h)])
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    cen = verts[faces].mean(1)
    is_cut = clip & (np.abs(cen[:, 1]) < 0.75 * g.h[1]) & (np.abs(fn[:, 1]) > 0.9)
    L = np.array([0.35, 0.75, 0.55]); L /= np.linalg.norm(L)
    V = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)), np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                  np.sin(np.radians(elev))])
    nn = np.where((fn @ V)[:, None] < 0, -fn, fn)                    # face the viewer
    diff = np.clip(nn @ L, 0, 1); hvec = (L + V) / np.linalg.norm(L + V)
    spec = np.clip(nn @ hvec, 0, 1) ** 60
    t = (0.25 + 0.75 * diff)[:, None]
    cols = SURF_DARK + (SURF_LIGHT - SURF_DARK) * t + 0.15 * spec[:, None]
    cols[is_cut] = CUT * (0.92 + 0.08 * diff[is_cut, None])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cols = np.clip(cols, 0, 1)
    pc = Poly3DCollection(verts[faces], facecolors=cols, edgecolors=cols, linewidths=0.15, zorder=1, rasterized=True)
    ax.computed_zorder = False
    ax.add_collection3d(pc)
    for a in A:
        ax.scatter(*a, marker="^", s=18, color=INK, depthshade=False, zorder=10, linewidths=0)
    if truth is not None:
        ax.scatter(*truth, marker="x", s=16, color=INK, depthshade=False, zorder=10, linewidths=1.0)
    c0 = verts.mean(0) if centre is None else np.asarray(centre, float)
    hw = (verts.max(0) - verts.min(0)).max() / 2 * 1.02 if half is None else half
    ax.set_xlim(c0[0] - hw, c0[0] + hw); ax.set_ylim(c0[1] - hw, c0[1] + hw); ax.set_zlim(c0[2] - hw, c0[2] + hw)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    return res


# ------------------------------------------------------------------ 1. hero: the staircase
def hero():
    from skimage import measure
    fig = plt.figure(figsize=(W1, 4.25))
    A2 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.35, -1.45]]); x2 = np.array([0.2, 1.1])
    titles = ["one range", "two ranges", "three ranges"]
    betti2 = [r"$\beta_0=1,\ \beta_1=1$", r"$\beta_0=2$", r"$\beta_0=1$"]
    for k in range(3):
        ax = fig.add_axes([0.02 + k * 0.33, 0.53, 0.29, 0.44])
        a = A2[:k + 1]; r = np.linalg.norm(a - x2, axis=1)
        region_panel(ax, a, r, 0.11, 2.9, truth=x2)
        ax.set_title(f"{titles[k]}\n" + betti2[k], fontsize=8, color=INK, pad=2)
        if k == 0:
            tag(ax, r"$d=2$", x=-0.08, y=0.9)
    A3 = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0.25, 1.35, 0.15]]); x3 = np.array([0.1, 0.45, 0.9])
    betti3 = [r"$\beta_2=1$ (shell)", r"$\beta_1=1$ (tube)", r"$\beta_0=2$"]
    for k in range(3):
        ax = fig.add_axes([0.01 + k * 0.33, 0.0, 0.31, 0.5], projection="3d")
        a = A3[:k + 1]; r = np.linalg.norm(a - x3, axis=1)
        mesh3d(ax, a, r, 0.11, lim=2.75, n=180, clip=(k < 2), elev=18, azim=60,
               centre=[0, 0, 0] if k < 2 else None, half=[1.85, 1.35, None][k], truth=x3 if k == 2 else None)
        ax.set_title(betti3[k], fontsize=8, color=INK, y=0.96)
        if k == 0:
            ax.text2D(-0.02, 0.9, r"$d=3$", transform=ax.transAxes, fontsize=9, fontweight="semibold")
    save(fig, "fig1_staircase")


# ------------------------------------------------------------------ 2. one range: the hole closes
def onerange():
    k2 = float(radial.kappa_exact(2)[0])
    ks = [3.0, 2.4, k2, 1.6]
    fig = plt.figure(figsize=(W1, 2.95))
    for i, k in enumerate(ks):
        ax = fig.add_axes([0.01 + i * 0.25, 0.42, 0.22, 0.56])
        region_panel(ax, np.zeros((1, 2)), [1.0], 1 / k, 1 + 3.2 / k, cells=480)
        lab = r"$r/\sigma=\kappa_2$" if abs(k - k2) < 1e-9 else fr"$r/\sigma={k:g}$"
        ax.set_title(lab, fontsize=8, pad=2)
        bx = fig.add_axes([0.03 + i * 0.25, 0.07, 0.18, 0.26])
        u = np.linspace(0, k + 4, 400); g_ = u * np.exp(-(u - k) ** 2 / 2); g_ /= g_.max()
        t = radial.hpd_halfwidth(2, k, ALPHA); lo, hi = max(0, k - t), k + t
        m = (u >= lo) & (u <= hi)
        bx.fill_between(u[m], 0, g_[m], color=ACCENT_SOFT, lw=0); bx.plot(u, g_, color=ACCENT, lw=0.9)
        bx.axvline(0, color=INK, lw=0.6); bx.axvline(2 * k, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        bx.set_yticks([]); bx.set_xticks([0, k, 2 * k]); bx.set_xticklabels(["0", r"$\kappa$", r"$2\kappa$"])
        bx.spines["left"].set_visible(False); bx.set_ylim(0, 1.1)
        if i == 0:
            bx.set_xlabel(r"$u=\|x-a\|/\sigma$", labelpad=1)
    save(fig, "fig2_one_range")


# ------------------------------------------------------------------ 3. the threshold kappa_d
def kappa():
    rows = json.loads((ROOT / "results/data/problem0_kappa_table.json").read_text())["rows"]
    ds = np.arange(1, 101); kd = np.array([radial.kappa_d(int(d), ALPHA) for d in ds])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.45), gridspec_kw=dict(wspace=0.32))
    dd = np.linspace(1, 100, 400)
    a1.plot(dd, np.sqrt(dd / 2) + 1.007263, color=MUTED, lw=0.9, label=r"$\sqrt{d/2}+z\sqrt{3/8}$")
    a1.plot(dd, np.sqrt(dd / 2), color=MUTED, lw=0.8, ls=(0, (3, 2)), label=r"$\sqrt{d/2}$")
    a1.scatter(ds, kd, s=5, color=ACCENT, zorder=3, label=r"$\kappa_d$", linewidths=0)
    a1.set_xscale("log"); a1.set_xlabel(r"dimension $d$"); a1.set_ylabel(r"critical ratio $r/\sigma$")
    a1.legend(loc="upper left"); tag(a1, "a", x=-0.18)
    nx = json.loads((ROOT / "results/data/kappa_next_order.json").read_text())   # scripts/kappa_next_order.py
    d_e = np.array([q["d"] for q in nx["rows"]]); v_e = np.array([q["value"] for q in nx["rows"]])
    a2.plot(d_e, v_e, "o-", ms=2.6, color=ACCENT, lw=0.9)
    a2.axhline(nx["c_limit"], color=AMBER, lw=0.9, ls=(0, (3, 2)))
    a2.text(12, 0.0712, r"$c_\alpha=(13z^2-28)/(72\sqrt{2})=0.0704$", color=AMBER, fontsize=7.5)
    a2.set_xscale("log"); a2.set_xlabel(r"dimension $d$"); a2.set_ylabel(r"$\sqrt{d}\,(\kappa_d-\sqrt{d/2}-z\sqrt{3/8})$")
    a2.set_ylim(0.02, 0.078); tag(a2, "b", x=-0.22)
    save(fig, "fig3_threshold")


# ------------------------------------------------------------------ 4. the mean in the hole
def mean():
    kap = 4.0; s = 1 / kap
    fig, ax = plt.subplots(figsize=(W2 * 0.86, W2 * 0.86))
    region_panel(ax, np.zeros((1, 2)), [1.0], s, 2.1, cells=560)
    f = lambda u, j: u ** j * np.exp(-(u - kap) ** 2 / 2)
    EU2 = integrate.quad(lambda u: f(u, 3), 0, kap + 30)[0] / integrate.quad(lambda u: f(u, 1), 0, kap + 30)[0]
    rad = np.sqrt(s * s * EU2 / 2 * stats.chi2.ppf(1 - ALPHA, 2)); th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(rad * np.cos(th), rad * np.sin(th), color=AMBER, lw=1.0, ls=(0, (4, 2.5)))
    ax.scatter([0], [0], s=70, facecolor="none", edgecolor=AMBER, linewidth=1.2, zorder=7)  # mean, around the anchor
    save(fig, "fig4_mean")


# ------------------------------------------------------------------ 5. the triangle theorem, geometrically
def triangle():
    D = 2.0; x = np.array([0.35, 1.25]); a1, a2 = np.array([-1.0, 0]), np.array([1.0, 0])
    r1, r2 = np.linalg.norm(x - a1), np.linalg.norm(x - a2); s = (D + r1 + r2) / 2
    ts, t1, t2 = s - D, s - r2, s - r1
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    th = np.linspace(0, 2 * np.pi, 500)
    for a, rr in ((a1, r1), (a2, r2)):
        ax.plot(a[0] + rr * np.cos(th), a[1] + rr * np.sin(th), color=RULE, lw=0.8)
    ax.plot([-4.4, 4.4], [0, 0], color=MUTED, lw=0.6)
    th = np.linspace(0, 2 * np.pi, 500)
    ax.plot([a1[0], x[0], a2[0], a1[0]], [a1[1], x[1], a2[1], a1[1]], color=INK, lw=0.9)
    area = 0.5 * D * x[1]; rin = area / s; Iz = (D * x[0] + r1 * a2[0] + r2 * a1[0]) / (D + r1 + r2)
    ax.plot(Iz + rin * np.cos(th), rin + rin * np.sin(th), color=ACCENT, lw=1.0)
    # excircle opposite a1 touches the line beyond a2 (radius Area/t1); opposite a2 beyond a1 (radius Area/t2);
    # both lie on the same side of the anchor line as x*
    for zt, rad in ((a2[0] + ts, area / t1), (a1[0] - ts, area / t2)):
        ax.plot(zt + rad * np.cos(th), rad + rad * np.sin(th), color=ACCENT, lw=0.8, ls=(0, (3, 2)))
    pts = [(a1[0] - ts, r"$(s-r_2)^2$", "outer"), ((r1 - r2) / 2, r"$(s-D)^2$", "inner"), (a2[0] + ts, r"$(s-r_1)^2$", "outer")]
    for z, lab, kind in pts:
        ax.scatter([z], [0], s=22, color=ACCENT, zorder=6, edgecolor="white", linewidth=0.6)
        ax.text(z, -0.42, lab, ha="center", fontsize=7.5, color=ACCENT)
    for a, lab, dx in ((a1, r"$a_1$", -0.28), (a2, r"$a_2$", 0.28)):
        anchor(ax, *a, s=30); ax.text(a[0] + dx, 0.1, lab, ha="center", fontsize=8)
    ax.scatter([x[0]], [x[1]], marker="x", s=24, color=INK, lw=1.1, zorder=6); ax.text(x[0] + 0.1, x[1] + 0.08, r"$x^*$", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-4.4, 4.4); ax.set_ylim(-0.55, 3.4); bare(ax)
    save(fig, "fig5_triangle")


# ------------------------------------------------------------------ 6. everything happens on the axis
def axis_ladder():
    B = np.array([[-1.0], [1.0]]); x = np.array([0.2, 1.1]); A = np.array([[-1.0, 0], [1.0, 0]])
    r = np.linalg.norm(A - x, axis=1)
    sig = [0.2, 0.40, 0.56, 0.66, 0.75, 0.85]
    names = ["two pieces", "one piece", "ring", "one piece", "ring", "one piece"]
    fig = plt.figure(figsize=(W1, 3.0))
    zz = np.linspace(-3.2, 3.2, 1200); f = ((np.abs(zz + 1) - r[0]) ** 2 + (np.abs(zz - 1) - r[1]) ** 2)
    for i, s in enumerate(sig):
        ax = fig.add_axes([0.005 + i * 0.166, 0.42, 0.155, 0.5])
        region_panel(ax, A, r, s, 3.0, cells=420, truth=x); ax.set_title(f"{names[i]}\n" + fr"$\sigma={s}$", fontsize=7.5, pad=1)
        res = analysis.analyze_reduced(2, B, r, s, cells=600)
        c = -2 * s * s * res.log_level  # cut-off in units of S
        bx = fig.add_axes([0.012 + i * 0.166, 0.08, 0.14, 0.26])
        bx.plot(zz, f, color=INK, lw=0.8); bx.axhline(c, color=ACCENT, lw=0.8)
        bx.fill_between(zz, 0, 1, where=f <= c, transform=bx.get_xaxis_transform(), color=ACCENT_SOFT, lw=0, alpha=0.7)
        bx.set_ylim(0, 4.2); bx.set_xlim(-3.2, 3.2); bx.set_yticks([]); bx.set_xticks([-1, 1]); bx.set_xticklabels([r"$a_1$", r"$a_2$"])
        bx.spines["left"].set_visible(False)
        m = int(np.sum(np.diff((f <= c).astype(int)) == 1) + (f[0] <= c))
        bx.text(0.5, 1.02, fr"$m={m}$", transform=bx.transAxes, ha="center", va="bottom", fontsize=7.5, color=ACCENT)
    save(fig, "fig6_axis_ladder")


# ------------------------------------------------------------------ 7. phase diagrams
def phase():
    EX = json.loads((ROOT / "results/data/exact_thresholds.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, d, letter in zip(axs, (2, 3), "ab"):
        z = np.load(ROOT / f"results/data/problem2_phase_d{d}.npz"); th, sg, Bt = z["theta_deg"], z["sigma"], z["betti"]
        top = np.array([[b[d - 1] if d > 1 else 0 for b in row] for row in Bt])        # cavities beta_{d-1}
        low = np.array([[b[d - 2] - (1 if d == 2 else 0) for b in row] for row in Bt])  # tube / extra piece
        code = np.where(top > 0, 1 + np.minimum(top, 2), np.where(low > 0, 0, 1))      # 0 staircase, 1 trivial, 2 one hole, 3 two
        cmap = matplotlib.colors.ListedColormap(["#7FA2CC", "#F2F4F7", "#E8C9A0", "#B0702A"])
        ax.pcolormesh(th, sg, code.T, cmap=cmap, vmin=-.5, vmax=3.5, shading="nearest", rasterized=True)
        for key, ls in (("inner", "-"), ("outer", (0, (3, 2))), ("anchor", (0, (1, 1.5)))):
            xs, ys = [], []
            for l in EX["ladders"]:                         # exact quadrature (scripts/exact_thresholds.py)
                if l["d"] != d:
                    continue
                for g in l["rungs"]:
                    names = " ".join(g["events"])
                    hit = {"inner": "inner contact" in names, "outer": "contact beyond" in names, "anchor": "anchor" in names}[key]
                    if hit and np.isfinite(g["sigma"]):
                        xs.append(l["theta"]); ys.append(g["sigma"])
            ax.plot(xs, ys, color=INK, lw=0.8, ls=ls)
        ax.set_yscale("log"); ax.set_xlabel(r"crossing angle $\theta$ (degrees)"); ax.set_xlim(15, 165); ax.set_ylim(0.02, 3.0)
        ax.axvline(60, color=INK, lw=0.4, ls=(0, (1, 2)))
        tag(ax, f"{letter}   $d={d}$", x=0.0)
    axs[0].set_ylabel(r"noise $\sigma$ (half-separation units)")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color="#7FA2CC", label="noise-free shape"), Patch(color="#F2F4F7", label="contractible"),
         Patch(color="#E8C9A0", label="one cavity"), Patch(color="#B0702A", label="two cavities"),
         Line2D([], [], color=INK, lw=0.8, label="inner contact"), Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label="outer contacts"),
         Line2D([], [], color=INK, lw=0.8, ls=(0, (1, 1.5)), label="anchors")]
    fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.1), fontsize=6.6, handlelength=1.6, columnspacing=1.0)
    save(fig, "fig7_phase")


# ------------------------------------------------------------------ 8. which event comes first
def first_event():
    zs, hs = np.linspace(-3, 3, 801), np.linspace(0.01, 3, 400); Z, H = np.meshgrid(zs, hs)
    r1, r2 = np.hypot(Z + 1, H), np.hypot(Z - 1, H); s = (2 + r1 + r2) / 2
    Din, Do1, Do2 = (s - 2) ** 2, (s - r1) ** 2, (s - r2) ** 2
    first = np.where(Din < np.minimum(Do1, Do2), 0, 1); two = (2 * np.abs(r1 - r2) < Din)
    fig, ax = plt.subplots(figsize=(W1 * 0.62, 2.0))
    ax.contourf(Z, H, first, levels=[-.5, .5, 1.5], colors=["#7FA2CC", "#E8C9A0"])
    gap = Din - 2 * np.abs(r1 - r2)          # > 0  <=>  two holes coexist (D = 2)
    ax.contour(Z, H, gap, levels=[0], colors=[INK], linewidths=0.7)
    anchor(ax, -1, 0, s=28); anchor(ax, 1, 0, s=28)
    ax.set_aspect("equal"); ax.set_xlabel(r"position of $x^*$ along the anchor line"); ax.set_ylabel(r"distance of $x^*$ from the line")
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    ax.text(0, 0.5, "merge between\nthe anchors", ha="center", fontsize=7, color=INK, bbox=box)
    ax.text(-2.15, 0.9, "merge around\nthe outside", ha="center", fontsize=7, color=INK, bbox=box)
    ax.text(0, 2.55, "outside the arcs: two holes coexist", ha="center", fontsize=7, color=INK, bbox=box)
    save(fig, "fig8_first_event")


# ------------------------------------------------------------------ 9. the sqrt 2 transition
def sqrt2():
    c = 1.007263
    kd = {d: float(radial.kappa_exact(d, ALPHA)[0]) for d in (2, 10, 100, 1000)}
    def ph(sig, d):
        f = lambda R: stats.chi.pdf(R, d) * stats.norm.cdf(R / sig - kd[d])
        return integrate.quad(f, max(0, np.sqrt(d) - 14), np.sqrt(d) + 14, limit=300)[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.4), gridspec_kw=dict(wspace=0.28))
    shades = ["#C9D7E8", "#8EADD2", "#4F7AB0", "#1F3F6B"]
    sg = np.linspace(0.2, 3, 150)
    for (d, col) in zip((2, 10, 100, 1000), shades):
        a1.plot(sg, [ph(s, d) for s in sg], color=col, label=fr"$d={d}$")
        if d >= 10:
            tau = np.linspace(max(-8, np.sqrt(d) * (0.3 - np.sqrt(2))), 5, 120)
            a2.plot(tau, [ph(np.sqrt(2) + t / np.sqrt(d), d) for t in tau], color=col)
    a1.axvline(np.sqrt(2), color=AMBER, lw=0.8, ls=(0, (3, 2))); a1.text(1.47, 0.9, r"$\sqrt{2}$", color=AMBER)
    a1.set_xlabel(r"noise $\sigma$"); a1.set_ylabel("P(95% region has a hole)"); a1.legend(loc="upper right"); tag(a1, "a", x=-0.16)
    tau = np.linspace(-8, 5, 200)
    a2.plot(tau, 1 - stats.norm.cdf((np.sqrt(2) * c + tau / np.sqrt(2)) / np.sqrt(2.5)), color=AMBER, lw=1.4, label="limit")
    a2.set_xlabel(r"$\tau=\sqrt{d}\,(\sigma-\sqrt{2})$"); a2.legend(loc="upper right"); tag(a2, "b", x=-0.12)
    save(fig, "fig9_sqrt2")


# ------------------------------------------------------------------ 10. residual coordinates
def residual():
    """Half-plane (a) and its image in residual coordinates (b): HPD region = disc cut by the walls of a half-strip."""
    D = 2.0; x = np.array([0.2, 1.1]); sig = 0.56
    A = np.array([[-1.0, 0.0], [1.0, 0.0]]); r1, r2 = np.linalg.norm(x - A[0]), np.linalg.norm(x - A[1])
    s = (D + r1 + r2) / 2; ts, t1, t2 = s - D, s - r2, s - r1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.35), gridspec_kw=dict(wspace=0.08, width_ratios=[1.25, 1]))
    res = region_panel(a1, A, [r1, r2], sig, 2.9, cells=640, truth=x)
    c = -2 * sig ** 2 * res.log_level
    a1.set_ylim(-0.02, 2.4)
    a1.plot([-2.9, -1], [0, 0], color=AMBER, lw=1.6, zorder=4); a1.plot([1, 2.9], [0, 0], color=AMBER, lw=1.6, zorder=4)
    a1.plot([-1, 1], [0, 0], color=INK, lw=1.6, zorder=4)
    a1.text(-1, -0.2, r"$a_1$", ha="center", va="top", fontsize=8); a1.text(1, -0.2, r"$a_2$", ha="center", va="top", fontsize=8)
    a1.text(x[0] + 0.12, x[1] + 0.08, r"$x^*$", fontsize=8)
    tag(a1, "a", x=0.0, y=0.98)
    # residual plane, rotated: p = (e1 - e2)/sqrt2, q = (e1 + e2)/sqrt2
    L, Rt, B = -np.sqrt(2) * t1, np.sqrt(2) * t2, -np.sqrt(2) * ts
    P, Q = np.meshgrid(np.linspace(-1.9, 1.6, 700), np.linspace(-0.95, 2.2, 640))
    inF = (P > L) & (P < Rt) & (Q > B)
    E2 = P ** 2 + Q ** 2
    a2.imshow(np.where(inF & (E2 > c), np.exp(-E2 / (2 * sig ** 2)), np.nan), origin="lower",
              extent=[P.min(), P.max(), Q.min(), Q.max()], cmap=DENS, vmin=0, vmax=1.15)
    a2.imshow(np.where(inF & (E2 <= c), 0.35 + 0.65 * np.sqrt(np.exp(-E2 / (2 * sig ** 2))), np.nan), origin="lower",
              extent=[P.min(), P.max(), Q.min(), Q.max()], cmap=REGION, vmin=0, vmax=1)
    th = np.linspace(0, 2 * np.pi, 600); pc, qc = np.sqrt(c) * np.cos(th), np.sqrt(c) * np.sin(th)
    inside = (pc > L) & (pc < Rt) & (qc > B)
    a2.plot(np.where(inside, pc, np.nan), np.where(inside, qc, np.nan), color=ACCENT, lw=0.9)
    a2.plot(np.where(~inside, pc, np.nan), np.where(~inside, qc, np.nan), color=ACCENT, lw=0.6, ls=(0, (2, 2)))
    a2.plot([L, Rt], [B, B], color=INK, lw=1.6); a2.plot([L, L], [B, 2.2], color=AMBER, lw=1.6)
    a2.plot([Rt, Rt], [B, 2.2], color=AMBER, lw=1.6)
    anchor(a2, L, B); anchor(a2, Rt, B)
    a2.text(L, B - 0.2, r"$a_1$", ha="center", va="top", fontsize=8); a2.text(Rt, B - 0.2, r"$a_2$", ha="center", va="top", fontsize=8)
    a2.scatter([0], [0], marker="x", s=18, color=INK, linewidths=1.0, zorder=6)
    a2.text(0.08, 0.08, r"$x^*$", fontsize=8)
    for (u, v), lab, ha in (((0, B), r"$\sqrt{2}\,t_*$", "left"), ((L, 0), r"$\sqrt{2}\,t_1$", "center"),
                            ((Rt, 0), r"$\sqrt{2}\,t_2$", "center")):
        a2.plot([0, u], [0, v], color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    lb = dict(fontsize=7.5, color=INK, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.75))
    a2.text(0.06, B / 2, r"$\sqrt{2}\,t_*$", va="center", ha="left", **lb)
    a2.text(L / 2, 0.06, r"$\sqrt{2}\,t_1$", ha="center", va="bottom", **lb)
    a2.text(Rt / 2, 0.06, r"$\sqrt{2}\,t_2$", ha="center", va="bottom", **lb)
    a2.set_xlim(-1.9, 1.6); a2.set_ylim(-0.95, 2.2); a2.set_aspect("equal"); bare(a2)
    tag(a2, "b", x=0.0, y=0.98)
    print(f"residual: c={c:.4f}, sqrt c={np.sqrt(c):.4f}, walls {np.sqrt(2)*ts:.3f} {np.sqrt(2)*t2:.3f} {np.sqrt(2)*t1:.3f},"
          f" corner a2 {np.sqrt(2*(ts**2+t2**2)):.3f}")
    save(fig, "fig10_residual")


# ------------------------------------------------------------------ 11. high dimension
def high_dim():
    """Two ranges, ranges = sqrt(d): finite-d first thresholds, the limit sigma_inf(theta), and sigma_inf - zC/sqrt(d)."""
    fin = [dict(theta=r["theta"], d=r["d"], sigma=r["sigma_scaled"])
           for r in json.loads((ROOT / "results/data/exact_thresholds.json").read_text())["large_d"]]
    z = stats.norm.ppf(1 - ALPHA)
    th = np.linspace(20, 165, 400); a = np.sin(np.radians(th) / 2)
    s_in = np.sqrt(8 * (1 - a) ** 2 / (2 - a)); s_out = np.sqrt(2 * a * (1 + 2 * a) / (1 + a))
    lim = np.where(th >= 60, s_in, s_out)
    C = lambda a_: 2 * (1 - a_) * np.sqrt(a_ * a_ - 4 * a_ + 6) / (2 - a_) ** 1.5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.45), gridspec_kw=dict(wspace=0.3))
    a1.plot(th, lim, color=INK, lw=1.2, label=r"$d\to\infty$")
    shades = ["#C9D7E8", "#8EADD2", "#4F7AB0", "#1F3F6B"]
    for d, col in zip((40, 160, 640, 2560), shades):
        rows = sorted((r["theta"], r["sigma"]) for r in fin if r["d"] == d)
        a1.plot(*zip(*rows), "o", ms=3.2, color=col, label=fr"$d={d}$")
    a1.axvline(60, color=MUTED, lw=0.5, ls=(0, (1, 2)))
    a1.set_xlabel(r"crossing angle $\theta$ (degrees)"); a1.set_ylabel(r"first transition $\sigma$ (ranges $=\sqrt{d}$)")
    a1.set_xlim(20, 165); a1.set_ylim(0, 1.25); a1.legend(loc="upper right", handlelength=1.2); tag(a1, "a", x=-0.17)
    x = np.linspace(0, 0.17, 100)
    for th0, col in ((75, shades[1]), (90, shades[2]), (120, shades[3])):
        a0 = np.sin(np.radians(th0) / 2); si = np.sqrt(8 * (1 - a0) ** 2 / (2 - a0))
        a2.plot(x, si - z * C(a0) * x, color=col, lw=1.0)
        rows = sorted((r["d"], r["sigma"]) for r in fin if r["theta"] == th0 and r["d"] >= 40)
        a2.plot([dd ** -0.5 for dd, _ in rows], [v for _, v in rows], "o", ms=3.2, color=col)
        a2.text(0.172, si - z * C(a0) * 0.17, fr"${th0}^\circ$", fontsize=7.5, va="center", color=INK)
    a2.set_xlabel(r"$1/\sqrt{d}$"); a2.set_xlim(0, 0.19); a2.set_ylim(0.2, 1.0); tag(a2, "b", x=-0.15)
    save(fig, "fig11_high_dim")


# ------------------------------------------------------------------ 12. the backwards staircase in space
def ladder3d():
    """Two ranges in R^3, symmetric, crossing angle 45 deg (anchors at +-1): tube -> one cavity -> two cavities -> solid,
    cut open along the plane through the anchors and the true point."""
    th = np.radians(45); r = 1 / np.sin(th / 2); h = r * np.cos(th / 2)
    A = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
    sig = [0.45, 0.72, 1.05, 1.40]
    names = ["tube", "one cavity", "two cavities", "solid"]
    betti = [r"$\beta_1=1$", r"$\beta_2=1$", r"$\beta_2=2$", "contractible"]
    fig = plt.figure(figsize=(W1, 2.15))
    for k, s_ in enumerate(sig):
        ax = fig.add_axes([k * 0.25 - 0.02, -0.04, 0.29, 0.92], projection="3d")
        res = mesh3d(ax, A, [r, r], s_, lim=r + 4.2 * s_ + 0.6, n=170, elev=16, azim=62, centre=[0, 0, 0],
                     half=3.55, truth=[0, 0, h])
        ax.set_title(f"{names[k]}\n" + betti[k] + fr",  $\sigma={s_}$", fontsize=7.5, y=0.97)
    save(fig, "fig12_ladder_3d")


# ------------------------------------------------------------------ 13. unequal noise: annihilation
def unequal():
    """(a) Residual coordinates for x* = (-1.5, 0.6), s = (2, 1): the ellipse through the corner a1, the weighted foot of the
    base line beyond a1 (open) and the foot of the ray beyond a1 inside its edge (filled). (b) Where annihilation occurs,
    over positions of x* (anchors at +-1), for noise ratio s1/s2 = 2."""
    def tri(x):
        r1, r2 = np.hypot(x[0] + 1, x[1]), np.hypot(x[0] - 1, x[1]); s = (2 + r1 + r2) / 2
        return r1, r2, s - 2, s - r2, s - r1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W1, 2.55), gridspec_kw=dict(wspace=0.12, width_ratios=[1, 1.25]))
    # (a) rotated residual plane p = (e1 - e2)/sqrt2, q = (e1 + e2)/sqrt2
    s1, s2 = 2.0, 1.0; r1, r2, ts, t1, t2 = tri(np.array([-1.5, 0.6])); S = s1 ** 2 + s2 ** 2; rho = (s1 ** 2 - s2 ** 2) / S
    rot = lambda e: np.array([(e[0] - e[1]) / np.sqrt(2), (e[0] + e[1]) / np.sqrt(2)])
    L, Rt, B = -np.sqrt(2) * t1, np.sqrt(2) * t2, -np.sqrt(2) * ts
    c1 = rot(np.array([-ts - t1, t1 - ts]))
    dA1 = ((ts + t1) ** 2 / s1 ** 2 + (t1 - ts) ** 2 / s2 ** 2) / 2
    th = np.linspace(0, 2 * np.pi, 800)
    E = np.sqrt(2 * dA1) * np.stack([s1 * np.cos(th), s2 * np.sin(th)])      # e^T W e = 2 dA1
    P = (E[0] - E[1]) / np.sqrt(2); Q = (E[0] + E[1]) / np.sqrt(2)
    inside = (P > L) & (P < Rt) & (Q > B)
    a1.fill_between([L, Rt], B, 2.2, color="#EEF2F7", lw=0)
    a1.plot(np.where(inside, P, np.nan), np.where(inside, Q, np.nan), color=ACCENT, lw=1.0)
    a1.plot(np.where(~inside, P, np.nan), np.where(~inside, Q, np.nan), color=ACCENT, lw=0.6, ls=(0, (2, 2)))
    a1.plot([L, Rt], [B, B], color=INK, lw=1.5); a1.plot([L, L], [B, 2.2], color=AMBER, lw=1.5); a1.plot([Rt, Rt], [B, 2.2], color=AMBER, lw=1.5)
    a1.plot([L - 1.3, L], [B, B], color=INK, lw=0.7, ls=(0, (1, 1.5)))            # base line beyond its edge
    fb = rot(-ts * np.array([1 + rho, 1 - rho])); f1 = rot(t1 * np.array([-(1 + rho), 1 - rho]))
    a1.scatter(*fb, s=22, facecolor="white", edgecolor=INK, lw=0.9, zorder=6)
    a1.scatter(*f1, s=18, color=INK, zorder=6)
    anchor(a1, L, B); anchor(a1, Rt, B)
    a1.text(L + 0.06, B - 0.22, r"$a_1$", fontsize=8, va="top"); a1.text(Rt, B - 0.22, r"$a_2$", fontsize=8, va="top", ha="center")
    a1.scatter([0], [0], marker="x", s=16, color=INK, lw=1.0, zorder=6)
    a1.set_xlim(L - 1.4, Rt + 0.3); a1.set_ylim(B - 0.55, 2.2); a1.set_aspect("equal"); bare(a1); tag(a1, "a", x=0.0, y=0.98)
    # (b) map over x*
    zs, hs = np.linspace(-3, 3, 601), np.linspace(0.02, 3, 300); Z, H = np.meshgrid(zs, hs)
    R1, R2 = np.hypot(Z + 1, H), np.hypot(Z - 1, H); Sm = (2 + R1 + R2) / 2; TS, T1, T2 = Sm - 2, Sm - R2, Sm - R1
    base_out = T1 <= rho * TS; ray_out = TS <= rho * T1
    code = np.where(base_out, 1, np.where(ray_out, 2, 0))
    a2.contourf(Z, H, code, levels=[-.5, .5, 1.5, 2.5], colors=["#7FA2CC", "#E8C9A0", "#B0702A"])
    anchor(a2, -1, 0, s=26); anchor(a2, 1, 0, s=26)
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
    a2.text(1.6, 2.2, "five events", fontsize=7, ha="center", bbox=box)
    a2.set_xlabel(r"position of $x^*$ along the anchor line"); a2.set_ylabel(r"distance from the line")
    a2.set_aspect("equal"); a2.set_ylim(0, 3); tag(a2, "b", x=-0.12)
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color="#E8C9A0", label=r"inner contact annihilates at $a_1$"),
                       Patch(color="#B0702A", label=r"outer contact annihilates at $a_1$")],
              loc="upper left", fontsize=6.8, frameon=True, framealpha=0.9, edgecolor="none")
    save(fig, "fig13_unequal")


def three_ranges():
    """(a) Positions of the noise-free point x* (unit equilateral anchors) at which S has a smooth local maximum
    (results/data/three_range_phase_scan.npz). (b-d) x* = (0, -8): the 95% region before the ring forms, as a ring around
    the anchors, and after the hole has closed at the Fermat point; each panel's Betti numbers are checked."""
    from scipy import ndimage
    A = np.array([[0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])
    fig, axs = plt.subplots(1, 4, figsize=(W1, 1.85), gridspec_kw=dict(wspace=0.08, width_ratios=[1.1, 1, 1, 1]))
    d = np.load(ROOT / "results/data/three_range_phase_scan.npz"); pts, res = d["pts"], d["res"]
    g = np.unique(pts[:, 0]); Zc = np.full((len(g), len(g)), np.nan)
    for (x, y), v in zip(pts, res): Zc[np.searchsorted(g, y), np.searchsorted(g, x)] = v
    a = axs[0]; a.imshow(Zc, origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap=LinearSegmentedColormap.from_list("m", ["#EEF2F7", ACCENT]), vmin=0, vmax=1, interpolation="nearest")
    for q in A: anchor(a, *q, s=14)
    a.set_xticks([-8, 0, 8]); a.set_yticks([-8, 0, 8]); a.set_aspect("equal"); a.tick_params(labelsize=6.5)
    a.set_title("dark: smooth maximum", fontsize=7.5, pad=3); tag(a, "a", x=-0.32, y=1.0)
    X = np.array([0, -8.0]); r = np.linalg.norm(A - X, axis=1); L, N = 30.0, 1801
    gg = np.linspace(-L, L, N); XX = np.stack(np.meshgrid(gg, gg, indexing="xy"), -1)
    D = sum((np.linalg.norm(XX - q, axis=-1) - ri) ** 2 for q, ri in zip(A, r)) / 2
    for ax, sig, want, title, t in [(axs[1], 0.75, (1, 0), "one piece", "b"), (axs[2], 3.0, (1, 1), "ring", "c"), (axs[3], 6.3, (1, 0), "one piece", "d")]:
        p_ = np.exp(-(D - D.min()) / sig ** 2); p_ /= p_.sum(); o = np.sort(p_.ravel())[::-1]; c = o[np.searchsorted(np.cumsum(o), 0.95)]
        Rg = p_ >= c; lab, k = ndimage.label(~Rg); border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        b0, b1 = ndimage.label(Rg)[1], k - len(border); assert (b0, b1) == want, (sig, b0, b1)
        ax.imshow(np.where(Rg, p_ / p_.max(), np.nan), origin="lower", extent=[-L, L, -L, L], cmap=REGION, vmin=0, vmax=1)
        ax.contour(gg, gg, Rg.astype(float), levels=[0.5], colors=[ACCENT], linewidths=0.7)
        for q in A: anchor(ax, *q, s=8)
        ax.scatter([0], [-8], marker="x", s=12, color=INK, lw=0.9, zorder=6)
        ax.scatter([0], [0.2214], s=6, color=AMBER, zorder=7)
        ax.set_xlim(-21, 21); ax.set_ylim(-21, 21); ax.set_aspect("equal"); bare(ax)
        ax.set_title(f"{title}, $\\sigma = {sig:g}$", fontsize=7.5, pad=3); tag(ax, t, x=-0.04, y=1.0)
    save(fig, "fig14_three")


ALL = dict(hero=hero, onerange=onerange, kappa=kappa, mean=mean, triangle=triangle, axis=axis_ladder,
           phase=phase, first=first_event, sqrt2=sqrt2, residual=residual, high=high_dim, ladder3d=ladder3d, unequal=unequal,
           three=three_ranges)
if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
