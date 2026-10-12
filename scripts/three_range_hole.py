"""Remark 4.7: the noise interval in which the 95% region has a hole, for equilateral anchors (0, 1), (+-sqrt3/2, -1/2)
and noise-free point (0, -2.5), on two grid resolutions; endpoints by bisection on beta_1. Writes
results/data/three_range_hole.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptopo import analysis  # noqa: E402

A = np.array([[0.0, 1.0], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]])
X = np.array([0.0, -2.5]); R = np.linalg.norm(A - X, axis=1)


def b1(sigma, n):
    lim = R.max() + 1.5 + 6 * sigma
    return int(analysis.analyze_full(A, R, sigma, cells=n, lo=[-lim, -lim], hi=[lim, lim]).betti[1])


def edge(lo, hi, n, want_lo):
    # b1(lo) == want_lo != b1(hi); bisect to relative width 1e-4
    while hi / lo - 1 > 1e-4:
        mid = np.sqrt(lo * hi)
        if (b1(mid, n) > 0) == want_lo:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


if __name__ == "__main__":
    out = {}
    for n in (3000, 4000):
        grid = np.round(np.arange(0.8, 1.9, 0.05), 3)
        hs = [(float(s), b1(s, n)) for s in grid]
        inside = [s for s, b in hs if b > 0]
        a, b = min(inside), max(inside)
        prev = max(s for s, _ in hs if s < a); nxt = min(s for s, _ in hs if s > b)
        lo_edge = edge(prev, a, n, want_lo=False); hi_edge = edge(b, nxt, n, want_lo=True)
        out[n] = dict(scan=hs, hole_from=lo_edge, hole_to=hi_edge)
        print(n, "hole for sigma in [%.4f, %.4f]" % (lo_edge, hi_edge), flush=True)
    (ROOT / "results/data/three_range_hole.json").write_text(json.dumps(out, indent=1))
