"""Exact first thresholds sigma1 (end of the staircase regime) on a fine reduced grid, and proved/exact ratios.
Solves Delta_alpha(sigma) = t_min^2 by Brent's method (cells=2000; spot-checked against cells=3000).
Writes results/data/stage4_exact_sigma1_fine.json."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import optimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptopo import analysis  # noqa: E402
from stage4_check import sigma1  # noqa: E402

CELLS = 2000

def solve(job):
    tag, d, B, r, guess = job
    s = (2 + r[0] + r[1]) / 2; tmin = min(s - 2, s - r[0], s - r[1])
    f = lambda sg: analysis.analyze_reduced(d, B, r, sg, cells=CELLS).depth * sg * sg - tmin ** 2
    root = optimize.brentq(f, guess * 0.9, guess * 1.1, xtol=1e-7)
    proved = sigma1(2.0, r[0], r[1], d)
    return dict(case=tag, d=d, exact=root, proved=proved, ratio=proved / root)

if __name__ == "__main__":
    old = {(q["theta_deg"], q["d"]): q["sigma1"] for q in json.load(open("results/data/stage4_exact_sigma1.json"))["rows"]}
    jobs = []
    for th in (30, 60, 90, 120, 150):
        B, r = analysis.symmetric_pair(np.radians(th))
        for d in (2, 3, 4, 6, 10):
            jobs.append((th, d, B, np.asarray(r, float), old[(th, d)]))
    x = np.array([0.2, 1.1]); B = np.array([[-1.0], [1.0]])
    r = np.array([np.hypot(x[0] + 1, x[1]), np.hypot(x[0] - 1, x[1])])
    jobs += [("asym", 2, B, r, 0.2821), ("asym", 3, B, r, 0.2880)]
    with Pool(8) as pool:
        rows = pool.map(solve, jobs)
    Path("results/data/stage4_exact_sigma1_fine.json").write_text(json.dumps(dict(cells=CELLS, alpha=0.05, rows=rows), indent=1, default=float))
    for q in rows:
        print(q["case"], q["d"], "exact %.5f proved %.5f ratio %.3f" % (q["exact"], q["proved"], q["ratio"]))
