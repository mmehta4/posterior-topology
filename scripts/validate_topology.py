"""Independent check of Proposition 4.4 against brute-force homology, at noise levels where the prediction is unambiguous.

The theorem is the ground truth; this script only checks that brute force agrees with it where a grid can resolve the
answer. Transition noise levels come from exact quadrature (results/data/exact_thresholds.json). Predeclared criterion:
every interval between consecutive transitions whose endpoints differ by at least 4% in sigma is sampled at its
geometric midpoint (plus one level below the first and one above the last transition); the predicted Betti numbers
(Proposition 4.4 with Lemma 4.1) are compared with
  (a) the reduced grid (half-plane, n x n/2 cells, n = 900), lifted by Lemma 4.1, for d in {2, 3, 4, 6, 10};
  (b) direct cubical homology of the full 3D region (GUDHI, no lift), for d = 3 at six angles and the asymmetric example.
Writes results/data/topology_validation.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptopo import analysis, topology  # noqa: E402

GAP = 1.04


def predicted(d, rungs, sigma):
    m = 0
    for g in rungs:
        if g["sigma"] < sigma:
            for name in g["events"]:
                m += 1 if "contact" in name else -1
    b = [0] * (d + 1)
    if m == 0:
        if d == 2:
            b[0] = 2
        else:
            b[0] = 1; b[d - 2] += 1
    else:
        b[0] = 1; b[d - 1] += m - 1
    return b, m


def sample_levels(rungs):
    s = [g["sigma"] for g in rungs if np.isfinite(g["sigma"])]
    lv = [s[0] / 1.3]
    for a, b in zip(s, s[1:]):
        if b / a >= GAP:
            lv.append(float(np.sqrt(a * b)))
    lv.append(s[-1] * 1.3)
    return lv, sum(1 for a, b in zip(s, s[1:]) if b / a < GAP)


def reduced_job(L):
    th, d, rungs = L["theta"], L["d"], L["rungs"]
    B, r = analysis.symmetric_pair(np.radians(th))
    levels, skipped = sample_levels(rungs)
    out = []
    for sg in levels:
        pb, m = predicted(d, rungs, sg)
        gb = analysis.analyze_reduced(d, B, r, sg, cells=900).betti.tolist()
        out.append(dict(sigma=sg, m=m, predicted=pb, grid=gb, ok=pb == gb))
    return dict(theta=th, d=d, checks=out, narrow_intervals_skipped=skipped)


def full3d_job(args):
    tag, rungs = args
    if tag == "asym":
        x = np.array([0.2, 1.1, 0.0]); A = np.array([[-1.0, 0, 0], [1.0, 0, 0]])
        r = np.linalg.norm(A - x, axis=1)
    else:
        a = np.sin(np.radians(tag) / 2); A = np.array([[-1.0, 0, 0], [1.0, 0, 0]]); r = np.array([1 / a, 1 / a])
    levels, skipped = sample_levels(rungs)
    out = []
    for sg in levels:
        pb, m = predicted(3, rungs, sg)
        lim = r.max() + 1 + 5 * sg
        res = analysis.analyze_full(A, r, sg, cells=170, lo=[-lim] * 3, hi=[lim] * 3)
        gb = list(topology.betti_gudhi(res.mask))[:4]
        gb = gb + [0] * (4 - len(gb))
        out.append(dict(sigma=sg, m=m, predicted=pb, gudhi_3d=gb, ok=pb == gb))
    return dict(case=tag, checks=out, narrow_intervals_skipped=skipped)


if __name__ == "__main__":
    E = json.loads((ROOT / "results/data/exact_thresholds.json").read_text())
    with Pool(8) as pool:
        red = pool.map(reduced_job, E["ladders"])
        jobs = [(L["theta"], L["rungs"]) for L in E["ladders"] if L["d"] == 3 and L["theta"] in (30, 45, 60, 75, 90, 120)]
        asym3 = [a for a in E["asym"] if a["d"] == 3][0]
        jobs.append(("asym", [dict(sigma=g["sigma"], events=[g["event"]]) for g in asym3["rungs"]]))
        full = pool.map(full3d_job, jobs)
    n_red = sum(len(x["checks"]) for x in red); ok_red = sum(c["ok"] for x in red for c in x["checks"])
    n_full = sum(len(x["checks"]) for x in full); ok_full = sum(c["ok"] for x in full for c in x["checks"])
    skipped = sum(x["narrow_intervals_skipped"] for x in red)
    (ROOT / "results/data/topology_validation.json").write_text(json.dumps(dict(
        criterion=f"intervals with sigma ratio >= {GAP} sampled at geometric midpoint", reduced=red, full3d=full,
        summary=dict(reduced_checks=n_red, reduced_ok=ok_red, full3d_checks=n_full, full3d_ok=ok_full,
                     narrow_intervals_not_sampled=skipped)), indent=1, default=float))
    print(f"reduced grid: {ok_red}/{n_red} agree; direct 3D GUDHI: {ok_full}/{n_full} agree; "
          f"intervals narrower than 4% not sampled: {skipped}")
    for x in red + full:
        for c in x["checks"]:
            if not c["ok"]:
                print("MISMATCH", x.get("theta", x.get("case")), x.get("d", 3), c)
