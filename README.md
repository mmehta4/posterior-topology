# The topology of range-measurement credible regions — code

Code for the paper *The topology of range-measurement credible regions* (Meet Mehta). Every number, table and
figure in the paper is computed by a script here.

## Setup

```bash
python3 -m venv ~/.venvs/posterior-topology
~/.venvs/posterior-topology/bin/pip install -r requirements.txt
```

Python 3.9 or later; the pinned versions are in `requirements.txt` (mpmath 1.4.1, GUDHI 3.11.0). Then run
`make test` (27 tests) and `make paper`, which runs every script below in order (about an hour on 8 cores).

## What computes what

| Paper | Script | Output (`results/data/`) |
|---|---|---|
| Table 1, κ_d(α) | `scripts/table1.py` | `table1.json` |
| Every closed-form number quoted in the text (21 checks) | `scripts/paper_numbers_check.py` | printed |
| κ_d for d ≤ 100 (Figures 2 and 3(a)) | `scripts/problem0.py` | `problem0_kappa_table.json` |
| Figure 3(b), next-order constant c_α | `scripts/kappa_next_order.py` | `kappa_next_order.json` |
| Two-range thresholds: the ladder (§4), Figures 7 and 11, all of Table 2 (exact and certified σ₁, escaping masses) | `scripts/exact_thresholds.py` (exact quadrature, `ptopo/exact.py`) | `exact_thresholds.json` |
| Proposition 4.4, independent topology check (536/536, 28/28) | `scripts/validate_topology.py` | `topology_validation.json` |
| Lemma 4.6, inconsistent data | `scripts/inconsistent_data_check.py`, `scripts/inconsistent_data_check_fine.py` | printed |
| Remark 4.7, the three-range hole interval | `scripts/three_range_hole.py` | `three_range_hole.json` |
| §4.5, three ranges in the plane: search for smooth maxima, the x* = (0, −8) example and its hole | `scripts/three_range_smooth_max_search.py`, `scripts/three_range_fermat_example.py`, `scripts/three_range_fermat_hole.py` | printed |
| §4.5: where a smooth maximum exists (Figure 11(a)); at most one at a time | `scripts/three_range_phase_scan.py`, `scripts/three_range_max_count.py` | `three_range_phase_scan.npz` |
| Theorem 4.8 and Proposition 4.9: Fermat limit, far critical points, the ring and hole noise levels | `scripts/stage6b_check.py`, `scripts/stage6b_levels.py` | printed |
| Theorem 4.11, the n = d = 3 example | `scripts/nranges_check.py` | printed |
| Theorem 4.13 and Corollary 4.14, unequal noise | `scripts/unequal_noise_check.py`, `scripts/unequal_noise_classification.py` | printed |
| Theorem 5.4, the unequal-noise certificate | `scripts/unequal_certificate_check.py` | printed |
| Corollary 6.4 and Figure 13 | `scripts/large_d_next_order.py` | `large_d_next_order.json` |
| Theorem 7.3, geometry of random anchors | `scripts/sqrt_law_geometry_check.py` | printed |
| Theorem 7.3 and Conjecture 7.4, the failure probability | `scripts/flip_sqrt_law.py` | `flip_sqrt_law.json` |
| Table 3, coverage | `scripts/coverage.py` | `coverage.json` |
| Figures 1–14 | `scripts/paper_figures.py` (after `exact_thresholds.py`; Figure 11 needs `three_range_phase_scan.py`) | `paper/full/figs/` |

The library is in `ptopo/`: `exact.py` (two-range quadrature in residual coordinates), `radial.py` (one range, 60-digit
arithmetic), `topology.py` (Betti numbers on grids, with a GUDHI cross-check). Tests are in `tests/`.

## The paper

The text lives once, in `paper/full/content/` (abstract, body, appendix, supplement), and is built two ways:

- `paper/full/main.tex`: the arXiv version, a single PDF with the supplement at the end. It compiles with XeLaTeX
  (`tectonic main.tex`). `python3 paper/full/arxiv/build_arxiv.py` makes the upload archive.
- `paper/full/aos/build_aos.py`: the Annals of Statistics version, in the journal's `imsart` class (VTeX, MIT licence,
  included). It writes one .tex file for the paper and one for the supplement, as the journal asks, with references
  between the two written out as plain numbers taken from the arXiv build.

Other scripts in `scripts/` are from earlier stages of the project and are not needed for the paper.

## Licence

MIT (see `LICENSE`).
