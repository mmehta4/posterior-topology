# arXiv submission sheet

Fill the arXiv form with exactly these values.

**Upload file:** `arxiv/arxiv-upload.tar.gz` (built by `python3 arxiv/build_arxiv.py`; it compiles stand-alone).

**Processor:** XeLaTeX (the paper uses OpenType fonts). TeX Live: 2025 (the default).

**Title:** The topology of range-measurement credible regions

**Authors:** Meet Mehta

**Abstract** (1843 characters, limit 1920):

Noisy distances from a point to known anchors give posteriors whose credible regions are rings, shells, tubes and disconnected pieces rather than ellipses. We determine their topology exactly. The key observation is that for two ranges, in residual coordinates (the differences between distances and observed ranges), the posterior is a Gaussian times an explicit weight on a convex polygonal domain, and every highest-posterior-density region is a disc cut by the walls of that domain. Its topology can therefore be read off the triangle whose sides are the anchor separation and the two observed ranges. This gives the complete sequence of topologies as the noise grows, in every dimension: each transition occurs when the cut-off reaches one of five depths, the squared incircle tangent lengths $t_*^2, t_1^2, t_2^2$ and the sums $t_*^2 + t_1^2$, $t_*^2 + t_2^2$, and more noise first creates holes and then removes them. With a different noise level on each range the disc becomes an ellipse, and a contact can annihilate with an anchor. For $n \le d$ ranges the noise-free shape persists while the region misses the affine hull of the anchors, and for $n = d$ only then. An explicit noise level, computable from the data, certifies the noise-free shape. In simulation, from the noise level at which the first hole forms, even a mixture of Laplace approximations at the two posterior modes covers the true point in only 84–93% of data sets at the 95% level. For one range the region has a hole exactly when $r/\sigma$ exceeds $\kappa_d = \sqrt{d/2} + z\sqrt{3/8} + O(d^{-1/2})$, and the posterior mean then lies in the hole. In high dimension every transition has an explicit limit with a $1/\sqrt d$ correction, and for random anchors in the plane the flip pair is lost with probability of order $\sqrt\sigma$, up to a logarithmic factor.

**Comments:** 41 pages, 13 figures, 3 tables, including a supplement. Code and data:
https://github.com/mmehta4/posterior-topology (add the Zenodo DOI once minted)

**Primary category:** math.ST (Statistics Theory). arXiv adds the stat.TH alias automatically.

**Cross-lists:** stat.ME (Methodology), eess.SP (Signal Processing)

**MSC classes:** 62F15 (Primary); 55N31, 57R70, 62E20, 60D05 (Secondary)

**ACM classes:** none

**Journal reference / DOI:** leave empty until published.

**Licence:** CC BY 4.0
