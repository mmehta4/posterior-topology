# arXiv submission sheet

Fill the arXiv form with exactly these values.

**Upload file:** `arxiv/arxiv-upload.tar.gz` (built by `python3 arxiv/build_arxiv.py`; it compiles stand-alone).

**Processor:** XeLaTeX (the paper uses OpenType fonts, shipped in `fonts/`, so no TeX Live font package is needed). TeX Live: 2025 (the default). Check the compiled preview arXiv shows before submitting.

**Title:** The topology of range-measurement credible regions

**Authors:** Meet Mehta

**Abstract** (1891 characters, limit 1920):

Noisy distances from a point to known anchors give posteriors whose credible regions are rings, shells, tubes and disconnected pieces rather than ellipses. We determine their topology exactly. For two ranges the posterior, in residual coordinates (the distances minus the observed ranges), is a Gaussian times an explicit weight on a convex polygonal domain, and every highest-posterior-density region is a disc cut by the walls of that domain. Its topology can be read off the triangle whose sides are the anchor separation and the two observed ranges, which gives the complete sequence of topologies as the noise grows, in every dimension: each transition occurs when the cut-off reaches one of five depths, the squared incircle tangent lengths $t_*^2$, $t_1^2$, $t_2^2$ and the sums $t_*^2 + t_1^2$, $t_*^2 + t_2^2$, and more noise first creates holes and then removes them. With a different noise level on each range the disc becomes an ellipse, and a contact can annihilate with an anchor. An explicit noise level, computable from the data, certifies the noise-free shape. For $n \le d$ ranges the noise-free shape persists while the region misses the affine hull of the anchors, and for $n = d$ only then. With three ranges in the plane a hole can surround no anchor: for distant truths it closes at the anchors' Fermat point. For one range the region has a hole exactly when $r/\sigma$ exceeds $\kappa_d = \sqrt{d/2} + z\sqrt{3/8} + O(d^{-1/2})$, and the posterior mean then lies in the hole. In simulation, past the first hole, a mixture of Laplace approximations at the two posterior modes falls from 93% to 84% coverage at the 95% level, while the HPD region stays at 94–98%. In high dimension every transition has an explicit limit with a $1/\sqrt d$ correction, and for random anchors in the plane the flip pair is lost with probability of order $\sqrt\sigma$, up to a logarithm.

**Comments:** 44 pages, 14 figures, 3 tables, including a supplement. Code and data:
https://github.com/mmehta4/posterior-topology (add the Zenodo DOI once minted)

**Primary category:** math.ST (Statistics Theory). arXiv adds the stat.TH alias automatically.

**Cross-lists:** stat.ME (Methodology), eess.SP (Signal Processing)

**MSC classes:** 62F15 (Primary); 55N31, 57R70, 62E20, 60D05 (Secondary)

**ACM classes:** none

**Journal reference / DOI:** leave empty until published.

**Licence:** CC BY 4.0
