"""Assemble the Annals of Statistics submission from the shared content files, as the journal asks: one .tex file
for the paper (main-aos.tex) and one for the Supplementary Material (supplement-aos.tex). The journal forbids
external references, so references between the two are written as plain numbers. Then compile both with tectonic.
Usage: python3 build_aos.py"""
import re, shutil, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
C = FULL / "content"

# The journal forbids external references, so references between the paper and its supplement are written as plain
# numbers, taken from the arXiv build (main.tex), which has the same numbering for sections, results and equations.
subprocess.run(["tectonic", "--keep-intermediates", "main.tex"], cwd=FULL, check=True, capture_output=True)
NUM = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", (FULL / "main.aux").read_text())}
SUPP_LABELS = set(re.findall(r"\\label\{([^}]+)\}", (C / "supplement.tex").read_text()))

def plain_refs(text, foreign):
    text = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + NUM[m.group(1)] + ")" if m.group(1) in foreign else m.group(0), text)
    return re.sub(r"\\ref\{([^}]+)\}", lambda m: NUM[m.group(1)] if m.group(1) in foreign else m.group(0), text)

ALL_LABELS = set(NUM)
read = lambda name: (C / f"{name}.tex").read_text().strip()

PREAMBLE = r"""\documentclass[aos]{imsart}
\RequirePackage{amsthm,amsmath,amsfonts,amssymb,mathtools}
\RequirePackage[authoryear]{natbib}
\RequirePackage[colorlinks,citecolor=blue,urlcolor=blue,linkcolor=blue]{hyperref}
\RequirePackage{graphicx,booktabs}
\RequirePackage[shortlabels]{enumitem}
\graphicspath{{../figs/}}

\startlocaldefs
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{conjecture}[theorem]{Conjecture}
\theoremstyle{definition}
\newtheorem{remark}[theorem]{Remark}
\newcommand{\R}{\mathbb{R}}
\newcommand{\HPD}{\mathcal{H}_\alpha}
\newcommand{\dd}{\,\mathrm{d}}
\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator{\tr}{tr}
\providecommand{\bseriesno}[1]{#1}
\endlocaldefs
"""
AUTHOR = r"""\begin{aug}
\author[A]{\fnms{Meet}~\snm{Mehta}\ead[label=e1]{meetmehta.5410@gmail.com}}
\address[A]{\printead{e1}}
\end{aug}"""
KEYWORDS = r"""\begin{keyword}[class=MSC]
\kwdgroup[type=primary]{\kwd{62F15}}
\kwdgroup[type=secondary]{\kwd{55N31}\kwd{57R70}\kwd{62E20}\kwd{60D05}}
\end{keyword}

\begin{keyword}
\kwd{Highest-posterior-density region}
\kwd{Betti number}
\kwd{Morse theory}
\kwd{range measurement}
\kwd{flip ambiguity}
\kwd{Laplace method}
\end{keyword}"""
TITLE = "The topology of range-measurement credible regions"

main = PREAMBLE + "\n" + r"""\begin{document}

\begin{frontmatter}
\title{""" + TITLE + r"""}
\runtitle{Topology of range-measurement credible regions}
""" + AUTHOR + r"""

\begin{abstract}
""" + read("abstract") + r"""
\end{abstract}

""" + KEYWORDS + r"""
\end{frontmatter}

""" + read("body") + r"""

\begin{appendix}
""" + read("appendix") + r"""
\end{appendix}

\begin{supplement}
\stitle{Supplement to ``""" + TITLE + r"""''}
\sdescription{Proofs of Theorem~\ref{thm:main}(c) and of the results of Section~\ref{sec:high}, the two-variable
Laplace lemma, and the computations behind every number, table and figure.}
\end{supplement}

\bibliographystyle{spr-ims-nameyear}
\bibliography{refs}

\end{document}
"""

supp = PREAMBLE + r"\renewcommand{\thesection}{S\arabic{section}}" + "\n\n" + r"""\begin{document}

\begin{frontmatter}
\title{Supplement to ``""" + TITLE + r"""''}
\runtitle{Supplement: topology of range-measurement credible regions}
""" + AUTHOR + r"""

\begin{abstract}
Proofs of Theorem~\ref{thm:main}(c) and of the results of Section~\ref{sec:high} of the main paper, the two-variable
Laplace lemma, and the computations behind every number, table and figure. References to the main paper are to its
numbering.
\end{abstract}
\end{frontmatter}

""" + read("supplement") + r"""

\bibliographystyle{spr-ims-nameyear}
\bibliography{refs}

\end{document}
"""

main = plain_refs(main, SUPP_LABELS)
supp = plain_refs(supp, ALL_LABELS - SUPP_LABELS)
(HERE / "main-aos.tex").write_text(main)
(HERE / "supplement-aos.tex").write_text(supp)
shutil.copy(HERE.parent / "refs.bib", HERE / "refs.bib")
for name in ["main-aos", "supplement-aos"]:
    r = subprocess.run(["tectonic", "--keep-intermediates", "--keep-logs", f"{name}.tex"], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"{name} failed")
print("built main-aos.pdf and supplement-aos.pdf")
