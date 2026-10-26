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
SUPP_LABELS = set(re.findall(r"\\label\{([^}]+)\}", (C / "supplement.tex").read_text() + (C / "appendix.tex").read_text()))

def plain_refs(text, foreign):
    text = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + NUM[m.group(1)] + ")" if m.group(1) in foreign else m.group(0), text)
    return re.sub(r"\\ref\{([^}]+)\}", lambda m: NUM[m.group(1)] if m.group(1) in foreign else m.group(0), text)

ALL_LABELS = set(NUM)
read = lambda name: (C / f"{name}.tex").read_text().strip()

# The journal asks for at most about 25 pages, so Appendix A and the proofs move to the supplement (as its last two
# sections), except the three that carry the main idea (Theorem 2.1, Lemma 4.2, the triangle theorem); every statement
# stays in the paper. Each proof is identified by the label of the result just before it.
MOVE = ["lem:shell", "thm:main", "thm:noise", "prop:laplace", "cor:mean", "prop:gauss", "lem:lift", "prop:axis",
        "cor:first", "rem:inconsistent", "lem:subspace", "thm:nranges", "thm:unequal",
        "cor:anchortest", "prop:criterion", "lem:weight", "thm:staircase", "thm:staircaseW", "thm:sqrt2",
        "thm:random2"]
KIND = {"theorem": "Theorem", "lemma": "Lemma", "proposition": "Proposition", "corollary": "Corollary"}
SUPP_SEC = str(1 + len(re.findall(r"\\section\{", read("supplement") + read("appendix"))))
SUPP_CITE = r"(\citet{mehta2026supp})"

def split_proofs(body):
    lines, keep, moved, last, proof = body.split("\n"), [], [], None, None
    for line in lines:
        m = re.search(r"\\begin\{(theorem|lemma|proposition|corollary)\}", line)
        if m:
            last = [m.group(1), None]
        lab = re.search(r"\\label\{([^}]+)\}", line)
        if lab and last and last[1] is None:
            last[1] = lab.group(1)
        if proof is None and "\\begin{proof}" in line and last and last[1] in MOVE:
            proof, part = [line], re.search(r"\\begin\{proof\}\[Proof of ([^\]]*)\]", line)
            continue
        if proof is not None:
            proof.append(line)
            if "\\end{proof}" in line:
                moved.append((last, "\n".join(proof)))
                # one pointer per result: Theorem 3.2 has separate proofs of (a), (b), (d)
                parts = PARTS.get(last[1], [])
                if not parts:
                    keep.append(f"The proof is in Section~S{SUPP_SEC} of the Supplementary Material {SUPP_CITE}.")
                elif part and part.group(1) == parts[0]:
                    names = ", ".join(parts[:-1]) + " and " + parts[-1]
                    keep.append(f"The proofs of {names} are in Section~S{SUPP_SEC} of the Supplementary Material "
                                f"{SUPP_CITE}.")
                proof = None
            continue
        keep.append(line)
    assert {lab for (_, lab), _ in moved} == set(MOVE), "a proof to move was not found"
    return "\n".join(keep), moved

PARTS = {}
for (kind, lab), proof in split_proofs(read("body"))[1]:
    m = re.search(r"\\begin\{proof\}\[Proof of ([^\]]*)\]", proof)
    if m:
        PARTS.setdefault(lab, []).append(m.group(1))
BODY, MOVED = split_proofs(read("body"))
# This remark explains the proof of Theorem 3.2(a), so it travels with that proof.
SORTING = "The hole's edge $v = 2$ is exactly where $(v-1)^2$ crosses $1$, which is why the sorting works."
assert BODY.count(SORTING) == 1
BODY = BODY.replace(SORTING + "\n", "")
MOVED = [(k, pr + "\n\n" + SORTING if k[1] == "thm:main" and "Proof of (a)" in pr else pr) for k, pr in MOVED]
MOVED_TEX = (r"\section{Proofs of results stated in the main paper}" + "\n\n"
             + "\n\n".join(re.sub(r"\\begin\{proof\}(\[[^\]]*\])?",
                                  lambda m: r"\begin{proof}[Proof of " + KIND[kind] + r"~\ref{" + lab + "}"
                                  + (" " + m.group(1)[1:-1].replace("Proof of ", "") if m.group(1) else "") + "]",
                                  proof, count=1)
                           for (kind, lab), proof in MOVED))
for old, new in [
        ("Appendix~\\ref{app:sqrt} proves the square-root law. The\nSupplementary Material contains the proofs of "
         "Theorem~\\ref{thm:main}(c) and of Sections~\\ref{sec:three}\nand~\\ref{sec:high} (Supplements~\\ref{app:c}--"
         "\\ref{app:high}) and the computations (Supplement~\\ref{app:comp}).",
         "The\nSupplementary Material " + SUPP_CITE + " contains the proofs of Theorem~\\ref{thm:main}(c) and of "
         "Sections~\\ref{sec:three}\nand~\\ref{sec:high} (Supplements~\\ref{app:c}--\\ref{app:high}), the computations "
         "(Supplement~\\ref{app:comp}), the proof of\nthe square-root law (Supplement~\\ref{app:sqrt}) and the other "
         "longer proofs (Supplement~S" + SUPP_SEC + ")."),
        ("The proof is in Appendix~\\ref{app:sqrt}.", "The proof is in Supplement~\\ref{app:sqrt}."),
        ("The decomposition of\nAppendix~\\ref{app:sqrt}", "The decomposition of\nSupplement~\\ref{app:sqrt}")]:
    assert BODY.count(old) == 1, old[:60]
    BODY = BODY.replace(old, new)

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
% number sections and subsections only, not the run-in paragraph headings
\setcounter{secnumdepth}{2}
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

""" + BODY + r"""

\begin{supplement}
\stitle{Supplement to ``""" + TITLE + r"""''}
\sdescription{Proofs of Theorem~\ref{thm:main}(c) and of the results of Section~\ref{sec:high}, the two-variable
Laplace lemma, the computations behind every number, table and figure, and the longer proofs of results stated in
the paper.}
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
Laplace lemma, the computations behind every number, table and figure, and the longer proofs of results stated in
the main paper. References to the main paper are to its numbering.
\end{abstract}
\end{frontmatter}

""" + read("supplement") + "\n\n" + read("appendix") + "\n\n" + MOVED_TEX + r"""

\bibliographystyle{spr-ims-nameyear}
\bibliography{refs}

\end{document}
"""

supp = plain_refs(supp, ALL_LABELS - SUPP_LABELS)
(HERE / "supplement-aos.tex").write_text(supp)
(HERE / "refs.bib").write_text((FULL / "refs.bib").read_text() + """
@misc{mehta2026supp,
  author = {Mehta, Meet},
  title  = {Supplement to ``""" + TITLE + """''},
  year   = {2026},
  note   = {DOI: 10.1214/[provided by typesetter]}
}
""")
def compile_(name):
    r = subprocess.run(["tectonic", "--keep-intermediates", "--keep-logs", f"{name}.tex"], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"{name} failed")

compile_("supplement-aos")
# The appendix is now a supplement section, so main's references to the supplement take the supplement's numbers.
NUM.update({k: v for k, v in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", (HERE / "supplement-aos.aux").read_text())
            if k in SUPP_LABELS})
(HERE / "main-aos.tex").write_text(plain_refs(main, SUPP_LABELS))
compile_("main-aos")
print("built main-aos.pdf and supplement-aos.pdf")
