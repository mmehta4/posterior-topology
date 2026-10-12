"""Build the arXiv upload: paper/full/arxiv/arxiv-upload.tar.gz with main.tex, content/, figs/, refs.bib and main.bbl.
arXiv must be told to use XeLaTeX (the paper uses OpenType fonts). Usage: python3 build_arxiv.py"""
import re, subprocess, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent
subprocess.run(["tectonic", "--keep-intermediates", "--keep-logs", "main.tex"], cwd=FULL, check=True,
               capture_output=True)
log = (FULL / "main.log").read_text(errors="ignore")
assert "Overfull" not in log and "undefined" not in log.lower().replace("font shape", ""), "fix warnings first"
files = ["main.tex", "main.bbl", "refs.bib"] + [f"content/{p.name}" for p in sorted((FULL / "content").glob("*.tex"))]
figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                             "".join((FULL / f).read_text() for f in files if f.endswith(".tex")))))
files += [f"figs/{f if f.endswith('.pdf') else f + '.pdf'}" for f in figs]
out = HERE / "arxiv-upload.tar.gz"
with tarfile.open(out, "w:gz") as tar:
    for f in files:
        tar.add(FULL / f, arcname=f)
print(f"{out.name}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB")
for f in files: print("  ", f)
