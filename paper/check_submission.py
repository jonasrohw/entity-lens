"""Check mechanical ICLR submission properties; not an author attestation."""

import hashlib
import re
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def run(*args):
    return subprocess.check_output(args, cwd=PAPER, text=True)


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"PASS: {message}")


def main():
    for line in (PAPER / "ICLR2027_TEMPLATE_SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        require(
            hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == digest,
            f"official template unchanged: {name}",
        )

    pdf = "iclr2027_submission.pdf"
    require(
        (PAPER / pdf).read_bytes() == (PAPER / "build/lens.pdf").read_bytes(),
        "submission PDF matches the current build",
    )
    log = (PAPER / "build/lens.log").read_text()
    require(
        not re.search(
            r"undefined|multiply defined|Overfull|LaTeX Error|^!", log, re.MULTILINE
        ),
        "no unresolved references, duplicate labels, overfull boxes, or LaTeX errors",
    )
    require(
        "Warning--" not in (PAPER / "build/lens.blg").read_text(),
        "BibTeX reports no warnings",
    )
    aux = (PAPER / "build/lens.aux").read_text()
    match = re.search(r"\\newlabel\{sec:main-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    require(match is not None, "main-text end is recorded")
    main_pages = int(match.group(1))
    require(main_pages <= 9, f"main text ends on page {main_pages} (limit 9)")

    info = run("pdfinfo", pdf)
    require(
        bool(re.search(r"^Author:\s*$", info, re.MULTILINE)),
        "PDF author metadata is empty",
    )
    require("Encrypted:       no" in info, "PDF is not encrypted")
    count = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    sizes = run("pdfinfo", "-f", "1", "-l", str(count), pdf)
    dimensions = re.findall(r"Page\s+\d+ size:\s+([\d.]+) x ([\d.]+)", sizes)
    require(
        len(dimensions) == count
        and all(float(w) == 612 and float(h) == 792 for w, h in dimensions),
        f"all {count} pages use US Letter",
    )
    fonts = run("pdffonts", pdf).splitlines()[2:]
    require(
        bool(fonts) and all(line.split()[-5] == "yes" for line in fonts),
        "all fonts are embedded",
    )
    require(all("Type 3" not in line for line in fonts), "no Type 3 fonts")

    text = run("pdftotext", "-layout", pdf, "-")
    pages = [page for page in text.split("\f") if page.strip()]
    require(
        len(pages) == count
        and all(
            "Under review as a conference paper at ICLR 2027" in page for page in pages
        ),
        "every page carries the ICLR 2027 review header",
    )
    require(
        "Anonymous authors" in pages[0]
        and "Paper under double-blind review" in pages[0],
        "anonymous review title block is present",
    )
    require(
        all(re.search(r"^\s*\d{3}\s*$", page, re.MULTILINE) for page in pages),
        "review line numbers are present on every page",
    )
    require(
        not re.search(
            r"TO BE EXPANDED|Appendix\s*\?|\?\?|just random stuff|Anonymous ACL", text
        ),
        "no visible draft placeholders or unresolved question marks",
    )
    require(
        not re.search(
            r"jonas|hinrich|cis\.uni-muenchen|/mounts/|/home/", text, re.IGNORECASE
        ),
        "no known local author identifiers in extracted text",
    )
    compact = re.sub(r"\s+", "", text).lower()
    require(
        "aiusestatement" in compact,
        "AI-use section is present (completeness requires author input)",
    )
    require(
        compact.index("references")
        < compact.index("jacobian–vectorproducts", compact.index("references")),
        "appendix follows references",
    )
    print(
        "Mechanical checks passed. Author declarations and scientific review remain separate."
    )


if __name__ == "__main__":
    main()
