"""Build the ICLR preprint and collect its local inputs for arXiv."""

from pathlib import Path
import shutil
import subprocess
import tarfile


PAPER = Path(__file__).resolve().parents[1]
BUILD = PAPER / "build" / "arxiv"
SOURCE = BUILD / "source"


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
            "-outdir=" + str(BUILD), "arxiv.tex",
        ],
        cwd=PAPER,
        check=True,
    )
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    SOURCE.mkdir()

    # TeX's recorder lists the actual inputs, including nested tables/figures.
    inputs = set()
    for line in (BUILD / "arxiv.fls").read_text().splitlines():
        if line.startswith("INPUT "):
            path = (PAPER / line[6:]).resolve()
            if path.is_relative_to(PAPER) and not path.is_relative_to(BUILD):
                if path.suffix in {".tex", ".sty", ".pdf", ".png", ".jpg"}:
                    inputs.add(path)
    inputs.update({PAPER / "custom.bib", PAPER / "iclr2027_conference.bst"})
    # Inline the wrapper so the archive has exactly one document entry point.
    inputs.difference_update({PAPER / "arxiv.tex", PAPER / "lens.tex"})
    for path in sorted(inputs):
        target = SOURCE / path.relative_to(PAPER)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (SOURCE / "arxiv.tex").write_text(
        "\\def\\arxivversion{1}\n" + (PAPER / "lens.tex").read_text()
    )
    shutil.copy2(BUILD / "arxiv.bbl", SOURCE / "arxiv.bbl")
    archive = BUILD / "arxiv-source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(SOURCE))
    print(f"PDF: {BUILD / 'arxiv.pdf'}")
    print(f"Source archive: {archive}")


if __name__ == "__main__":
    main()
