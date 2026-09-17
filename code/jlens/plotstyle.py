"""One figure style for every plot in the paper, matched to the ICLR template.

Typography.  ``lens.tex`` is ``\\documentclass{article}`` + ``times``, so the
body face is Times (URW's ``Nimbus Roman``) at 10pt on 11pt leading, with
``\\small`` at 9pt and ``\\scriptsize`` at 8pt.  The figures use the same face
and quote those same sizes, so a label in a figure is the same ink as a label
in the text.  ``times`` leaves maths in Computer Modern, so the figures do too
(``mathtext.fontset="cm"``): ``$\\log_{10}$`` in an axis label is then set from
the same design as ``$\\log_{10}$`` in a sentence.

Scale.  A point size in a figure only equals a point size on the page if the
figure is placed at its natural width.  ``FULL_WIDTH`` and ``BODY_WIDTH`` are
the two widths ``lens.tex`` actually uses; pick the figsize from those and save
without ``bbox_inches="tight"`` (which silently rescales), and the two match.

Colour.  https://coolors.co/palette/f4f1de-e07a5f-3d405b-81b29a-f2cc8f, plus
tints of the palette's navy where a recessive neutral is needed.  Series are
separated by shape and line style as well as by hue, so the figures survive
greyscale printing and the common colour-vision deficiencies.

Output.  PDF, vector throughout: no raster resolution to be limited by, and
text stays selectable and searchable.  Fonts are embedded as Type 42 (a
publisher requirement, and what keeps maths from reflowing on another machine),
and the time stamp is dropped so a rebuild is byte-identical.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# https://coolors.co/palette/f4f1de-e07a5f-3d405b-81b29a-f2cc8f
CREAM = "#F4F1DE"
TERRACOTTA = "#E07A5F"
NAVY = "#3D405B"
SAGE = "#81B29A"
SAND = "#F2CC8F"
PALETTE = (NAVY, TERRACOTTA, SAGE, SAND, CREAM)

# Recessive neutrals: the palette's navy over white at ~45% / ~25% / ~10%.
MUTED = "#8D8FA0"
FAINT = "#AEAFBB"
GRID = "#E2E2E8"

# Semantic roles, so a colour means the same thing in every figure.
INK = NAVY  # text, axis labels, primary marks
HIT = NAVY  # subjects the LRE reproduced
MISS = TERRACOTTA  # subjects it did not
FIT = SAGE  # regression over all points
FIT_MISS = TERRACOTTA  # regression over the misses only
ACCENT = SAND  # called-out marks (e.g. the rescored year relations)
DEGENERATE = MUTED  # panels with nothing to measure
EDGE = "white"  # marker halo, to separate overlapping points

# Times at the ICLR template's sizes.  10pt body, \small 9pt, \scriptsize 8pt.
BODY_PT = 10.0
SMALL_PT = 9.0
SCRIPT_PT = 8.0
TINY_PT = 7.0

# \textwidth is 5.5in (iclr2027_conference.sty); \includegraphics[width=...]
# in lens.tex uses \linewidth and 0.8\linewidth.
FULL_WIDTH = 5.5
BODY_WIDTH = 0.8 * FULL_WIDTH

SERIF_STACK = [
    "Nimbus Roman",  # what \usepackage{times} resolves to on TeX Live
    "TeX Gyre Termes",
    "Times New Roman",
    "Tinos",
    "Liberation Serif",
    "STIXGeneral",  # last Times-metric fallback before the default face
    "DejaVu Serif",
]

RC = {
    "font.family": "serif",
    "font.serif": SERIF_STACK,
    "font.size": SMALL_PT,
    "mathtext.fontset": "cm",  # what \usepackage{times} leaves maths in
    "axes.labelsize": SMALL_PT,
    "axes.titlesize": SMALL_PT,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.titlepad": 5.0,
    "axes.labelpad": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "xtick.labelsize": SCRIPT_PT,
    "ytick.labelsize": SCRIPT_PT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "text.color": INK,
    "legend.fontsize": SCRIPT_PT,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.borderpad": 0.1,
    "legend.labelspacing": 0.25,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.9,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.5,
    "figure.dpi": 300,  # on-screen and for the PNG companions
    "savefig.dpi": 600,  # only bites where something is rasterised
    "savefig.bbox": None,  # keep the figure at its declared size; see above
    "savefig.pad_inches": 0.0,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "pdf.fonttype": 42,  # embed as TrueType, not as subsetted Type 3
    "ps.fonttype": 42,
    "pdf.compression": 9,
}


def use_paper_style():
    """Apply the style globally.  Idempotent; call it before building a figure."""
    plt.rcParams.update(RC)


def collect_handles(axes):
    """Legend entries across a row of panels, first occurrence wins.

    Taking them from one panel is not enough: a panel can be missing a
    series (no misses, so no miss fit) and would then drop it from the key.
    """
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def label_bottom_row(axes, ncol, n, xlabel, size=None):
    """Put the x label and its tick labels on the lowest panel of each column.

    ``sharex`` strips tick labels from every row but the last, which on a
    ragged grid leaves the panels above the short final row with an axis
    label and no numbers under it.
    """
    for column in range(min(ncol, n)):
        ax = axes[max(i for i in range(n) if i % ncol == column)]
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(xlabel, fontsize=size or TINY_PT)


def deemphasise(ax, degenerate):
    """Grey a panel out when it has no variance left to measure."""
    if degenerate:
        for spine in ax.spines.values():
            spine.set_color(FAINT)


def save(fig, path, *, png=False):
    """Write a print-ready, reproducible PDF (and optionally a PNG companion).

    ``CreationDate=None`` drops the time stamp, so identical inputs give an
    identical file; the empty ``Author`` keeps the double-blind submission from
    carrying a user name in its metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": path.stem.replace("_", " "),
        "Author": "",
        "Subject": "",
        "Keywords": "",
        "Creator": "matplotlib",
        "Producer": "matplotlib",
        "CreationDate": None,
    }
    fig.savefig(path.with_suffix(".pdf"), metadata=metadata)
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=RC["savefig.dpi"])
    return path.with_suffix(".pdf")
