from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict

from entity_lens.config import Project
from entity_lens.records import Readout
from entity_lens.summary import RelationSummary


class Style(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    width_inches: float
    height_inches: float
    font_family: str
    serif_fonts: tuple[str, ...]
    math_fontset: str
    label_size: float
    tick_size: float
    annotation_size: float
    ink: str
    muted: str
    grid: str
    fit: str
    accent: str
    edge: str
    marker_size: float
    accent_marker_size: float
    y_limits: tuple[float, float]
    x_padding: float


class AgreementFigure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: dict[str, tuple[float, float]]
    x_label: str
    y_label: str


class FigureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    style: Style
    agreement: AgreementFigure


def scientific(value: float) -> str:
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def apply_style(style: Style) -> None:
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.serif": list(style.serif_fonts),
            "mathtext.fontset": style.math_fontset,
            "font.size": style.label_size,
            "axes.labelsize": style.label_size,
            "axes.labelcolor": style.ink,
            "axes.edgecolor": style.muted,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "xtick.color": style.muted,
            "ytick.color": style.muted,
            "xtick.labelcolor": style.ink,
            "ytick.labelcolor": style.ink,
            "text.color": style.ink,
            "legend.frameon": False,
            "grid.color": style.grid,
            "grid.linewidth": 0.4,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def agreement(project: Project, summaries: list[RelationSummary], rho: float, p: float, output: Path) -> Path:
    settings = FigureConfig.model_validate(project.load("figures"))
    style, figure_settings = settings.style, settings.agreement
    apply_style(style)
    x = np.array([s.scores[Readout.CEL] for s in summaries])
    y = np.array([s.faithfulness for s in summaries])
    years = np.array([s.relation in project.protocol.years.relations for s in summaries])
    figure, axis = plt.subplots(figsize=(style.width_inches, style.height_inches), layout="constrained")
    slope, intercept = np.polyfit(x, y, 1)
    line = np.linspace(x.min() - style.x_padding, x.max() + style.x_padding, 50)
    axis.plot(line, intercept + slope * line, lw=1.0, color=style.fit, zorder=1)
    axis.scatter(
        x[~years], y[~years], s=style.marker_size, facecolor=style.ink, edgecolor=style.edge, linewidth=0.5, zorder=3
    )
    axis.scatter(
        x[years],
        y[years],
        s=style.accent_marker_size,
        marker="^",
        facecolor=style.accent,
        edgecolor=style.ink,
        linewidth=0.7,
        zorder=4,
    )
    for summary, xi, yi in zip(summaries, x, y, strict=True):
        if summary.relation in figure_settings.labels:
            offset = figure_settings.labels[summary.relation]
            axis.annotate(
                summary.relation,
                (xi, yi),
                xytext=offset,
                textcoords="offset points",
                ha="right" if offset[0] < 0 else "left",
                fontsize=style.annotation_size,
                color=style.ink,
                arrowprops={"arrowstyle": "-", "lw": 0.45, "color": style.muted},
            )
    axis.text(
        0.035,
        0.965,
        rf"$\rho = {rho:.2f}$   $p = {scientific(p)}$",
        transform=axis.transAxes,
        fontsize=style.tick_size,
        va="top",
    )
    axis.set_xlabel(figure_settings.x_label)
    axis.set_ylabel(figure_settings.y_label)
    axis.grid(visible=True, zorder=0)
    axis.set_ylim(*style.y_limits)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": output.stem,
        "Author": "",
        "Subject": "",
        "Keywords": "",
        "Creator": "matplotlib",
        "Producer": "matplotlib",
        "CreationDate": None,
    }
    figure.savefig(output, metadata=metadata)
    plt.close(figure)
    return output
