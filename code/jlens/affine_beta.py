"""Beta selection for the neutral-prompt affine Jacobian baseline."""

import argparse

import numpy as np

from jlens.matched_baseline import aggregate

DEFAULT_BETAS = "0.5,1,1.5,2,3,5"
SELECTION_METRICS = ("top1", "median-logrank")


def parse_betas(value):
    try:
        betas = sorted(set(float(item) for item in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "betas must be comma-separated numbers"
        ) from error
    if not betas or any(not np.isfinite(beta) or beta <= 0 for beta in betas):
        raise argparse.ArgumentTypeError("betas must be finite and positive")
    if 1.0 not in betas:
        raise argparse.ArgumentTypeError("betas must include 1 for the fixed baseline")
    return betas


def select(grid, betas, metric):
    """Match the LRE convention: first grid maximum wins. Betas are ascending."""
    if metric not in SELECTION_METRICS:
        raise ValueError(f"unknown beta selection metric: {metric}")
    return max(
        sorted(betas),
        key=lambda beta: (
            grid[str(beta)]["agree_at1"]
            if metric == "top1"
            else -grid[str(beta)]["median_lograng"]
        ),
    )


def tuned_score(sweep, metric):
    """Re-select beta from a saved sweep; every beta's aggregate is retained."""
    best = select(sweep["agg"], sweep["betas"], metric)
    return best, -sweep["agg"][str(best)]["median_lograng"]


def summarize(rows, betas, metric):
    grid = {
        str(beta): aggregate([row["rank_lre_beta"][str(beta)] for row in rows])
        for beta in betas
    }
    best = select(grid, betas, metric)
    return dict(
        betas=betas,
        selection_metric=metric,
        selection_split="scored_subjects",
        tie_break="smallest_beta",
        best_beta=best,
        agg=grid,
    )
