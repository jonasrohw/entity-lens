from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy import stats

Array = NDArray[np.float64]


class Correlation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rho: float | None
    p: float | None
    n: int


class Interval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: float
    low: float
    high: float


def _array(values: Sequence[float]) -> Array:
    return np.asarray(values, dtype=np.float64)


def spearman(x: Sequence[float], y: Sequence[float]) -> Correlation:
    a, b = _array(x), _array(y)
    if len(a) != len(b):
        message = f"Length mismatch: {len(a)} and {len(b)}"
        raise ValueError(message)
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return Correlation(rho=None, p=None, n=len(a))
    result = stats.spearmanr(a, b)
    return Correlation(rho=float(result.statistic), p=float(result.pvalue), n=len(a))


def partial_spearman(x: Sequence[float], y: Sequence[float], controls: Sequence[Sequence[float]]) -> Correlation:
    a = stats.rankdata(_array(x))
    b = stats.rankdata(_array(y))
    design = np.column_stack([np.ones(len(a)), *[stats.rankdata(_array(c)) for c in controls]])
    a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    rho = float(stats.pearsonr(a, b).statistic)
    dof = len(a) - len(controls) - 2
    t = abs(rho) * np.sqrt(dof / (1 - rho * rho))
    return Correlation(rho=rho, p=float(2 * stats.t.sf(t, dof)), n=len(a))


def ols_residuals(x: Sequence[float], y: Sequence[float]) -> Array:
    a, b = _array(x), _array(y)
    slope, intercept = np.polyfit(a, b, 1)
    residuals: Array = b - (slope * a + intercept)
    return residuals


def leave_one_group_out(x: Sequence[float], y: Sequence[float], groups: Sequence[str]) -> dict[str, Correlation]:
    a, b = _array(x), _array(y)
    labels = np.asarray(groups)
    return {group: spearman(list(a[labels != group]), list(b[labels != group])) for group in sorted(set(groups))}


def bootstrap_rho(x: Array, y: Array, indices: NDArray[np.int64]) -> Array:
    a = stats.rankdata(x[indices], axis=1)
    b = stats.rankdata(y[indices], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    result: Array = np.divide((a * b).sum(1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)
    return result


def paired_difference_interval(
    first: Sequence[float],
    second: Sequence[float],
    outcome: Sequence[float],
    *,
    seed: int,
    resamples: int,
    level: float,
) -> Interval:
    a, b, y = _array(first), _array(second), _array(outcome)
    indices = np.random.default_rng(seed).integers(0, len(y), size=(resamples, len(y)))
    differences = bootstrap_rho(a, y, indices) - bootstrap_rho(b, y, indices)
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(differences, [tail, 100 - tail])
    estimate = float(stats.spearmanr(a, y).statistic - stats.spearmanr(b, y).statistic)
    return Interval(estimate=estimate, low=float(low), high=float(high))


def median_log_rank(ranks: Sequence[int]) -> float:
    return float(np.median(np.log10(_array(ranks) + 1)))


def summarize_range(values: Mapping[str, Correlation]) -> tuple[float, float, float]:
    defined = [c.rho for c in values.values() if c.rho is not None]
    return min(defined), max(defined), float(np.median(defined))


def _rho(a: Array, b: Array) -> float:
    return float(stats.spearmanr(a, b).statistic)


def held_out_selection_rate(
    candidates: Mapping[float, Sequence[float]],
    baseline: Sequence[float],
    outcome: Sequence[float],
    *,
    train_size: int,
    splits: int,
    seed: int,
    min_distinct: int,
) -> float | None:
    y = _array(outcome)
    base = _array(baseline)
    scores = {beta: _array(values) for beta, values in candidates.items()} | {float("inf"): base}
    generator = np.random.default_rng(seed)
    wins = kept = 0
    for _ in range(splits):
        order = generator.permutation(len(y))
        train, test = order[:train_size], order[train_size:]
        if len(set(y[test])) < min_distinct or len(set(y[train])) < min_distinct:
            continue
        best = max(scores, key=lambda beta: _rho(scores[beta][train], y[train]))
        wins += _rho(scores[best][test], y[test]) > _rho(base[test], y[test])
        kept += 1
    return wins / kept if kept else None
