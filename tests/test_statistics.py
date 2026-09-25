from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from entity_lens.statistics import (
    Correlation,
    bootstrap_rho,
    held_out_selection_rate,
    leave_one_group_out,
    median_log_rank,
    ols_residuals,
    paired_difference_interval,
    partial_spearman,
    spearman,
    summarize_range,
)

X = [0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.55]
Y = [0.0, 0.3, 0.5, 0.6, 1.0, 0.1, 0.9, 0.4]
Z = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]


def test_spearman_matches_scipy() -> None:
    result = spearman(X, Y)
    expected = stats.spearmanr(X, Y)
    assert result.rho == pytest.approx(expected.statistic)
    assert result.p == pytest.approx(expected.pvalue)
    assert result.n == 8


def test_spearman_constant_is_undefined() -> None:
    assert spearman([1, 1, 1], [1, 2, 3]).rho is None


def test_spearman_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        spearman([1, 2], [1, 2, 3])


def test_partial_spearman_without_association_to_control() -> None:
    result = partial_spearman(X, Y, [Z])
    rx, ry, rz = stats.rankdata(X), stats.rankdata(Y), stats.rankdata(Z)
    design = np.column_stack([np.ones(8), rz])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    assert result.rho == pytest.approx(stats.pearsonr(ex, ey).statistic)
    assert result.p is not None
    assert 0 < result.p < 1


def test_residuals_sum_to_zero() -> None:
    residuals = ols_residuals(X, Y)
    assert residuals.sum() == pytest.approx(0.0, abs=1e-12)


def test_leave_one_group_out() -> None:
    groups = ["a", "a", "b", "b", "c", "c", "c", "d"]
    result = leave_one_group_out(X, Y, groups)
    assert set(result) == {"a", "b", "c", "d"}
    assert result["c"].n == 5
    low, high, median = summarize_range(result)
    assert low <= median <= high


def test_bootstrap_rho_matches_spearman_for_identity() -> None:
    indices = np.arange(8)[None, :].repeat(2, axis=0)
    values = bootstrap_rho(np.asarray(X), np.asarray(Y), indices)
    assert values == pytest.approx([stats.spearmanr(X, Y).statistic] * 2)


def test_paired_interval_is_deterministic() -> None:
    first = paired_difference_interval(X, Z, Y, seed=1, resamples=200, level=0.95)
    second = paired_difference_interval(X, Z, Y, seed=1, resamples=200, level=0.95)
    assert first == second
    assert first.low <= first.high


def test_median_log_rank() -> None:
    assert median_log_rank([0, 9, 99]) == pytest.approx(1.0)


def test_held_out_selection_rate_bounds() -> None:
    candidates = {1.0: X, 2.0: Z}
    rate = held_out_selection_rate(candidates, Y, Y, train_size=4, splits=20, seed=0, min_distinct=3)
    assert rate == 0.0
    perfect = held_out_selection_rate({1.0: Y}, Z, Y, train_size=4, splits=20, seed=0, min_distinct=3)
    assert perfect == 1.0


def test_correlation_model_is_frozen() -> None:
    correlation = Correlation(rho=0.5, p=0.1, n=3)
    with pytest.raises(ValueError, match="frozen"):
        correlation.rho = 0.2
