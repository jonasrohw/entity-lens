import numpy as np
import torch

from jlens.beta_intervals import rank_candidates, top1_intervals


def bounds_for_intervals(lower, upper):
    a = torch.tensor(
        [[0.0, lo, -hi] for lo, hi in zip(lower, upper)], dtype=torch.float64
    )
    b = torch.tensor([[0.0, -1.0, 1.0]] * len(lower), dtype=torch.float64)
    return top1_intervals(a, b, [0] * len(lower))


def test_continuous_optimum_between_grid_points():
    bounds = bounds_for_intervals([1.1, 1.15, 2.0], [1.2, 1.3, 3.0])
    values, scores = rank_candidates(
        bounds, [0, 1, 2], [0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    )
    assert scores[0] == 2
    assert 1.15 < values[0] < 1.2


def test_calibration_selection_cannot_see_heldout_boundaries():
    original = bounds_for_intervals([1.1, 1.15, 2.0], [1.2, 1.3, 3.0])
    changed = bounds_for_intervals([1.1, 1.15, 100.0], [1.2, 1.3, 200.0])
    for a, b in zip(
        rank_candidates(original, [0, 1]), rank_candidates(changed, [0, 1])
    ):
        np.testing.assert_array_equal(a, b)


def test_impossible_and_unbounded_intervals():
    a = torch.tensor(
        [[0.0, 2.0, -1.0], [0.0, -1.0, -1.0], [0.0, 1.0, 1.0]], dtype=torch.float64
    )
    b = torch.tensor(
        [[0.0, -1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    bounds = top1_intervals(a, b, [0, 0, 0])
    assert bounds["valid"].tolist() == [False, True, False]
    assert bounds["lower"][1] == 0 and np.isposinf(bounds["upper"][1])


def test_intervals_match_direct_argmax_away_from_boundaries():
    generator = torch.Generator().manual_seed(42)
    a = torch.randn(30, 20, generator=generator, dtype=torch.float64)
    b = torch.randn(30, 20, generator=generator, dtype=torch.float64)
    targets = torch.randint(0, 20, (30,), generator=generator)
    bounds = top1_intervals(a, b, targets)
    for beta in np.geomspace(0.001, 1000, 400):
        predicted = (
            bounds["valid"] & (bounds["lower"] <= beta) & (beta <= bounds["upper"])
        )
        actual = (a + beta * b).argmax(-1).eq(targets).numpy()
        np.testing.assert_array_equal(predicted, actual)
