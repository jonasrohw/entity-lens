"""Continuous positive-beta top-1 regions for a bias-free RMSNorm readout."""

import numpy as np
import torch


def top1_intervals(intercept, slope, targets):
    """Intersect target-versus-token affine inequalities for every subject."""
    targets = torch.as_tensor(targets, device=slope.device)
    intercept = intercept.expand_as(slope)
    a = intercept.gather(1, targets[:, None]) - intercept
    b = slope.gather(1, targets[:, None]) - slope
    threshold = -a / torch.where(b == 0, torch.ones_like(b), b)
    lower = torch.where(b > 0, threshold, -torch.inf).amax(-1).clamp_min(0)
    upper = torch.where(b < 0, threshold, torch.inf).amin(-1)
    valid = ~((b == 0) & (a < 0)).any(-1) & (upper >= lower) & (upper > 0)
    return dict(
        lower=lower.cpu().numpy(), upper=upper.cpu().numpy(), valid=valid.cpu().numpy()
    )


def candidate_betas(bounds, selection, anchors=()):
    selection = np.asarray(selection, dtype=int)
    valid = bounds["valid"][selection]
    endpoints = np.concatenate(
        [bounds["lower"][selection][valid], bounds["upper"][selection][valid]]
    )
    endpoints = np.unique(
        np.r_[0.0, endpoints[np.isfinite(endpoints) & (endpoints >= 0)]]
    )
    candidates = list(anchors)
    candidates += ((endpoints[:-1] + endpoints[1:]) / 2).tolist()
    candidates.append(float(endpoints[-1] + max(1.0, abs(endpoints[-1]))))
    return np.asarray(sorted(set(b for b in candidates if np.isfinite(b) and b > 0)))


def rank_candidates(bounds, selection, anchors=()):
    """Candidate creation and ranking depend only on calibration subjects."""
    values = candidate_betas(bounds, selection, anchors)
    selection = np.asarray(selection, dtype=int)
    matches = (
        (values[:, None] >= bounds["lower"][selection])
        & (values[:, None] <= bounds["upper"][selection])
        & bounds["valid"][selection]
    )
    scores = matches.sum(-1)
    order = np.lexsort((values, -scores))
    return values[order], scores[order]
