"""Selection-aware summaries of the expanded grids (all maxima are exploratory)."""

import numpy as np
from scipy.stats import rankdata, spearmanr

COARSE_LAYERS = [5, 11, 14, 17, 20, 23]
COARSE_BETAS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
METRICS = ["median_logrank", "mean_logrank", "top1", "top10", "mrr"]


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=None, p=None, n=len(x))
    result = spearmanr(x, y)
    return dict(rho=float(result.statistic), p=float(result.pvalue), n=len(x))


def all_correlations(x, y):
    xr = rankdata(x, axis=1)
    yr = rankdata(y)
    xr -= xr.mean(axis=1, keepdims=True)
    yr -= yr.mean()
    denominator = np.linalg.norm(xr, axis=1) * np.linalg.norm(yr)
    return np.divide(
        xr @ yr, denominator, out=np.full(len(x), np.nan), where=denominator > 0
    )


def maximum_permutation_test(x, y, permutations=2000, seed=20260919):
    """Search-adjusted diagnostic, conditional on the chosen faithfulness target."""
    xr = rankdata(x, axis=1)
    yr = rankdata(y)
    xr -= xr.mean(axis=1, keepdims=True)
    yr -= yr.mean()
    norms = np.linalg.norm(xr, axis=1)
    xr = xr[norms > 0] / norms[norms > 0, None]
    yr /= np.linalg.norm(yr)
    observed = float((xr @ yr).max())
    rng = np.random.default_rng(seed)
    maxima = []
    for start in range(0, permutations, 200):
        shuffled = np.stack(
            [rng.permutation(yr) for _ in range(min(200, permutations - start))]
        )
        maxima.extend((xr @ shuffled.T).max(axis=0).tolist())
    maxima = np.asarray(maxima)
    return dict(
        observed_max_rho=observed,
        permutations=permutations,
        seed=seed,
        adjusted_p=float((1 + np.sum(maxima >= observed - 1e-12)) / (permutations + 1)),
        null_95th_percentile=float(np.quantile(maxima, 0.95)),
        scope="Maximum over this lens grid, conditional on this LRE target; not adjusted for prior model selection or all research decisions.",
    )


def selected_faith(row, refined, scenario, refs=8):
    if scenario in ("coarse", "dense"):
        cells = []
        for layer in COARSE_LAYERS if scenario == "coarse" else row["layers"]:
            if (
                scenario == "dense"
                and not refined["grid"][str(layer)][str(refs)]["input_dependent"]
            ):
                continue
            betas = (
                COARSE_BETAS
                if scenario == "coarse"
                else [b for b in row["betas"] if b > 0]
            )
            for beta in betas:
                cell = row["faith"]["grid"][str(layer)][str(refs)][str(beta)]
                cells.append(
                    dict(
                        layer=layer,
                        beta=beta,
                        score=float(np.mean(cell["hits"])),
                        cell=cell,
                    )
                )
        return max(cells, key=lambda c: c["score"])
    key = "calibration" if scenario == "calibrated" else "all_subjects"
    cells = []
    for layer in row["layers"]:
        cell = refined["grid"][str(layer)][str(refs)]
        fit = cell[key]
        if fit is not None and cell["input_dependent"]:
            cells.append(
                dict(
                    layer=layer, beta=fit["beta"], score=fit["selection_faith"], fit=fit
                )
            )
    if not cells:
        return None
    best = max(cells, key=lambda c: c["score"])
    if scenario == "calibrated":
        best["selection_score"] = best["score"]
        best["score"] = best["fit"]["evaluation_faith"]
    return best


def candidate_configs(row, metric="median_logrank"):
    for prompt, layers in row["lens"]["prompts"].items():
        for layer, references in layers.items():
            for refs, cells in references.items():
                yield dict(
                    prompt=prompt,
                    layer=int(layer),
                    references=int(refs),
                    readout="diff",
                    beta=None,
                    metric=metric,
                )
                for beta in cells["affine"]:
                    if float(beta) > 0:
                        yield dict(
                            prompt=prompt,
                            layer=int(layer),
                            references=int(refs),
                            readout="affine",
                            beta=float(beta),
                            metric=metric,
                        )


def lens_value(row, config):
    cell = row["lens"]["prompts"][config["prompt"]][str(config["layer"])][
        str(config["references"])
    ]
    if cell["diff"]["n_constant_outputs"]:
        return np.nan
    cell = (
        cell["diff"]
        if config["readout"] == "diff"
        else cell["affine"][str(config["beta"])]
    )
    if cell["n_constant_outputs"]:
        return np.nan
    value = cell[config["metric"]]
    return -value if config["metric"].endswith("logrank") else value


def select_on_relations(x, y, selection):
    scores = all_correlations(x[:, selection], y[selection])
    return int(np.nanargmax(scores)) if np.isfinite(scores).any() else None


def search(
    rows, refined, folds, scenario, refs=8, metric="median_logrank", permutations=0
):
    selections = [
        selected_faith(row, refined[row["relation"]], scenario, refs) for row in rows
    ]
    kept = [
        (row, selection)
        for row, selection in zip(rows, selections)
        if selection is not None
    ]
    rows, selections = map(list, zip(*kept))
    relations = [row["relation"] for row in rows]
    y = np.asarray([selection["score"] for selection in selections])
    configs = list(candidate_configs(rows[0], metric))
    x = np.asarray([[lens_value(row, config) for row in rows] for config in configs])
    valid = np.isfinite(x).all(axis=1)
    x = x[valid]
    configs = [c for c, include in zip(configs, valid) if include]
    scores = all_correlations(x, y)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    winners = [dict(config=configs[i], **correlation(x[i], y)) for i in order[:10]]
    selection = [
        i for i, relation in enumerate(relations) if relation in folds["selection"]
    ]
    evaluation = [
        i for i, relation in enumerate(relations) if relation in folds["evaluation"]
    ]
    chosen = select_on_relations(x, y, selection)
    heldout = (
        dict(
            config=configs[chosen],
            selection=correlation(x[chosen, selection], y[selection]),
            evaluation=correlation(x[chosen, evaluation], y[evaluation]),
            selection_relations=[relations[i] for i in selection],
            evaluation_relations=[relations[i] for i in evaluation],
        )
        if chosen is not None
        else None
    )
    baseline = dict(
        prompt="known_for",
        layer=17,
        references=8,
        readout="diff",
        beta=None,
        metric=metric,
    )
    baseline_x = [lens_value(row, baseline) for row in rows]
    return dict(
        scenario=scenario,
        lre_references=refs,
        metric=metric,
        n_candidates=len(configs),
        relations=relations,
        baseline_lens=correlation(baseline_x, y),
        exploratory_top10=winners,
        heldout_relations=heldout,
        selected_faith={
            name: {k: v for k, v in cell.items() if k not in ("cell", "fit")}
            for name, cell in zip(relations, selections)
        },
        faithfulness_mean=float(y.mean()),
        maximum_permutation_test=maximum_permutation_test(x, y, permutations)
        if permutations and np.ptp(y) > 0
        else None,
    )


def matched_control(rows, refined):
    selected = [selected_faith(row, refined[row["relation"]], "dense") for row in rows]
    return dict(
        **correlation(
            [-cell["cell"]["median_logrank"] for cell in selected],
            [cell["score"] for cell in selected],
        ),
        warning="Same predictions and targets define both rank and top-1 faithfulness; this is a positive control, not independent lens evidence.",
    )
