"""Build the prompt-end lens report from all 52 completed GPU measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2] / "code"
STEM = "last-position-lre-correlation"
LABELS = {"qwen": "Qwen3 8B Base", "olmo": "OLMo 3 7B"}


def fit(x, y):
    x, y = np.asarray(x), np.asarray(y)
    a, b = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    residual = y - a - b * x
    np.testing.assert_allclose([residual.sum(), residual @ x], [0, 0], atol=1e-12)
    rho, p = stats.spearmanr(x, y)
    r, rp = stats.pearsonr(x, y)
    return {
        "n": len(x),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pearson_r": float(r),
        "pearson_p": float(rp),
        "intercept": float(a),
        "slope": float(b),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=BASE / "generated/last_position_correlation"
    )
    parser.add_argument("--output-dir", type=Path, default=BASE / "docs")
    args = parser.parse_args()
    inputs = BASE / "docs" / (STEM + "-inputs.json")
    frozen = json.loads(inputs.read_text())
    evidence = {
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "excluded_relations": frozen["excluded_relations"],
        "models": {},
    }
    rows = []
    for key, spec in frozen["models"].items():
        measured = [
            json.loads((args.input_dir / key / (r["relation"] + ".json")).read_text())
            for r in spec["relations"]
        ]
        assert len(measured) == 26
        for d, original in zip(measured, spec["relations"], strict=True):
            assert d["relation"] == original["relation"]
            assert d["faithfulness"] == original["faithfulness"]
            assert d["provenance"]["input_sha256"] == evidence["inputs_sha256"]
            assert d["provenance"]["layer"] == spec["layer"]
            assert (
                d["provenance"]["script_sha256"]
                == hashlib.sha256(
                    Path(__file__)
                    .with_name("last_position_correlation.py")
                    .read_bytes()
                ).hexdigest()
            )
            assert (
                d["provenance"]["intervention_sha256"]
                == hashlib.sha256((BASE / "jlens/model.py").read_bytes()).hexdigest()
            )
            for mode in ("subject", "last"):
                rs = d["modes"][mode]["rows"]
                ranks = np.asarray([r["rank"] for r in rs], dtype=float)
                assert d["modes"][mode]["agg"]["median_lograng"] == float(
                    np.median(np.log10(ranks + 1))
                )
                assert [r["subject"] for r in rs] == [
                    r["subject"] for r in original["targets"]
                ]
                for r, t in zip(rs, original["targets"], strict=True):
                    assert r["model_top1"] == t["model_top1"]
                    assert r["n_references"] == sum(
                        s != r["subject"] for s in original["references"]
                    )
                    if mode == "last":
                        assert r["position"] == len(r["input_ids"]) - 1
            assert [r["model_top1_id"] for r in d["modes"]["subject"]["rows"]] == [
                r["model_top1_id"] for r in d["modes"]["last"]["rows"]
            ]
        y = [d["faithfulness"] for d in measured]
        fits = {"original": fit([d["original_x"] for d in measured], y)}
        for mode in ("subject", "last"):
            fits[mode] = fit(
                [-d["modes"][mode]["agg"]["median_lograng"] for d in measured], y
            )
        rank_errors = [
            r["rank"] - old
            for d in measured
            for r, old in zip(
                d["modes"]["subject"]["rows"], d["original_ranks"], strict=True
            )
        ]
        x_error = max(
            abs(d["original_x"] + d["modes"]["subject"]["agg"]["median_lograng"])
            for d in measured
        )
        # Permit deep-tail rank rounding on a different GPU, but fail on protocol drift.
        assert x_error < 0.001, (key, x_error)
        for d in measured:
            new_x = -d["modes"]["last"]["agg"]["median_lograng"]
            old_x = d["original_x"]
            f = fits["last"]
            pred = f["intercept"] + f["slope"] * new_x
            rows.append(
                {
                    "model": key,
                    "relation": d["relation"],
                    "n": len(d["original_ranks"]),
                    "original_x": old_x,
                    "rerun_subject_x": -d["modes"]["subject"]["agg"]["median_lograng"],
                    "last_x": new_x,
                    "faithfulness": d["faithfulness"],
                    "last_y_hat": pred,
                    "last_residual": d["faithfulness"] - pred,
                    "original_median_rank": d["modes"]["subject"]["agg"]["median_rank"],
                    "last_median_rank": d["modes"]["last"]["agg"]["median_rank"],
                    "original_top10": d["modes"]["subject"]["agg"]["agree_at10"],
                    "last_top10": d["modes"]["last"]["agg"]["agree_at10"],
                }
            )
        evidence["models"][key] = {
            "fits": fits,
            "baseline_validation": {
                "n_subjects": len(rank_errors),
                "exact_ranks": sum(e == 0 for e in rank_errors),
                "max_abs_rank_error": max(abs(e) for e in rank_errors),
                "max_abs_x_error": x_error,
            },
            "measurements": measured,
        }
        evidence["models"][key]["absolute_performance"] = {}
        for mode in ("subject", "last"):
            all_rows = [r for d in measured for r in d["modes"][mode]["rows"]]
            evidence["models"][key]["absolute_performance"][mode] = {
                "n": len(all_rows),
                "top1_count": sum(r["rank"] == 0 for r in all_rows),
                "top10_count": sum(r["rank"] < 10 for r in all_rows),
                "semit_top1_count": sum(
                    r["top_tokens"][0] == "-Semit" for r in all_rows
                ),
                "state_norm_min": min(r["state_norm"] for r in all_rows),
                "state_norm_max": max(r["state_norm"] for r in all_rows),
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / (STEM + ".json")).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / (STEM + ".csv")).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    for ax, key in zip(axes, LABELS, strict=True):
        rs = [r for r in rows if r["model"] == key]
        for r in rs:
            ax.plot(
                [r["original_x"], r["last_x"]],
                [r["faithfulness"]] * 2,
                color="0.83",
                lw=0.7,
                zorder=1,
            )
        ax.scatter(
            [r["original_x"] for r in rs],
            [r["faithfulness"] for r in rs],
            color="0.55",
            marker="x",
            label="Last subject token",
            zorder=2,
        )
        ax.scatter(
            [r["last_x"] for r in rs],
            [r["faithfulness"] for r in rs],
            color="#2166ac",
            s=27,
            label="Final prompt token: for",
            zorder=3,
        )
        xs = np.linspace(-5.3, 0, 100)
        for mode, color in [("original", "0.55"), ("last", "#2166ac")]:
            f = evidence["models"][key]["fits"][mode]
            ax.plot(xs, f["intercept"] + f["slope"] * xs, color=color, lw=1, ls="--")
        for r in rs:
            if r["relation"] in {"country_language", "company_ceo", "name_gender"}:
                ax.annotate(
                    r["relation"],
                    (r["last_x"], r["faithfulness"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
        old, new = (
            evidence["models"][key]["fits"][m]["spearman_rho"]
            for m in ("original", "last")
        )
        ax.set_title(f"{LABELS[key]}: Spearman ρ {old:.3f} → {new:.3f}")
        ax.set_xlabel("−median(log10(target rank + 1)); higher is better")
        ax.set_ylim(-0.04, 1.09)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("LRE faithfulness (unchanged)")
    axes[0].legend(fontsize=8, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(args.output_dir / (STEM + "." + ext), dpi=190)
    plt.close(fig)

    text = [
        "# Final-prompt-position lens versus LRE faithfulness\n",
        "Measured 20 September 2026. This is a controlled **26-relation ablation** in Qwen3 8B Base and OLMo 3 7B; only `presidents_birth_year` and `presidents_election_year` are excluded. `name_gender` remains included.\n",
        "## Correlations\n",
        "Each point is one relation: **x = −median(log10(rank_diff + 1))**, **y = the retained LRE top-1 faithfulness**. Ranks are zero-based and targets are the model's first answer token under the original relation-specific ICL prompt. Higher x and y are better. P-values below are two-sided and test each correlation against zero; they do not test the change between correlations.\n",
        "| Model | Lens input/intervention position | Spearman ρ | p | Pearson r | p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in LABELS.items():
        for mode, name in [
            ("original", "Last subject token (retained)"),
            ("subject", "Last subject token (rerun)"),
            ("last", "Last prompt token: `for`"),
        ]:
            f = evidence["models"][key]["fits"][mode]
            text.append(
                f"| {label} | {name} | {f['spearman_rho']:.6f} | {f['spearman_p']:.6g} | {f['pearson_r']:.6f} | {f['pearson_p']:.6g} |"
            )
    text += [
        f"\n![Position ablation]({STEM}.png)\n",
        f"[PDF]({STEM}.pdf), [all 52 relation rows as CSV]({STEM}.csv).\n",
        "## Interpretation\n",
        "**Moving to `for` does not recover a positive correlation for Qwen.** OLMo retains a positive correlation, with a smaller point estimate than before. These are observed differences, not a formal test that one correlation exceeds another.\n",
        "Absolute decoding also deteriorates. The table below pools the 491 scored rows per model, including the retained duplicate rows; these counts are descriptive and are not the relation-level correlation statistic.\n",
        "| Model | Position | Targets in top 1 | Targets in top 10 | `-Semit` top 1 | Input-state norm range |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in LABELS.items():
        for mode, name in [("subject", "Last subject token"), ("last", "`for`")]:
            a = evidence["models"][key]["absolute_performance"][mode]
            text.append(
                f"| {label} | {name} | {a['top1_count']}/{a['n']} | {a['top10_count']}/{a['n']} | {a['semit_top1_count']}/{a['n']} | {a['state_norm_min']:.2f}–{a['state_norm_max']:.2f} |"
            )
    text += [
        "\nQwen's extreme norm disparity and repeated `-Semit` top predictions disappear, but no target reaches the top 10. For `country_language`, its median target rank worsens from 75,290.5 to 95,010.5; OLMo's worsens from 9 to 1,960.5. Removing the observed position-0 artifact is therefore insufficient to obtain a useful differential readout. OLMo's remaining correlation describes relative differences among relations despite poor absolute ranks.\n",
        "## What changes\n",
        "Both variants use exactly `{subject} is known for`, without an added prefix. The original variant caches h at the last subject token and differentiates the final prompt state with respect to that position. The new variant caches h at the final token `for` and differentiates with respect to that same final position. The output is still the raw final-block state at `for`.\n",
        "In both cases, `v_e = mean_{i != e} J_i (h_e − h_i)`, decoded through the model's native final normalization and output head. Each Jacobian is evaluated in its reference's own prompt. We retain state 19 for Qwen and state 17 for OLMo, the first eight references among up to 20 main-protocol test subjects, and self-exclusion. We recompute both variants, rather than simply changing the readout position.\n",
        "The relation-specific target prompts, target tokens and LRE y-values are fixed. We also retain duplicate rows from the original protocol: `Sofia` occurs twice among the `name_gender` references, so its two test rows each exclude both matching references and average six contributions. There is no LRE refit, layer search, beta tuning, added prefix, or normalization of the input states. This changes the representation being tested: `for` already contextualizes the subject and the suffix. It is an alternative lens definition, not proof that the original estimator is repaired. Absolute final-token indices can still vary with subject length.\n",
        "## Why a large norm can be a problem\n",
        "A large norm alone is not a bug: uniform rescaling can be harmless, and some large differences can still be well approximated. The observed Qwen issue is that a large **position-dependent component** dominates `h_e − h_i`. Subtracting states from different positional regimes creates a large, largely nonsemantic direction. The local derivative then extrapolates far from its reference state, and a few reference contributions dominate the average.\n",
        "The earlier [implementation review](qwen3-jlens-implementation-review.md) measured this directly for Germany: with Peru as reference, the state displacement is 80.77 times the reference norm, the linear prediction has norm 7,170, and the actual full-step output change has norm 389 (relative vector error 18.26). With Australia as reference, the displacement ratio is 0.042 and the relative error is 0.046. Small-step derivative checks pass: the derivative is correct, but its finite-step extrapolation fails for those cross-position changes.\n",
        "Final RMS normalization cannot undo a wrong direction: it rescales the already mixed vector. With Qwen's unbiased output head, the scalar RMS divisor cannot change token ordering. Normalizing subject states beforehand would instead change the intervention and its coordinate system; it is not an algebraically equivalent fix.\n",
        "## All relation points and refitted residuals\n",
        "Each model gets its own OLS fit on the 26 new x-values, with equal weight per relation. Residual = y − fitted y; a positive value means greater LRE faithfulness than the line predicts.\n",
    ]
    for key, label in LABELS.items():
        f = evidence["models"][key]["fits"]["last"]
        text += [
            f"### {label}\n",
            f"`ŷ = {f['intercept']:.12f} + {f['slope']:.12f} × x`\n",
            "| Relation | n | Original x | Final-position x | LRE y | Fitted y | Residual |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in sorted(
            (r for r in rows if r["model"] == key),
            key=lambda r: -abs(r["last_residual"]),
        ):
            text.append(
                f"| {r['relation']} | {r['n']} | {r['original_x']:.6f} | {r['last_x']:.6f} | {r['faithfulness']:.6f} | {r['last_y_hat']:.6f} | {r['last_residual']:+.6f} |"
            )
        text.append("")
    text += ["## Validation and reproduction\n"]
    for key, label in LABELS.items():
        e = evidence["models"][key]
        v = e["baseline_validation"]
        checks = [c for d in e["measurements"] for c in d["derivative_checks"]]
        fd = max(
            min(q["relative_error"] for q in c["finite_differences"]) for c in checks
        )
        text.append(
            f"- {label}: {v['exact_ranks']}/{v['n_subjects']} original ranks reproduced exactly; maximum rank difference {v['max_abs_rank_error']}, maximum relation-x difference {v['max_abs_x_error']:.9g}. {len(checks)} final-position derivative checks; worst best-step central-finite-difference relative error {fd:.6g}."
        )
    text += [
        "- All 982 target predictions are re-evaluated and checked against the frozen decoded target strings. Qwen's retained numeric token IDs also match exactly; OLMo's older snapshot stored strings only, so its re-evaluated IDs are recorded in the new evidence. Both positions use identical target IDs, subjects and reference counts.",
        "- Every reference in both variants passes exact no-op intervention checks; captured raw final states reproduce native logits. All new cached/intervened positions equal the final prompt index and decode to `for`.",
        "- Computation uses pinned cached checkpoints, float32 and eager attention on A100 80 GB GPUs. Input, source and dataset hashes, individual ranks/top-five tokens, state norms, and per-reference contribution norms are retained in the JSON evidence.\n",
        f"[Frozen input snapshot]({STEM}-inputs.json), [complete numerical results]({STEM}.json), [GPU runner](../../paper/repro/last_position_correlation.py), [CPU report generator](../../paper/repro/report_last_position_correlation.py).\n",
        "```sh\ncd code\nCUDA_VISIBLE_DEVICES=0 uv run --extra gpu python ../paper/repro/last_position_correlation.py --model qwen\nCUDA_VISIBLE_DEVICES=1 uv run --extra gpu python ../paper/repro/last_position_correlation.py --model olmo\nuv run python ../paper/repro/report_last_position_correlation.py\n```\n",
        "For multiple workers, add `--shards 2 --shard 0` / `--shards 2 --shard 1` and use a separate GPU for each worker. No historical measurement files are overwritten. These are exploratory comparisons over the retained relations, not validation on a new relation set.\n",
    ]
    (args.output_dir / (STEM + ".md")).write_text("\n".join(text))
    print(
        json.dumps(
            {key: value["fits"] for key, value in evidence["models"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
