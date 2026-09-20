"""Report the 26-relation Qwen raw-versus-differential J-lens ablation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from report_last_position_correlation import fit

BASE = Path(__file__).resolve().parents[2] / "code"
STEM = "qwen3-8b-base-raw-jlens"
POSITIONS = {"subject": "Last subject token", "last": "Final prompt token `for`"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=BASE / "generated/qwen_raw_jlens"
    )
    args = parser.parse_args()
    docs = BASE / "docs"
    inputs = docs / "last-position-lre-correlation-inputs.json"
    specs = json.loads(inputs.read_text())["models"]["qwen"]
    baseline = json.loads((docs / "last-position-lre-correlation.json").read_text())[
        "models"
    ]["qwen"]
    previous = {r["relation"]: r for r in baseline["measurements"]}
    measurements = []
    rank_errors = []
    score_errors = []
    for spec in specs["relations"]:
        d = json.loads((args.input_dir / (spec["relation"] + ".json")).read_text())
        assert d["faithfulness"] == spec["faithfulness"]
        assert d["provenance"]["layer"] == 19
        for field, path in [
            ("input_sha256", inputs),
            ("script_sha256", Path(__file__).with_name("qwen_raw_jlens.py")),
            ("helper_sha256", Path(__file__).with_name("last_position_correlation.py")),
            ("intervention_sha256", BASE / "jlens/model.py"),
        ]:
            assert (
                d["provenance"][field] == hashlib.sha256(path.read_bytes()).hexdigest()
            )
        for position in POSITIONS:
            controls = d["positions"][position]["modes"]["diff"]
            old = previous[d["relation"]]["modes"][position]
            score_errors.append(
                abs(controls["agg"]["median_lograng"] - old["agg"]["median_lograng"])
            )
            rank_errors.extend(
                r["rank"] - p["rank"]
                for r, p in zip(controls["rows"], old["rows"], strict=True)
            )
            for kind in ("diff", "raw"):
                mode = d["positions"][position]["modes"][kind]
                assert [r["subject"] for r in mode["rows"]] == [
                    r["subject"] for r in spec["targets"]
                ]
                for r, t in zip(mode["rows"], spec["targets"], strict=True):
                    assert r["target_id"] == t["model_top1_id"]
                    assert r["n_references"] == sum(
                        s != r["subject"] for s in spec["references"]
                    )
                assert mode["agg"]["median_lograng"] == float(
                    np.median(np.log10([r["rank"] + 1 for r in mode["rows"]]))
                )
        measurements.append(d)
    assert len(measurements) == 26
    assert max(score_errors) < 0.001
    summary = {}
    rows = []
    y = [d["faithfulness"] for d in measurements]
    for position in POSITIONS:
        summary[position] = {}
        for kind in ("diff", "raw"):
            x = [
                -d["positions"][position]["modes"][kind]["agg"]["median_lograng"]
                for d in measurements
            ]
            all_rows = [
                r
                for d in measurements
                for r in d["positions"][position]["modes"][kind]["rows"]
            ]
            f = fit(x, y)
            summary[position][kind] = {
                "fit": f,
                "n_rows": len(all_rows),
                "top1_count": sum(r["rank"] == 0 for r in all_rows),
                "top10_count": sum(r["rank"] < 10 for r in all_rows),
                "semit_top1_count": sum(
                    r["top_tokens"][0] == "-Semit" for r in all_rows
                ),
            }
        for d in measurements:
            old = d["positions"][position]["modes"]["diff"]["agg"]
            new = d["positions"][position]["modes"]["raw"]["agg"]
            raw_x = -new["median_lograng"]
            f = summary[position]["raw"]["fit"]
            predicted = f["intercept"] + f["slope"] * raw_x
            rows.append(
                {
                    "relation": d["relation"],
                    "position": position,
                    "n": len(d["positions"][position]["modes"]["raw"]["rows"]),
                    "diff_x": -old["median_lograng"],
                    "raw_x": raw_x,
                    "faithfulness": d["faithfulness"],
                    "raw_y_hat": predicted,
                    "raw_residual": d["faithfulness"] - predicted,
                    "diff_median_rank": old["median_rank"],
                    "raw_median_rank": new["median_rank"],
                    "diff_top10": old["agree_at10"],
                    "raw_top10": new["agree_at10"],
                }
            )
    validation = {
        "baseline_rank_comparisons": len(rank_errors),
        "baseline_exact_ranks": sum(e == 0 for e in rank_errors),
        "baseline_max_rank_error": max(abs(e) for e in rank_errors),
        "baseline_max_x_error": max(score_errors),
        "max_linearity_relative_error": max(
            e
            for d in measurements
            for p in d["positions"].values()
            for c in p["contributions"]
            for e in c["linearity_relative_errors"]
        ),
        "finite_difference_checks": sum(len(d["checks"]) for d in measurements),
        "worst_best_step_fd_relative_error": max(
            min(e["relative_error"] for e in c["finite_differences"])
            for d in measurements
            for c in d["checks"]
        ),
    }
    evidence = {
        "summary": summary,
        "validation": validation,
        "measurements": measurements,
    }
    (docs / (STEM + ".json")).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )
    with (docs / (STEM + ".csv")).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (position, label) in zip(axes, POSITIONS.items(), strict=True):
        rs = [r for r in rows if r["position"] == position]
        for r in rs:
            ax.plot(
                [r["diff_x"], r["raw_x"]], [r["faithfulness"]] * 2, color="0.85", lw=0.8
            )
        for kind, color, marker in [("diff", "0.55", "x"), ("raw", "#2166ac", "o")]:
            f = summary[position][kind]["fit"]
            ax.scatter(
                [r[kind + "_x"] for r in rs],
                [r["faithfulness"] for r in rs],
                color=color,
                marker=marker,
                s=28,
                label=f"{kind}: ρ = {f['spearman_rho']:.3f}",
            )
            xs = np.linspace(-5.3, 0, 100)
            ax.plot(xs, f["intercept"] + f["slope"] * xs, color=color, ls="--", lw=1)
        for r in rs:
            if r["relation"] in {"country_language", "company_ceo", "name_gender"}:
                ax.annotate(
                    r["relation"],
                    (r["raw_x"], r["faithfulness"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.set_title(label.replace("`", ""))
        ax.set_xlabel("−median(log10(target rank + 1)); higher is better")
        ax.set_ylim(-0.04, 1.1)
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("LRE faithfulness (unchanged)")
    for extension in ("png", "pdf"):
        fig.savefig(docs / (STEM + "." + extension), dpi=190)
    plt.close(fig)

    text = [
        "# Qwen3 8B Base: J-lens without subtraction\n",
        "Measured 20 September 2026, on the same 26 non-year relations as the [position experiment](last-position-lre-correlation.md). `name_gender` is retained.\n",
        "The requested raw J-lens is the repository's matched baseline: **`v_raw(e) = mean_{i != e} J_i h_e`**. The differential control is **`v_diff(e) = mean_{i != e} J_i (h_e − h_i)`**. Each Jacobian is evaluated in its reference's neutral prompt; this retains reference averaging and self-exclusion. No reference output or affine intercept is added.\n",
        "Both readouts are computed at the original last-subject-token position and at the final token `for`. The prompt remains `{subject} is known for`, with no prefix. The checkpoint, state 19, targets, references, and LRE faithfulness values are fixed. The output remains the raw final-block state at the last prompt position, decoded with the native final normalization and output head.\n",
        "**At the original subject position, removing subtraction gives almost the same correlation:** Spearman ρ = 0.294661 (p = 0.143944), versus 0.298768 with subtraction. The repeated `-Semit` predictions persist, including 10/16 country-language subjects. This is not a remedy for that artifact.\n",
        "At the final `for` position, the raw variant has a more positive correlation than the differential variant: ρ = 0.364819 (p = 0.066881), versus −0.067077. Its Pearson r is 0.416935 (p = 0.034092). However, none of its 491 targets is in the top 10, so the correlation concerns relative differences among very poor target rankings. These are exploratory results over the retained relations.\n",
        "## Correlation with LRE faithfulness\n",
        "The relation-level score is `x = −median(log10(rank + 1))`; y is the retained LRE top-1 faithfulness. Higher is better. Ranks are zero-based against the original model answer token, not full gold-answer accuracy. P-values test each correlation against zero, not the difference between variants.\n",
        "| Position | Readout | Spearman ρ | p | Pearson r | p | Top 1 | Top 10 | `-Semit` top 1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for position, label in POSITIONS.items():
        for kind in ("diff", "raw"):
            s = summary[position][kind]
            f = s["fit"]
            text.append(
                f"| {label} | {kind} | {f['spearman_rho']:.6f} | {f['spearman_p']:.6g} | {f['pearson_r']:.6f} | {f['pearson_p']:.6g} | {s['top1_count']}/{s['n_rows']} | {s['top10_count']}/{s['n_rows']} | {s['semit_top1_count']}/{s['n_rows']} |"
            )
    text += [
        "\nTop-k counts pool 491 scored rows, including the original duplicate rows; the correlation instead gives each of the 26 relations equal weight. No hyperparameters were selected using these new results.\n",
        f"![Raw versus differential lens]({STEM}.png)\n",
        f"[PDF]({STEM}.pdf) · [Relation CSV]({STEM}.csv) · [Full per-subject numerical evidence]({STEM}.json).\n",
        "## What removing subtraction changes\n",
        "For a fixed reference, `J_i h_e = J_i(h_e − h_i) + J_i h_i`. Thus removing subtraction adds a reference-dependent vector before normalization and decoding. It does not add the affine Taylor term `f_i(h_i)`. Nor does it remove the large input coordinates or ensure that reference and subject states belong to the same positional regime. The raw result must be measured; it does not automatically cure the positional artifact.\n",
        "This refines the [earlier diagnosis](qwen3-jlens-implementation-review.md): **subtraction is not necessary for the repeated-token artifact.** The raw lens still applies reference Jacobians to the large position-0 subject state. For Germany, the two later-position references still dominate, and dropping the subtraction barely changes their contribution norms:\n",
        "| Reference for Germany | Differential contribution norm | Raw contribution norm | Norm of removed term J_i h_i |",
        "| --- | ---: | ---: | ---: |",
        "| Australia | 9.060 | 37.661 | 37.365 |",
        "| Peru | 7170.070 | 7173.538 | 72.497 |",
        "| United States | 4933.006 | 4938.585 | 49.286 |",
        "\nThese are vector norms; the norms themselves are not additive. The direct vector identity is checked numerically below. The persistent artifact implicates the combination of positional state differences and reference Jacobians, rather than the subtraction operation alone.\n",
        "## All relation points\n",
        "Raw-readout OLS residuals are refitted independently for each position over all 26 relations: residual = y − fitted y.\n",
    ]
    for position, label in POSITIONS.items():
        f = summary[position]["raw"]["fit"]
        text += [
            f"### {label}\n",
            f"`ŷ = {f['intercept']:.12f} + {f['slope']:.12f} × x_raw`\n",
            "| Relation | n | Differential x | Raw x | LRE y | Fitted y | Raw residual |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in sorted(
            (r for r in rows if r["position"] == position),
            key=lambda r: -abs(r["raw_residual"]),
        ):
            text.append(
                f"| {r['relation']} | {r['n']} | {r['diff_x']:.6f} | {r['raw_x']:.6f} | {r['faithfulness']:.6f} | {r['raw_y_hat']:.6f} | {r['raw_residual']:+.6f} |"
            )
        text.append("")
    text += [
        "## Country-language raw predictions\n",
        "The target is the model's first answer token under the unchanged relation-specific ICL prompt.\n",
        "| Subject | Target | Subject-position raw top 1 | Target rank | Final-position raw top 1 | Target rank |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    country = next(d for d in measurements if d["relation"] == "country_language")
    for a, b in zip(
        country["positions"]["subject"]["modes"]["raw"]["rows"],
        country["positions"]["last"]["modes"]["raw"]["rows"],
        strict=True,
    ):
        safe = lambda s: s.replace("|", "\\|").replace("\n", "\\n")
        text.append(
            f"| {a['subject']} | {safe(a['target'])} | {safe(a['top_tokens'][0])} | {a['rank']} | {safe(b['top_tokens'][0])} | {b['rank']} |"
        )
    text += [
        "\n## Validation and reproduction\n",
        f"- Differential controls: {validation['baseline_exact_ranks']}/{validation['baseline_rank_comparisons']} ranks match the preceding position experiment exactly; maximum rank difference {validation['baseline_max_rank_error']}, maximum relation-score difference {validation['baseline_max_x_error']:.9g}.",
        f"- The linear identity `J_i h_e − J_i h_i = J_i(h_e − h_i)` is checked for every reference and scored row at both positions. Maximum relative discrepancy, normalized by the sum of term norms: {validation['max_linearity_relative_error']:.6g}.",
        f"- {validation['finite_difference_checks']} raw-product derivative checks cover both positions. Worst best-step central-finite-difference relative error: {validation['worst_best_step_fd_relative_error']:.6g}; batched/scalar JVP checks also pass.",
        "- Every reference passes exact no-op intervention checks. Cached raw final states reproduce native logits. All 491 frozen target IDs are regenerated and match exactly; reference counts and duplicate/self-exclusion behavior are preserved.",
        "- Pinned Qwen revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; float32, eager attention, A100 80 GB. Source and input hashes, per-reference contribution norms, and all ranks/top-five predictions are recorded in the JSON.\n",
        "[GPU runner](../../paper/repro/qwen_raw_jlens.py), [CPU report generator](../../paper/repro/report_qwen_raw_jlens.py), [frozen inputs](last-position-lre-correlation-inputs.json).\n",
        "```sh\ncd code\nCUDA_VISIBLE_DEVICES=0 uv run --extra gpu python ../paper/repro/qwen_raw_jlens.py\nuv run python ../paper/repro/report_qwen_raw_jlens.py\n```\n",
        "For parallel GPU workers add `--shards 4 --shard 0` through `--shard 3`, assigning each worker a separate GPU. Historical measurements are retained.\n",
    ]
    (docs / (STEM + ".md")).write_text("\n".join(text))
    print(json.dumps({"summary": summary, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
