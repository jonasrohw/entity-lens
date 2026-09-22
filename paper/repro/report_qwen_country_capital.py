"""Export inspectable country-capital raw/differential J-lens predictions.

Reads committed measurements only; no model inference or GPU is required.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2] / "code"
DOCS = BASE / "docs"
RELATION = "country_capital_city"
STEM = "qwen3-8b-base-country-capital-predictions"
POSITIONS = {"subject": "Last subject token", "last": "Final prompt token `for`"}


def code(value):
    text = str(value).replace("|", "&#124;").replace("\n", "\\n")
    delimiter = "``" if "`" in text else "`"
    return f"{delimiter}{text}{delimiter}"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = DOCS / "qwen3-8b-base-raw-jlens.json"
    inputs = DOCS / "last-position-lre-correlation-inputs.json"
    dataset = BASE / "data/relations" / (RELATION + ".json")
    all_results = json.loads(source.read_text())
    measured = next(d for d in all_results["measurements"] if d["relation"] == RELATION)
    language = next(
        d for d in all_results["measurements"] if d["relation"] == "country_language"
    )
    specs = json.loads(inputs.read_text())["models"]["qwen"]
    spec = next(d for d in specs["relations"] if d["relation"] == RELATION)
    names = [r["subject"] for r in spec["targets"]]
    objects = {
        r["subject"]: r["object"] for r in json.loads(dataset.read_text())["samples"]
    }
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == spec["dataset_sha256"]
    assert (
        hashlib.sha256(inputs.read_bytes()).hexdigest()
        == measured["provenance"]["input_sha256"]
    )
    assert measured["faithfulness"] == spec["faithfulness"]
    assert len(names) == len(set(names)) == 16
    predictions, contributions, aggregates = [], [], []
    for position, label in POSITIONS.items():
        modes = measured["positions"][position]["modes"]
        for kind in ("diff", "raw"):
            result = modes[kind]
            assert [r["subject"] for r in result["rows"]] == names
            ranks = np.asarray([r["rank"] for r in result["rows"]], dtype=float)
            assert (
                float(np.median(np.log10(ranks + 1))) == result["agg"]["median_lograng"]
            )
            assert float(np.median(ranks)) == result["agg"]["median_rank"]
            assert float((ranks < 10).mean()) == result["agg"]["agree_at10"]
            # The neutral lens sees the same countries and references in both relations.
            assert [r["top_ids"] for r in result["rows"]] == [
                r["top_ids"]
                for r in language["positions"][position]["modes"][kind]["rows"]
            ]
            x = -result["agg"]["median_lograng"]
            fit = all_results["summary"][position][kind]["fit"]
            prediction = fit["intercept"] + fit["slope"] * x
            aggregates.append(
                {
                    "position": position,
                    "readout": kind,
                    **result["agg"],
                    "top1_count": int((ranks == 0).sum()),
                    "top10_count": int((ranks < 10).sum()),
                    "semit_top1_count": sum(
                        r["top_tokens"][0] == "-Semit" for r in result["rows"]
                    ),
                    "x": x,
                    "faithfulness": measured["faithfulness"],
                    "fitted_y": prediction,
                    "residual": measured["faithfulness"] - prediction,
                }
            )
            for row, target in zip(result["rows"], spec["targets"], strict=True):
                assert row["target_id"] == target["model_top1_id"]
                assert row["target"] == target["model_top1"]
                assert row["n_references"] == sum(
                    s != row["subject"] for s in spec["references"]
                )
                for token_id, token in zip(
                    row["top_ids"], row["top_tokens"], strict=True
                ):
                    assert isinstance(token_id, int) and isinstance(token, str)
                flat = {
                    "subject": row["subject"],
                    "dataset_answer": objects[row["subject"]],
                    "position": position,
                    "position_index": row["position"],
                    "readout": kind,
                    "state_norm": row["state_norm"],
                    "n_references": row["n_references"],
                    "target": row["target"],
                    "target_id": row["target_id"],
                    "rank": row["rank"],
                }
                for i, (token_id, token) in enumerate(
                    zip(row["top_ids"], row["top_tokens"], strict=True), 1
                ):
                    flat[f"top{i}_token"] = token
                    flat[f"top{i}_id"] = token_id
                predictions.append(flat)
        by_name = {r["subject"]: r for r in modes["raw"]["rows"]}
        for ref in measured["positions"][position]["contributions"]:
            anchor = by_name[ref["reference"]]
            for i, name in enumerate(names):
                if name == ref["reference"]:
                    continue
                row = by_name[name]
                contributions.append(
                    {
                        "position": position,
                        "subject": name,
                        "reference": ref["reference"],
                        "subject_position": row["position"],
                        "reference_position": anchor["position"],
                        "subject_state_norm": row["state_norm"],
                        "reference_state_norm": anchor["state_norm"],
                        "weight": 1 / row["n_references"],
                        "raw_jvp_norm": ref["raw_norms"][i],
                        "diff_jvp_norm": ref["diff_norms"][i],
                        "anchor_product_norm": ref["anchor_product_norm"],
                        "linearity_relative_error": ref["linearity_relative_errors"][i],
                    }
                )
    assert len(predictions) == 64
    assert len(contributions) == 240
    target_context = "".join(
        spec["template"].format(d["subject"]) + " " + d["object"] + "\n"
        for d in spec["demonstrations"]
    )
    evidence = {
        "relation": RELATION,
        "provenance": measured["provenance"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_sha256": spec["dataset_sha256"],
        "specification": spec,
        "icl_context": target_context,
        "target_prompt_template": spec["template"],
        "aggregates": aggregates,
        "predictions": predictions,
        "contributions": contributions,
        "measurement": measured,
        "validation": {
            "prediction_rows": 64,
            "included_reference_contributions": 240,
            "all_target_ids_match": True,
            "all_top5_ids_match_country_language": True,
            "aggregates_recomputed_from_ranks": True,
        },
    }
    (DOCS / (STEM + ".json")).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )
    write_csv(DOCS / (STEM + ".csv"), predictions)
    write_csv(DOCS / (STEM + "-contributions.csv"), contributions)

    text = [
        "# Qwen3 8B Base: country-capital predictions with and without subtraction\n",
        "Relation: `country_capital_city`. These are the 16 scored countries from the completed [raw J-lens experiment](qwen3-8b-base-raw-jlens.md), measured on 20 September 2026. This report exports the retained results; it does not rerun or change the experiment.\n",
        "**Without subtraction at the subject position, 4/16 capital targets are in the top 10, versus 2/16 with subtraction. Neither has a top-1 hit, and both still predict `-Semit` first for the same ten countries.** At the final `for` position, neither variant places a target in the top 10.\n",
        "The raw subject-position top-10 cases are Peru → `Lima` (rank 2), Chile → `Santiago` (rank 4), Saudi Arabia → `Riyadh` (rank 2), and South Korea → `Seoul` (rank 2). The differential lens has only the latter two in the top 10; its Peru/Chile ranks are 11/10. All ranks below are zero-based: top 10 means rank < 10.\n",
        f"[Prediction CSV: tokens and IDs]({STEM}.csv) · [Per-reference contributions CSV]({STEM}-contributions.csv) · [Full relation JSON]({STEM}.json).\n",
        "## Protocol and interpretation\n",
        "- **Differential:** `mean_{i != e} J_i (h_e − h_i)`. **Raw / no subtraction:** `mean_{i != e} J_i h_e`. Both average the same references and exclude the scored country itself. Raw does not add an affine reference output.",
        "- **Model and layer:** `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, state 19 (output of block 18).",
        "- **Lens input:** `{country} is known for`, without a prefix. At each position, both cached input states and Jacobian interventions use that position. Output/readout is at the final prompt token, with the native final normalization and output head.",
        "- **Target:** the model's first answer token under `The capital city of {country} is`, after the eight demonstrations shown below. A target such as `Bras`, `Car`, `Buenos`, or `New` is a token fragment, not a complete capital name. Ranks measure agreement with that model token, not complete-answer correctness.",
        "- **LRE faithfulness:** retained y = 0.625, from the separate relation-specific LRE evaluation. The predictions below are neutral-prompt lenses, not those relation-specific LRE predictions.",
        "- **Token labels:** the retained top-token strings have outer whitespace stripped. Distinct token IDs may therefore have identical displayed text (for example `Saudi`). The CSV and JSON preserve every target and top-five token ID. Dataset answers and ICL strings are retained verbatim, including literal Unicode escape sequences.",
        "- **Same outputs as country-language:** these two relations share the 16 subjects, eight references, neutral prompts, and layer. All four variants' top-five token IDs match the country-language run exactly; only the relation-specific target tokens and their ranks differ. This is expected for a relation-agnostic lens.\n",
        "References, in their averaging order: "
        + ", ".join(code(s) for s in spec["references"])
        + ". Reference countries receive seven contributions; other countries receive eight.\n",
        "## Relation summaries\n",
        "The fitted y and residual use the corresponding variant's OLS fit over all 26 relations, not a fit to these 16 countries. x is `−median(log10(rank + 1))`; residual is y − fitted y.\n",
        "| Position | Readout | Median rank | Top 1 | Top 10 | `-Semit` top 1 | x | Fitted y | Residual |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in aggregates:
        text.append(
            f"| {POSITIONS[a['position']]} | {a['readout']} | {a['median_rank']:.1f} | {a['top1_count']}/16 | {a['top10_count']}/16 | {a['semit_top1_count']}/16 | {a['x']:.6f} | {a['fitted_y']:.6f} | {a['residual']:+.6f} |"
        )
    for position, label in POSITIONS.items():
        modes = measured["positions"][position]["modes"]
        text += [
            f"\n## {label}: all predictions\n",
            "Top-five lists are ordered from highest to lowest score.\n",
            "| Country | Model target | Differential top 5 | Target rank | Raw top 5 (no subtraction) | Target rank |",
            "| --- | --- | --- | ---: | --- | ---: |",
        ]
        for diff, raw in zip(modes["diff"]["rows"], modes["raw"]["rows"], strict=True):
            text.append(
                f"| {raw['subject']} | {code(raw['target'])} | {', '.join(code(t) for t in diff['top_tokens'])} | {diff['rank']} | {', '.join(code(t) for t in raw['top_tokens'])} | {raw['rank']} |"
            )
    text += [
        "\n## Input positions and norms\n",
        "The subject position is the last token of the country name, not necessarily position 0. Position indices are zero-based.\n",
        "| Country | Dataset answer (verbatim) | Model target ID | Subject position | Subject norm | Final position | Final norm | References |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subject, last in zip(
        measured["positions"]["subject"]["modes"]["raw"]["rows"],
        measured["positions"]["last"]["modes"]["raw"]["rows"],
        strict=True,
    ):
        text.append(
            f"| {subject['subject']} | {code(objects[subject['subject']])} | {subject['target_id']} | {subject['position']} | {subject['state_norm']:.3f} | {last['position']} | {last['state_norm']:.3f} | {subject['n_references']} |"
        )
    text += [
        "\n## Reference contributions for Germany\n",
        "Germany is not among the references, so each contribution has weight 1/8. These are norms before averaging and decoding; norms themselves are not additive. The Peru and United States subject-position contributions remain very large without subtraction. The downloadable contributions CSV contains the equivalent rows for every scored country and both positions.\n",
        "| Position | Reference | Reference position | Reference-state norm | Differential JVP norm | Raw JVP norm | J_i h_i norm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in contributions:
        if r["subject"] == "Germany":
            text.append(
                f"| {r['position']} | {r['reference']} | {r['reference_position']} | {r['reference_state_norm']:.3f} | {r['diff_jvp_norm']:.3f} | {r['raw_jvp_norm']:.3f} | {r['anchor_product_norm']:.3f} |"
            )
    text += [
        "\n## Exact target-prompt context\n",
        "Append `The capital city of {country} is` to this context to obtain that country's target prompt. Literal backslashes below are part of the retained dataset strings.\n",
        "```text\n" + target_context + "```\n",
        "## Reproduction and checks\n",
        "The exporter reads only versioned snapshots. It checks all 64 prediction rows against frozen subjects/target IDs/reference counts, recomputes the aggregates from individual ranks, and exports all 240 included reference contributions. It also verifies the shared top-five token IDs against country-language. The underlying GPU run passed raw-JVP finite-difference and batched/scalar checks; its 982 differential control ranks exactly reproduced the preceding position experiment.\n",
        "```sh\ncd code\nuv run python ../paper/repro/report_qwen_country_capital.py\n```\n",
        "[Exporter](../../paper/repro/report_qwen_country_capital.py) · [Source measurements](qwen3-8b-base-raw-jlens.json) · [Frozen split/targets](last-position-lre-correlation-inputs.json). Source hashes and the full retained relation measurement are included in the relation JSON.\n",
    ]
    (DOCS / (STEM + ".md")).write_text("\n".join(text))
    print(
        json.dumps(
            {
                "predictions": len(predictions),
                "contributions": len(contributions),
                "aggregates": aggregates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
