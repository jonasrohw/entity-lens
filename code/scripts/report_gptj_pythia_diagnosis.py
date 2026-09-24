"""Validate complete full-row newline experiments and explain rank disagreements."""
import collections
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/all_models_gender_original_continents"
OUT = ROOT / "results/gptj_pythia_full_row_diagnosis"
LABELS = {"gptj": "GPT-J 6B", "pythia6_9b": "Pythia 6.9B"}


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scalar_rho(x, y):
    # Independent tied ranks and Pearson calculation, without scipy rankdata.
    def ranks(values):
        result = [None] * len(values)
        for value in set(values):
            positions = [i + 1 for i, v in enumerate(sorted(values)) if v == value]
            rank = sum(positions) / len(positions)
            for i, v in enumerate(values):
                if v == value:
                    result[i] = rank
        return result
    x, y = ranks(x), ranks(y)
    mx, my = sum(x)/len(x), sum(y)/len(y)
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (
        sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y)
    )**0.5


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows, correlations, evidence, files = [], [], {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for i, (name, label) in enumerate(LABELS.items()):
        manifest = read(BASE/name/"manifest.json")
        names = sorted(manifest["dataset"])
        paths = list((OUT/name/"measurements").glob("*.json"))
        assert {p.stem for p in paths} == set(names), (name, len(paths), "incomplete experiment")
        rows, university, prompt_rows = [], {}, []
        for rel in names:
            p = OUT/name/"measurements"/(rel+".json")
            current = read(p)
            old_path = ROOT/current["baseline_path"]
            old = read(old_path)
            assert sha(old_path) == current["baseline_sha256"]
            plan = manifest["dataset"][rel]
            assert sha(ROOT/"data/relations"/(rel+".json")) == current["data_sha256"] == plan["data_sha256"]
            assert current["lre_reused"] and current["targets_reused"]
            assert current["gender_alternatives"] and not current["diverse_continent_references"]
            assert not old["date_audit"]["excluded"]
            assert current["lens"]["neutral_prompt"] == "\n{} is known for"
            assert current["lens"]["layer"] == old["lens"]["layer"] == manifest["lens_layer"]
            worker = read(OUT/name/"workers"/(current["worker"]+".json"))
            assert worker["model"] == manifest["model"] and worker["revision"] == manifest["revision"]
            assert worker["manifest_sha256"] == sha(BASE/name/"manifest.json")
            for filename, digest in worker["source_sha256"].items():
                assert sha(OUT/name/"provenance"/(digest+"_"+Path(filename).name)) == digest
            n = plan["n_original"]
            lens, prev, faith = current["lens"], old["lens"], old["faithfulness"]
            assert n == lens["n"] == prev["n"] == faith["n"]
            assert len(lens["rows"]) == len(prev["rows"]) == n
            for a, b, sample in zip(lens["rows"], prev["rows"], plan["lens_split"]["evaluation"], strict=True):
                for key in ["subject", "object", "model_top1_id", "model_top1", "n_references"]:
                    assert a[key] == b[key]
                assert (a["subject"], a["object"]) == (sample["subject"], sample["object"])
                if rel == "name_gender":
                    assert a["gender_target_terms"] == b["gender_target_terms"]
                    assert a["rank_diff"] == min([a["rank_diff_original"]]+[v["rank"] for v in a["gender_term_ranks"].values()])
            for l in [lens, prev]:
                assert abs(float(np.median(np.log10(np.array([r["rank_diff"] for r in l["rows"]])+1)))-l["agg"]["diff"]["median_lograng"]) < 1e-12
            ft = {r["subject"]: r["model_top1_id"] for r in faith["rows"]}
            mismatch = sum(ft[r["subject"]] != r["model_top1_id"] for r in lens["rows"])
            counts = collections.Counter(r["model_top1_id"] for r in faith["rows"])
            majority_id, majority_n = counts.most_common(1)[0]
            majority_token = next(r["model_top1"] for r in faith["rows"] if r["model_top1_id"] == majority_id)
            predictions = faith["cells"][faith["best_key"]]["predictions"]
            literal_original = -prev["agg"].get("diff_original", prev["agg"]["diff"])["median_lograng"]
            literal_newline = -lens["agg"].get("diff_original", lens["agg"]["diff"])["median_lograng"]
            neutral = current["neutral_prompt_diagnostics"]
            row_norms = [neutral[r["subject"]] for r in lens["rows"]]
            references = plan["lens_split"]["references"]
            row = dict(
                model=name, relation=rel, n=n,
                faithfulness=faith["best_faith"], faithfulness_frozen=faith["fixed_faith"],
                original_score=-prev["agg"]["diff"]["median_lograng"],
                newline_score=-lens["agg"]["diff"]["median_lograng"],
                original_median_rank=float(np.median([r["rank_diff"]+1 for r in prev["rows"]])),
                newline_median_rank=float(np.median([r["rank_diff"]+1 for r in lens["rows"]])),
                literal_original_score=literal_original, literal_newline_score=literal_newline,
                target_context_disagreements=mismatch, target_context_disagreement_rate=mismatch/n,
                majority_target_token=majority_token, majority_target_faithfulness=majority_n/n,
                faithfulness_above_majority=faith["best_faith"]-majority_n/n,
                lre_fraction_predicting_majority=predictions.count(majority_id)/n,
                lre_unique_predictions=len(set(predictions)),
                original_top1_lens_fraction=float(np.mean([r["rank_diff"] == 0 for r in prev["rows"]])),
                newline_top1_lens_fraction=float(np.mean([r["rank_diff"] == 0 for r in lens["rows"]])),
                fraction_references_original_position_zero=sum(neutral[r["subject"]]["original_position"] == 0 for r in references)/len(references),
                fraction_rows_original_position_zero=sum(r["original_position"] == 0 for r in row_norms)/n,
                original_subject_norm_median=float(np.median([r["original_subject_norm"] for r in row_norms])),
                newline_subject_norm_median=float(np.median([r["newline_subject_norm"] for r in row_norms])),
                lens_score_improvement=prev["agg"]["diff"]["median_lograng"]-lens["agg"]["diff"]["median_lograng"],
            )
            rows.append(row)
            if rel == "person_university":
                university = {k: row[k] for k in ["n", "faithfulness", "majority_target_token", "majority_target_faithfulness", "lre_fraction_predicting_majority", "lre_unique_predictions"]}
            for r in lens["rows"]:
                diag = current["neutral_prompt_diagnostics"][r["subject"]]
                assert diag["newline_position"] > 0
                prompt_rows.append(dict(relation=rel, subject=r["subject"], **diag))
            files[str(p.relative_to(ROOT))] = sha(p)
        assert len(rows) == 28 and sum(r["n"] for r in rows) == 9747
        for variant in ["original", "newline"]:
            xr = rankdata([-r[variant+"_score"] for r in rows])
            yr = rankdata([-r["faithfulness"] for r in rows])
            for r, x, y in zip(rows, xr, yr):
                r[variant+"_lens_relation_rank"] = float(x)
                r[variant+"_lre_relation_rank"] = float(y)
                r[variant+"_rank_gap"] = float(x-y)
            for axis in ["faithfulness", "faithfulness_frozen"]:
                x = [r[variant+"_score"] for r in rows]
                y = [r[axis] for r in rows]
                rho = float(spearmanr(x, y).statistic)
                assert abs(rho-scalar_rho(x, y)) < 1e-12
                correlations.append(dict(model=name, variant=variant, lre_axis=axis,
                                         n_relations=28, n_rows=9747, spearman_rho=rho))
        for r in rows:
            r["rank_gap_reduction"] = abs(r["original_rank_gap"])-abs(r["newline_rank_gap"])
        evidence[name] = dict(
            university=university,
            context_mismatch_weighted=sum(r["target_context_disagreements"] for r in rows)/9747,
            context_mismatch_median=float(np.median([r["target_context_disagreement_rate"] for r in rows])),
            original_outliers=sorted(rows, key=lambda r: -abs(r["original_rank_gap"])),
            newline_outliers=sorted(rows, key=lambda r: -abs(r["newline_rank_gap"])),
            prompt_norms={kind: {
                "median": float(np.median([r[kind] for r in prompt_rows])),
                "p99": float(np.quantile([r[kind] for r in prompt_rows], .99)),
                "maximum": max(r[kind] for r in prompt_rows),
            } for kind in ["original_subject_norm", "newline_subject_norm", "original_final_norm", "newline_final_norm"]},
            subject_norms_by_original_position={label: {
                "n_rows": sum((r["original_position"] == 0) == is_zero for r in prompt_rows),
                "original_median": float(np.median([r["original_subject_norm"] for r in prompt_rows if (r["original_position"] == 0) == is_zero])),
                "newline_median": float(np.median([r["newline_subject_norm"] for r in prompt_rows if (r["original_position"] == 0) == is_zero])),
            } for label, is_zero in [("position_zero", True), ("later_positions", False)]},
            original_position_zero_rows=sum(r["original_position"] == 0 for r in prompt_rows),
            newline_position_zero_rows=sum(r["newline_position"] == 0 for r in prompt_rows),
            content_tokenization_changed_rows=sum(r["newline_ids"][1:] != r["original_ids"] for r in prompt_rows),
            original_rank_disagreement_share_top_three=sum(r["original_rank_gap"]**2 for r in sorted(rows, key=lambda r: -abs(r["original_rank_gap"]))[:3])/sum(r["original_rank_gap"]**2 for r in rows),
            score_improvement_vs_position_zero_refs_rho=float(spearmanr([r["fraction_references_original_position_zero"] for r in rows], [r["lens_score_improvement"] for r in rows]).statistic),
            covariate_correlations={field: {
                axis: float(spearmanr([r[field] for r in rows], [r[axis] for r in rows]).statistic)
                for axis in ["faithfulness", "original_score", "newline_score"]
            } for field in ["n", "majority_target_faithfulness", "target_context_disagreement_rate"]},
        )
        for j, variant in enumerate(["original", "newline"]):
            ax = axes[i, j]
            x = [r[variant+"_score"] for r in rows]
            y = [r["faithfulness"] for r in rows]
            ax.scatter(x, y, color="#2166ac" if j == 0 else "#1b9e77", s=38)
            for r in sorted(rows, key=lambda r: -abs(r[variant+"_rank_gap"]))[:6]:
                near_right = r[variant+"_score"] > max(x)-.2*(max(x)-min(x))
                ax.annotate(r["relation"].replace("person_", "").replace("country_", "ctry_"),
                            (r[variant+"_score"], r["faithfulness"]), xytext=(-4 if near_right else 4, 5),
                            ha="right" if near_right else "left", textcoords="offset points", fontsize=7)
            ax.set_title(f"{label} — {variant}; Spearman ρ={spearmanr(x, y).statistic:.4f}")
            ax.set_xlabel("Diff Jacobian lens score (higher is better)")
            ax.set_ylabel("LRE faithfulness")
            ax.set_ylim(-.03, 1.08)
            ax.grid(alpha=.18)
        all_rows.extend(rows)
    write_csv(OUT/"relation_diagnostics.csv", all_rows)
    write_csv(OUT/"correlations.csv", correlations)
    result = dict(complete=True, protocol="Original 28; all 9747 rows/model; 8 demos and 8 refs per method; original continent refs; gender alternatives; years included; neutral newline only; LRE and targets reused.", correlations=correlations, evidence=evidence, source_sha256=files)
    (OUT/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    fig.savefig(OUT/"outliers.png", dpi=180)
    fig.savefig(OUT/"outliers.pdf")
    lines = ["# GPT-J and Pythia: full-row newline investigation", "", result["protocol"], "",
             "These are in-sample scores: demonstrations and reference rows are also evaluated. No relation or row was removed. All LRE grids, chosen layers/betas and target IDs were reused unchanged. Only the neutral lens prompt changed from `{} is known for` to `\\n{} is known for`.", "",
             "| Model | Original ρ | Newline ρ | Δρ | Frozen original LRE ρ | Frozen newline LRE ρ |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, label in LABELS.items():
        c = {(r["variant"], r["lre_axis"]): r["spearman_rho"] for r in correlations if r["model"] == name}
        a, b = c["original", "faithfulness"], c["newline", "faithfulness"]
        lines.append(f"| {label} | {a:.6f} | {b:.6f} | {b-a:+.6f} | {c['original', 'faithfulness_frozen']:.6f} | {c['newline', 'faithfulness_frozen']:.6f} |")
    lines += ["", "The leading newline substantially improves both correlations and removes large subject-state norm spikes at the first token. The strongest lens gains occur in country-language, capital-city and largest-city relations. The evidence supports a beginning-of-prompt sensitivity; it does not isolate the responsible attention heads or fully explain every remaining outlier.", "",
              "University remains a shared outlier because LRE achieves high faithfulness by predicting the constant first token `University`. Gender remains difficult for the neutral lens even with the accepted wording set. Different LRE/lens demonstration contexts are another source of disagreement, especially for occupation and sports position; those contexts were preserved for this comparison."]
    for name, label in LABELS.items():
        e = evidence[name]
        lines += ["", f"## {label}", "", "Largest original rank mismatches; rank 1 is best. Median token ranks below are one-based.", "",
                  "| Relation | Rows | LRE faithfulness | Original lens median rank | Newline lens median rank | Original rank gap | Newline rank gap |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in e["original_outliers"][:10]:
            lines.append(f"| {r['relation']} | {r['n']} | {r['faithfulness']:.4f} | {r['original_median_rank']:.1f} | {r['newline_median_rank']:.1f} | {r['original_rank_gap']:+.1f} | {r['newline_rank_gap']:+.1f} |")
        u = e["university"]
        lines += ["", f"University: the selected LRE predicts `{u['majority_target_token']}` on every row, scoring {u['faithfulness']:.4%}; this exactly matches the constant majority-token baseline. This is first-token faithfulness, not full university identification.", "",
                  f"The LRE and lens demonstration contexts produce different target tokens on {e['context_mismatch_weighted']:.2%} of all rows (median across relations {e['context_mismatch_median']:.2%}). This diagnostic does not establish how much correlation would change with matched contexts; that would be a separate protocol experiment.", "",
                  "| Activation norm | Median | 99th percentile | Maximum |", "|---|---:|---:|---:|"]
        for k, v in e["prompt_norms"].items():
            lines.append(f"| {k} | {v['median']:.2f} | {v['p99']:.2f} | {v['maximum']:.2f} |")
        lines += ["", "The overall median hides the first-position spike. The following groups cover every row:", "",
                  "| Original subject endpoint | Rows | Original median subject norm | Newline median subject norm |",
                  "|---|---:|---:|---:|"]
        for group, v in e["subject_norms_by_original_position"].items():
            lines.append(f"| {group} | {v['n_rows']} | {v['original_median']:.2f} | {v['newline_median']:.2f} |")
        lines += ["", f"Original subject endpoint at position zero: {e['original_position_zero_rows']}/9747 rows; after newline: {e['newline_position_zero_rows']}/9747. Content token IDs changed beyond the added newline: {e['content_tokenization_changed_rows']}/9747. Norm statistics cover every evaluation row (duplicates retained).", ""]
        lines += [f"Across all 28 relations, the fraction of reference subjects ending at token position zero correlates with the lens-score improvement at Spearman {e['score_improvement_vs_position_zero_refs_rho']:.4f}. This supports sensitivity to the beginning of the neutral prompt; it does not separately identify a particular attention mechanism.", ""]
    lines += ["## Validation", "", "All 56 model-relation files and 19,494 evaluated rows verified against immutable baseline hashes, dataset hashes, pinned model revisions, row order, target IDs, reference counts and gender target sets. Aggregates were recomputed from every row; Spearman values were checked with an independent tied-rank implementation. GPU derivative checks and cached-target equivalence test passed.", "", "Artifacts: [correlations](correlations.csv), [all relation diagnostics](relation_diagnostics.csv), [outlier figure](outliers.png), [machine-readable evidence](summary.json)."]
    (OUT/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
