"""Subject-paired inference and complete prediction export for anchor swaps."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/anchor_swap_gemini"
DATA = ROOT / "data/anchor_swap_gemini"
NBOOT = 20000
CONDITIONS = [
    "own",
    "foreign",
    "neutral",
    "anchor",
    "wrong_foreign",
    "wrong_neutral",
    "cyclic_foreign",
    "cyclic_neutral",
    "neutral_alone",
]
CONTRASTS = {
    "foreign_minus_own": ("foreign", "own"),
    "foreign_minus_wrong": ("foreign", "wrong_foreign"),
    "neutral_minus_wrong": ("neutral", "wrong_neutral"),
    "neutral_minus_anchor": ("neutral", "anchor"),
    "foreign_minus_cyclic": ("foreign", "cyclic_foreign"),
    "neutral_minus_cyclic": ("neutral", "cyclic_neutral"),
}


def interval(reps, level=0.95):
    q = (1 - level) / 2
    return np.quantile(reps, [q, 1 - q]).tolist()


def holm(ps):
    order = np.argsort(ps)
    out = np.zeros(len(ps))
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, ps[j] * (len(ps) - i))
        out[j] = min(1.0, running)
    return out.tolist()


def calc(arrays, rng):
    # Arrays have one value per subject, already averaging both directions.
    n = len(arrays["own"])
    if not n:
        return {"n": 0}, None
    idx = rng.integers(n, size=(NBOOT, n))
    reps = {k: v[idx].mean(1) for k, v in arrays.items()}
    summary = {
        "n": n,
        "rates": {
            k: {"mean": float(v.mean()), "ci95": interval(reps[k])}
            for k, v in arrays.items()
        },
        "contrasts": {},
    }
    for k, (a, b) in CONTRASTS.items():
        rep = reps[a] - reps[b]
        summary["contrasts"][k] = {
            "mean": float((arrays[a] - arrays[b]).mean()),
            "ci95": interval(rep),
            "ci90": interval(rep, 0.90),
        }
    lo, hi = summary["contrasts"]["foreign_minus_own"]["ci90"]
    summary["equivalent_within_10pp"] = bool(lo > -0.1 and hi < 0.1)
    summary["foreign_beats_wrong"] = bool(
        summary["contrasts"]["foreign_minus_wrong"]["ci95"][0] > 0
    )
    return summary, reps


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    rng = np.random.default_rng(220922)
    report = {"n_bootstrap": NBOOT, "domains": {}, "macro": {}, "direction_tests": []}
    allreps = {}
    allmeans = {}
    predictions = []
    domain_rows = []
    for name, d in manifest["domains"].items():
        p = OUT / f"{name}.json"
        r = json.loads(p.read_text())
        assert r["complete"]
        assert (
            r["provenance"]["manifest_sha256"]
            == hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest()
        )
        test = np.array([x["split"] == "test" for x in r["rows"]])
        val = ~test
        selected = {
            m: max(
                [1.0, 2.0, 3.0, 5.0, 10.0],
                key=lambda b: (
                    np.mean(
                        np.array(r["scores"][m][str(b)]["own"]["top"])[val]
                        == np.array(r["targets"][m])[val]
                    ),
                    -b,
                ),
            )
            for m in ["a", "b"]
        }
        val_faith = {
            m: float(
                np.mean(
                    np.array(r["scores"][m]["3.0"]["own"]["top"])[val]
                    == np.array(r["targets"][m])[val]
                )
            )
            for m in ["a", "b"]
        }
        distinct = np.array(r["targets"]["a"]) != np.array(r["targets"]["b"])
        bothcorrect = np.array(r["tokenization"]["a"]["model_equals_gold"]) & np.array(
            r["tokenization"]["b"]["model_equals_gold"]
        )
        unique = np.array(r["tokenization"]["a"]["gold_first_token_unique"]) & np.array(
            r["tokenization"]["b"]["gold_first_token_unique"]
        )
        gd = np.array(r["gold"]["a"]) != np.array(r["gold"]["b"])
        masks = {
            "all": test,
            "different_model_tokens": test & distinct,
            "both_model_correct_distinct": test & bothcorrect & gd,
            "identifying_both_correct_distinct": test & bothcorrect & gd & unique,
        }
        ds = {
            "n_test": int(test.sum()),
            "n_validation": int(val.sum()),
            "selected_beta": selected,
            "validation_own_beta3": val_faith,
            "strong_operators": all(v >= 0.5 for v in val_faith.values()),
            "subsets": {},
            "directions": {},
            "checks": r["checks"],
            "seconds": r["seconds"],
        }
        for setting in ["primary", "validation_beta", "gold"]:
            hit = {}
            for c in CONDITIONS:
                hit[c] = np.stack(
                    [
                        np.array(
                            r["scores"][m][
                                str(selected[m])
                                if setting == "validation_beta"
                                else "3.0"
                            ][c]["top"]
                        )
                        == np.array(r["gold" if setting == "gold" else "targets"][m])
                        for m in ["a", "b"]
                    ],
                    1,
                )
            for subset, mask in masks.items():
                key = setting + "/" + subset
                summary, reps = calc({k: v[mask].mean(1) for k, v in hit.items()}, rng)
                ds["subsets"][key] = summary
                if reps is not None:
                    allreps.setdefault(key, []).append(reps)
                    allmeans.setdefault(key, []).append(
                        {k: float(v[mask].mean()) for k, v in hit.items()}
                    )
                    if ds["strong_operators"]:
                        strong_key = "strong_operators/" + key
                        allreps.setdefault(strong_key, []).append(reps)
                        allmeans.setdefault(strong_key, []).append(
                            {k: float(v[mask].mean()) for k, v in hit.items()}
                        )
            if setting == "primary":
                for j, m in enumerate(["a", "b"]):
                    entry = {c: float(hit[c][test, j].mean()) for c in CONDITIONS}
                    entry["model_gold_accuracy"] = float(
                        np.array(r["tokenization"][m]["model_equals_gold"])[test].mean()
                    )
                    entry["relation"] = d["spec"]["relations"][j]
                    other = "b" if m == "a" else "a"
                    foreign_top = np.array(r["scores"][m]["3.0"]["foreign"]["top"])
                    informative = test & distinct
                    entry["n_different_targets"] = int(informative.sum())
                    entry["foreign_matches_source_relation"] = (
                        float(
                            (
                                foreign_top[informative]
                                == np.array(r["targets"][other])[informative]
                            ).mean()
                        )
                        if informative.any()
                        else None
                    )
                    own_correct = test & hit["own"][:, j]
                    entry["own_correct_n"] = int(own_correct.sum())
                    entry["foreign_retains_own_success"] = (
                        float(hit["foreign"][own_correct, j].mean())
                        if own_correct.any()
                        else None
                    )
                    ds["directions"][m] = entry
                    x = hit["foreign"][test, j]
                    y = hit["own"][test, j]
                    wins = int((x & ~y).sum())
                    losses = int((~x & y).sum())
                    report["direction_tests"].append(
                        {
                            "domain": name,
                            "relation": entry["relation"],
                            "foreign_only": wins,
                            "own_only": losses,
                            "mcnemar_p": float(binomtest(wins, wins + losses).pvalue)
                            if wins + losses
                            else 1.0,
                        }
                    )
        for i, row in enumerate(r["rows"]):
            assert r["donors"][i] != i and r["cyclic_donors"][i] != i
            for j, m in enumerate(["a", "b"]):
                rec = {
                    "domain": name,
                    "subject": row["subject"],
                    "split": row["split"],
                    "relation": d["spec"]["relations"][j],
                    "gold_object": row["object_" + m],
                    "gold_first_token": r["tokenization"][m]["gold_text"][i],
                    "model_first_token": r["tokenization"][m]["target_text"][i],
                    "wrong_donor": r["rows"][r["donors"][i]]["subject"],
                    "donor_different_both": r["donor_different_both"][i],
                    "gold_token_identifying": r["tokenization"][m][
                        "gold_first_token_unique"
                    ][i],
                }
                for c in CONDITIONS:
                    rec[c + "_token"] = r["scores"][m]["3.0"][c]["text"][i]
                    rec[c + "_faithful"] = int(
                        r["scores"][m]["3.0"][c]["top"][i] == r["targets"][m][i]
                    )
                predictions.append(rec)
        report["domains"][name] = ds
        s = ds["subsets"]["primary/all"]
        domain_rows.append(
            {
                "domain": name,
                "test_subjects": ds["n_test"],
                **{k: v["mean"] for k, v in s["rates"].items()},
                "foreign_minus_own": s["contrasts"]["foreign_minus_own"]["mean"],
                "equivalent_10pp": s["equivalent_within_10pp"],
            }
        )
    for key, repslist in allreps.items():
        reps = {k: np.mean([x[k] for x in repslist], 0) for k in CONDITIONS}
        means = {k: float(np.mean([x[k] for x in allmeans[key]])) for k in CONDITIONS}
        summary = {
            "n_domains": len(repslist),
            "rates": {
                k: {"mean": means[k], "ci95": interval(reps[k])} for k in CONDITIONS
            },
            "contrasts": {},
        }
        for k, (a, b) in CONTRASTS.items():
            summary["contrasts"][k] = {
                "mean": means[a] - means[b],
                "ci95": interval(reps[a] - reps[b]),
                "ci90": interval(reps[a] - reps[b], 0.90),
            }
        lo, hi = summary["contrasts"]["foreign_minus_own"]["ci90"]
        summary["equivalent_within_10pp"] = bool(lo > -0.1 and hi < 0.1)
        report["macro"][key] = summary
    adj = holm([x["mcnemar_p"] for x in report["direction_tests"]])
    for x, p in zip(report["direction_tests"], adj):
        x["holm_p"] = p
    for name, rows in [("predictions", predictions), ("domain_summary", domain_rows)]:
        with (OUT / f"{name}.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    usage = {}
    calls = []
    for p in sorted((ROOT / "generated/anchor_swap_gemini/api").glob("*.json")):
        v = json.loads(p.read_text())
        if "response" not in v:
            continue
        u = v["response"].get("usageMetadata", {})
        for k, x in u.items():
            if isinstance(x, int):
                usage[k] = usage.get(k, 0) + x
        calls.append(
            {
                "file": p.name,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "model": v["response"].get("modelVersion"),
                "response_id": v["response"].get("responseId"),
                "usage": u,
            }
        )
    report["api"] = {"calls": len(calls), "token_usage": usage}
    (DATA / "API_PROVENANCE.json").write_text(
        json.dumps({"calls": calls, "totals": usage}, indent=2) + "\n"
    )
    (OUT / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "domains": domain_rows,
                "primary_macro": report["macro"]["primary/all"],
                "api": report["api"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
