"""Audit matched subjects and prepare a sourced, unmeasured paired country cohort.

Run from code/: HF_HUB_OFFLINE=1 .venv/bin/python -m jlens.paired_relation_audit
Source snapshots must already exist in output/sources; no model inference is run.
"""

import argparse
import csv
import hashlib
import itertools
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binom
from transformers import AutoTokenizer

from jlens.splits import filter_identifying_objects

MODEL = "allenai/Olmo-3-1025-7B"
REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
SEED = 2009


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def unique_samples(samples):
    values = defaultdict(set)
    for x in samples:
        values[x["subject"]].add(x["object"])
    return {s: next(iter(v)) for s, v in values.items() if len(v) == 1}


def normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value).casefold() if c.isalnum())


def paired_power(n, difference, discordance, alpha=0.05):
    """Exact two-sided McNemar power under independent identically distributed pairs.

    discordance = P(A succeeds, B fails) + P(B succeeds, A fails).
    difference = P(A succeeds, B fails) - P(B succeeds, A fails).
    Condition on discordant count, then average its binomial distribution.
    """
    assert 0 < difference <= discordance <= 1
    probability_a = (discordance + difference) / (2 * discordance)
    power = 0.0
    for m in range(1, n + 1):
        k = list(range(m + 1))
        pvalues = 2 * binom.cdf([min(j, m - j) for j in k], m, 0.5)
        reject = pvalues <= alpha
        power += binom.pmf(m, n, discordance) * binom.pmf(k, m, probability_a)[reject].sum()
    return float(power)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    out = args.output or root / "generated/paired_relation_feasibility_20260920"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    paths = sorted((root / "data/relations").glob("*.json"))
    data = {p.stem: json.loads(p.read_text()) for p in paths}
    raw = {n: unique_samples(d["samples"]) for n, d in data.items()}
    filtered = {
        n: unique_samples(filter_identifying_objects(d["samples"], tok, n.startswith("presidents_")))
        for n, d in data.items()
    }
    previous = root / "generated/mechanistic_20260919_rho2"
    with (previous / "analysis/by_relation.csv").open() as handle:
        metrics = {r["relation"]: r for r in csv.DictReader(handle) if r["selection"] == "17"}
    results = {
        n: json.loads((previous / f"{n}.json").read_text())
        for n in data if (previous / f"{n}.json").exists()
    }
    measured = {}
    for name, d in results.items():
        layer = d["layers"]["17"]
        scores = layer["relation_grids"][str(layer["beta"])]
        measured[name] = {
            x["subject"]: {
                "faithful": scores["hit"][i], "model_correct": d["model_correct"][i],
                "dataset_object": x["object"], "model_token": tok.decode([d["targets"][i]]),
                "lre_token": tok.decode([scores["top"][i]]),
            }
            for i, x in enumerate(d["subjects"]) if i >= d["nval"]
        }
    overlap, examples = [], []
    for a, b in itertools.combinations(sorted(data), 2):
        common = sorted(filtered[a].keys() & filtered[b].keys())
        raw_common = raw[a].keys() & raw[b].keys()
        tested = sorted(measured.get(a, {}).keys() & measured.get(b, {}).keys())
        if raw_common:
            overlap.append({
                "relation_a": a, "relation_b": b, "raw_overlap": len(raw_common),
                "identifying_overlap": len(common), "previous_common_test": len(tested),
                "previous_unmatched_faith_a": metrics.get(a, {}).get("faith", ""),
                "previous_unmatched_faith_b": metrics.get(b, {}).get("faith", ""),
            })
        if common:
            write_json(out / "existing_pairs" / f"{a}__{b}.json", [
                {"subject": s, a: filtered[a][s], b: filtered[b][s]} for s in common
            ])
        for s in tested:
            examples.append({"subject": s, "relation_a": a, "relation_b": b,
                             "a": measured[a][s], "b": measured[b][s]})
    with (out / "overlap.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlap[0]))
        writer.writeheader()
        writer.writerows(overlap)
    write_json(out / "existing_matched_measurements.json", examples)
    cases = []
    for name, subject in [
        ("country_capital_city", "Saudi Arabia"), ("country_language", "Saudi Arabia"),
        ("country_currency", "Australia"), ("country_largest_city", "Australia"),
    ]:
        d = results[name]
        i = next(i for i, x in enumerate(d["subjects"]) if x["subject"] == subject)
        j = i - d["nval"]
        assert j >= 0
        def selected_score(score, index):
            return {k: score[k][index] for k in ["hit", "logp", "margin"]}
        for layer_id, layer in d["layers"].items():
            cases.append({
                "relation": name, "subject": subject, "layer": int(layer_id), "beta": layer["beta"],
                "lre": selected_score(layer["relation_grids"][str(layer["beta"])], i),
                "neutral_transplant_lre": selected_score(layer["transplant_lre"], i),
                "component_lre": {c: selected_score(layer["component_lre"][c], j) for c in ["attention", "mlp"]},
                "full_model": {c: selected_score(layer["full_model"][c], j) for c in ["clean", "attention", "mlp"]},
            })
    write_json(out / "case_studies.json", cases)

    # Sources are joined by ISO code. Cross-source disagreements are excluded,
    # not silently resolved. This is a candidate dataset, not a factuality audit.
    source_dir = out / "sources"
    secondary = {x["cca2"]: x for x in json.loads((source_dir / "countries.json").read_text())}
    rows, excluded = [], []
    old_subjects = {
        normalized(x["subject"])
        for n, d in data.items() if n.startswith("country_") for x in d["samples"]
    }
    for line in (source_dir / "countryInfo.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 17:
            continue
        iso, subject, capital, code, currency = (fields[i].strip() for i in [0, 4, 5, 10, 11])
        other = secondary.get(iso)
        reason = None
        if not other or not other.get("unMember"):
            reason = "not a UN member in secondary source"
        elif not capital or not code or not currency:
            reason = "missing field"
        elif len(other["capital"]) != 1 or len(other["currencies"]) != 1:
            reason = "multiple capitals or currencies in secondary source"
        elif normalized(capital) != normalized(other["capital"][0]):
            reason = "capital sources disagree (including unresolved spelling variants)"
        elif code not in other["currencies"]:
            reason = "currency codes disagree"
        if reason:
            excluded.append({"iso": iso, "subject": subject, "reason": reason})
            continue
        aliases = [subject, other["name"]["common"], other["name"]["official"], *other.get("altSpellings", [])]
        rows.append({
            "subject": subject, "iso": iso, "geonames_id": fields[16],
            "capital": capital, "currency": currency, "currency_code": code,
            "currency_full_name_secondary": other["currencies"][code]["name"],
            "previous_country_dataset_subject": any(normalized(s) in old_subjects for s in aliases),
        })
    n_source_agreement = len(rows)
    valid = []
    for field in ["capital", "currency"]:
        sample = [{"subject": r["subject"], "object": r[field]} for r in rows]
        valid.append({x["subject"] for x in filter_identifying_objects(sample, tok)})
    keep = valid[0] & valid[1]
    for row in rows:
        if row["subject"] not in keep:
            excluded.append({"iso": row["iso"], "subject": row["subject"], "reason": "non-identifying first token"})
    rows = sorted((r for r in rows if r["subject"] in keep), key=lambda r: r["iso"])
    assert len({r["subject"] for r in rows}) == len(rows)
    assert len({r["iso"] for r in rows}) == len(rows)
    old = [r for r in rows if r["previous_country_dataset_subject"]]
    fresh = [r for r in rows if not r["previous_country_dataset_subject"]]
    random.Random(SEED).shuffle(old)
    random.Random(SEED + 1).shuffle(fresh)
    # Previously available countries stay outside confirmation. Every relation
    # receives exactly the same subject partitions, including demonstrations.
    development = old + fresh
    n_development = max(len(old), 8 + 8 + 24 + 16)
    dev, test = development[:n_development], development[n_development:]
    splits = {"icl": dev[:8], "references": dev[8:16], "train": dev[16:40],
              "validation": dev[40:], "test": test}
    flat = [r["iso"] for p in splits.values() for r in p]
    assert len(flat) == len(set(flat)) == len(rows)
    assert all(not r["previous_country_dataset_subject"] for r in test)
    relations = {
        "country_capital_city": ("capital", "The capital city of {} is", "{} has the capital city of"),
        "country_currency": ("currency", "The official currency of {} is the", "{} uses the currency called"),
    }
    plans = {}
    for name, (field, template, causal_template) in relations.items():
        samples = [{"subject": r["subject"], "object": r[field]} for r in rows]
        write_json(out / "relations" / f"{name}.json", {
            "name": name, "prompt_templates": [template],
            "prompt_templates_subject_first": [causal_template],
            "properties": {"domain_name": "country", "range_name": field, "relation_type": "factual", "symmetric": False},
            "samples": samples,
        })
        plans[name] = {
            "template": template, "subject_first_template": causal_template,
            "parts": {p: [{"subject": r["subject"], "object": r[field]} for r in items] for p, items in splits.items()},
        }
    write_json(out / "country_pairs.json", rows)
    write_json(out / "excluded.json", excluded)
    write_json(out / "paired_manifest.json", {
        "status": "prepared; not evaluated by the model", "model": MODEL, "revision": REVISION,
        "seed": SEED, "layers": [5, 11, 17, 23], "relations": plans,
        "note": "Do not run make_manifest independently per relation; it would destroy the paired split.",
    })
    assumptions = [(0.2, 0.4), (0.2, 0.6), (0.3, 0.5)]
    power = [{"n": n, "difference": d, "discordance": q, "power": paired_power(n, d, q)}
             for n in sorted({len(test), 100, 150}) for d, q in assumptions]
    source_urls = {
        "countryInfo.txt": "https://download.geonames.org/export/dump/countryInfo.txt",
        "countries.json": "https://raw.githubusercontent.com/mledoze/countries/master/countries.json",
    }
    summary = {
        "source_agreement_countries": n_source_agreement,
        "identifying_paired_countries": len(rows), "previous_country_subjects": len(old),
        "splits": {k: len(v) for k, v in splits.items()},
        "test_currency_counts": dict(Counter(r["currency"] for r in test)),
        "power_assumptions": power,
        "cautions": [
            "No new model predictions or mechanistic interventions have been run.",
            "Source agreement is not independent verification or alignment to model training date.",
            "Currency labels are generic units (Dollar, Franc), consistent with existing data; they do not identify a unique ISO currency.",
            "First-token uniqueness is assessed within each relation, not across the two relations.",
            "Power assumes independent subject pairs and a single planned comparison; not power for circuit localization or a guaranteed effect.",
            "Conditioning on both model answers being correct will reduce confirmation sample size.",
        ],
        "sources": {name: {"url": url, "retrieved": "2026-09-20", "sha256": hashlib.sha256((source_dir/name).read_bytes()).hexdigest()}
                    for name, url in source_urls.items()},
        "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["input_sha256", "sources", "test_currency_counts"]}, indent=2))


if __name__ == "__main__":
    main()
