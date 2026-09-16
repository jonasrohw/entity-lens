"""Compare free-generation object prominence with gold objects in lens top-five tokens.

A distinctive gold word matches a lens token on a prefix of at least three
characters. Report both raw prominence and a name-leakage correction."""

import argparse
import re

from scipy.stats import spearmanr

from jlens.config import OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json
from jlens.matching import distinctive


def gold_in_lens(relation, lens_results):
    data = read_json(ROOT / f"data/relations/{relation}.json")
    objects_by_subject = {
        sample["subject"]: sample["object"] for sample in data["samples"]
    }
    hit = n = 0
    for r in lens_results[relation]["rows"]:
        if r["subject"] not in objects_by_subject:
            continue
        d = distinctive(objects_by_subject[r["subject"]], r["subject"])
        if not d:
            continue
        toks = [t.strip().lower() for t in r["top_diff"]]
        hit += any(
            len(t) >= 3 and any(w.startswith(t) or t.startswith(w) for w in d)
            for t in toks
        )
        n += 1
    return (hit / n, n) if n else (None, 0)


def leaks(row):
    """Genuine name leakage: a gold word that SURVIVED `distinctive()` --- so
    it is not simply the shared surname --- yet still agrees with a subject
    word on 4 characters.  `measure_prominence.py`'s own flag tested the raw object
    and therefore fired on every `person_father` row whose son shares the
    surname, which `distinctive()` had already removed.  Recomputed here."""
    sw = [w for w in re.findall(r"[A-Za-z]+", row["subject"]) if len(w) >= 4]
    return any(
        len(w) >= 4 and any(w[:4].lower() == s[:4].lower() for s in sw)
        for w in row["words"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "prominence_statistics.json"))
    args = parser.parse_args()
    lens_results = read_json(f"{ROOT}/results/jacobian_lens.json")
    prominence_results = read_json(f"{ROOT}/results/prominence.json")
    rows = []
    for relation, p in prominence_results.items():
        if relation not in lens_results:
            continue
        gold_agreement, gold_count = gold_in_lens(relation, lens_results)
        if gold_agreement is None:
            continue
        clean = [r for r in p["rows"] if not leaks(r)]
        rows.append(
            dict(
                rel=relation,
                prom=p["prominence"],
                n_prom=p["scored"],
                leak=1 - len(clean) / len(p["rows"]),
                prom_clean=(
                    sum(r["hit"] for r in clean) / len(clean)
                    if len(clean) >= 5
                    else None
                ),
                n_clean=len(clean),
                gold_lens=gold_agreement,
                n_gl=gold_count,
            )
        )
    rows.sort(key=lambda r: -r["prom"])

    hdr = f"{'relation':24s} {'promin':>7s} {'n':>4s} {'leak':>6s} {'clean':>6s} {'nc':>4s} {'goldlens':>9s} {'n':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pc = "n/a" if r["prom_clean"] is None else f"{r['prom_clean']:.0%}"
        print(
            f"{r['rel']:24s} {r['prom']:7.0%} {r['n_prom']:4d} {r['leak']:6.0%} "
            f"{pc:>6s} {r['n_clean']:4d} {r['gold_lens']:9.0%} {r['n_gl']:4d}"
        )

    def report_correlation(label, sel, key):
        s = [r for r in rows if sel(r) and r[key] is not None]
        rho, pv = spearmanr([r[key] for r in s], [r["gold_lens"] for r in s])
        print(f"  {label:52s} rho={rho:+.3f}  p={pv:.4f}  n={len(s)}")

    print("\nSpearman, prominence vs gold-in-lens:")
    report_correlation("raw prominence, all relations", lambda r: True, "prom")
    report_correlation("leakage-corrected prominence", lambda r: True, "prom_clean")
    report_correlation(
        "raw, excluding relations with >25% leakage",
        lambda r: r["leak"] <= 0.25,
        "prom",
    )

    write_json(args.out, rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
