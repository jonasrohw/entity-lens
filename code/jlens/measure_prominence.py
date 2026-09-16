"""Measure object prominence independently of the Jacobian lens.

Generate up to 90 tokens greedily from each of two prose prompts. Record whether
identifying object words appear, plus raw name-leakage flags. Score at most 40
shared-split subjects per relation. See prominence_statistics for the corrected
leakage rule that excludes words already removed from the object name."""

import argparse

from jlens import splits
from jlens.config import MODEL_ID, OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json
from jlens.matching import distinctive, leaky

RELATIONS = [
    "city_in_country",
    "country_largest_city",
    "country_capital_city",
    "person_plays_pro_sport",
    "product_by_company",
    "landmark_in_country",
    "person_occupation",
    "person_father",
    "person_mother",
    "superhero_archnemesis",
    "company_hq",
    "star_constellation",
    "country_currency",
    "person_university",
]
# Titles, particles and generic name prefixes.  Without this, "The Joker"
# matches any generation via "the", and "King Shark"/"Doctor Doom" match on
# "king"/"doctor" -- the same collapse problem as first-token scoring, one
# level up.  HS caught this before the run.


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "prominence.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    out = {}
    for relation in RELATIONS:
        data = read_json(ROOT / f"data/relations/{relation}.json")
        S = splits.filter_identifying_objects(data["samples"], tokenizer)
        demonstrations, references, test_samples = splits.split_samples(S)
        test_samples = test_samples[:40]
        if len(test_samples) < 5:
            continue
        rows = []
        dropped = 0
        with torch.no_grad():
            for x in test_samples:
                d = distinctive(x["object"], x["subject"])
                if not d:  # nothing identifying left to look for
                    dropped += 1
                    continue
                # NB: "<X> is known for" makes this model emit a
                # multiple-choice quiz rather than prose, so the object
                # never gets named.  Two prose prompts, union of both,
                # and a longer window.
                cont = ""
                for tpl in (
                    "{} is best known for",
                    "Here are some facts about {}. First,",
                ):
                    p = tpl.format(x["subject"])
                    input_ids = tokenizer(p, return_tensors="pt").to(args.device)[
                        "input_ids"
                    ]
                    g2 = model.generate(
                        input_ids,
                        max_new_tokens=90,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    cont += " " + tokenizer.decode(g2[0, input_ids.shape[1] :]).lower()
                rows.append(
                    {
                        "subject": x["subject"],
                        "object": x["object"],
                        "words": sorted(d),
                        "hit": bool(any(t in cont for t in d)),
                        "strict": bool(all(t in cont for t in d)),
                        "leak": bool(leaky(x["object"], x["subject"])),
                    }
                )
        scored = len(rows)
        clean = [r for r in rows if not r["leak"]]
        if scored < 5:
            print(f"  {relation:30s} only {scored} scorable, skipped", flush=True)
            continue
        hit = sum(r["hit"] for r in rows)
        out[relation] = {
            "prominence": hit / scored,
            "strict": sum(r["strict"] for r in rows) / scored,
            "scored": scored,
            "dropped": dropped,
            "leak_rate": sum(r["leak"] for r in rows) / scored,
            "n_clean": len(clean),
            "prominence_clean": (
                sum(r["hit"] for r in clean) / len(clean) if len(clean) >= 5 else None
            ),
            "rows": rows,
        }
        pc = out[relation]["prominence_clean"]
        print(
            f"  {relation:30s} any-word {hit / scored:>4.0%}   leak {out[relation]['leak_rate']:>4.0%}"
            f"   clean {('n/a' if pc is None else f'{pc:.0%}'):>4s}"
            f"   (n={scored}, {dropped} dropped)",
            flush=True,
        )
    write_json(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
