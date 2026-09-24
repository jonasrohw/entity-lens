"""Per-relation object-frequency covariate.

Proxy = the MODEL's own prior mass on the answer token, averaged over
several neutral contexts.  That is the confound we care about: "the LRE
hits it because the token is a priori likely."  Reported for both the
model's top-1 (what x and y are actually scored against) and the gold
object.

Protocol below mirrors jlens/measure_jacobian_lens.py exactly (same seed, same
ICL split), so the model-top-1 ids line up with that run.
"""

import argparse
import random

import numpy as np

from jlens.config import MODEL_ID, OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json

N_SUBJECTS = 20
N_ICL = 8
NEUTRAL_CONTEXTS = [
    "",
    "The",
    "In the",
    "It was",
    "She said the",
    "Here is",
    "One of the",
    "According to the",
    "There is a",
    "Yesterday the",
]


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "answer_prior.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    RELATIONS = sorted(p.stem for p in (ROOT / "data/relations").glob("*.json"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def first_token(s):
        return tokenizer.encode(" " + s, add_special_tokens=False)[0]

    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    # --- neutral prior over the whole vocabulary ---
    log_probabilities = []
    with torch.no_grad():
        for c in NEUTRAL_CONTEXTS:
            input_ids = tokenizer(
                c if c else tokenizer.bos_token or "<|endoftext|>", return_tensors="pt"
            ).to(args.device)["input_ids"]
            logits = model(input_ids=input_ids).logits[0, -1].float()
            log_probabilities.append(torch.log_softmax(logits, -1))
    prior = torch.stack(log_probabilities).mean(0)  # mean log-prob per token
    print(
        f"prior over {prior.shape[0]} tokens from {len(NEUTRAL_CONTEXTS)} contexts",
        flush=True,
    )
    out = {}
    for relation in RELATIONS:
        relation_data = read_json(ROOT / f"data/relations/{relation}.json")
        template = relation_data["prompt_templates"][0]
        pool = list(relation_data["samples"])
        random.Random(17).shuffle(pool)
        demonstrations = pool[:N_ICL]
        rest = pool[N_ICL:][:N_SUBJECTS]
        if len(rest) < 9:
            print(f"!! {relation} too small")
            continue
        context = "".join(
            template.format(x["subject"]) + " " + x["object"] + "\n"
            for x in demonstrations
        )
        model_priors = []
        gold_priors = []
        with torch.no_grad():
            for x in rest:
                ts = " " if relation.startswith("presidents_") else ""
                input_ids = tokenizer(
                    context + template.format(x["subject"]) + ts, return_tensors="pt"
                ).to(args.device)["input_ids"]
                m = int(model(input_ids=input_ids).logits[0, -1].argmax())
                model_priors.append(float(prior[m]))
                gold_priors.append(
                    float(
                        prior[
                            (
                                tokenizer.encode(x["object"], add_special_tokens=False)[
                                    0
                                ]
                                if ts
                                else first_token(x["object"])
                            )
                        ]
                    )
                )
        out[relation] = dict(
            n=len(rest),
            prior_modeltop1=float(np.mean(model_priors)),
            prior_gold=float(np.mean(gold_priors)),
        )
        print(
            f"{relation:26s} n={len(rest):<3} prior(model_top1)={np.mean(model_priors):>8.2f} "
            f"prior(gold)={np.mean(gold_priors):>8.2f}",
            flush=True,
        )
    write_json(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
