"""Relation-agnostic logit-lens baseline for the paper's 28 relations.

At the neutral prompt and the configured lens layer, apply the model's final
RMS norm and unembedding directly to the subject's hidden state. Score the
rank of the same model-answer token used by the differential Jacobian lens.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT, OUTPUT_DIR, ROOT

YEARS = {"presidents_birth_year", "presidents_election_year"}


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUTPUT_DIR / "logit_lens.json"))
    parser.add_argument("--layer", type=int, default=LENS_LAYER)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    main_ref = json.loads((ROOT / "results/jacobian_lens.json").read_text())
    year_ref = json.loads((ROOT / "results/jacobian_lens_years.json").read_text())
    relations = sorted(r for r in main_ref if not r.startswith("_"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="eager"
        )
        .to(args.device)
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    norm_weight = model.model.norm.weight.detach().float()
    unembed = model.lm_head.weight.detach().float()
    norm_epsilon = float(getattr(model.model.norm, "variance_epsilon", 1e-5))

    def rank_of(hidden, token_id):
        vector = hidden.float()
        vector = vector / torch.sqrt(vector.pow(2).mean() + norm_epsilon)
        logits = (norm_weight * vector) @ unembed.T
        rank = int((logits > logits[token_id]).sum())
        top = [tokenizer.decode([int(i)]).strip() for i in logits.topk(5).indices]
        return rank, top

    output = {}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    for relation in relations:
        reference = year_ref[relation] if relation in YEARS else main_ref[relation]
        if reference["layer"] != args.layer:
            raise ValueError(f"{relation}: reference layer differs from baseline")
        data = json.loads((ROOT / f"data/relations/{relation}.json").read_text())
        template = data["prompt_templates"][0]
        if template != reference["template"]:
            raise ValueError(f"{relation}: prompt template differs from reference")
        pool = list(data["samples"])
        random.Random(17).shuffle(pool)
        demonstrations = pool[:8]
        subjects = [row["subject"] for row in pool[8:28]]
        expected = [row["subject"] for row in reference["rows"]]
        if subjects != expected:
            raise ValueError(f"{relation}: scored subjects differ from reference")
        context = "".join(
            template.format(row["subject"]) + " " + row["object"] + "\n"
            for row in demonstrations
        )
        rows = []
        with torch.no_grad():
            for ref_row in reference["rows"]:
                subject = ref_row["subject"]
                target_prompt = context + template.format(subject)
                if relation in YEARS:
                    target_prompt += " "
                target_ids = tokenizer(target_prompt, return_tensors="pt").to(
                    args.device
                )["input_ids"]
                target_id = int(model(input_ids=target_ids).logits[0, -1].argmax())
                target_text = tokenizer.decode([target_id]).strip()
                if target_text != ref_row["model_top1"]:
                    raise ValueError(
                        f"{relation}/{subject}: model answer {target_text!r} "
                        f"differs from reference {ref_row['model_top1']!r}"
                    )

                neutral_prompt = NEUTRAL_PROMPT.format(subject)
                neutral_ids = tokenizer(neutral_prompt, return_tensors="pt").to(
                    args.device
                )["input_ids"]
                end = neutral_prompt.rindex(subject) + len(subject)
                position = min(
                    len(tokenizer(neutral_prompt[:end]).input_ids) - 1,
                    neutral_ids.shape[1] - 1,
                )
                hidden = model(
                    input_ids=neutral_ids, output_hidden_states=True
                ).hidden_states[args.layer][0, position, :]
                rank, top = rank_of(hidden, target_id)
                rows.append(
                    dict(
                        subject=subject,
                        model_top1=target_text,
                        model_top1_id=target_id,
                        rank_logit=rank,
                        top_logit=top,
                    )
                )
        ranks = np.array([row["rank_logit"] for row in rows], dtype=float)
        aggregate = dict(
            median_rank=float(np.median(ranks)),
            median_lograng=float(np.median(np.log10(ranks + 1))),
            agree_at1=float((ranks == 0).mean()),
            agree_at10=float((ranks < 10).mean()),
        )
        output[relation] = dict(
            n=len(rows),
            layer=args.layer,
            neutral_prompt=NEUTRAL_PROMPT,
            legacy_split=True,
            year_trailing_space=relation in YEARS,
            agg=aggregate,
            rows=rows,
        )
        path.write_text(json.dumps(output, indent=1, ensure_ascii=False) + "\n")
        print(
            f"{relation:28s} n={len(rows):>2} "
            f"median_lograng={aggregate['median_lograng']:.3f}",
            flush=True,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
