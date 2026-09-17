"""Differential logit-lens baseline: the lens's difference, without its Jacobian.

At the neutral prompt and the configured lens layer, subtract the mean reference subject's
hidden state from the test subject's and decode the residual directly with the
model's final RMS norm and unembedding:

    v_e = s_e - mean_i s_i .

This is Eq. 5 with each Jacobian replaced by the identity, so the comparison
with `measure_jacobian_lens.py` isolates what the Jacobian contributes, and the
comparison with `measure_logit_lens.py` isolates what differencing contributes.
Subjects, layer, and target tokens are matched to `results/jacobian_lens.json`.
"""

import argparse

from jlens import matched_baseline
from jlens.config import (
    LENS_LAYER,
    MODEL_ID,
    NEUTRAL_PROMPT,
    OUTPUT_DIR,
    ROOT,
    YEAR_RELATIONS,
)
from jlens.io import write_json


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUTPUT_DIR / "logit_lens_diff.json"))
    parser.add_argument("--layer", type=int, default=LENS_LAYER)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    references = matched_baseline.load_references(ROOT)
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
    rank_of = matched_baseline.make_decoder(model, tokenizer)

    output = {}
    for relation, reference in references.items():
        template, context, subjects, anchors = matched_baseline.matched_split(
            relation, reference, args.layer, ROOT
        )
        answers = matched_baseline.model_answers(
            model, tokenizer, args.device, relation, reference, template, context
        )
        cache = matched_baseline.subject_states(
            model, tokenizer, args.device, subjects, args.layer, NEUTRAL_PROMPT
        )
        rows = []
        for subject in subjects:
            # Skip self-comparisons exactly as the differential lens does.
            others = [cache[a][2] for a in anchors if a != subject]
            residual = cache[subject][2] - sum(others) / len(others)
            token_id, text = answers[subject]
            rank, top = rank_of(residual, token_id)
            rows.append(
                dict(
                    subject=subject,
                    model_top1=text,
                    model_top1_id=token_id,
                    n_references=len(others),
                    rank_logit_diff=rank,
                    top_logit_diff=top,
                )
            )
        output[relation] = dict(
            n=len(rows),
            layer=args.layer,
            neutral_prompt=NEUTRAL_PROMPT,
            template=template,
            legacy_split=True,
            year_trailing_space=relation in YEAR_RELATIONS,
            agg=matched_baseline.aggregate([row["rank_logit_diff"] for row in rows]),
            rows=rows,
        )
        write_json(args.out, output)
        print(
            f"{relation:28s} n={len(rows):>2} "
            f"median_lograng={output[relation]['agg']['median_lograng']:.3f}",
            flush=True,
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
