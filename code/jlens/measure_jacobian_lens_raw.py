"""Standard Jacobian-lens baseline: the Jacobian, without the lens's difference.

The Jacobian is averaged over the same reference subjects and applied to the
test subject's *raw* activation rather than to its residual:

    v_e = mean_i J_i s_e ,

which is the construction of Gurnee et al. (2026), restricted to the same
reference set and neutral prompt as our lens. It is Eq. 5 with the subtraction
of the reference state dropped, so the comparison with `measure_jacobian_lens.py`
isolates what differencing contributes at fixed Jacobian cost. Subjects, layer,
and target tokens are matched to `results/jacobian_lens.json`.
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
    parser.add_argument("--out", default=str(OUTPUT_DIR / "jacobian_lens_raw.json"))
    parser.add_argument("--layer", type=int, default=LENS_LAYER)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import HiddenStateIntervention

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
    intervention = HiddenStateIntervention(model)

    output = {}
    for relation, reference in references.items():
        template, context, subjects, anchors = matched_baseline.matched_split(
            relation, reference, args.layer, ROOT
        )
        # Every unhooked forward pass must happen before the hook is installed.
        answers = matched_baseline.model_answers(
            model, tokenizer, args.device, relation, reference, template, context
        )
        cache = matched_baseline.subject_states(
            model, tokenizer, args.device, subjects, args.layer, NEUTRAL_PROMPT
        )
        handle = model.model.layers[args.layer - 1].register_forward_hook(
            intervention.hook
        )
        rows = []
        for subject in subjects:
            total = None
            n = 0
            for anchor in anchors:
                # Skip self-comparisons exactly as the differential lens does.
                if anchor == subject:
                    continue
                input_ids, position, reference_hidden = cache[anchor]
                product = intervention.jvp(
                    reference_hidden, cache[subject][2], input_ids, position
                )
                total = product if total is None else total + product
                n += 1
            token_id, text = answers[subject]
            rank, top = rank_of(total / n, token_id)
            rows.append(
                dict(
                    subject=subject,
                    model_top1=text,
                    model_top1_id=token_id,
                    n_references=n,
                    rank_raw=rank,
                    top_raw=top,
                )
            )
        handle.remove()
        output[relation] = dict(
            n=len(rows),
            layer=args.layer,
            neutral_prompt=NEUTRAL_PROMPT,
            template=template,
            legacy_split=True,
            year_trailing_space=relation in YEAR_RELATIONS,
            agg=matched_baseline.aggregate([row["rank_raw"] for row in rows]),
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
