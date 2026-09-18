"""Relation-level sanity check: does a RELATION-AGNOSTIC entity J-lens predict
how linear a relation is?

x (new)  per relation: how well the lens's ranked list agrees with the MODEL's
         own top-1.  The lens sees only the subject at a neutral prompt --
         no relation prompt, no objects, no (subject,object) pairs, and a
         fixed layer for every relation (tuning the layer per relation would
         itself be relation-specific).
y (have) per relation: LRE faithfulness, already measured.

Relation-specificity lives ONLY in the target: the model's top-1 is taken
under the same 8-shot ICL prompt that the faithfulness runs used, so the two
quantities are commensurable.

Two lens forms, same cost:
  diff : v = mean_i J_i (s_e - s_i)                  pure difference readout
  lre  : v = mean_i [ f(s_i) + J_i (s_e - s_i) ]     affine (+zeroth-order) baseline
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np

from jlens import splits
from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT, OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json

LAYER = LENS_LAYER
N_REFERENCES = 8
N_SUBJECTS = 20
N_ICL = 8


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--out", default=str(OUTPUT_DIR / "jacobian_lens.json"))
    parser.add_argument("--rels", default="", help="comma-list; merges into --out")
    parser.add_argument("--n-test", type=int, default=splits.N_TEST)
    parser.add_argument(
        "--legacy-split",
        action="store_true",
        help="reproduce the pre-alignment split: Random(17), anchors drawn "
        "from the scored subjects.  The paper's numbers used this.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--distinct-first-token",
        action="store_true",
        help="see measure_faithfulness.py; must match that run or the split "
        "fingerprints will differ and the analysis will refuse to run.",
    )
    parser.add_argument(
        "--model-correct",
        action="store_true",
        help="keep only the triples the model answers correctly; must match "
        "the other run or the split fingerprints will differ.",
    )
    parser.add_argument("--model-correct-dir", default=f"{ROOT}/results/model_correct")
    parser.add_argument(
        "--trailing-space",
        action="store_true",
        help="append a space to the ICL query so the model's top-1 is a CONTENT "
        "token rather than the bare space that precedes every number.  "
        "Affects only the TARGET; the neutral lens prompt is unchanged.",
    )
    parser.add_argument("--data-dir", default=f"{ROOT}/data/relations")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import (
        HiddenStateIntervention,
        final_logits,
        forward_capturing_final_block,
        resolve_blocks,
        resolve_final_norm,
    )

    rels = (
        args.rels.split(",")
        if args.rels
        else sorted(path.stem for path in Path(args.data_dir).glob("*.json"))
    )
    layer = args.layer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="eager"
        )
        .to(args.device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach().float()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias = lm_bias.detach().float() if lm_bias is not None else None
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    final_block = resolve_blocks(model)[-1]
    intervention = HiddenStateIntervention(model, final_block=final_block)

    def rank_of(v, tid):
        lg = final_logits(v, norm, unembedding, lm_bias, softcap)
        return int((lg > lg[tid]).sum()), [
            tokenizer.decode([int(i)]).strip() for i in lg.topk(5).indices
        ]

    results = read_json(args.out) if (args.rels and os.path.exists(args.out)) else {}
    for relation in rels:
        relation_data = read_json(f"{args.data_dir}/{relation}.json")
        template = relation_data["prompt_templates"][0]
        samples = relation_data["samples"]
        if args.distinct_first_token:
            samples = splits.filter_identifying_objects(
                samples, tokenizer, args.trailing_space
            )
        if args.legacy_split:
            pool = list(samples)
            random.Random(17).shuffle(pool)
            demonstrations = pool[:N_ICL]
            rest = pool[N_ICL:]
            subjects = [x["subject"] for x in rest[:N_SUBJECTS]]
            anchor_names = subjects[:N_REFERENCES]  # anchors inside the scored set
            split_fingerprint = None
        elif args.model_correct:
            demonstrations, references, test_samples = (
                splits.split_model_correct_samples(
                    samples,
                    relation,
                    args.model_correct_dir,
                    seed=args.split_seed,
                    n_icl=N_ICL,
                    n_ref=N_REFERENCES,
                    n_test=args.n_test,
                )
            )
        else:
            demonstrations, references, test_samples = splits.split_samples(
                samples,
                seed=args.split_seed,
                n_icl=N_ICL,
                n_ref=N_REFERENCES,
                n_test=args.n_test,
            )
        if not args.legacy_split:
            subjects = [x["subject"] for x in test_samples]
            anchor_names = [x["subject"] for x in references]  # disjoint from `subs`
            split_fingerprint = splits.split_fingerprint(
                demonstrations, references, test_samples
            )
        if len(subjects) < 2:
            print(
                f"!! {relation}: too few samples after filtering "
                f"({len(subjects)} scorable subjects), skipping",
                flush=True,
            )
            continue
        context = "".join(
            template.format(x["subject"]) + " " + x["object"] + "\n"
            for x in demonstrations
        )
        # --- target: the MODEL's own top-1 under the relation's ICL prompt ---
        target_tokens = {}
        with torch.no_grad():
            for s in subjects:
                input_ids = tokenizer(
                    context + template.format(s) + (" " if args.trailing_space else ""),
                    return_tensors="pt",
                ).to(args.device)["input_ids"]
                target_tokens[s] = int(
                    model(input_ids=input_ids).logits[0, -1].argmax()
                )
        # --- lens: neutral prompt only ---
        cache = {}
        with torch.no_grad():
            for subject in list(dict.fromkeys(subjects + anchor_names)):
                p = NEUTRAL_PROMPT.format(subject)
                input_ids = tokenizer(p, return_tensors="pt").to(args.device)[
                    "input_ids"
                ]
                i = p.rindex(subject) + len(subject)
                subject_position = min(
                    len(tokenizer(p[:i]).input_ids) - 1, input_ids.shape[1] - 1
                )
                o, raw_final = forward_capturing_final_block(
                    model, final_block, input_ids, output_hidden_states=True
                )
                cache[subject] = (
                    input_ids,
                    subject_position,
                    o.hidden_states[layer][0, subject_position, :].detach().clone(),
                    raw_final.detach().clone(),
                )
        handle = resolve_blocks(model)[layer - 1].register_forward_hook(
            intervention.hook
        )
        anchors = anchor_names
        rows = []
        for subject in subjects:
            difference_sum = None
            linear_sum = None
            n = 0
            for anchor in anchors:
                if anchor == subject:
                    continue
                input_ids, subject_position, reference_hidden, reference_output = cache[
                    anchor
                ]
                v = intervention.jvp(
                    reference_hidden,
                    cache[subject][2] - reference_hidden,
                    input_ids,
                    subject_position,
                )
                difference_sum = v if difference_sum is None else difference_sum + v
                w = v + reference_output
                linear_sum = w if linear_sum is None else linear_sum + w
                n += 1
            difference_rank, difference_top = rank_of(
                difference_sum / n, target_tokens[subject]
            )
            linear_rank, linear_top = rank_of(linear_sum / n, target_tokens[subject])
            rows.append(
                dict(
                    subject=subject,
                    model_top1=tokenizer.decode([target_tokens[subject]]).strip(),
                    rank_diff=difference_rank,
                    rank_lre=linear_rank,
                    top_diff=difference_top,
                    top_lre=linear_top,
                )
            )
        handle.remove()
        aggregate = {}
        for k in ("diff", "lre"):
            r = np.array([x["rank_" + k] for x in rows], dtype=float)
            aggregate[k] = dict(
                median_rank=float(np.median(r)),
                median_lograng=float(np.median(np.log10(r + 1))),
                agree_at1=float((r == 0).mean()),
                agree_at10=float((r < 10).mean()),
            )
        results[relation] = dict(
            n=len(subjects),
            layer=layer,
            template=template,
            agg=aggregate,
            rows=rows,
            split_fingerprint=split_fingerprint,
            legacy_split=bool(args.legacy_split),
        )
        print(
            f"{relation:26s} n={len(subjects):<3} diff: med_rank={aggregate['diff']['median_rank']:>9.0f} "
            f"a@1={aggregate['diff']['agree_at1']:.2f} | affine (+zeroth-order): med_rank={aggregate['lre']['median_rank']:>9.0f} "
            f"a@1={aggregate['lre']['agree_at1']:.2f}",
            flush=True,
        )
    write_json(args.out, results)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
