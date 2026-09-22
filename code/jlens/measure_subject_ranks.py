"""Measure per-subject LRE ranks at an already-selected layer and beta.

Reconstruct the shared split, check its fingerprint, and verify that rank zero
reproduces every stored hit/miss outcome before saving continuous ranks."""

import argparse
import os

import numpy as np

from jlens import splits
from jlens.config import MODEL_ID, OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation", required=True)
    parser.add_argument("--data-dir", default=f"{ROOT}/data/relations")
    parser.add_argument("--faith-dir", default=f"{ROOT}/results/shared_split")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--trailing-space", action="store_true")
    parser.add_argument("--distinct-first-token", action="store_true", default=True)
    parser.add_argument(
        "--no-distinct-first-token",
        dest="distinct_first_token",
        action="store_false",
        help="score every object, colliding first tokens included. "
        "This is the protocol behind the paper's Figure 1, and "
        "the only one the smallest relations can satisfy: with "
        "the filter on, the two year relations retain 10 and 8 "
        "samples against the 17 a split needs.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-icl", type=int, default=8)
    parser.add_argument("--n-ref", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import (
        HiddenStateIntervention,
        final_logits,
        forward_with_raw_hidden_states,
        resolve_blocks,
        resolve_final_norm,
    )

    device = args.device

    faithfulness = read_json(f"{args.faith_dir}/faithfulness_{args.relation}.json")
    selected_layer, selected_beta = (
        int(faithfulness["best_layer"]),
        float(faithfulness["best_beta"]),
    )
    print(
        f"{args.relation}: selected cell layer {selected_layer}, beta {selected_beta}, "
        f"faith {faithfulness['best_faith']:.3f}",
        flush=True,
    )

    relation_data = read_json(f"{args.data_dir}/{args.relation}.json")
    samples = relation_data["samples"]
    template = relation_data["prompt_templates"][0]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if args.distinct_first_token:
        samples = splits.filter_identifying_objects(
            samples, tokenizer, args.trailing_space
        )
    demonstrations, references, test_samples = splits.split_samples(
        samples,
        seed=args.split_seed,
        n_icl=args.n_icl,
        n_ref=args.n_ref,
        n_test=args.n_test,
    )
    split_fingerprint = splits.split_fingerprint(
        demonstrations, references, test_samples
    )
    if split_fingerprint != faithfulness.get("split_fingerprint"):
        raise SystemExit(
            f"FATAL {args.relation}: split fingerprint {split_fingerprint} != stored "
            f"{faithfulness.get('split_fingerprint')}; not the same subjects."
        )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    intervention = HiddenStateIntervention(model, final_block=resolve_blocks(model)[-1])

    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias = lm_bias.detach() if lm_bias is not None else None
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in demonstrations
    )

    def prompt(sub):
        return context + template.format(sub) + (" " if args.trailing_space else "")

    def sub_pos(sub):
        p = prompt(sub)
        i = p.rindex(sub) + len(sub)
        return len(tokenizer(p[:i]).input_ids) - 1

    cache = {}
    model_top_tokens = {}
    with torch.no_grad():
        for x in references + test_samples:
            sub = x["subject"]
            input_ids = tokenizer(prompt(sub), return_tensors="pt").to(device)[
                "input_ids"
            ]
            subject_position = min(sub_pos(sub), input_ids.shape[1] - 1)
            out, hidden_states = forward_with_raw_hidden_states(model, input_ids)
            cache[sub] = (
                input_ids,
                subject_position,
                hidden_states[selected_layer][0, subject_position, :].detach().clone(),
            )
            model_top_tokens[sub] = int(out.logits[0, -1].argmax())

    handle = resolve_blocks(model)[selected_layer - 1].register_forward_hook(
        intervention.hook
    )
    with torch.no_grad():
        base = torch.stack(
            [
                intervention.forward(
                    cache[x["subject"]][2],
                    cache[x["subject"]][0],
                    cache[x["subject"]][1],
                ).detach()
                for x in references
            ]
        ).mean(0)
    rows = {}
    for x in test_samples:
        sub = x["subject"]
        subject_hidden = cache[sub][2]
        acc = torch.zeros_like(base)
        for reference_sample in references:
            reference_subject = reference_sample["subject"]
            reference_hidden = cache[reference_subject][2]
            acc = acc + intervention.jvp(
                reference_hidden,
                (subject_hidden - reference_hidden),
                cache[reference_subject][0],
                cache[reference_subject][1],
            )
        predicted_hidden = base + selected_beta * (acc / len(references))
        logits = final_logits(predicted_hidden, norm, unembedding, lm_bias, softcap)
        t = model_top_tokens[sub]
        rows[sub] = {
            "rank": int((logits > logits[t]).sum()),
            "lre_top1": tokenizer.decode([int(logits.argmax())]).strip(),
            "model_top1": tokenizer.decode([t]).strip(),
        }
    handle.remove()

    # The check that makes this trustworthy: rank 0 must reproduce the
    # stored per-subject hits, subject for subject.
    stored_outcomes = faithfulness["per_subject"]
    mismatches = [
        s
        for s in rows
        if s in stored_outcomes and int(rows[s]["rank"] == 0) != int(stored_outcomes[s])
    ]
    measured_faithfulness = np.mean([rows[s]["rank"] == 0 for s in rows])
    print(
        f"  rank==0 on {measured_faithfulness:.3f} of subjects; stored faithfulness "
        f"{faithfulness['best_faith']:.3f}; {len(mismatches)} per-subject mismatches",
        flush=True,
    )
    if mismatches:
        raise SystemExit(
            f"FATAL {args.relation}: {len(mismatches)} subjects disagree with "
            f"the stored outcome, e.g. {mismatches[:5]}. The operator or "
            f"the split has drifted; ranks are not usable."
        )

    out = {
        "relation": args.relation,
        "layer": selected_layer,
        "beta": selected_beta,
        "split_fingerprint": split_fingerprint,
        "best_faith": faithfulness["best_faith"],
        "trailing_space": args.trailing_space,
        "ranks": rows,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = f"{args.out_dir}/subject_ranks_{args.relation}.json"
    write_json(path, out)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
