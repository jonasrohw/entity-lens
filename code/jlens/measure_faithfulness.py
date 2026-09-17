"""Measure LRE faithfulness on the configured model using a layer-by-beta sweep.

Faithfulness is agreement with the full model's top-1 token over the entire
vocabulary. For each test subject, compute mean_i[f(s_i) + beta J_i(s - s_i)]
using Jacobian-vector products. Average each grid cell over trials before
selecting its maximum. The retained paper runs use one trial; this differs
from the 24-trial protocol described by Hernandez et al. (2024).

Double backward requires eager attention. Float32 is the paper default."""

import argparse
import random

import numpy as np

from jlens import splits
from jlens.config import LRE_LAYERS, MODEL_ID, OUTPUT_DIR, ROOT
from jlens.io import read_json, write_json


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation", required=True)
    parser.add_argument("--data-dir", default=f"{ROOT}/data/relations")
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--trailing-space",
        action="store_true",
        help="append a space to the query and score the object's CONTENT first "
        'token.  Needed for numeric objects: " 1743" tokenizes as '
        "[space,'174','3'], so without this both the model top-1 and the "
        "gold are the bare space token 220.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--legacy-split",
        action="store_true",
        help="reproduce the pre-alignment split (Random(0) then Random(1000+t)); "
        "the paper's current numbers were produced this way",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--distinct-first-token",
        action="store_true",
        help="score only objects whose first token identifies them within "
        "the relation.  An LRE emits one vector, so it cannot tell "
        "'University of Chicago' from 'University of Michigan'; where "
        "first tokens collide, agreement measures nothing.",
    )
    parser.add_argument(
        "--model-correct",
        action="store_true",
        help="keep only the triples the model answers correctly; must match "
        "the other run or the split fingerprints will differ.",
    )
    parser.add_argument("--model-correct-dir", default=f"{ROOT}/results/model_correct")
    parser.add_argument("--n-ref", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--n-icl", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default=LRE_LAYERS)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import (
        HiddenStateIntervention,
        final_logits,
        resolve_blocks,
        resolve_final_norm,
    )

    device = args.device
    relation_data = read_json(f"{args.data_dir}/{args.relation}.json")
    samples = relation_data["samples"]
    template = relation_data["prompt_templates"][0]
    print(
        f"{args.relation}: {len(samples)} samples, "
        f"{len(set(s['object'] for s in samples))} distinct objects",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if args.distinct_first_token:
        n0 = len(samples)
        samples = splits.filter_identifying_objects(
            samples, tokenizer, args.trailing_space
        )
        print(
            f"  distinct-first-token filter: {n0} -> {len(samples)} samples", flush=True
        )
    need = args.n_icl + args.n_ref + 1
    if len(samples) < need:
        print(
            f"SKIP {args.relation}: too few samples ({len(samples)}) for "
            f"n_icl={args.n_icl}+n_ref={args.n_ref}+1={need}; nothing to score.",
            flush=True,
        )
        write_json(
            f"{args.out_dir}/SKIPPED_{args.relation}.json",
            dict(
                relation=args.relation,
                skipped="too_few_samples",
                n_samples=len(samples),
                needed=need,
            ),
        )
        return
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=getattr(torch, args.dtype), attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)

    layers = [int(x) for x in args.layers.split(",")]
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    intervention = HiddenStateIntervention(model, final_block=resolve_blocks(model)[-1])

    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias = lm_bias.detach() if lm_bias is not None else None
    trials = np.zeros((args.trials, len(layers), len(betas)))
    subject_outcomes = {}
    split_fp = None
    correct_counts = []
    handle = None
    for trial in range(args.trials):
        if args.legacy_split:
            legacy_pool = list(samples)
            random.Random(0).shuffle(legacy_pool)
            rng = random.Random(1000 + trial)
            pool = list(legacy_pool)
            rng.shuffle(pool)
            demonstrations = pool[: args.n_icl]
            rest = pool[args.n_icl :]
            references = rest[: args.n_ref]
            test_samples = rest[args.n_ref : args.n_ref + args.n_test]
        elif args.model_correct:
            demonstrations, references, test_samples = (
                splits.split_model_correct_samples(
                    samples,
                    args.relation,
                    args.model_correct_dir,
                    seed=args.split_seed + trial,
                    n_icl=args.n_icl,
                    n_ref=args.n_ref,
                    n_test=args.n_test,
                )
            )
        else:
            demonstrations, references, test_samples = splits.split_samples(
                samples,
                seed=args.split_seed + trial,
                n_icl=args.n_icl,
                n_ref=args.n_ref,
                n_test=args.n_test,
            )
        split_fingerprint = splits.split_fingerprint(
            demonstrations, references, test_samples
        )
        context = "".join(
            template.format(x["subject"]) + " " + x["object"] + "\n"
            for x in demonstrations
        )

        def prompt(sub):
            return context + template.format(sub) + (" " if args.trailing_space else "")

        def sub_pos(sub):
            p = prompt(sub)
            i = p.rindex(sub) + len(sub)
            return len(tokenizer(p[:i]).input_ids) - 1

        if handle is not None:
            handle.remove()
            handle = None
        cache = {}
        model_top_tokens = {}
        correct_count = 0
        split_fp = split_fingerprint
        with torch.no_grad():
            for x in references + test_samples:
                sub = x["subject"]
                input_ids = tokenizer(prompt(sub), return_tensors="pt").to(device)[
                    "input_ids"
                ]
                subject_position = min(sub_pos(sub), input_ids.shape[1] - 1)
                out = model(input_ids=input_ids, output_hidden_states=True)
                cache[sub] = (
                    input_ids,
                    subject_position,
                    {
                        layer: out.hidden_states[layer][0, subject_position, :]
                        .detach()
                        .clone()
                        for layer in layers
                    },
                )
                model_top_tokens[sub] = int(out.logits[0, -1].argmax())
        for x in test_samples:
            correct_count += int(
                model_top_tokens[x["subject"]]
                == tokenizer.encode(
                    ("" if args.trailing_space else " ") + x["object"],
                    add_special_tokens=False,
                )[0]
            )
        correct_counts.append(correct_count)
        for layer_index, layer in enumerate(layers):
            if handle is not None:
                handle.remove()
            handle = resolve_blocks(model)[layer - 1].register_forward_hook(
                intervention.hook
            )
            with torch.no_grad():
                base = torch.stack(
                    [
                        intervention.forward(
                            cache[x["subject"]][2][layer],
                            cache[x["subject"]][0],
                            cache[x["subject"]][1],
                        ).detach()
                        for x in references
                    ]
                ).mean(0)
            jacobian_products = []
            for x in test_samples:
                sub = x["subject"]
                subject_hidden = cache[sub][2][layer]
                acc = torch.zeros_like(base)
                for reference_sample in references:
                    reference_subject = reference_sample["subject"]
                    reference_hidden = cache[reference_subject][2][layer]
                    acc = acc + intervention.jvp(
                        reference_hidden,
                        (subject_hidden - reference_hidden),
                        cache[reference_subject][0],
                        cache[reference_subject][1],
                    )
                jacobian_products.append(acc / len(references))
            jacobian_products = torch.stack(jacobian_products)
            for beta_index, b in enumerate(betas):
                predicted_hidden = base.unsqueeze(0) + b * jacobian_products
                pred = final_logits(
                    predicted_hidden, norm, unembedding, lm_bias
                ).argmax(-1)
                hits = [
                    int(int(p) == model_top_tokens[x["subject"]])
                    for p, x in zip(pred, test_samples)
                ]
                trials[trial, layer_index, beta_index] = float(np.mean(hits))
                # per-subject outcomes, so the unit of analysis can be the
                # subject rather than the relation
                subject_outcomes[(trial, layers[layer_index], b)] = {
                    x["subject"]: h for x, h in zip(test_samples, hits)
                }
        print(
            f"  trial {trial + 1}/{args.trials}: LM correct {correct_count}/{len(test_samples)}, "
            f"best cell this trial {trials[trial].max():.3f}",
            flush=True,
        )

    mean = trials.mean(0)
    std = trials.std(0)
    layer_index, beta_index = np.unravel_index(mean.argmax(), mean.shape)
    best = (
        float(mean[layer_index, beta_index]),
        layers[layer_index],
        betas[beta_index],
    )
    print(f"\n{'layer':>6} " + " ".join(f"b={b:<9}" for b in betas))
    for i, layer in enumerate(layers):
        print(
            f"{layer:>6} "
            + " ".join(f"{mean[i, j]:.3f}+-{std[i, j]:.2f} " for j in range(len(betas)))
        )
    grid = {
        str(layer): {
            str(b): [float(mean[i, j]), float(std[i, j])] for j, b in enumerate(betas)
        }
        for i, layer in enumerate(layers)
    }
    print(
        f"\nselected on the MEAN over {args.trials} draws: layer {best[1]}, beta {best[2]}"
    )
    print(
        "single-draw grid maxima were: "
        + " ".join(f"{trials[trial].max():.3f}" for trial in range(args.trials))
    )
    print(
        f"  mean of per-draw maxima {trials.max(axis=(1, 2)).mean():.3f} "
        f"(this is the biased statistic)"
    )
    correct_count = int(np.mean(correct_counts))
    if handle is not None:
        handle.remove()
    print(
        f"\nBEST faithfulness {best[0]:.3f} +- {std[layer_index, beta_index]:.3f} "
        f"at layer {best[1]}, beta {best[2]}  ({args.trials} draws)"
    )
    # per-subject outcomes at the SELECTED cell, plus the split fingerprint
    # so the lens run can be asserted to have scored the same subjects
    selected_outcomes = subject_outcomes.get((args.trials - 1, best[1], best[2]), {})
    output_path = f"{args.out_dir}/faithfulness_{args.relation}{args.tag}{('_bf16' if args.dtype == 'bfloat16' else '')}.json"
    write_json(
        output_path,
        dict(
            relation=args.relation,
            n_test=args.n_test,
            trials=args.trials,
            grid=grid,
            best_faith=best[0],
            best_sd=float(std[layer_index, beta_index]),
            best_layer=best[1],
            best_beta=best[2],
            lm_correct=correct_count,
            split_fingerprint=split_fp,
            legacy_split=bool(args.legacy_split),
            distinct_first_token=bool(args.distinct_first_token),
            per_subject=selected_outcomes,
            biased_single_draw_mean=float(trials.max(axis=(1, 2)).mean()),
        ),
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
