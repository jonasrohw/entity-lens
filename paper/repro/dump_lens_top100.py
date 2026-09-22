"""Dump the top-k differential-Jacobian-lens readout for every test subject.

HS: the paper says the lens returns *category* vocabulary for
`star_constellation` -- "constellation, brightness, stellar" rather than
the particular constellation -- and asks the obvious follow-up: why does
the category survive the differencing here when the references are stars
too?  He wants the top 100 readouts per star to build an intuition.

The claim in the manuscript (S6, the prominence outlier) rests on six
top-5 lists; `notes/prominence.md` says so explicitly and warns it should
not be reported as more than an observation.  This script produces the
measurement that claim needs: the full ranked readout per subject, plus
two aggregates --

  * how far down the list the gold object actually sits, which separates
    "the lens does not know" from "the lens knows but the category
    outranks it", and
  * which tokens are *shared* across subjects, which is what "category
    vocabulary" has to mean if it means anything measurable.

Everything is the paper's canonical configuration: neutral prompt, layer
17, eight references, `--distinct-first-token`, seed 0.  The split
fingerprint is asserted against the stored faithfulness run so the
subjects here are the subjects the paper scored.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

from pathlib import Path

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data
# tree (data/, e2/, results/) one level above the repo.  Override the
# data tree with JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))

from jlens import splits
from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))
N_REFERENCES = 8
N_ICL = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rels", default="star_constellation")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-test", type=int, default=splits.N_TEST)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--data-dir", default=f"{ROOT}/paper/code/data/relations")
    ap.add_argument("--faith-dir", default=f"{ROOT}/e2/corrected",
                    help="stored faithfulness run whose split we must match")
    ap.add_argument("--out-dir", default=f"{ROOT}/e2/lenstop")
    ap.add_argument("--device", default=None,
                    help="default: claim a genuinely idle card")
    return ap.parse_args()


def claim_gpu():
    """Index of a card nobody is computing on.  Never hardcode a card: the
    machines are shared and an earlier version of the job runner landed on
    top of another user's work."""
    import subprocess
    busy = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.split()
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in rows.strip().splitlines():
        i, u, used, tot = [t.strip() for t in line.split(",")]
        if u in busy:
            continue
        if int(tot) - int(used) < 30000:
            continue
        return f"cuda:{i}"
    raise SystemExit("no idle GPU with 30 GB free; not doubling up on a card")


def main():
    args = parse_args()
    device = args.device or claim_gpu()
    print(f"using {device}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        .to(device).eval())
    for p in model.parameters():
        p.requires_grad_(False)
    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach().float()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias = lm_bias.detach().float() if lm_bias is not None else None
    softcap = getattr(model.config.get_text_config(),
                      "final_logit_softcapping", None)
    blocks = resolve_blocks(model)
    intervention = HiddenStateIntervention(model, final_block=blocks[-1])

    for relation in args.rels.split(","):
        run_relation(relation, args, device, torch, tokenizer, model, norm,
                     unembedding, lm_bias, softcap, blocks, intervention,
                     final_logits, forward_capturing_final_block)


def run_relation(relation, args, device, torch, tokenizer, model, norm,
                 unembedding, lm_bias, softcap, blocks, intervention,
                 final_logits, forward_capturing_final_block):
    data = json.load(open(f"{args.data_dir}/{relation}.json"))
    template = data["prompt_templates"][0]
    samples = splits.filter_identifying_objects(data["samples"], tokenizer, False)
    demonstrations, references, test_samples = splits.split_samples(
        samples, seed=args.split_seed, n_icl=N_ICL, n_ref=N_REFERENCES,
        n_test=args.n_test)
    fingerprint = splits.split_fingerprint(demonstrations, references, test_samples)

    stored = f"{args.faith_dir}/lrefaithT_{relation}.json"
    if os.path.exists(stored):
        want = json.load(open(stored)).get("split_fingerprint")
        if want and want != fingerprint:
            raise SystemExit(
                f"FATAL {relation}: fingerprint {fingerprint} != stored {want}; "
                f"these would not be the subjects the paper scored.")
        print(f"{relation}: split fingerprint {fingerprint} matches {stored}")

    subjects = [x["subject"] for x in test_samples]
    gold = {x["subject"]: x["object"] for x in test_samples}
    anchor_names = [x["subject"] for x in references]
    layer = args.layer

    # every distinct object in the WHOLE relation, so we can ask whether any
    # answer-space token at all makes the list, not just the gold one
    object_first = {}
    for obj in sorted({x["object"] for x in data["samples"]}):
        object_first.setdefault(splits.first_object_token(obj, tokenizer), obj)

    context = "".join(template.format(x["subject"]) + " " + x["object"] + "\n"
                      for x in demonstrations)
    target_tokens = {}
    with torch.no_grad():
        for s in subjects:
            ids = tokenizer(context + template.format(s),
                            return_tensors="pt").to(device)["input_ids"]
            target_tokens[s] = int(model(input_ids=ids).logits[0, -1].argmax())

    cache = {}
    with torch.no_grad():
        for subject in list(dict.fromkeys(subjects + anchor_names)):
            p = NEUTRAL_PROMPT.format(subject)
            ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
            i = p.rindex(subject) + len(subject)
            pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
            o, raw_final = forward_capturing_final_block(
                model, blocks[-1], ids, output_hidden_states=True)
            cache[subject] = (ids, pos,
                              o.hidden_states[layer][0, pos, :].detach().clone(),
                              raw_final.detach().clone())

    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    rows = []
    for subject in subjects:
        difference_sum, n = None, 0
        for anchor in anchor_names:
            if anchor == subject:
                continue
            ids, pos, reference_hidden, _ = cache[anchor]
            v = intervention.jvp(reference_hidden,
                                 cache[subject][2] - reference_hidden, ids, pos)
            difference_sum = v if difference_sum is None else difference_sum + v
            n += 1
        v = difference_sum / n
        logits = final_logits(v, norm, unembedding, lm_bias, softcap)
        order = logits.argsort(descending=True)
        top_ids = [int(t) for t in order[:args.topk]]
        top = [tokenizer.decode([t]).strip() for t in top_ids]

        gold_tok = splits.first_object_token(gold[subject], tokenizer)
        gold_rank = int((logits > logits[gold_tok]).sum())
        model_rank = int((logits > logits[target_tokens[subject]]).sum())
        # any object of this relation appearing in the top-k, with its rank
        answers_in_top = [(r, object_first[t]) for r, t in enumerate(top_ids)
                          if t in object_first]
        rows.append(dict(
            subject=subject, gold=gold[subject],
            model_top1=tokenizer.decode([target_tokens[subject]]).strip(),
            gold_rank=gold_rank, model_rank=model_rank,
            top=top, top_ids=top_ids,
            answers_in_topk=answers_in_top,
            best_answer_rank=answers_in_top[0][0] if answers_in_top else None,
        ))
    handle.remove()

    write_outputs(relation, args, rows, template, fingerprint, len(object_first))


def write_outputs(relation, args, rows, template, fingerprint, n_answers):
    K = args.topk
    shared = collections.Counter()
    for r in rows:
        shared.update(set(r["top"]))
    gr = np.array([r["gold_rank"] for r in rows], dtype=float)
    summary = dict(
        relation=relation, n=len(rows), layer=args.layer, topk=K,
        neutral_prompt=NEUTRAL_PROMPT, icl_template=template,
        split_fingerprint=fingerprint, n_distinct_answer_tokens=n_answers,
        gold_in_top5=float((gr < 5).mean()),
        gold_in_top20=float((gr < 20).mean()),
        gold_in_topk=float((gr < K).mean()),
        median_gold_rank=float(np.median(gr)),
        any_answer_in_topk=float(np.mean([r["best_answer_rank"] is not None
                                          for r in rows])),
        shared_tokens=[[t, c] for t, c in shared.most_common(60)],
    )
    jout = f"{args.out_dir}/lenstop{K}_{relation}.json"
    json.dump(dict(summary=summary, rows=rows), open(jout, "w"), indent=1)

    tout = f"{args.out_dir}/lenstop{K}_{relation}.txt"
    with open(tout, "w") as fh:
        fh.write(f"top-{K} differential-Jacobian-lens readout, {relation}\n")
        fh.write(f"neutral prompt {NEUTRAL_PROMPT!r}, layer {args.layer}, "
                 f"8 references, split {fingerprint}\n")
        fh.write(f"gold in top-5 {summary['gold_in_top5']:.2f}   "
                 f"top-20 {summary['gold_in_top20']:.2f}   "
                 f"top-{K} {summary['gold_in_topk']:.2f}   "
                 f"median gold rank {summary['median_gold_rank']:.0f}\n\n")
        fh.write(f"tokens shared by the most subjects (of {len(rows)}):\n")
        for t, c in summary["shared_tokens"]:
            fh.write(f"  {c:4d}  {t}\n")
        fh.write("\n" + "=" * 72 + "\n\n")
        for r in rows:
            fh.write(f"{r['subject']}   gold={r['gold']}  "
                     f"(gold rank {r['gold_rank']})   "
                     f"model top-1={r['model_top1']!r} "
                     f"(lens rank {r['model_rank']})\n")
            if r["answers_in_topk"]:
                fh.write("  answer-space tokens in the list: "
                         + ", ".join(f"{o}@{rk}" for rk, o in r["answers_in_topk"][:8])
                         + "\n")
            for i in range(0, K, 10):
                fh.write("   " + " ".join(f"{t!r}" for t in r["top"][i:i + 10]) + "\n")
            fh.write("\n")
    print(f"wrote {jout}\nwrote {tout}")
    print(f"{relation}: gold in top-5 {summary['gold_in_top5']:.2f}, "
          f"top-{K} {summary['gold_in_topk']:.2f}, "
          f"median gold rank {summary['median_gold_rank']:.0f}")


if __name__ == "__main__":
    main()
