"""Prefix ablation over every subject of a relation.

HS: *"run the test we did for the two star constellation examples for the
entire dataset. so for each subject, shorten it one by one (from the
start) and record the top 20 tokens in j lens."*

For each test subject we drop leading subject tokens one at a time and
reread the lens, so `Alnair` yields `Al n air`, `n air`, `air`.  The
surviving tokens are **spliced by id** from the full prompt's own
tokenization rather than re-tokenized from a truncated string.

What this measures, and why it is worth measuring: if the readout barely
changes when the beginning of the name is removed, the lens is not using
the whole subject.  That is the signature shared by both failure modes we
have found by hand --- the namesake collision (`Barry Sobel` reading as
the rabbi `Henry Sobel`, where the surname suffices) and the sub-lexical
collision (`Alnair` reading as aviation, where `n`+`air` suffices).
Neither was detectable at scale before; `notes/star-constellation-readout.md`
S11 lists exactly this as the open item.

Two caveats, both real, recorded here so results are not over-read:

1. **Position is not held constant.**  Dropping tokens also moves the
   survivors earlier in the sequence, so some of the change at every level
   is positional rather than lexical.  This cannot be designed away while
   still deleting tokens.

2. **The one-token endpoint is known to be degenerate.**  In
   `lens_lasttoken_control.py` the single-final-token readouts collapsed
   to one generic list across subjects (pairwise Jaccard 0.485 against
   0.070 for full subjects).  The last level of every subject is
   therefore expected to be junk and should be read as a floor, not as a
   measurement.

The quantity to look at is consequently the **first** drop, d=0 vs d=1:
how much of the readout survives losing just the start of the name.
"""

import argparse
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
    ap.add_argument("--relation", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-test", type=int, default=splits.N_TEST)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--data-dir", default=f"{ROOT}/paper/code/data/relations")
    ap.add_argument("--faith-dir", default=f"{ROOT}/e2/corrected")
    ap.add_argument("--out-dir", default=f"{ROOT}/e2/lenstop/prefix")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def resolve_device():
    """A queue worker is handed a card via CUDA_VISIBLE_DEVICES, in which
    case the only visible device is index 0.  Standalone, claim an idle
    card -- never hardcode one, the machines are shared."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
    import subprocess
    busy = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.split()
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in rows.strip().splitlines():
        i, u, used, tot = [t.strip() for t in line.split(",")]
        if u not in busy and int(tot) - int(used) >= 30000:
            return f"cuda:{i}"
    raise SystemExit("no idle GPU with 30 GB free; not doubling up on a card")


def main():
    args = parse_args()
    device = args.device or resolve_device()
    os.makedirs(args.out_dir, exist_ok=True)
    trailing = args.relation.endswith("_year")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    data = json.load(open(f"{args.data_dir}/{args.relation}.json"))
    template = data["prompt_templates"][0]
    samples = splits.filter_identifying_objects(data["samples"], tokenizer, trailing)
    demonstrations, references, test_samples = splits.split_samples(
        samples, seed=args.split_seed, n_icl=N_ICL, n_ref=N_REFERENCES,
        n_test=args.n_test)
    if len(test_samples) < 1:
        print(f"{args.relation}: {len(test_samples)} test subjects after "
              f"filtering, nothing to measure -- skipping", flush=True)
        return
    fingerprint = splits.split_fingerprint(demonstrations, references, test_samples)
    stored = f"{args.faith_dir}/lrefaithT_{args.relation}.json"
    if os.path.exists(stored):
        want = json.load(open(stored)).get("split_fingerprint")
        if want and want != fingerprint:
            raise SystemExit(f"FATAL {args.relation}: fingerprint {fingerprint} "
                             f"!= stored {want}")

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
    layer = args.layer
    n_special = len(tokenizer("", add_special_tokens=True).input_ids)

    subjects = [x["subject"] for x in test_samples]
    gold = {x["subject"]: x["object"] for x in test_samples}
    anchor_names = [x["subject"] for x in references]

    def full_ids(subject):
        p = NEUTRAL_PROMPT.format(subject)
        ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
        i = p.rindex(subject) + len(subject)
        pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
        return ids, pos

    def state(ids, pos):
        o, raw = forward_capturing_final_block(
            model, blocks[-1], ids, output_hidden_states=True)
        return o.hidden_states[layer][0, pos, :].detach().clone()

    # the model's own top-1 under the relation's ICL prompt, per subject
    context = "".join(template.format(x["subject"]) + " " + x["object"] + "\n"
                      for x in demonstrations)
    target = {}
    with torch.no_grad():
        for s in subjects:
            ids = tokenizer(context + template.format(s)
                            + (" " if trailing else ""),
                            return_tensors="pt").to(device)["input_ids"]
            target[s] = int(model(input_ids=ids).logits[0, -1].argmax())
        refs = {}
        for anchor in anchor_names:
            ids, pos = full_ids(anchor)
            refs[anchor] = (ids, pos, state(ids, pos))

    # Phase 1: every truncation's hidden state, BEFORE the hook goes on.
    # The intervention hook rejects an ordinary forward pass, so all plain
    # forwards have to happen first.
    plan = []
    with torch.no_grad():
        for subject in subjects:
            ids, pos = full_ids(subject)
            n_tokens = pos + 1 - n_special
            for d in range(n_tokens):
                sub_ids = (ids if d == 0 else
                           torch.cat([ids[:, :n_special],
                                      ids[:, n_special + d:]], dim=1))
                plan.append((subject, d, n_tokens,
                             [tokenizer.decode([int(t)])
                              for t in sub_ids[0, n_special:pos - d + 1]],
                             state(sub_ids, pos - d)))
    print(f"  {args.relation}: {len(plan)} truncation levels cached",
          flush=True)

    handle = blocks[layer - 1].register_forward_hook(intervention.hook)

    def lens(hidden):
        total, n = None, 0
        for anchor in anchor_names:
            a_ids, a_pos, ref_hidden = refs[anchor]
            v = intervention.jvp(ref_hidden, hidden - ref_hidden, a_ids, a_pos)
            total = v if total is None else total + v
            n += 1
        return total / n

    # Phase 2: the lens for each cached state, with the hook on.
    by_subject = {s: dict(subject=s, gold=gold[s], n_tokens=0,
                          model_top1=tokenizer.decode([target[s]]).strip(),
                          levels=[]) for s in subjects}
    base = {}
    # NOT under no_grad: the JVP needs the autograd graph.
    for k, (subject, d, n_tokens, pieces, hidden) in enumerate(plan):
        gold_tok = splits.first_object_token(gold[subject], tokenizer,
                                             trailing)
        v = lens(hidden)
        logits = final_logits(v, norm, unembedding, lm_bias, softcap)
        top_ids = [int(t) for t in logits.topk(args.topk).indices]
        if d == 0:
            base[subject] = (v, set(top_ids))
        base_v, base_top = base[subject]
        by_subject[subject]["n_tokens"] = n_tokens
        by_subject[subject]["levels"].append(dict(
            drop=d, tokens=pieces,
            top=[tokenizer.decode([t]).strip() for t in top_ids],
            gold_rank=int((logits > logits[gold_tok]).sum()),
            model_rank=int((logits > logits[target[subject]]).sum()),
            cos_to_full=float(torch.nn.functional.cosine_similarity(
                v, base_v, dim=0)),
            jaccard_to_full=len(base_top & set(top_ids))
            / len(base_top | set(top_ids)),
        ))
        if (k + 1) % 200 == 0:
            print(f"  {args.relation}: {k + 1}/{len(plan)} levels",
                  flush=True)
    handle.remove()
    rows = [by_subject[s] for s in subjects]

    drop1 = [r["levels"][1]["cos_to_full"] for r in rows if r["n_tokens"] > 1]
    summary = dict(
        relation=args.relation, n_subjects=len(rows), layer=layer,
        topk=args.topk, split_fingerprint=fingerprint,
        n_levels=sum(r["n_tokens"] for r in rows),
        mean_n_tokens=float(np.mean([r["n_tokens"] for r in rows])),
        n_multitoken=len(drop1),
        mean_cos_drop1=float(np.mean(drop1)) if drop1 else None,
        median_cos_drop1=float(np.median(drop1)) if drop1 else None,
        mean_jaccard_drop1=float(np.mean(
            [r["levels"][1]["jaccard_to_full"] for r in rows
             if r["n_tokens"] > 1])) if drop1 else None,
    )
    out = f"{args.out_dir}/prefix_{args.relation}.json"
    json.dump(dict(summary=summary, rows=rows), open(out, "w"), indent=1)
    print(f"wrote {out}")
    print(f"{args.relation}: n={len(rows)} levels={summary['n_levels']} "
          f"mean cos(full, drop1)="
          f"{summary['mean_cos_drop1'] if drop1 else float('nan'):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
