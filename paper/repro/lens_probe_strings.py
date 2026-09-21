"""Run the star lens on arbitrary strings, to test whether a readout is
driven by the subject's ORTHOGRAPHY rather than by the entity.

HS's last-token control (`lens_lasttoken_control.py`) came back
uninformative: 65 of the 100 star names end in a contentless fragment
(`is` 17x, `ae` 10x, `e` 8x), so "the lens of the last token" collapses to
one generic list for almost every star -- mean pairwise Jaccard 0.485
across subjects, against 0.070 for the full-subject lens.  The control
cannot discriminate, so its negative result says nothing about the
hypothesis.

The hypothesis is still live, and the qualitative evidence for it is
strong: `Alnair` (tokenized `Al`/`n`/`air`) reads as *flying, flown,
Flight, flights, fly, flew, rescue, sailing, airline, airspace, Airways*,
and `rho01 Cnc` reads as *machining, drilled, CNC, Sculpt, drill,
drilling, manufacturing*.  But it operates on the whole string, not on the
final token, which is why truncating to that token destroyed it.

The direct test is a minimal pair: perturb the accidental word while
keeping the string star-like, and see whether the semantic field goes with
it.  If `Alnaer` loses the aviation vocabulary that `Alnair` has, the
orthography caused it.

Same construction as the paper throughout -- neutral prompt, layer 17, the
eight `star_constellation` reference subjects, corrected norm path.
"""

import argparse
import json
import os
import sys

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


def claim_gpu():
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
    ap = argparse.ArgumentParser()
    ap.add_argument("strings", nargs="+")
    ap.add_argument("--rel", default="star_constellation")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=f"{ROOT}/e2/lenstop/probe_strings.json")
    args = ap.parse_args()
    device = args.device or claim_gpu()
    print(f"using {device}", flush=True)

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
    layer = args.layer

    data = json.load(open(f"{ROOT}/paper/code/data/relations/{args.rel}.json"))
    samples = splits.filter_identifying_objects(data["samples"], tokenizer, False)
    _, references, _ = splits.split_samples(samples, seed=0)
    anchor_names = [x["subject"] for x in references]
    print("references:", ", ".join(anchor_names), flush=True)

    def state(subject):
        p = NEUTRAL_PROMPT.format(subject)
        ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
        i = p.rindex(subject) + len(subject)
        pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
        o, raw = forward_capturing_final_block(
            model, blocks[-1], ids, output_hidden_states=True)
        return (ids, pos, o.hidden_states[layer][0, pos, :].detach().clone(),
                raw.detach().clone())

    cache = {}
    with torch.no_grad():
        for name in list(dict.fromkeys(list(args.strings) + anchor_names)):
            cache[name] = state(name)

    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    out = {}
    for name in args.strings:
        total, n = None, 0
        for anchor in anchor_names:
            ids, pos, reference_hidden, _ = cache[anchor]
            v = intervention.jvp(reference_hidden,
                                 cache[name][2] - reference_hidden, ids, pos)
            total = v if total is None else total + v
            n += 1
        logits = final_logits(total / n, norm, unembedding, lm_bias, softcap)
        top = [tokenizer.decode([int(t)]).strip()
               for t in logits.topk(args.topk).indices]
        ids, pos, _, _ = cache[name]
        pieces = [tokenizer.decode([int(t)]) for t in ids[0, :pos + 1]]
        out[name] = dict(top=top, tokens=pieces)
        print(f"\n{name!r}  tokens={pieces[-4:]}")
        print("   " + ", ".join(top))
    handle.remove()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
