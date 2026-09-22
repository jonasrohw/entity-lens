"""Prefix ablation: drop leading subject tokens and reread the lens.

HS: *"can you compare the top 20 tokens of 'rho 01 C nc' and 'C nc' and
the top 20 tokens of 'Al n air' and 'n air'? this would tell us if 'Al' is
involved (which is plausible for an airline) or not."*

The point is that `Al` is itself suggestive -- airline codes, `Al Italia`
-- so `Alnair`'s aviation readout might come from `Al`+`air` jointly
rather than from `air` alone.

Specs are `SUBJECT:D`, where D leading tokens of the subject span are
dropped.  The remaining tokens are **spliced by id** from the full
prompt's own tokenization rather than re-tokenized from a truncated
string, so `Alnair:1` is exactly the ids for `n`,`air` that `Alnair`
produces -- re-tokenizing the string "nair" would not be.

Caveat inherited from `lens_lasttoken_control.py`: dropping tokens also
moves the surviving ones earlier in the sequence, so position is not held
constant.  With two tokens left this is far less severe than truncating to
one (where 65 of 100 star names collapsed to a contentless fragment), but
it is not nothing, and a difference of one or two ranks should not be read
as signal.
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
    ap.add_argument("specs", nargs="+", help="SUBJECT:NDROP")
    ap.add_argument("--rel", default="star_constellation")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=f"{ROOT}/e2/lenstop/probe_prefix.json")
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
    n_special = len(tokenizer("", add_special_tokens=True).input_ids)

    data = json.load(open(f"{ROOT}/paper/code/data/relations/{args.rel}.json"))
    samples = splits.filter_identifying_objects(data["samples"], tokenizer, False)
    _, references, _ = splits.split_samples(samples, seed=0)
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
        return (ids, pos, o.hidden_states[layer][0, pos, :].detach().clone(),
                raw.detach().clone())

    cache = {}
    with torch.no_grad():
        for anchor in anchor_names:
            ids, pos = full_ids(anchor)
            cache[anchor] = state(ids, pos)
        probes = {}
        for spec in args.specs:
            subject, _, d = spec.rpartition(":")
            subject, d = (subject, int(d)) if subject else (spec, 0)
            ids, pos = full_ids(subject)
            if d:
                ids = torch.cat([ids[:, :n_special], ids[:, n_special + d:]], dim=1)
                pos = pos - d
            pieces = [tokenizer.decode([int(t)]) for t in ids[0, n_special:pos + 1]]
            probes[spec] = (state(ids, pos), pieces)

    handle = blocks[layer - 1].register_forward_hook(intervention.hook)
    out = {}
    for spec, ((ids, pos, hidden, _), pieces) in probes.items():
        total, n = None, 0
        for anchor in anchor_names:
            a_ids, a_pos, reference_hidden, _ = cache[anchor]
            v = intervention.jvp(reference_hidden, hidden - reference_hidden,
                                 a_ids, a_pos)
            total = v if total is None else total + v
            n += 1
        logits = final_logits(total / n, norm, unembedding, lm_bias, softcap)
        top = [tokenizer.decode([int(t)]).strip()
               for t in logits.topk(args.topk).indices]
        out[spec] = dict(tokens=pieces, top=top)
        print(f"\n{spec:16s} tokens={pieces}")
        print("   " + ", ".join(top))
    handle.remove()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
