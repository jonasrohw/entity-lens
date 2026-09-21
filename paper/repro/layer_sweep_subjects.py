"""Is the last-token subject representation valid only at some layers?

HS: *"in the prefix ablation experiment we found that for many subjects,
the last-token subject representation we're working with is actually not
a valid representation of the subject. could hits be layer-dependent?"*

The prefix ablation says that at layer 17 the readout often barely
changes when the start of the name is deleted -- so the state we read is
not carrying the whole subject.  Subject integration into the final token
is progressive, though, so that could be a fact about layer 17 rather
than about the model.

Two quantities per layer, for a handful of named subjects only (HS asked
for a manageable amount of output):

  integ  cos( h_last(full), h_last(drop-1) ) on the HIDDEN STATE.
         High means the prefix has not been integrated into the final
         token -- the state would be much the same without it.  This
         needs no Jacobian and is the direct measure of HS's premise.

  lens   the differential lens AT THAT LAYER, with the same relation's
         eight references also taken at that layer, plus the gold
         object's rank in it.

If integration completes at some depth, `integ` should fall with layer,
and the interesting question is whether the lens starts finding the gold
object at the layers where it does.
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
from jlens.config import MODEL_ID, NEUTRAL_PROMPT

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))


def resolve_device():
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", required=True)
    ap.add_argument("--subjects", required=True, help="comma-separated")
    ap.add_argument("--layers", default="")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--data-dir", default=f"{ROOT}/paper/code/data/relations")
    ap.add_argument("--out-dir", default=f"{ROOT}/e2/lenstop/layers")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or resolve_device()
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
    n_layers = len(blocks)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else sorted(set(list(range(4, n_layers, 4)) + [17, n_layers - 1])))
    n_special = len(tokenizer("", add_special_tokens=True).input_ids)
    print(f"{n_layers} blocks; sweeping layers {layers}", flush=True)

    data = json.load(open(f"{args.data_dir}/{args.relation}.json"))
    samples = splits.filter_identifying_objects(data["samples"], tokenizer, False)
    _, references, test_samples = splits.split_samples(samples, seed=0)
    anchor_names = [x["subject"] for x in references]
    gold_of = {x["subject"]: x["object"] for x in test_samples}

    def prompt_ids(subject):
        p = NEUTRAL_PROMPT.format(subject)
        ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
        i = p.rindex(subject) + len(subject)
        pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
        return ids, pos

    def all_layer_states(ids, pos):
        """hidden state at `pos` for every layer, from one forward pass."""
        o, _ = forward_capturing_final_block(
            model, blocks[-1], ids, output_hidden_states=True)
        return [o.hidden_states[L][0, pos, :].detach().clone()
                for L in range(len(o.hidden_states))]

    subjects = [s.strip() for s in args.subjects.split(",")]
    with torch.no_grad():
        refs = {a: (*prompt_ids(a), None) for a in anchor_names}
        refs = {a: (ids, pos, all_layer_states(ids, pos))
                for a, (ids, pos, _) in refs.items()}
        full, drop1 = {}, {}
        for s in subjects:
            ids, pos = prompt_ids(s)
            full[s] = (ids, pos, all_layer_states(ids, pos))
            n_tok = pos + 1 - n_special
            if n_tok > 1:
                d_ids = torch.cat([ids[:, :n_special],
                                   ids[:, n_special + 1:]], dim=1)
                drop1[s] = all_layer_states(d_ids, pos - 1)
            else:
                drop1[s] = None

    out = {}
    for L in layers:
        handle = blocks[L - 1].register_forward_hook(intervention.hook)
        for s in subjects:
            ids, pos, states = full[s]
            total, n = None, 0
            for a in anchor_names:
                a_ids, a_pos, a_states = refs[a]
                v = intervention.jvp(a_states[L], states[L] - a_states[L],
                                     a_ids, a_pos)
                total = v if total is None else total + v
                n += 1
            logits = final_logits(total / n, norm, unembedding, lm_bias, softcap)
            top = [tokenizer.decode([int(t)]).strip()
                   for t in logits.topk(args.topk).indices]
            gold = gold_of.get(s)
            grank = None
            if gold is not None:
                gt = splits.first_object_token(gold, tokenizer)
                grank = int((logits > logits[gt]).sum())
            integ = None
            if drop1[s] is not None:
                integ = float(torch.nn.functional.cosine_similarity(
                    states[L], drop1[s][L], dim=0))
            out.setdefault(s, []).append(
                dict(layer=L, integ=integ, gold=gold, gold_rank=grank, top=top))
        handle.remove()

    tag = args.relation
    path = f"{args.out_dir}/layers_{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    for s in subjects:
        g = gold_of.get(s)
        print(f"\n=== {s}   gold={g} ===")
        print(f"{'layer':>5} {'integ':>6} {'goldrank':>9}   lens top-5")
        for r in out[s]:
            i = "  --  " if r["integ"] is None else f"{r['integ']:6.3f}"
            gr = "    --   " if r["gold_rank"] is None else f"{r['gold_rank']:9d}"
            print(f"{r['layer']:5d} {i} {gr}   {', '.join(r['top'])}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
