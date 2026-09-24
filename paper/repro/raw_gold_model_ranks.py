"""Per-subject gold and model-top-1 ranks under the raw lens.

HS: a table of the 100 subjects with gold answer, model top-1 answer, and
the rank each receives under the raw readout  v_e = mean_i J_i s_e.

References are the eight of the relation's own split, reconstructed the
same way measure_faithfulness does under --legacy-split (Random(0), then
Random(1000 + trial)), so the subjects and references match
e2/mainlens and e2/mainrank.  Gold is reported here only as a column; it
is not used to select anything.
"""
import argparse, glob, json, os, random, sys
import numpy as np
from pathlib import Path

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data
# tree (data/, e2/) one level above the repo.  Override the data tree
# with JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))
from jlens import splits
from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--relation", default="person_native_language")
ap.add_argument("--layer", type=int, default=LENS_LAYER)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--n-icl", type=int, default=8)
ap.add_argument("--n-ref", type=int, default=8)
ap.add_argument("--n-test", type=int, default=100)
ap.add_argument("--self-jacobian", action="store_true",
                help="use v_e = J_e s_e, the subject read through its OWN "
                     "local map, instead of averaging over the split's "
                     "references. This is the diagonal of the raw matrix in "
                     "e2/single_ref_*.json, which stores only the model-top-1 "
                     "rank; gold ranks need this pass.")
a = ap.parse_args()
ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.model import (HiddenStateIntervention, final_logits,
                         forward_capturing_final_block, resolve_blocks,
                         resolve_final_norm)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
data = json.load(open(f"{ROOT}/paper/code/data/relations/{a.relation}.json"))
samples = data["samples"]
pool = list(samples); random.Random(0).shuffle(pool)
p2 = list(pool); random.Random(1000).shuffle(p2)
demos, rest = p2[: a.n_icl], p2[a.n_icl :]
refs, tests = rest[: a.n_ref], rest[a.n_ref : a.n_ref + a.n_test]
fp = splits.split_fingerprint(demos, refs, tests)
stored = json.load(open(f"{ROOT}/e2/mainlens/lens_{a.relation}.json"))[a.relation]
assert fp == stored.get("split_fingerprint"), f"{fp} != {stored.get('split_fingerprint')}"
gold = {x["subject"]: x["object"] for x in samples}
model_top1 = {r["subject"]: r["model_top1"] for r in stored["rows"]}
print(f"{a.relation}: {len(tests)} subjects, {len(refs)} references, split {fp}", flush=True)

model = (AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager").to(a.device).eval())
for p in model.parameters(): p.requires_grad_(False)
norm = resolve_final_norm(model); Wh = model.lm_head.weight.detach().float()
bias = getattr(model.lm_head, "bias", None)
bias = bias.detach().float() if bias is not None else None
cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
blocks = resolve_blocks(model); iv = HiddenStateIntervention(model, final_block=blocks[-1])
L = a.layer

def state(s):
    p = NEUTRAL_PROMPT.format(s)
    ids = tok(p, return_tensors="pt").to(a.device)["input_ids"]
    i = p.rindex(s) + len(s)
    pos = min(len(tok(p[:i]).input_ids) - 1, ids.shape[1] - 1)
    o, _ = forward_capturing_final_block(model, blocks[-1], ids, output_hidden_states=True)
    return ids, pos, o.hidden_states[L][0, pos, :].detach().clone()

names = [x["subject"] for x in tests]; rnames = [x["subject"] for x in refs]
with torch.no_grad():
    cache = {s: state(s) for s in set(names + ([] if a.self_jacobian else rnames))}
h = blocks[L - 1].register_forward_hook(iv.hook)
rows = []
for s in names:
    if a.self_jacobian:
        acc = iv.jvp(cache[s][2], cache[s][2], cache[s][0], cache[s][1])
        denom = 1
    else:
        acc, denom = None, len(rnames)
        for r in rnames:
            v = iv.jvp(cache[r][2], cache[s][2], cache[r][0], cache[r][1])
            acc = v if acc is None else acc + v
    lg = final_logits(acc / denom, norm, Wh, bias, cap)
    gt = tok.encode(" " + gold[s], add_special_tokens=False)[0]
    mt = tok.encode(" " + model_top1[s], add_special_tokens=False)[0]
    rows.append(dict(subject=s, gold=gold[s], model_top1=model_top1[s],
                     rank_gold=int((lg > lg[gt]).sum()),
                     rank_model=int((lg > lg[mt]).sum()),
                     top5=[tok.decode([int(t)]).strip() for t in lg.topk(5).indices]))
h.remove()
tag = "selfjac" if a.self_jacobian else "raw"
out = f"{ROOT}/e2/{tag}_gold_model_{a.relation}.json"
json.dump(rows, open(out, "w"), indent=1)
print(f"wrote {out}\n")
print(f"{'subject':30s} {'gold':12s} {'model top-1':12s} {'rk gold':>8} {'rk model':>9}")
for r in rows:
    print(f"{r['subject'][:29]:30s} {r['gold'][:11]:12s} {r['model_top1'][:11]:12s} "
          f"{r['rank_gold']:8d} {r['rank_model']:9d}")
g = np.array([r["rank_gold"] for r in rows]); m = np.array([r["rank_model"] for r in rows])
print(f"\nmedian rank: gold {np.median(g):.0f}   model top-1 {np.median(m):.0f}")
print(f"gold at rank 0: {(g==0).sum()}/{len(g)}   model top-1 at rank 0: {(m==0).sum()}/{len(m)}")
print(f"agree (gold == model top-1): {sum(r['gold']==r['model_top1'] for r in rows)}/{len(rows)}")
