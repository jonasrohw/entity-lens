"""Token readouts for chosen (subject, reference) pairs, plus the constant.

HS asked why one reference works and another does not, when both share the
subject's answer.  Ranks say *that* they differ; this says *what* each
returns.

For reference i and subject e the contrastive readout is
J_i s_e - J_i s_i.  The second term does not depend on e, so it is the
thing that distinguishes one reference from another.  We therefore decode
three vectors per pair --- the raw J_i s_e, the constant J_i s_i, and the
difference --- and report their norms, since a constant that is large
relative to the signal will swamp it.
"""
import argparse, json, sys
import numpy as np
from pathlib import Path

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data
# tree (data/, e2/) one level above the repo.  Override the data tree
# with JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))
from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--subjects", required=True)
ap.add_argument("--references", required=True)
ap.add_argument("--topk", type=int, default=15)
ap.add_argument("--layer", type=int, default=LENS_LAYER)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
subs = [x.strip() for x in a.subjects.split(",")]
refs = [x.strip() for x in a.references.split(",")]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.model import (HiddenStateIntervention, final_logits,
                         forward_capturing_final_block, resolve_blocks,
                         resolve_final_norm)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
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

with torch.no_grad():
    cache = {s: state(s) for s in set(subs + refs)}
h = blocks[L - 1].register_forward_hook(iv.hook)
def top(v):
    lg = final_logits(v, norm, Wh, bias, cap)
    return [tok.decode([int(t)]).strip() for t in lg.topk(a.topk).indices]

for r in refs:
    r_ids, r_pos, r_h = cache[r]
    Jss = iv.jvp(r_h, r_h, r_ids, r_pos)
    print(f"\n{'='*74}\nREFERENCE {r}")
    print(f"  the subtracted constant  J_i s_i   ||.||={float(Jss.norm()):7.2f}")
    print(f"     decodes to: {', '.join(top(Jss))}")
    for s in subs:
        Jse = iv.jvp(r_h, cache[s][2], r_ids, r_pos)
        dif = Jse - Jss
        print(f"\n  subject {s}")
        print(f"    raw  J_i s_e         ||.||={float(Jse.norm()):7.2f}  {', '.join(top(Jse)[:10])}")
        print(f"    diff J_i(s_e - s_i)  ||.||={float(dif.norm()):7.2f}  {', '.join(top(dif)[:10])}")
h.remove()
