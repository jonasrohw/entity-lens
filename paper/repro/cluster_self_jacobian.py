"""Cluster the 100 self-Jacobian readouts J_e s_e.

HS: *"can you cluster the 100 J_e s_e vectors? maybe this would show the
three classes you identified and then we could pick better references?"*

S6 of notes/single-reference-matrix.md found that a good reference is one
whose local Jacobian points at the relation's answer space, diagnosed by
what J_i s_i decodes to.  J_e s_e is exactly that object for every
subject, so clustering these 100 vectors asks whether "good reference" is
a structural property that can be read off without consulting any answer.

The "three classes" -- nationality, historical era, surface form of the
name -- came from FOUR hand-picked examples and are an impression, not a
taxonomy.  This tests it.

The payoff would be a selection rule: if cluster membership predicts a
reference's column mean in the contrastive matrix, references can be
chosen structurally rather than by trying them.
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
ap.add_argument("--relation", default="person_native_language")
ap.add_argument("--layer", type=int, default=LENS_LAYER)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.model import (HiddenStateIntervention, final_logits,
                         forward_capturing_final_block, resolve_blocks,
                         resolve_final_norm)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
stored = json.load(open(f"{ROOT}/e2/mainlens/lens_{a.relation}.json"))[a.relation]
subs = [r["subject"] for r in stored["rows"]]
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
    cache = {s: state(s) for s in subs}
h = blocks[L - 1].register_forward_hook(iv.hook)
V, TOP = [], []
for s in subs:
    ids, pos, hs = cache[s]
    v = iv.jvp(hs, hs, ids, pos)
    V.append(v.detach().cpu().numpy())
    lg = final_logits(v, norm, Wh, bias, cap)
    TOP.append([tok.decode([int(t)]).strip() for t in lg.topk(10).indices])
h.remove()
V = np.stack(V)
np.save(f"{ROOT}/e2/selfjac_vectors_{a.relation}.npy", V)
json.dump(dict(subjects=subs, top10=TOP),
          open(f"{ROOT}/e2/selfjac_vectors_{a.relation}.json", "w"), indent=1)
print(f"wrote {V.shape} vectors and their top-10 readouts")
