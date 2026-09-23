"""Split "typical" from "which answer" for the self-Jacobian readouts.

HS: *"we have two opposing goals here in clustering. on the one hand we'd
like to find subjects that have 'typical' associations (as opposed to
idiosyncratic ones like French literature prizes). On the other hand we
probably have cluster structure *within* the subjects that have typical
associations: French vs English vs German etc."*

Right, and a flat clustering conflates them.  They are separable:

  TYPICALITY  how much of J_e s_e lands in the relation's answer space at
              all -- a scalar.  Measured as the probability mass the
              readout puts on the set of answer tokens, and as the rank
              of the best answer token.

  IDENTITY    WHICH answer it points at -- a direction.  Measured as the
              argmax over answer tokens only.

The prediction the decomposition makes: TYPICALITY should predict how
good a subject is as a reference, for every subject; IDENTITY should
predict only WHICH subjects it helps, through the same-language effect.
"""
import json, os, sys
import numpy as np
from pathlib import Path

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data
# tree (data/, e2/) one level above the repo.  Override the data tree
# with JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))
from jlens.config import MODEL_ID
ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))
REL = "person_native_language"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.model import final_logits, resolve_final_norm
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = (AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval())
norm = resolve_final_norm(model); Wh = model.lm_head.weight.detach().float()
bias = getattr(model.lm_head, "bias", None)
bias = bias.detach().float() if bias is not None else None
cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

V = np.load(f"{ROOT}/e2/selfjac_vectors_{REL}.npy")
M = json.load(open(f"{ROOT}/e2/selfjac_vectors_{REL}.json"))
S = M["subjects"]
data = json.load(open(f"{ROOT}/paper/code/data/relations/{REL}.json"))
answers = sorted({x["object"] for x in data["samples"]})
aid = {}
for o in answers:
    t = tok.encode(" " + o, add_special_tokens=False)[0]
    aid.setdefault(t, o)
ids = torch.tensor(sorted(aid), device="cuda:0")
print(f"{len(answers)} distinct answers -> {len(aid)} distinct first tokens")

rows = []
for s, v in zip(S, V):
    lg = final_logits(torch.tensor(v, device="cuda:0"), norm, Wh, bias, cap)
    p = torch.softmax(lg, -1)
    mass = float(p[ids].sum())
    best = int(ids[int(lg[ids].argmax())])
    rows.append(dict(subject=s, answer_mass=mass,
                     best_answer=aid[best],
                     best_answer_rank=int((lg > lg[best]).sum())))
json.dump(rows, open(f"{ROOT}/e2/answer_space_{REL}.json", "w"), indent=1)
print(f"wrote e2/answer_space_{REL}.json")
