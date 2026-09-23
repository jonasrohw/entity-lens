"""Every reference set of size one, for one relation.

HS: *"for person native language, i'd like to understand how the reference
set influences the prediction of the contrastive jacobian. let's do
reference sets of size 1. can you compute lens log_10 rank for *all*
reference sets of size 1? so this is an nxn matrix. also provide log_10
rank of the raw jacobian for reference."*

Entry (e, i) is the log10 rank of the model's own answer for subject e
when the reference set is the single subject i:

    contrastive   v = J_i (s_e - s_i)
    raw           v = J_i  s_e

**One pass gives both matrices.**  J_i is linear, so

    J_i (s_e - s_i) = J_i s_e - J_i s_i ,

and J_i s_i is the DIAGONAL of the raw matrix.  So n^2 JVPs suffice: the
contrastive matrix is the raw matrix minus a per-column constant vector.
That is also the cleanest statement of what the subtraction does here --
with one reference it removes a constant that depends on the reference
and not on the subject, exactly as Eq. 5 of the paper says for the
averaged case.

The contrastive DIAGONAL is degenerate: s_e - s_e = 0, so the readout is
the decoding of the zero vector and carries no information about e.  It is
written as null rather than as a number.
"""

import argparse
import json
import os
import sys

import numpy as np

from pathlib import Path

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data
# tree (data/, e2/) one level above the repo.  Override the data tree
# with JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))

from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="person_native_language")
    ap.add_argument("--lens-file", default=None,
                    help="run whose subjects and targets we reuse")
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lens_file = a.lens_file or f"{ROOT}/e2/mainlens/lens_{a.relation}.json"
    out = a.out or f"{ROOT}/e2/single_ref_{a.relation}.json"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    L = json.load(open(lens_file))[a.relation]
    subjects = [r["subject"] for r in L["rows"]]
    n = len(subjects)
    print(f"{a.relation}: {n} subjects -> {n*n} JVPs", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        .to(a.device).eval())
    for p in model.parameters():
        p.requires_grad_(False)
    norm = resolve_final_norm(model)
    Wh = model.lm_head.weight.detach().float()
    bias = getattr(model.lm_head, "bias", None)
    bias = bias.detach().float() if bias is not None else None
    cap = getattr(model.config.get_text_config(),
                  "final_logit_softcapping", None)
    blocks = resolve_blocks(model)
    iv = HiddenStateIntervention(model, final_block=blocks[-1])
    Lay = a.layer

    # the model's own top-1, reused from the stored run so the target is
    # identical to the one the paper scores
    target = {r["subject"]: r["model_top1"] for r in L["rows"]}
    tgt_id = {s: tok.encode(" " + t, add_special_tokens=False)[0]
              for s, t in target.items()}

    with torch.no_grad():
        cache = {}
        for s in subjects:
            p = NEUTRAL_PROMPT.format(s)
            ids = tok(p, return_tensors="pt").to(a.device)["input_ids"]
            i = p.rindex(s) + len(s)
            pos = min(len(tok(p[:i]).input_ids) - 1, ids.shape[1] - 1)
            o, _ = forward_capturing_final_block(
                model, blocks[-1], ids, output_hidden_states=True)
            cache[s] = (ids, pos, o.hidden_states[Lay][0, pos, :].detach().clone())

    handle = blocks[Lay - 1].register_forward_hook(iv.hook)
    raw = np.full((n, n), np.nan)
    con = np.full((n, n), np.nan)
    for j, ref in enumerate(subjects):                    # column = reference
        r_ids, r_pos, r_h = cache[ref]
        Jss = iv.jvp(r_h, r_h, r_ids, r_pos)              # J_i s_i, the constant
        for k, sub in enumerate(subjects):                # row = subject
            Jse = iv.jvp(r_h, cache[sub][2], r_ids, r_pos)
            t = tgt_id[sub]
            lg = final_logits(Jse, norm, Wh, bias, cap)
            raw[k, j] = np.log10(int((lg > lg[t]).sum()) + 1)
            if k != j:
                lg = final_logits(Jse - Jss, norm, Wh, bias, cap)
                con[k, j] = np.log10(int((lg > lg[t]).sum()) + 1)
        if (j + 1) % 10 == 0:
            print(f"  reference {j+1}/{n}", flush=True)
    handle.remove()

    base = {r["subject"]: np.log10(r["rank_diff"] + 1) for r in L["rows"]}
    json.dump(dict(relation=a.relation, layer=Lay, subjects=subjects,
                   targets=target,
                   contrastive=[[None if v != v else v for v in row] for row in con],
                   raw=[[None if v != v else v for v in row] for row in raw],
                   eight_reference_baseline=[base[s] for s in subjects]),
              open(out, "w"))
    print(f"wrote {out}")
    off = ~np.eye(n, dtype=bool)
    print(f"\ncontrastive: mean {np.nanmean(con[off]):.2f}  median "
          f"{np.nanmedian(con[off]):.2f}  min {np.nanmin(con[off]):.2f}")
    print(f"raw        : mean {np.nanmean(raw):.2f}  median "
          f"{np.nanmedian(raw):.2f}  min {np.nanmin(raw):.2f}")
    print(f"8-reference baseline: median {np.median(list(base.values())):.2f}")


if __name__ == "__main__":
    main()
