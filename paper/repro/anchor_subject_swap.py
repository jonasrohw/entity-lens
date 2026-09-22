"""Mix and match anchors and subject terms across two relations.

HS: *"i would be more interested in mixing and matching the anchors and
subject terms, e.g., for country-capital and country-largest. or this
doesn't make sense?"*

It makes a great deal of sense: it is the interchange test of the
decomposition, and it is far stronger than the cosines in S3 of
notes/affine-decomposition.md, which are correlational.  An LRE is

    LRE_r(e) = A_r + beta * T_r(e)

and the claim is that A_r carries the RELATION and T_r(e) the SUBJECT.
If that is right, then feeding the capital anchor the largest-city
subject term should still name the capital:

    A_capital + beta * T_largest(e)   -> the capital of e
    A_largest + beta * T_capital(e)   -> the largest city of e

and if it is wrong, the subject term should drag the answer with it.
This is falsifiable in one run.

Only countries whose capital differs from their largest city can tell the
two apart.  There are nine of the eighteen shared by the two relations:
Australia, Brazil, Canada, China, India, Nigeria, Pakistan, Turkey and
the United States.

**The reference set is the subtlety that can make this vacuous.**  A_r is
mean_i f_r(s_i) over the references.  If the references are countries
whose capital IS their largest city, then f_capital and f_largest agree
on every one of them, the two anchors come out nearly identical, and
swapping them does nothing -- the experiment would "confirm" the
decomposition for a reason that has nothing to do with it.  So the
references must themselves be discriminating countries.  We therefore
run leave-one-out over the nine: each is tested with the other eight as
references.

In-context demonstrations are drawn from the relation's own samples
outside the nine, so no test or reference country appears in its prompt.
"""

import argparse
import glob
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

from jlens.config import MODEL_ID, NEUTRAL_PROMPT

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))


def claim_gpu():
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
    ap.add_argument("--rel-a", default="country_capital_city")
    ap.add_argument("--rel-b", default="country_largest_city")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--n-icl", type=int, default=8)
    ap.add_argument("--betas", default="1,2,3,5")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=f"{ROOT}/e2/anchor_subject_swap.json")
    a = ap.parse_args()
    device = a.device or claim_gpu()
    betas = [float(b) for b in a.betas.split(",")]
    print(f"using {device}; layer {a.layer}; betas {betas}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    def load(r):
        p = (glob.glob(f"{ROOT}/data/lre_full/*/{r}.json")
             or glob.glob(f"{ROOT}/data/lre_all/{r}.json"))[0]
        d = json.load(open(p))
        return ({x["subject"]: x["object"] for x in d["samples"]},
                d["prompt_templates"][0])

    A, tplA = load(a.rel_a)
    B, tplB = load(a.rel_b)
    shared = sorted(set(A) & set(B))
    differing = [c for c in shared if A[c] != B[c]]
    print(f"shared {len(shared)}, discriminating {len(differing)}: "
          f"{', '.join(differing)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        .to(device).eval())
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
    L = a.layer

    def first_tok(obj):
        return tokenizer.encode(" " + obj, add_special_tokens=False)[0]

    def dec(v):
        return final_logits(v, norm, Wh, bias, cap)

    rows = []
    for held in differing:
        refs = [c for c in differing if c != held]
        terms = {}
        for rel, gold, tpl in ((a.rel_a, A, tplA), (a.rel_b, B, tplB)):
            icl = [c for c in gold if c not in differing][:a.n_icl]
            ctx = "".join(tpl.format(c) + " " + gold[c] + "\n" for c in icl)

            def cache(sub):
                p = ctx + tpl.format(sub)
                ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
                i = p.rindex(sub) + len(sub)
                pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
                o, _ = forward_capturing_final_block(
                    model, blocks[-1], ids, output_hidden_states=True)
                return ids, pos, o.hidden_states[L][0, pos, :].detach().clone()

            with torch.no_grad():
                C = {s: cache(s) for s in refs + [held]}
            h = blocks[L - 1].register_forward_hook(iv.hook)
            with torch.no_grad():
                anchor = torch.stack([iv.forward(C[s][2], C[s][0], C[s][1]).detach()
                                      for s in refs]).mean(0)
            acc = None
            for s in refs:
                v = iv.jvp(C[s][2], C[held][2] - C[s][2], C[s][0], C[s][1])
                acc = v if acc is None else acc + v
            h.remove()
            terms[rel] = (anchor, acc / len(refs))

        # the CONTRASTIVE LENS: same references, but the neutral prompt and
        # no relation anywhere.  Pairing it with a relation anchor asks
        # whether the anchor can supply the relation to a subject readout
        # that never saw one -- a much more foreign subject term than the
        # other relation's, cos(v, T) being 0.18-0.25 against 0.72.
        def ncache(sub):
            p = NEUTRAL_PROMPT.format(sub)
            ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
            i = p.rindex(sub) + len(sub)
            pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
            o, _ = forward_capturing_final_block(
                model, blocks[-1], ids, output_hidden_states=True)
            return ids, pos, o.hidden_states[L][0, pos, :].detach().clone()

        with torch.no_grad():
            N = {s_: ncache(s_) for s_ in refs + [held]}
        h = blocks[L - 1].register_forward_hook(iv.hook)
        acc = None
        for s_ in refs:
            vv = iv.jvp(N[s_][2], N[held][2] - N[s_][2], N[s_][0], N[s_][1])
            acc = vv if acc is None else acc + vv
        h.remove()
        v_lens = acc / len(refs)

        Aa, Ta = terms[a.rel_a]
        Ab, Tb = terms[a.rel_b]
        cap_tok, lgst_tok = first_tok(A[held]), first_tok(B[held])
        row = dict(country=held, capital=A[held], largest=B[held], combos={})
        for name, anc, sub in (("A_cap + T_cap", Aa, Ta),
                               ("A_lgst + T_lgst", Ab, Tb),
                               ("A_cap + T_lgst", Aa, Tb),
                               ("A_lgst + T_cap", Ab, Ta),
                               ("A_cap + v", Aa, v_lens),
                               ("A_lgst + v", Ab, v_lens),
                               ("v alone", torch.zeros_like(Aa), v_lens)):
            per_beta = {}
            for b in betas:
                lg = dec(anc + b * sub)
                t = int(lg.argmax())
                per_beta[str(b)] = dict(
                    top1=tokenizer.decode([t]).strip(),
                    is_capital=bool(t == cap_tok),
                    is_largest=bool(t == lgst_tok),
                    rank_capital=int((lg > lg[cap_tok]).sum()),
                    rank_largest=int((lg > lg[lgst_tok]).sum()))
            row["combos"][name] = per_beta
        rows.append(row)
        print(f"  {held:16s} " + "  ".join(
            f"{k}->{row['combos'][k][str(betas[-1])]['top1'][:10]}"
            for k in row["combos"]), flush=True)

    json.dump(dict(layer=L, betas=betas, differing=differing, rows=rows),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

    print(f"\n{'combination':>18}  " + "  ".join(
        f"b={b:g}: cap/lgst" for b in betas))
    for name in ("A_cap + T_cap", "A_lgst + T_lgst",
                 "A_cap + T_lgst", "A_lgst + T_cap",
                 "A_cap + v", "A_lgst + v", "v alone"):
        cells = []
        for b in betas:
            c = sum(r["combos"][name][str(b)]["is_capital"] for r in rows)
            l = sum(r["combos"][name][str(b)]["is_largest"] for r in rows)
            cells.append(f"{c:>2}/{l:<2}")
        print(f"{name:>18}  " + "        ".join(cells))
    print(f"\n(of {len(rows)} countries; 'cap' = top-1 is the capital, "
          f"'lgst' = the largest city)")


if __name__ == "__main__":
    main()
