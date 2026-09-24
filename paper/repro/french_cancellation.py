"""Does a same-answer reference set cancel the answer?

HS: *"if we use persons across relations, then results on french people
drop in name-language, right? what happens if we use *only* french
people from name-language as references for predicting french people in
name-language? I would have expected that 'French' gets canceled out in
the vector difference?"*

The prediction follows directly from the construction.  With

    v_e = mean_i J_i (s_e - s_i) = W s_e - c,    c = mean_i J_i s_i

if every reference is a French speaker then c is a readout pointing at
`French`, and subtracting it suppresses exactly the token we are trying
to predict.  The raw lens W s_e has no such term and should be far less
affected.  S4 of notes/single-reference-matrix.md already saw a version
of this -- same-answer references scored 3.05 against 2.18 -- but never
isolated it.

Why person_native_language is the right place to see it: 69 of its 100
subjects are French speakers and the model's own top-1 is the single
token `French` for 65 of those 69, so "the answer" is one token and
cancellation has somewhere visible to show up.

THE DESIGN.  Eight French and eight non-French references are designated
up front and removed from the test set, so all four arms score the
SAME fixed 61 French and 23 non-French subjects.  Reference composition
is then the only thing that varies:

    french8      8 French speakers        -- c should point at `French`
    nonfrench8   8 non-French speakers    -- c should point elsewhere
    mixed44      4 of each                -- half the cancellation
    cross8       8 persons from person_occupation / person_plays_instrument,
                 outside the relation entirely (the S5 condition)

The constant c is decoded for each arm and printed, which is the
qualitative half of the result: it shows what is being subtracted.
"""

import argparse
import json
import os
import random
import subprocess
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
REL = "person_native_language"
OTHER = ["person_occupation", "person_plays_instrument"]


def claim_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
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
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-ref", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=f"{ROOT}/e2/french_cancellation.json")
    a = ap.parse_args()
    device = a.device or claim_gpu()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    stored = json.load(open(f"{ROOT}/e2/mainlens/lens_{REL}.json"))[REL]
    subs = [r["subject"] for r in stored["rows"]]
    target = {r["subject"]: r["model_top1"] for r in stored["rows"]}
    gold = {x["subject"]: x["object"] for x in
            json.load(open(f"{ROOT}/paper/code/data/relations/{REL}.json"))["samples"]}
    french = [s for s in subs if gold.get(s) == "French"]
    nonfrench = [s for s in subs if gold.get(s) != "French"]

    rng = random.Random(a.seed)
    fr_ref = sorted(rng.sample(french, a.n_ref))
    nf_ref = sorted(rng.sample(nonfrench, a.n_ref))
    fr_test = [s for s in french if s not in fr_ref]
    nf_test = [s for s in nonfrench if s not in nf_ref]

    cross = []
    for r in OTHER:
        pool = [x["subject"] for x in
                json.load(open(f"{ROOT}/e2/mainlens/lens_{r}.json"))[r]["rows"]]
        cross += rng.sample(pool, a.n_ref // 2)

    arms = {"french8": fr_ref,
            "nonfrench8": nf_ref,
            "mixed44": fr_ref[: a.n_ref // 2] + nf_ref[: a.n_ref // 2],
            "cross8": cross}

    print(f"device {device}; test sets fixed at {len(fr_test)} French and "
          f"{len(nf_test)} non-French subjects", flush=True)
    for k, v in arms.items():
        print(f"  {k:11s} " + ", ".join(v), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        .to(device).eval())
    for p in model.parameters():
        p.requires_grad_(False)
    norm = resolve_final_norm(model)
    Wh = model.lm_head.weight.detach().float()
    bias = getattr(model.lm_head, "bias", None)
    bias = bias.detach().float() if bias is not None else None
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    blocks = resolve_blocks(model)
    iv = HiddenStateIntervention(model, final_block=blocks[-1])
    L = a.layer

    def state(s):
        p = NEUTRAL_PROMPT.format(s)
        ids = tok(p, return_tensors="pt").to(device)["input_ids"]
        i = p.rindex(s) + len(s)
        pos = min(len(tok(p[:i]).input_ids) - 1, ids.shape[1] - 1)
        o, _ = forward_capturing_final_block(
            model, blocks[-1], ids, output_hidden_states=True)
        return ids, pos, o.hidden_states[L][0, pos, :].detach().clone()

    need = set(subs) | {r for v in arms.values() for r in v}
    with torch.no_grad():
        cache = {s: state(s) for s in sorted(need)}
    tgt = {s: tok.encode(" " + target[s], add_special_tokens=False)[0]
           for s in subs}
    french_tok = tok.encode(" French", add_special_tokens=False)[0]

    def top10(v):
        lg = final_logits(v, norm, Wh, bias, cap)
        return [tok.decode([int(i)]).strip()
                for i in torch.topk(lg, 10).indices.tolist()]

    handle = blocks[L - 1].register_forward_hook(iv.hook)
    res = {}
    for name, refs in arms.items():
        const = {}
        for r in refs:
            ids, pos, h = cache[r]
            const[r] = iv.jvp(h, h, ids, pos)
        cvec = torch.stack([const[r] for r in refs]).mean(0)
        row = {"references": refs, "c_top10": top10(cvec),
               "c_norm": float(cvec.norm())}
        ex = {}
        for group, test in (("french", fr_test), ("nonfrench", nf_test)):
            raw_r, con_r, fr_rank = [], [], []
            for s in test:
                acc_raw = acc_con = None
                for r in refs:
                    ids, pos, h = cache[r]
                    jse = iv.jvp(h, cache[s][2], ids, pos)
                    acc_raw = jse if acc_raw is None else acc_raw + jse
                    d = jse - const[r]
                    acc_con = d if acc_con is None else acc_con + d
                vcon, vraw = acc_con / len(refs), acc_raw / len(refs)
                lgr = final_logits(vraw, norm, Wh, bias, cap)
                lgc = final_logits(vcon, norm, Wh, bias, cap)
                raw_r.append(np.log10(int((lgr > lgr[tgt[s]]).sum()) + 1))
                con_r.append(np.log10(int((lgc > lgc[tgt[s]]).sum()) + 1))
                fr_con = int((lgc > lgc[french_tok]).sum())
                fr_rank.append(np.log10(fr_con + 1))
                if group == "french":
                    # every French subject, so an example can be chosen on
                    # the numbers rather than on the first few dumped
                    ex[s] = {"french_rank_con": fr_con,
                             "french_rank_raw": int((lgr > lgr[french_tok]).sum()),
                             "target_rank_con": int((lgc > lgc[tgt[s]]).sum()),
                             "target": target[s],
                             "contrastive": top10(vcon), "raw": top10(vraw)}
            row[group] = dict(
                n=len(test),
                con=float(np.median(con_r)), raw=float(np.median(raw_r)),
                con_mean=float(np.mean(con_r)), raw_mean=float(np.mean(raw_r)),
                french_token_rank=float(np.median(fr_rank)))
        row["examples"] = ex
        res[name] = row
        print(f"\n  {name}  ||c||={row['c_norm']:.2f}")
        print(f"    c decodes to: {', '.join(row['c_top10'])}")
        for g in ("french", "nonfrench"):
            v = row[g]
            print(f"    {g:10s} n={v['n']:3d}  contrastive {v['con']:.2f}"
                  f"  raw {v['raw']:.2f}"
                  f"  [rank of ' French' {v['french_token_rank']:.2f}]",
                  flush=True)
    handle.remove()

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
