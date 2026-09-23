"""Two disjoint reference sets from the MIDDLE of the norm range.

HS: *"i would like to see if we take two disjoint sets from the middle
range of the norm. can we compare that to the other arms? i'm looking
for a simpler and more robust selection criterion."*

The motivation is what the French readouts in e2/selfjac_french_by_norm.txt
show at the two ends.  The smallest norms include the degenerate cases
(`Marc Alyn` -> A, A, A; `Marina Hands` -> Hands, Hand), which is why the
S1 procedure needs a hand-written step 3 at all.  The largest norms are
one dominant fact swamping everything (`Charles Chamberland` ->
filtering; `Jean-Antoine Chaptal` -> sweet, sugar).  If both extremes are
bad for different reasons, the middle may be the simple rule: no
thresholds fitted per relation, no degenerate list, just "avoid both
ends".

Two DISJOINT sets, which also fixes the weakness of S3 of
notes/plan-reference-selection.md: the nine survivor draws there take 8
from a pool of 16 and share most of their members, so they are not
independent replicates.  Two disjoint sets of 8 are.

Everything else follows reference_selection_test.py exactly -- same
subjects, same targets (the model's own top-1), same neutral prompt,
same layer, one JVP per (subject, reference) giving both readouts -- so
the output is directly comparable to the arms already measured in
e2/reference_selection_{relation}_draws_b.json.
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


def claim_gpu():
    """Never take a card someone else is on; the machines are shared."""
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
    ap.add_argument("--relation", default="person_native_language")
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-ref", type=int, default=8)
    ap.add_argument("--lo", type=int, default=34,
                    help="first rank of the middle band, 1-based")
    ap.add_argument("--hi", type=int, default=67,
                    help="last rank of the middle band, inclusive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = a.device or claim_gpu()
    out = a.out or f"{ROOT}/e2/mid_band_{a.relation}.json"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    stored = json.load(
        open(f"{ROOT}/e2/mainlens/lens_{a.relation}.json"))[a.relation]
    subs = [r["subject"] for r in stored["rows"]]
    target = {r["subject"]: r["model_top1"] for r in stored["rows"]}
    V = np.load(f"{ROOT}/e2/selfjac_vectors_{a.relation}.npy")
    meta = json.load(open(f"{ROOT}/e2/selfjac_vectors_{a.relation}.json"))
    assert meta["subjects"] == subs, "selfjac vectors are not aligned to the rows"
    nrm = np.linalg.norm(V, axis=1)

    order = list(np.argsort(nrm))          # ascending norm, 0-based
    band = order[a.lo - 1: a.hi]           # the middle band, by rank
    rng = random.Random(a.seed)
    pool = list(band)
    rng.shuffle(pool)
    mid_a = [int(i) for i in pool[: a.n_ref]]
    mid_b = [int(i) for i in pool[a.n_ref: 2 * a.n_ref]]
    assert not (set(mid_a) & set(mid_b)), "the two middle sets must be disjoint"

    print(f"device {device}; middle band = norm ranks {a.lo}-{a.hi} "
          f"({len(band)} candidates), norms "
          f"{nrm[band].min():.2f}-{nrm[band].max():.2f}", flush=True)
    for nm, s in (("mid_a", mid_a), ("mid_b", mid_b)):
        print(f"  {nm}: " + ", ".join(subs[i] for i in s), flush=True)

    arms = {"mid_a": [subs[i] for i in mid_a],
            "mid_b": [subs[i] for i in mid_b]}

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

    with torch.no_grad():
        cache = {s: state(s) for s in subs}
    tgt = {s: tok.encode(" " + target[s], add_special_tokens=False)[0]
           for s in subs}

    handle = blocks[L - 1].register_forward_hook(iv.hook)
    res = {}
    for name, refs in arms.items():
        const = {}
        for r in refs:
            ids, pos, h = cache[r]
            const[r] = iv.jvp(h, h, ids, pos)
        test = [s for s in subs if s not in refs]
        raw_r, con_r = [], []
        for s in test:
            acc_raw = acc_con = None
            for r in refs:
                ids, pos, h = cache[r]
                jse = iv.jvp(h, cache[s][2], ids, pos)
                acc_raw = jse if acc_raw is None else acc_raw + jse
                d = jse - const[r]
                acc_con = d if acc_con is None else acc_con + d
            for acc, box in ((acc_raw, raw_r), (acc_con, con_r)):
                lg = final_logits(acc / len(refs), norm, Wh, bias, cap)
                box.append(np.log10(int((lg > lg[tgt[s]]).sum()) + 1))
        res[name] = dict(
            references=refs, n=len(test),
            raw=float(np.median(raw_r)), con=float(np.median(con_r)),
            raw_mean=float(np.mean(raw_r)), con_mean=float(np.mean(con_r)),
            const_norm=float(np.linalg.norm(
                np.stack([const[r].cpu().numpy() for r in refs]).mean(0))))
        print(f"  {name:6s} n={len(test):3d}  contrastive {res[name]['con']:.2f}"
              f"  raw {res[name]['raw']:.2f}"
              f"  ||c||={res[name]['const_norm']:.2f}", flush=True)
    handle.remove()

    res["band"] = dict(lo=a.lo, hi=a.hi, seed=a.seed,
                       norm_min=float(nrm[band].min()),
                       norm_max=float(nrm[band].max()))
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
