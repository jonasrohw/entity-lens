"""End-to-end test of the reference-selection procedure.

HS: *"run this on the 100 data points and compare results for raw lens
and contrastive lens."*

The procedure is notes/plan-reference-selection.md:

    score J_e s_e  ->  keep smallest-norm fifth  ->  drop degenerates
    ->  diversify by minimizing ||mean_i J_i s_i||

Arms, all scored on the same subjects and the same targets:

    original    the split's own eight references (the 2.52 baseline)
    selected    the procedure's eight
    survivors   eight drawn at random from the norm filter's survivors,
                which isolates what steps 3-4 add over step 2 alone
    random      eight drawn at random from all 100, the null

One JVP per (subject, reference) gives BOTH readouts, since
J_i(s_e - s_i) = J_i s_e - J_i s_i and the second term is one extra JVP
per reference.  A subject that is itself a reference is dropped from its
own arm's test set, so each arm reports its own n.
"""

import argparse
import json
import os
import random
import re
import sys

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

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))


def degenerate(name, top5):
    """Step 3, made mechanical so the run is reproducible.

    Flags a readout whose top-5 is dominated by the subject's own name,
    by single characters, or by bare numbers.  Deliberately crude: a
    false positive costs one of eleven survivors.
    """
    nm = re.sub(r"[^a-z]", "", name.lower())
    echo = single = numeric = 0
    for t in top5:
        tt = re.sub(r"[^A-Za-z0-9]", "", t)
        low = tt.lower()
        if len(low) >= 3 and (low in nm or nm in low):
            echo += 1
        if len(tt) <= 2 and tt.isalpha():
            single += 1
        if tt.isdigit():
            numeric += 1
    return echo >= 2 or single >= 2 or numeric >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="person_native_language")
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-ref", type=int, default=8)
    ap.add_argument("--keep", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=0,
                    help="extra random and survivor draws, to measure how "
                         "much of any arm gap is draw-to-draw noise")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"{ROOT}/e2/reference_selection_{a.relation}.json"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    stored = json.load(open(f"{ROOT}/e2/mainlens/lens_{a.relation}.json"))[a.relation]
    subs = [r["subject"] for r in stored["rows"]]
    target = {r["subject"]: r["model_top1"] for r in stored["rows"]}
    V = np.load(f"{ROOT}/e2/selfjac_vectors_{a.relation}.npy")
    TOP = json.load(open(f"{ROOT}/e2/selfjac_vectors_{a.relation}.json"))["top10"]
    nrm = np.linalg.norm(V, axis=1)

    # ---- the procedure, on data already computed ----
    survivors = list(np.argsort(nrm)[: a.keep])
    dropped = [i for i in survivors if degenerate(subs[i], TOP[i][:5])]
    clean = [i for i in survivors if i not in dropped]
    print(f"step 2: kept {len(survivors)} of {len(subs)} by norm")
    print(f"step 3: dropped {len(dropped)} degenerate -> "
          + ", ".join(subs[i] for i in dropped))
    sel = [int(clean[int(np.argmin(nrm[clean]))])]
    while len(sel) < a.n_ref:
        best, bv = None, None
        for i in clean:
            if i in sel:
                continue
            c = np.linalg.norm(V[sel + [i]].mean(0))
            if bv is None or c < bv:
                bv, best = c, int(i)
        sel.append(best)
    print(f"step 4: selected {[subs[i] for i in sel]}")
    print(f"        ||mean constant|| = {np.linalg.norm(V[sel].mean(0)):.2f}")

    rng = random.Random(0)
    arms = {"selected": [subs[i] for i in sel],
            "survivors": [subs[i] for i in rng.sample(clean, a.n_ref)],
            "random": [subs[i] for i in rng.sample(range(len(subs)), a.n_ref)]}

    # the split's own eight, which live outside the 100 test subjects
    data = json.load(open(f"{ROOT}/paper/code/data/relations/{a.relation}.json"))
    pool = list(data["samples"]); random.Random(0).shuffle(pool)
    p2 = list(pool); random.Random(1000).shuffle(p2)
    arms["original"] = [x["subject"] for x in p2[8:16]]

    # Extra draws.  With one random arm per relation there is no way to tell
    # a real gap from the luck of the draw, and on person_plays_instrument
    # the single random arm beat the procedure.  Seeded separately so the
    # four arms above keep their published values.
    for k in range(a.n_random):
        r2 = random.Random(100 + k)
        arms[f"random{k}"] = [subs[i] for i in r2.sample(range(len(subs)), a.n_ref)]
        arms[f"survivors{k}"] = [subs[i] for i in r2.sample(clean, a.n_ref)]

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
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    blocks = resolve_blocks(model)
    iv = HiddenStateIntervention(model, final_block=blocks[-1])
    L = a.layer

    def state(s):
        p = NEUTRAL_PROMPT.format(s)
        ids = tok(p, return_tensors="pt").to(a.device)["input_ids"]
        i = p.rindex(s) + len(s)
        pos = min(len(tok(p[:i]).input_ids) - 1, ids.shape[1] - 1)
        o, _ = forward_capturing_final_block(
            model, blocks[-1], ids, output_hidden_states=True)
        return ids, pos, o.hidden_states[L][0, pos, :].detach().clone()

    need = set(subs) | {r for v in arms.values() for r in v}
    with torch.no_grad():
        cache = {s: state(s) for s in need}
    tgt = {s: tok.encode(" " + target[s], add_special_tokens=False)[0] for s in subs}

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
            for tag, acc, box in (("raw", acc_raw, raw_r), ("con", acc_con, con_r)):
                lg = final_logits(acc / len(refs), norm, Wh, bias, cap)
                box.append(np.log10(int((lg > lg[tgt[s]]).sum()) + 1))
        res[name] = dict(references=refs, n=len(test),
                         raw=float(np.median(raw_r)), con=float(np.median(con_r)),
                         raw_mean=float(np.mean(raw_r)), con_mean=float(np.mean(con_r)),
                         const_norm=float(np.linalg.norm(
                             np.stack([const[r].cpu().numpy() for r in refs]).mean(0))))
        print(f"  {name:10s} n={len(test):3d}  contrastive {res[name]['con']:.2f}  "
              f"raw {res[name]['raw']:.2f}  ||c||={res[name]['const_norm']:.2f}",
              flush=True)
    handle.remove()
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    print(f"\n{'arm':12s} {'n':>4} {'contrastive':>12} {'raw':>7} {'||c||':>7}")
    order = [k for k in ("original", "random", "survivors", "selected")
             if k in res]
    order += [k for k in res if k not in order]
    for k in order:
        r = res[k]
        print(f"{k:12s} {r['n']:4d} {r['con']:12.2f} {r['raw']:7.2f} "
              f"{r['const_norm']:7.2f}")

    for tag in ("random", "survivors"):
        draws = [res[k]["con"] for k in res
                 if k == tag or k.startswith(tag) and k[len(tag):].isdigit()]
        if len(draws) > 1:
            print(f"\n{tag} draws (n={len(draws)}): median contrastive "
                  f"{np.median(draws):.2f}, range {min(draws):.2f}-{max(draws):.2f}")
    print(f"selected: {res['selected']['con']:.2f}")


if __name__ == "__main__":
    main()
