"""Relation-agnostic reference sets: three mixed sets, all three relations.

HS: *"one issue is the per-relation imbalance of the sets. to look at the
impact of this factor, run the following. select three disjoint sets
randomly from the pool of all persons in person-occupation,
person-language, person-instrument. but make sure that we have at least
2 from each in a set. then use these three sets to score each of the
relations."*

Every reference set measured so far was drawn from the relation it was
scoring, so a set inherits that relation's composition --- and the
compositions are lopsided (`person_native_language` is 69% French
speakers).  Any apparent "reference quality" could be an artifact of
that, and S3/S4 of notes/plan-reference-selection.md cannot separate the
two.

Mixing across relations breaks the confound.  It is also the sharper
test of the construction: the Jacobians are taken under the NEUTRAL
prompt, so

    W = mean_i J_i        and        c = mean_i J_i s_i

depend on the references ALONE.  One mixed set therefore defines ONE
operator, and scoring it against all three relations asks whether a
single relation-agnostic operator reads out three different relations.
That is what S4(i) of lens.tex claims the type-specific reference set
buys, stated in a directly falsifiable form.

The three relations share no subjects at all (verified: 300 distinct
persons, pairwise intersections empty), so "at least 2 from each" is
unambiguous and the sets are cleanly disjoint.

Set construction depends only on the pooled subject lists and the seed,
never on which relation is being scored, so the three relations can run
as separate processes and still use identical sets.
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
RELATIONS = ["person_native_language", "person_occupation",
             "person_plays_instrument"]
SHORT = {"person_native_language": "lang", "person_occupation": "occ",
         "person_plays_instrument": "inst"}


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


def load_rows(rel):
    d = json.load(open(f"{ROOT}/e2/mainlens/lens_{rel}.json"))[rel]
    return ([r["subject"] for r in d["rows"]],
            {r["subject"]: r["model_top1"] for r in d["rows"]})


def build_sets(n_sets, n_ref, min_per, seed):
    """Disjoint sets from the pooled 300, each with >= min_per per relation.

    Deterministic in (seed, the stored subject order), so every process
    builds the same sets.
    """
    members = {r: load_rows(r)[0] for r in RELATIONS}
    for a in RELATIONS:
        for b in RELATIONS:
            if a < b:
                assert not (set(members[a]) & set(members[b])), \
                    f"{a} and {b} share subjects; set construction is ambiguous"
    rng = random.Random(seed)
    avail = {r: list(members[r]) for r in RELATIONS}
    for r in RELATIONS:
        rng.shuffle(avail[r])
    sets = []
    for _ in range(n_sets):
        chosen = []
        for r in RELATIONS:                      # the quota
            chosen += [(avail[r].pop(), r) for _ in range(min_per)]
        rest = [(s, r) for r in RELATIONS for s in avail[r]]
        rng.shuffle(rest)
        while len(chosen) < n_ref:               # free slots, any relation
            s, r = rest.pop()
            avail[r].remove(s)
            chosen.append((s, r))
        rng.shuffle(chosen)
        sets.append(chosen)
    flat = [s for st in sets for s, _ in st]
    assert len(flat) == len(set(flat)), "sets are not disjoint"
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="person_native_language",
                    choices=RELATIONS)
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-ref", type=int, default=8)
    ap.add_argument("--n-sets", type=int, default=3)
    ap.add_argument("--min-per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = a.device or claim_gpu()
    out = a.out or f"{ROOT}/e2/mixed_refs_{a.relation}.json"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    sets = build_sets(a.n_sets, a.n_ref, a.min_per, a.seed)
    subs, target = load_rows(a.relation)
    print(f"device {device}; scoring {a.relation}", flush=True)
    for k, st in enumerate(sets):
        comp = {r: sum(1 for _, rr in st if rr == r) for r in RELATIONS}
        print(f"  set {k}: " + " ".join(f"{SHORT[r]}={comp[r]}"
                                        for r in RELATIONS)
              + " | " + ", ".join(f"{s} [{SHORT[r]}]" for s, r in st),
              flush=True)

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

    need = set(subs) | {s for st in sets for s, _ in st}
    with torch.no_grad():
        cache = {s: state(s) for s in sorted(need)}
    tgt = {s: tok.encode(" " + target[s], add_special_tokens=False)[0]
           for s in subs}

    handle = blocks[L - 1].register_forward_hook(iv.hook)
    res = {}
    for k, st in enumerate(sets):
        refs = [s for s, _ in st]
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
        res[f"set{k}"] = dict(
            references=[{"subject": s, "from": r} for s, r in st],
            composition={r: sum(1 for _, rr in st if rr == r)
                         for r in RELATIONS},
            n=len(test),
            raw=float(np.median(raw_r)), con=float(np.median(con_r)),
            raw_mean=float(np.mean(raw_r)), con_mean=float(np.mean(con_r)),
            const_norm=float(np.linalg.norm(
                np.stack([const[r].cpu().numpy() for r in refs]).mean(0))))
        v = res[f"set{k}"]
        print(f"  set{k}  n={v['n']:3d}  contrastive {v['con']:.2f}"
              f"  raw {v['raw']:.2f}  ||c||={v['const_norm']:.2f}", flush=True)
    handle.remove()

    res["config"] = dict(relation=a.relation, seed=a.seed, n_ref=a.n_ref,
                         n_sets=a.n_sets, min_per=a.min_per, layer=L)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
