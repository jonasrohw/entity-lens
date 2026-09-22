"""The S10 experiment on PERSON subjects.

HS: *"can we do the same experiment for person?"*

Only as a two-way test.  The four-way country design has no person
analogue: the person relations partition their subjects (footballers for
sports, musicians for instruments, notable figures for parents), so among
person relations whose operators work at all -- faithfulness >= 0.25 --
no three-way intersection reaches eight subjects.

The structurally ideal pair is unusable.  person_father x person_mother
share 748 subjects and both answers are person names, which is the hard
case, exactly like capital/largest.  But their LRE faithfulness is 0.06
and 0.12: the operators cannot produce the answer, so nothing can
substitute for them, and the matrix would be empty for reasons that have
nothing to do with the decomposition.  That is the `country_currency`
problem at its worst.

What remains is person_plays_position_in_sport x person_plays_pro_sport:
24 shared athletes, faithfulness 0.86 and 0.91, genuinely different
templates, and answers of different kinds (quarterback/football,
midfielder/soccer, pitcher/baseball).

CAUTION on reading the result: only five distinct sports cover the 24
athletes, `football` alone covering eight, so an anchor that simply
emits the majority answer could score well with the subject term
contributing nothing.  The `v alone` row and the cross pairings are what
show whether that is happening.

S9 of notes/affine-decomposition.md showed that the relation's anchor
plus the NEUTRAL-prompt contrastive lens,

    A_r + beta * v_e ,

names the right capital for 6 of 9 countries, although v_e alone never
names a city at all.  Four relations in the benchmark take a country
subject -- capital city, currency, language, largest city -- and twelve
countries are shared by all four.  That turns S9's two-way test into a
four-way one, and into a confusion matrix rather than a score: if the
anchor supplies the relation, then holding v_e fixed and changing only
A_r should change which relation's object comes out.

The four answers are of quite different kinds (a city, a currency, a
language), so a diagonal matrix is a weaker result here than in S8, where
both answers were cities.  The informative cells are off-diagonal, and
one in particular: v_e alone returns NATIONALITIES (`Japanese`,
`Turkish`, `Brazilian`), which is most of the way to the language answer.
So `country_language` is the relation where the lens may already suffice
and the anchor may add nothing -- a prediction worth recording before
looking.

Leave-one-out over the twelve shared countries: each is read with the
other eleven as references.  In-context demonstrations come from each
relation's own samples outside the twelve, so no reference or test
country appears in any prompt.
"""

import argparse
import collections
import glob
import json
import os
import sys

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
RELATIONS = ["person_plays_position_in_sport", "person_plays_pro_sport"]
SHORT = {"person_plays_position_in_sport": "position",
         "person_plays_pro_sport": "sport"}


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
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--n-icl", type=int, default=8)
    ap.add_argument("--betas", default="3,5,10")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=f"{ROOT}/e2/country_anchor_lens.json")
    a = ap.parse_args()
    device = a.device or claim_gpu()
    betas = [float(b) for b in a.betas.split(",")]
    print(f"using {device}; layer {a.layer}; betas {betas}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    data = {}
    for r in RELATIONS:
        p = f"{ROOT}/paper/code/data/relations/{r}.json"
        d = json.load(open(p))
        data[r] = ({x["subject"]: x["object"] for x in d["samples"]},
                   d["prompt_templates"][0])
    shared = sorted(set.intersection(*[set(data[r][0]) for r in RELATIONS]))
    print(f"{len(shared)} subjects shared by all {len(RELATIONS)} relations: "
          f"{', '.join(shared)}", flush=True)

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

    def state(prompt, sub):
        ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        i = prompt.rindex(sub) + len(sub)
        pos = min(len(tokenizer(prompt[:i]).input_ids) - 1, ids.shape[1] - 1)
        o, _ = forward_capturing_final_block(
            model, blocks[-1], ids, output_hidden_states=True)
        return ids, pos, o.hidden_states[L][0, pos, :].detach().clone()

    rows = []
    for held in shared:
        refs = [c for c in shared if c != held]
        gold_tok = {r: first_tok(data[r][0][held]) for r in RELATIONS}
        terms = {}
        for r in RELATIONS:
            gold, tpl = data[r]
            icl = [c for c in gold if c not in shared][:a.n_icl]
            ctx = "".join(tpl.format(c) + " " + gold[c] + "\n" for c in icl)
            with torch.no_grad():
                C = {s: state(ctx + tpl.format(s), s) for s in refs + [held]}
            h = blocks[L - 1].register_forward_hook(iv.hook)
            with torch.no_grad():
                anchor = torch.stack([iv.forward(C[s][2], C[s][0], C[s][1]).detach()
                                      for s in refs]).mean(0)
            acc = None
            for s in refs:
                v = iv.jvp(C[s][2], C[held][2] - C[s][2], C[s][0], C[s][1])
                acc = v if acc is None else acc + v
            h.remove()
            terms[r] = (anchor, acc / len(refs))

        with torch.no_grad():
            N = {s: state(NEUTRAL_PROMPT.format(s), s) for s in refs + [held]}
        h = blocks[L - 1].register_forward_hook(iv.hook)
        acc = None
        for s in refs:
            v = iv.jvp(N[s][2], N[held][2] - N[s][2], N[s][0], N[s][1])
            acc = v if acc is None else acc + v
        h.remove()
        v_lens = acc / len(refs)

        # CONTROLS.  (1) the anchor alone: if A[r] already decodes to a
        # plausible object, some "hits" are free and nothing is being
        # composed.  (2) a wrong donor: the same anchor with ANOTHER
        # subject's lens vector.  If that scores as well, the lens is
        # contributing nothing subject-specific.
        donor = shared[(shared.index(held) + 1) % len(shared)]
        if donor == held:
            donor = shared[(shared.index(held) + 2) % len(shared)]
        h = blocks[L - 1].register_forward_hook(iv.hook)
        acc = None
        for s_ in refs:
            vv = iv.jvp(N[s_][2], N[donor][2] - N[s_][2], N[s_][0], N[s_][1])
            acc = vv if acc is None else acc + vv
        h.remove()
        v_donor = acc / len(refs)

        row = dict(country=held, donor=donor,
                   gold={r: data[r][0][held] for r in RELATIONS}, combos={})
        combos = [("v alone", torch.zeros_like(v_lens), v_lens)]
        for r in RELATIONS:
            combos.append((f"A[{SHORT[r]}] ALONE", terms[r][0],
                           torch.zeros_like(v_lens)))
            combos.append((f"A[{SHORT[r]}] + v[WRONG]", terms[r][0], v_donor))
        for r in RELATIONS:
            combos.append((f"A[{SHORT[r]}] + v", terms[r][0], v_lens))
        # the full LRE-to-LRE mix and match: every anchor against every
        # subject term, the diagonal being the true operators
        for r in RELATIONS:
            for r2 in RELATIONS:
                combos.append((f"A[{SHORT[r]}] + T[{SHORT[r2]}]",
                               terms[r][0], terms[r2][1]))
        for name, anc, sub in combos:
            per = {}
            for b in betas:
                lg = dec(anc + b * sub)
                t = int(lg.argmax())
                per[str(b)] = dict(
                    top1=tokenizer.decode([t]).strip(),
                    matches=[SHORT[r] for r in RELATIONS if t == gold_tok[r]])
            row["combos"][name] = per
        rows.append(row)
        best = str(betas[min(1, len(betas) - 1)])
        print(f"  {held:14s} v->{row['combos']['v alone'][best]['top1'][:12]:12s} "
              + " ".join(
                  f"{SHORT[r][:4]}->{row['combos'][f'A[{SHORT[r]}] + v'][best]['top1'][:11]:11s}"
                  for r in RELATIONS), flush=True)

    json.dump(dict(layer=L, betas=betas, shared=shared, rows=rows),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

    n = len(rows)
    for b in betas:
        print(f"\n=== beta = {b:g} ===")
        print(f"{'combination':>28}  " + "  ".join(
            f"{SHORT[r][:8]:>8}" for r in RELATIONS) + "   (matches which "
            f"relation's gold, of {n})")
        for name in [f"A[{SHORT[r]}] ALONE" for r in RELATIONS] + \
                    [f"A[{SHORT[r]}] + v[WRONG]" for r in RELATIONS] + \
                    [f"A[{SHORT[r]}] + v" for r in RELATIONS] + \
                    [f"A[{SHORT[r]}] + T[{SHORT[r2]}]"
                     for r in RELATIONS for r2 in RELATIONS] + \
                    ["v alone"]:
            counts = collections.Counter()
            for row in rows:
                for m in row["combos"][name][str(b)]["matches"]:
                    counts[m] += 1
            print(f"{name:>28}  " + "  ".join(
                f"{counts[SHORT[r]]:>8}" for r in RELATIONS))


if __name__ == "__main__":
    main()
