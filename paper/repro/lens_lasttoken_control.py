"""Is the lens reading the entity, or the subject's last token?

HS, on seeing that `Alnair` returns `flying, flown, Flight, flights, fly,
flew, ... airline, airspace, Airways`: *"wouldn't a more direct test be to
look at how similar the ranked lens list is to the ranked lens list of the
last token?"*

It is, and it is the same control `person-occupation.md` already runs for
the namesake collisions -- bare `Sobel` against `Barry Sobel`.  The lens
reads the hidden state at the **last token of the subject span** under the
neutral prompt, and `Alnair` tokenizes as `['Al', 'n', 'air']`.  So the
question has a precise form: replace the subject span with its final token
alone, keep everything else identical, and ask how much of the readout
survives.

For each subject we build a second prompt from the SAME token ids with the
subject span truncated to its last token -- `['Al','n','air',' is',...]`
becomes `['air',' is',...]` -- run the same lens against the same eight
reference stars, and compare.

Three comparisons, because top-k overlap alone is arbitrary:

  * cosine between the two lens vectors (continuous, no cutoff)
  * Jaccard of the two top-100 lists
  * Spearman over the union of the two top-100s

Each needs a null, and we have a good one: the same quantity between two
*different* stars.  A subject whose readout is its own last token scores
far above that null; a subject the model has actually resolved does not.

The payoff is the correlation with gold rank.  If surface-driven subjects
are the ones whose constellation is missing, this is a failure *mechanism*
and not only an observation about one star's name.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

from pathlib import Path

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data
# tree (data/, e2/, results/) one level above the repo.  Override the
# data tree with JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))

from jlens import splits
from jlens.config import LENS_LAYER, MODEL_ID, NEUTRAL_PROMPT

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))
N_REFERENCES = 8
N_ICL = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel", default="star_constellation")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--layer", type=int, default=LENS_LAYER)
    ap.add_argument("--n-test", type=int, default=splits.N_TEST)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--data-dir", default=f"{ROOT}/paper/code/data/relations")
    ap.add_argument("--faith-dir", default=f"{ROOT}/e2/corrected")
    ap.add_argument("--out-dir", default=f"{ROOT}/e2/lenstop")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def claim_gpu():
    """Index of a card nobody is computing on.  Never hardcode a card."""
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
    args = parse_args()
    device = args.device or claim_gpu()
    print(f"using {device}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import (HiddenStateIntervention, final_logits,
                             forward_capturing_final_block, resolve_blocks,
                             resolve_final_norm)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        .to(device).eval())
    for p in model.parameters():
        p.requires_grad_(False)
    norm = resolve_final_norm(model)
    unembedding = model.lm_head.weight.detach().float()
    lm_bias = getattr(model.lm_head, "bias", None)
    lm_bias = lm_bias.detach().float() if lm_bias is not None else None
    softcap = getattr(model.config.get_text_config(),
                      "final_logit_softcapping", None)
    blocks = resolve_blocks(model)
    intervention = HiddenStateIntervention(model, final_block=blocks[-1])
    layer = args.layer

    data = json.load(open(f"{args.data_dir}/{args.rel}.json"))
    samples = splits.filter_identifying_objects(data["samples"], tokenizer, False)
    demonstrations, references, test_samples = splits.split_samples(
        samples, seed=args.split_seed, n_icl=N_ICL, n_ref=N_REFERENCES,
        n_test=args.n_test)
    fingerprint = splits.split_fingerprint(demonstrations, references, test_samples)
    stored = f"{args.faith_dir}/lrefaithT_{args.rel}.json"
    if os.path.exists(stored):
        want = json.load(open(stored)).get("split_fingerprint")
        if want and want != fingerprint:
            raise SystemExit(f"FATAL: fingerprint {fingerprint} != stored {want}")
    print(f"split fingerprint {fingerprint} matches {stored}", flush=True)

    subjects = [x["subject"] for x in test_samples]
    gold = {x["subject"]: x["object"] for x in test_samples}
    anchor_names = [x["subject"] for x in references]

    # how many leading special tokens the tokenizer prepends
    n_special = len(tokenizer("", add_special_tokens=True).input_ids)

    def states_for(input_ids, position):
        o, raw = forward_capturing_final_block(
            model, blocks[-1], input_ids, output_hidden_states=True)
        return (input_ids, position,
                o.hidden_states[layer][0, position, :].detach().clone(),
                raw.detach().clone())

    def full_prompt(subject):
        p = NEUTRAL_PROMPT.format(subject)
        ids = tokenizer(p, return_tensors="pt").to(device)["input_ids"]
        i = p.rindex(subject) + len(subject)
        pos = min(len(tokenizer(p[:i]).input_ids) - 1, ids.shape[1] - 1)
        return ids, pos

    cache, last_tok = {}, {}
    with torch.no_grad():
        for subject in list(dict.fromkeys(subjects + anchor_names)):
            ids, pos = full_prompt(subject)
            cache[subject] = states_for(ids, pos)
            last_tok[subject] = tokenizer.decode([int(ids[0, pos])])
        # the last-token-only variant, built from the SAME ids
        pseudo = {}
        for subject in subjects:
            ids, pos = full_prompt(subject)
            keep = torch.cat([ids[:, :n_special], ids[:, pos:]], dim=1)
            pseudo[subject] = states_for(keep, n_special)

    handle = blocks[layer - 1].register_forward_hook(intervention.hook)

    def lens_vector(state):
        total, n = None, 0
        for anchor in anchor_names:
            ids, pos, reference_hidden, _ = cache[anchor]
            v = intervention.jvp(reference_hidden, state - reference_hidden, ids, pos)
            total = v if total is None else total + v
            n += 1
        return total / n

    def ranked(v):
        logits = final_logits(v, norm, unembedding, lm_bias, softcap)
        order = logits.argsort(descending=True)
        return logits, [int(t) for t in order[:args.topk]]

    rows, vfull, vpseudo = [], {}, {}
    for subject in subjects:
        vf = lens_vector(cache[subject][2])
        vp = lens_vector(pseudo[subject][2])
        vfull[subject], vpseudo[subject] = vf, vp
        lf, tf = ranked(vf)
        lp, tp = ranked(vp)
        cos = float(torch.nn.functional.cosine_similarity(vf, vp, dim=0))
        jac = len(set(tf) & set(tp)) / len(set(tf) | set(tp))
        union = list(set(tf) | set(tp))
        rf = np.array([int((lf > lf[t]).sum()) for t in union])
        rp = np.array([int((lp > lp[t]).sum()) for t in union])
        rho = float(stats.spearmanr(rf, rp).statistic)
        gold_tok = splits.first_object_token(gold[subject], tokenizer)
        rows.append(dict(
            subject=subject, gold=gold[subject], last_token=last_tok[subject],
            n_subject_tokens=int(cache[subject][0].shape[1]
                                 - n_special - 3),  # " is known for" = 3
            cos=cos, jaccard=jac, spearman=rho,
            overlap_at10=len(set(tf[:10]) & set(tp[:10])),
            gold_rank=int((lf > lf[gold_tok]).sum()),
            gold_rank_lasttoken=int((lp > lp[gold_tok]).sum()),
            top_full=[tokenizer.decode([t]).strip() for t in tf[:12]],
            top_lasttoken=[tokenizer.decode([t]).strip() for t in tp[:12]],
        ))
    handle.remove()

    # --- the null: the same quantities between two DIFFERENT stars ---
    pairs = list(itertools.combinations(subjects, 2))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), size=min(600, len(pairs)), replace=False)
    null_cos = [float(torch.nn.functional.cosine_similarity(
        vfull[pairs[i][0]], vfull[pairs[i][1]], dim=0)) for i in idx]

    summarize(args, rows, null_cos, fingerprint)


def summarize(args, rows, null_cos, fingerprint):
    cos = np.array([r["cos"] for r in rows])
    jac = np.array([r["jaccard"] for r in rows])
    gr = np.array([r["gold_rank"] for r in rows], dtype=float)
    nullm = float(np.mean(null_cos))
    summary = dict(
        relation=args.rel, n=len(rows), layer=args.layer, topk=args.topk,
        split_fingerprint=fingerprint,
        mean_cos_full_vs_lasttoken=float(cos.mean()),
        median_cos_full_vs_lasttoken=float(np.median(cos)),
        mean_cos_between_different_stars_NULL=nullm,
        mean_jaccard_full_vs_lasttoken=float(jac.mean()),
        median_jaccard=float(np.median(jac)),
        mean_spearman=float(np.mean([r["spearman"] for r in rows])),
        frac_cos_above_null=float((cos > nullm).mean()),
        spearman_cos_vs_gold_rank=float(stats.spearmanr(cos, gr).statistic),
        spearman_cos_vs_gold_rank_p=float(stats.spearmanr(cos, gr).pvalue),
    )
    jout = f"{args.out_dir}/lasttoken_{args.rel}.json"
    json.dump(dict(summary=summary, rows=rows), open(jout, "w"), indent=1)

    order = sorted(rows, key=lambda r: -r["cos"])
    tout = f"{args.out_dir}/lasttoken_{args.rel}.txt"
    with open(tout, "w") as fh:
        fh.write(f"full-subject lens vs last-token-only lens, {args.rel}\n")
        for k, v in summary.items():
            fh.write(f"  {k}: {v}\n")
        fh.write("\nsorted by cosine (most surface-driven first)\n\n")
        for r in order:
            fh.write(f"{r['subject']:22s} last={r['last_token']!r:12s} "
                     f"cos={r['cos']:+.3f} jac={r['jaccard']:.3f} "
                     f"ov@10={r['overlap_at10']:2d} gold={r['gold']:<12s} "
                     f"rank {r['gold_rank']:>6d} (last-tok {r['gold_rank_lasttoken']:>6d})\n")
            fh.write(f"    full: {', '.join(r['top_full'])}\n")
            fh.write(f"    last: {', '.join(r['top_lasttoken'])}\n\n")
    print(f"wrote {jout}\nwrote {tout}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
