"""Compare the differential Jacobian lens and the LRE as affine maps.

HS's three questions, on `country_capital_city` vs `country_largest_city`
-- two relations over the same countries.

Both maps are affine on the subject's hidden state:

    lens   f_j(s) = mean_i J_i^neut (s - s_i)                    [no anchor]
    LRE    f_r(s) = A_r + beta * mean_i J_i^r (s - s_i)
           where the anchor A_r = mean_i f(s_i) does not depend on s

So a priori the bridge f' with f'(f_j(s)) ~ f_r(s) is NOT a general
4096x4096 affine map -- fitting one from ~12 countries would leave it
unconstrained on a ~4084-dimensional complement. It is

    f'(x) = beta * x + A_r

which has 1 + 4096 parameters and is derived rather than fitted. What is
worth measuring is the RESIDUAL after applying it, which is exactly the
prompt-induced difference between J^neut and J^r.

Sizes are reported three ways, because Euclidean norm alone misleads in
the residual stream: the decoder applies RMS normalization, so a
displacement parallel to the vector changes no logit.

**Norm handling is delegated to `jlens`, not reimplemented here.** The
first version of this script took `hidden_states[-1]` and applied the
final norm itself, which double-normalizes: on this transformers version
that tuple entry is ALREADY post-norm (`code/jlens/BUG-final-norm-twice.md`,
found by Adrian). Reimplementing the arithmetic is how the two codebases
diverged in the first place, so this now imports
`HiddenStateIntervention`, `final_logits`, `resolve_blocks` and
`resolve_final_norm` from the fixed package.

The prediction under test: relation-specific information lives in the
ANCHOR (a city of the right kind) and subject-specific information in the
JACOBIAN term (which country is this). If so the two relations should
have clearly different anchors and near-identical subject terms.
"""
import argparse, json, glob, os, sys
from pathlib import Path

import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Repo-relative, matching the other scripts here: this file lives at
# <repo>/paper/repro/, the jlens package at <repo>/code/, and the data tree
# (data/, e2/) one level above the repo.  Override the data tree with
# JLENS_DATA_ROOT when it sits elsewhere.
_CODE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(_CODE))
from jlens.model import (HiddenStateIntervention, final_logits, resolve_blocks,
                         resolve_final_norm)

ROOT = os.environ.get("JLENS_DATA_ROOT",
                      str(Path(__file__).resolve().parents[3]))
M = "allenai/Olmo-3-1025-7B"
NEUTRAL = "{} is known for"
LENS_LAYER = 17


def claim_gpu():
    """Index of a card nobody is computing on."""
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
    ap.add_argument("--n-icl", type=int, default=8)
    ap.add_argument("--n-ref", type=int, default=6)
    ap.add_argument("--device", default=None,
                    help="default: claim a genuinely idle card.  Never "
                         "hardcode an index; these machines are shared.")
    ap.add_argument("--force-layer", type=int, default=None,
                    help="evaluate BOTH LRE subject terms at this layer instead "
                         "of each relation's own selected cell, so that the lens "
                         "and the operator are layer-matched and the only "
                         "remaining difference is the prompt. beta is re-picked "
                         "from the stored grid at that layer.")
    ap.add_argument("--zero-shot", action="store_true",
                    help="estimate the LRE with NO in-context examples, so the "
                         "operator and the lens differ only in their template "
                         "rather than in template AND 8 demonstrations. beta "
                         "cannot come from the stored grid (that was fitted "
                         "with ICL), so it is re-picked here by measuring "
                         "zero-shot faithfulness directly.")
    ap.add_argument("--tpl-a", default=None,
                    help="override relation A's template. Two reasons to, in "
                         "the zero-shot run: the benchmark templates do not "
                         "elicit an answer without demonstrations ('The "
                         "capital city of France is' continues 'known', i.e. "
                         "'is known for'), and a SUBJECT-FIRST template makes "
                         "the subject's hidden state identical to the lens's, "
                         "since causal attention means nothing precedes the "
                         "subject in either prompt. That isolates the prompt's "
                         "effect on the Jacobian from its effect on the input.")
    ap.add_argument("--tpl-b", default=None, help="override relation B's template")
    ap.add_argument("--out", default=f"{ROOT}/e2/affine_compare.json")
    a = ap.parse_args()
    DEV = a.device or claim_gpu()
    print(f"  device {DEV}")

    def load(r):
        g = glob.glob(f"{ROOT}/data/lre_full/*/{r}.json") or glob.glob(f"{ROOT}/data/lre_all/{r}.json")
        d = json.load(open(g[0]))
        return {x["subject"]: x["object"] for x in d["samples"]}, d["prompt_templates"][0]

    A, tplA = load(a.rel_a); B, tplB = load(a.rel_b)
    if a.tpl_a: tplA = a.tpl_a
    if a.tpl_b: tplB = a.tpl_b
    print(f"  templates: A {tplA!r} | B {tplB!r}")
    shared = sorted(set(A) & set(B))
    F = {r: json.load(open(f"{ROOT}/e2/unfilt/lrefaithT_{r}.json")) for r in (a.rel_a, a.rel_b)}
    if a.zero_shot:
        # beta is selected later, by measuring zero-shot faithfulness; the
        # stored grid was fitted under the 8-shot prompt and does not transfer.
        L0 = a.force_layer or LENS_LAYER
        cfg = {r: (L0, None) for r in F}
        print(f"  zero-shot: both operators at layer {L0}, beta to be measured",
              flush=True)
    elif a.force_layer is None:
        cfg = {r: (int(F[r]["best_layer"]), float(F[r]["best_beta"])) for r in F}
    else:
        cfg = {}
        for r in F:
            row = F[r]["grid"][str(a.force_layer)]
            best = max(float(v[0]) for v in row.values())
            # smallest beta attaining the maximum at this layer
            b = min(float(k) for k, v in row.items() if float(v[0]) == best)
            cfg[r] = (a.force_layer, b)
            print(f"  forced layer {a.force_layer}: {r} beta={b} "
                  f"(faith {best:.3f} there)", flush=True)

    # same references and same test countries for both relations, so the two
    # operators differ only in their prompt -- otherwise anchors and subject
    # terms would not be comparable
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(shared))
    ref_names, test_names = perm[:a.n_ref], perm[a.n_ref:]
    if a.zero_shot:
        icl = {r: [] for r in (a.rel_a, a.rel_b)}
    else:
        icl = {r: [s for s in (A if r == a.rel_a else B) if s not in shared][:a.n_icl]
               for r in (a.rel_a, a.rel_b)}
        for r in icl:                   # top up from shared-but-unused if needed
            pool = (A if r == a.rel_a else B)
            icl[r] += [s for s in pool if s not in icl[r] and s not in ref_names
                       and s not in test_names][:a.n_icl - len(icl[r])]
    print(f"shared countries {len(shared)}; refs {len(ref_names)}; test {len(test_names)}")
    print(f"  cells: {a.rel_a} layer {cfg[a.rel_a][0]} beta {cfg[a.rel_a][1]} | "
          f"{a.rel_b} layer {cfg[a.rel_b][0]} beta {cfg[a.rel_b][1]}", flush=True)

    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(
        M, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    # Intervention, JVP and final-norm handling all come from the fixed
    # jlens package; see the module docstring.
    blocks = resolve_blocks(model)
    iv = HiddenStateIntervention(model, final_block=blocks[-1])
    hook = iv.hook
    f_of_h = iv.forward
    jvp = iv.jvp
    norm = resolve_final_norm(model)
    Whead = model.lm_head.weight.detach()
    lm_bias = getattr(model.lm_head, "bias", None)
    softcap = getattr(model.config, "final_logit_softcapping", None)

    def cache_for(prompt_of, layer, names):
        out = {}
        with torch.no_grad():
            for s in names:
                p = prompt_of(s)
                ids = tok(p, return_tensors="pt").to(DEV)["input_ids"]
                i = p.rindex(s) + len(s)
                si = min(len(tok(p[:i]).input_ids) - 1, ids.shape[1] - 1)
                o = model(input_ids=ids, output_hidden_states=True)
                out[s] = (ids, si, o.hidden_states[layer][0, si, :].detach().clone())
        return out

    def terms(rel, tpl, layer, beta):
        """anchor A_r, and the subject term T_r(e) for each test country."""
        ctx = "".join(tpl.format(s) + " " + (A if rel == a.rel_a else B)[s] + "\n"
                      for s in icl[rel])
        pr = lambda s: ctx + tpl.format(s)
        C = cache_for(pr, layer, ref_names + test_names)
        h = blocks[layer - 1].register_forward_hook(hook)
        with torch.no_grad():
            anchor = torch.stack([f_of_h(C[s][2], C[s][0], C[s][1]).detach()
                                  for s in ref_names]).mean(0)
        T = {}
        for e in test_names:
            acc = torch.zeros_like(anchor)
            for s in ref_names:
                acc = acc + jvp(C[s][2], C[e][2] - C[s][2], C[s][0], C[s][1])
            T[e] = acc / len(ref_names)
        h.remove()
        return anchor, T, {s: C[s][2] for s in test_names}

    # the lens, under the neutral prompt, at the fixed lens layer
    prn = lambda s: NEUTRAL.format(s)
    Cn = cache_for(prn, LENS_LAYER, ref_names + test_names)
    hn = blocks[LENS_LAYER - 1].register_forward_hook(hook)
    V = {}
    for e in test_names:
        acc = torch.zeros(model.config.hidden_size, device=DEV)
        for s in ref_names:
            acc = acc + jvp(Cn[s][2], Cn[e][2] - Cn[s][2], Cn[s][0], Cn[s][1])
        V[e] = acc / len(ref_names)
    hn.remove()
    print("lens vectors done", flush=True)

    res = {}
    for rel, tpl in ((a.rel_a, tplA), (a.rel_b, tplB)):
        L, beta = cfg[rel]
        res[rel] = terms(rel, tpl, L, beta)
        print(f"{rel} terms done", flush=True)

    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x, y, dim=0))
    nrm = lambda x: float(x.norm())
    def decode(v):
        return final_logits(v, norm, Whead, lm_bias, softcap)

    # ---- the gate: does the LRE work at all without demonstrations? ----
    # Faithfulness is LRE top-1 against the MODEL's own top-1 under the same
    # prompt, never against gold -- the same definition as everywhere else.
    gate = {}
    if a.zero_shot:
        for rel, tpl in ((a.rel_a, tplA), (a.rel_b, tplB)):
            gold = A if rel == a.rel_a else B
            anchor, T, _ = res[rel]
            top1 = {}
            with torch.no_grad():
                for e in test_names:
                    p = tpl.format(e)
                    ids = tok(p, return_tensors="pt").to(DEV)["input_ids"]
                    top1[e] = int(model(input_ids=ids).logits[0, -1].argmax())
            gold_tok = {e: tok.encode(" " + gold[e], add_special_tokens=False)[0]
                        for e in test_names}
            model_gold = np.mean([top1[e] == gold_tok[e] for e in test_names])
            sweep = {}
            for b in [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]:
                hit = np.mean([int(decode(anchor + b * T[e]).argmax()) == top1[e]
                               for e in test_names])
                sweep[b] = float(hit)
            best = max(sweep.values())
            beta = min(k for k, v in sweep.items() if v == best)
            cfg[rel] = (cfg[rel][0], float(beta))
            gate[rel] = dict(model_matches_gold=float(model_gold),
                             faith_by_beta=sweep, best_faith=float(best),
                             beta=float(beta),
                             model_top1={e: tok.decode([top1[e]]).strip()
                                         for e in test_names})
            print(f"\nGATE {rel}: model top-1 = gold for "
                  f"{model_gold:.2f} of {len(test_names)} countries zero-shot")
            print(f"  LRE faithfulness by beta: "
                  + ", ".join(f"{k}:{v:.2f}" for k, v in sweep.items()))
            print(f"  -> beta {beta}, faithfulness {best:.3f}", flush=True)

    (Aa, Ta, Ha), (Ab, Tb, Hb) = res[a.rel_a], res[a.rel_b]
    ba, bb = cfg[a.rel_a][1], cfg[a.rel_b][1]
    rows = []
    for e in test_names:
        same = A[e] == B[e]
        fa = Aa + ba * Ta[e]; fb = Ab + bb * Tb[e]
        rows.append(dict(
            country=e, capital=A[e], largest=B[e], same_answer=bool(same),
            cos_lens_subjterm_a=cos(V[e], Ta[e]),
            cos_lens_subjterm_b=cos(V[e], Tb[e]),
            cos_subjterm_a_b=cos(Ta[e], Tb[e]),
            norm_anchor_a=nrm(Aa), norm_beta_subj_a=nrm(ba * Ta[e]),
            norm_anchor_b=nrm(Ab), norm_beta_subj_b=nrm(bb * Tb[e]),
            cos_out_in_a=cos(fa, Ha[e]), cos_out_in_b=cos(fb, Hb[e]),
            disp_a=nrm(fa - Ha[e]), disp_b=nrm(fb - Hb[e]),
            resid_bridge_a=float((fa - (ba * V[e] + Aa)).norm() / fa.norm()),
            resid_bridge_b=float((fb - (bb * V[e] + Ab)).norm() / fb.norm()),
            resid_anchor_only_a=float((fa - Aa).norm() / fa.norm()),
            resid_anchor_only_b=float((fb - Ab).norm() / fb.norm()),
            norm_lens=nrm(V[e]), cos_anchor_subj_a=cos(Aa, Ta[e]),
        ))
    summary = dict(
        zero_shot=bool(a.zero_shot), gate=gate,
        n_test=len(test_names), n_same=sum(r["same_answer"] for r in rows),
        refs=ref_names, cells={r: cfg[r] for r in cfg},
        cos_anchor_a_b=cos(Aa, Ab), norm_anchor_a=nrm(Aa), norm_anchor_b=nrm(Ab),
        dist_anchors=nrm(Aa - Ab),
        anchor_top5_a=[tok.decode([int(i)]).strip() for i in decode(Aa).topk(5).indices],
        anchor_top5_b=[tok.decode([int(i)]).strip() for i in decode(Ab).topk(5).indices],
    )
    json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print(f"\nANCHORS: cos = {summary['cos_anchor_a_b']:+.3f}, "
          f"||A_a|| = {summary['norm_anchor_a']:.1f}, ||A_b|| = {summary['norm_anchor_b']:.1f}, "
          f"||A_a - A_b|| = {summary['dist_anchors']:.1f}")
    print(f"  anchor A decodes to {summary['anchor_top5_a']}")
    print(f"  anchor B decodes to {summary['anchor_top5_b']}")
    m = lambda k, sel=None: np.mean([r[k] for r in rows if sel is None or r["same_answer"] == sel])
    print(f"\nSUBJECT TERMS  cos(T_a, T_b): all {m('cos_subjterm_a_b'):+.3f} | "
          f"same-answer {m('cos_subjterm_a_b', True):+.3f} | differ {m('cos_subjterm_a_b', False):+.3f}")
    print(f"LENS vs subject term: cos(v, T_a) {m('cos_lens_subjterm_a'):+.3f}, "
          f"cos(v, T_b) {m('cos_lens_subjterm_b'):+.3f}")
    print(f"SIZE  ||anchor|| {m('norm_anchor_a'):.1f} vs ||beta*T|| {m('norm_beta_subj_a'):.1f} (A); "
          f"{m('norm_anchor_b'):.1f} vs {m('norm_beta_subj_b'):.1f} (B)")
    print(f"BRIDGE residual (relative):     A {m('resid_bridge_a'):.3f}  B {m('resid_bridge_b'):.3f}")
    print(f"  baseline, anchor alone:       A {m('resid_anchor_only_a'):.3f}  "
          f"B {m('resid_anchor_only_b'):.3f}   <- does adding beta*v help?")
    print(f"  ||lens v|| {m('norm_lens'):.1f};  cos(anchor, subject term) A {m('cos_anchor_subj_a'):+.3f}")


if __name__ == "__main__":
    main()
