"""Exact paired-relation experiments; float32/eager, immutable subject splits."""

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.mechanistic import Runner, cpu_tree, save_json
from jlens.paired_relation_audit import MODEL, REVISION
from jlens.splits import first_object_token

LAYERS = [5, 11, 17, 23]
RELATIONS = ["country_capital_city", "country_currency"]
BETAS = [0., .5, 1., 1.5, 2., 3., 5.]
WIDE_BETAS = [0., .01, .03, .1, .3, .5, 1., 1.5, 2., 3., 5., 10., 30., 100., 300., 1000., 10000.]
GROUPS = ["subject_last", "subject_other", "suffix", "answer"]


def move(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move(v, device) for v in x]
    return x


def save_torch(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(cpu_tree(x), tmp)
    tmp.replace(path)


def load_runner():
    torch.set_num_threads(2)
    torch.manual_seed(2009)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    ).eval().to("cuda:0")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
        assert p.dtype == torch.float32 and p.device.type == "cuda"
    return Runner(model, tokenizer, LAYERS)


def provenance():
    return {"host": socket.gethostname(), "model": MODEL, "revision": REVISION,
            "dtype": "float32", "attention": "eager", "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def prompts(manifest, relation, mode):
    plans = manifest["relations"]
    plan = plans[relation]
    if mode == "original":
        template = plan["template"]
        context = "".join(template.format(x["subject"]) + " " + x["object"] + "\n" for x in plan["parts"]["icl"])
    else:
        template = plan["subject_first_template"]
        context = ""
        if mode == "unit":
            context = ("Answer with only a city name or a currency unit. For currencies, omit country adjectives; "
                       "use unit names such as Dollar, Euro, or Pound.\n")
        for i in range(len(plan["parts"]["icl"])):
            for name in RELATIONS:
                sample = plans[name]["parts"]["icl"][i]
                context += plans[name]["subject_first_template"].format(sample["subject"]) + " " + sample["object"] + "\n"
    return context, template


def score_one(runner, final, target, foil):
    with torch.no_grad():
        logits = runner.logits(final)
        return {"top": int(logits.argmax()), "hit": int(logits.argmax() == target),
                "logp": float(logits.log_softmax(-1)[target]),
                "margin": float(logits[target] - logits[foil])}


def group_positions(runner, prompt, subject):
    enc = runner.tokenizer(prompt, return_offsets_mapping=True)
    end = prompt.rindex(subject) + len(subject)
    start = end - len(subject)
    positions = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    last = len(enc["input_ids"]) - 1
    return {"subject_last": [positions[-1]], "subject_other": positions[:-1],
            "suffix": list(range(positions[-1] + 1, last)), "answer": [last]}


def best_beta(grids, values, nval):
    return max(values, key=lambda b: (np.mean(grids[str(b)]["hit"][:nval]),
                                     np.mean(grids[str(b)]["logp"][:nval]), -b))


def baseline(runner, args, manifest):
    stem = args.out / f"baseline_{args.mode}_{args.relation}"
    if stem.with_suffix(".json").exists() and json.loads(stem.with_suffix(".json").read_text()).get("complete"):
        return
    start = time.time()
    plan = manifest["relations"][args.relation]
    context, template = prompts(manifest, args.relation, args.mode)
    samples = sum([plan["parts"][p] for p in ["references", "train", "validation", "test"]], [])
    cache = {}
    for x in samples:
        subject = x["subject"]
        prompt = context + template.format(subject)
        c = runner.cache(prompt, subject)
        c.pop("components")
        c["groups"] = group_positions(runner, prompt, subject)
        c["prompt"] = prompt
        c["gold"] = first_object_token(x["object"], runner.tokenizer)
        cache[subject] = c
    scored = plan["parts"]["validation"] + plan["parts"]["test"]
    subjects = [x["subject"] for x in scored]
    targets = [cache[s]["top"] for s in subjects]
    gold = [cache[s]["gold"] for s in subjects]
    nv = len(plan["parts"]["validation"])
    refs = [cache[x["subject"]] for x in plan["parts"]["references"]]
    base = torch.stack([c["final"] for c in refs]).mean(0)
    real = torch.stack([cache[s]["final"] for s in subjects])
    tensors = {"cache": cache, "base": base, "products": {}}
    result = {"provenance": provenance(), "relation": args.relation, "mode": args.mode,
              "subjects": subjects, "nval": nv, "targets": targets, "gold": gold,
              "target_text": [runner.tokenizer.decode([x]) for x in targets],
              "gold_text": [runner.tokenizer.decode([x]) for x in gold],
              "model_correct": [int(a == b) for a, b in zip(targets, gold)],
              "full": runner.score(real, targets), "layers": {}}
    print(f"Cached {args.mode} {args.relation} in {time.time()-start:.1f}s", flush=True)
    for layer in LAYERS:
        tick = time.time()
        states = torch.stack([cache[s]["states"][layer] for s in subjects])
        product = runner.products(refs, layer, states, subtract_reference=True)
        grids = {str(b): runner.score(base + b * product, targets) for b in WIDE_BETAS}
        beta, wide = best_beta(grids, BETAS, nv), best_beta(grids, WIDE_BETAS, nv)
        tensors["products"][layer] = product
        result["layers"][str(layer)] = {"beta": beta, "wide_beta": wide,
            "grid": grids, "relative_vector_error": ((base + beta * product - real).norm(dim=-1) / real.norm(dim=-1)).tolist()}
        save_torch(stem.with_suffix(".pt"), tensors)
        save_json(stem.with_suffix(".json"), result)
        print(f"baseline {args.mode} {args.relation} L{layer} beta={beta} faith={np.mean(grids[str(beta)]['hit'][nv:]):.3f} {time.time()-tick:.1f}s", flush=True)
    result["seconds"] = time.time() - start
    result["complete"] = True
    save_json(stem.with_suffix(".json"), result)


def load_baseline(args, relation, mode="controlled", device="cuda:0"):
    stem = args.out / f"baseline_{mode}_{relation}"
    meta = json.loads(stem.with_suffix(".json").read_text())
    assert meta["complete"]
    tensors = torch.load(stem.with_suffix(".pt"), map_location=device, weights_only=False)
    return meta, tensors


def choose_donors(manifest, caches):
    """Training donors: different gold and model tokens in both relations.

    Prioritize equal subject-token counts, then a deterministic subject-name hash.
    The same donor is used for the two relations and all tested layers.
    """
    train = manifest["relations"][RELATIONS[0]]["parts"]["train"]
    result = {}
    for subject in caches[0]:
        candidates = []
        for x in train:
            donor = x["subject"]
            if donor == subject:
                continue
            if all(c[subject]["top"] != c[donor]["top"] and c[subject]["gold"] != c[donor]["gold"] for c in caches):
                length_diff = abs(len(caches[0][subject]["groups"]["subject_other"]) - len(caches[0][donor]["groups"]["subject_other"]))
                order = hashlib.sha256((subject + "|" + donor).encode()).hexdigest()
                candidates.append((length_diff, order, donor))
        if not candidates:
            raise RuntimeError(f"No paired donor for {subject}")
        result[subject] = min(candidates)[2]
    return result


def gradients(runner, args, manifest):
    stem = args.out / "channels_basis"
    if stem.with_suffix(".json").exists():
        return
    packs = [load_baseline(args, r) for r in RELATIONS]
    caches = [p[1]["cache"] for p in packs]
    donors = choose_donors(manifest, caches)
    # Exact causal-prefix control, checked for every reference/train/validation/test subject.
    errors = {}
    for layer in LAYERS:
        errors[layer] = max(float((caches[0][s]["states"][layer] - caches[1][s]["states"][layer]).abs().max()) for s in caches[0])
    assert max(errors.values()) < 2e-5, errors
    for s in caches[0]:
        a, b = caches[0][s], caches[1][s]
        assert a["pos"] == b["pos"]
        assert torch.equal(a["ids"][:, :a["pos"]+1], b["ids"][:, :b["pos"]+1])
    bases, spectra, gradients_all = {}, {}, {}
    train = manifest["relations"][RELATIONS[0]]["parts"]["train"]
    for layer in LAYERS:
        bases[layer], spectra[layer], gradients_all[layer] = {}, {}, {}
        for relation, cache in zip(RELATIONS, caches):
            vectors = []
            for x in train:
                subject = x["subject"]
                c, donor = cache[subject], cache[donors[subject]]
                state = c["states"][layer].clone().requires_grad_(True)
                final = runner.forward(c["ids"], c["pos"], state_patch=(layer, state))[0]
                logits = runner.logits(final)
                gradient = torch.autograd.grad(logits[c["top"]] - logits[donor["top"]], state)[0]
                vectors.append(gradient.detach())
            g = torch.stack(vectors)
            _, singular, v = torch.linalg.svd(g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12), full_matrices=False)
            assert singular[7] > singular[0] * 1e-6
            bases[layer][relation] = v[:8].T.contiguous()
            gradients_all[layer][relation] = g
            spectra[layer][relation] = singular.tolist()
        generator = torch.Generator(device=runner.device).manual_seed(2009 + layer)
        bases[layer]["random"] = [torch.linalg.qr(torch.randn(runner.dim, 8, device=runner.device, generator=generator)).Q for _ in range(3)]
        print(f"gradient bases L{layer} finished", flush=True)
    means = {layer: torch.stack([caches[0][x["subject"]]["states"][layer] for x in train]).mean(0) for layer in LAYERS}
    overlap = {layer: float((bases[layer][RELATIONS[0]].T @ bases[layer][RELATIONS[1]]).square().sum() / 8) for layer in LAYERS}
    save_torch(stem.with_suffix(".pt"), {"bases": bases, "means": means, "gradients": gradients_all})
    save_json(stem.with_suffix(".json"), {"complete": True, "provenance": provenance(), "rank": 8,
        "donors": donors, "identical_state_max_error": errors, "subspace_overlap": overlap, "spectra": spectra})


def channels(runner, args, manifest):
    stem = args.out / f"channels_{args.relation}_{args.layer}"
    if stem.with_suffix(".json").exists():
        return
    meta, pack = load_baseline(args, args.relation)
    basis_meta = json.loads((args.out / "channels_basis.json").read_text())
    learned = torch.load(args.out / "channels_basis.pt", map_location=runner.device, weights_only=False)
    layer = args.layer
    bases, mean = learned["bases"][layer], learned["means"][layer]
    cache = pack["cache"]
    plan = manifest["relations"][args.relation]
    refs = [cache[x["subject"]] for x in plan["parts"]["references"]]
    all_bases = [bases[r] for r in RELATIONS] + bases["random"]
    response = runner.products(refs, layer, torch.cat([q.T for q in all_bases]))
    beta = meta["layers"][str(layer)]["beta"]
    rows = []
    finals = []
    for x in plan["parts"]["test"]:
        subject = x["subject"]
        c = cache[subject]
        target, foil = c["top"], cache[basis_meta["donors"][subject]]["top"]
        state = c["states"][layer]
        index = meta["subjects"].index(subject)
        prediction = pack["base"] + beta * pack["products"][layer][index]
        clean = score_one(runner, c["final"], target, foil)
        clean_lre = score_one(runner, prediction, target, foil)
        for source_i, source in enumerate(RELATIONS):
            q = bases[source]
            norm = ((state - mean) @ q).norm()
            for control_i in [-1, 0, 1, 2]:
                basis_i = source_i if control_i == -1 else control_i + 2
                basis = all_bases[basis_i]
                coefficients = (state - mean) @ basis
                if control_i != -1:
                    coefficients = coefficients * norm / coefficients.norm().clamp_min(1e-12)
                delta = coefficients @ basis.T
                linear_delta = coefficients @ response[basis_i*8:(basis_i+1)*8]
                for strength in [.5, 1.]:
                    with torch.no_grad():
                        final = runner.forward(c["ids"], c["pos"], state_patch=(layer, state - strength * delta))[0]
                        lre_final = prediction - strength * beta * linear_delta
                    rows.append({"subject": subject, "relation": args.relation, "layer": layer, "source": source,
                        "control": control_i, "strength": strength, "perturbation_norm": float(strength*delta.norm()),
                        "relative_perturbation_norm": float(strength*delta.norm()/state.norm()),
                        "clean": clean, "clean_lre": clean_lre, "full": score_one(runner, final, target, foil),
                        "lre": score_one(runner, lre_final, target, foil),
                        "clean_vector_error": float((c["final"]-prediction).norm()),
                        "patched_vector_error": float((final-lre_final).norm())})
            print(f"channels L{layer} {args.relation} {subject} source={source_i} done", flush=True)
    save_json(stem.with_suffix(".json"), {"complete": True, "provenance": provenance(), "rows": rows})


def nonlinear(runner, args, manifest):
    stem = args.out / f"nonlinear_{args.relation}_{args.layer}"
    if stem.with_suffix(".json").exists():
        return
    meta, pack = load_baseline(args, args.relation)
    donors = json.loads((args.out / "channels_basis.json").read_text())["donors"]
    cache, layer = pack["cache"], args.layer
    plan = manifest["relations"][args.relation]
    samples = plan["parts"]["test"]
    refs = [cache[x["subject"]] for x in plan["parts"]["references"]]
    directions = torch.stack([cache[donors[x["subject"]]]["states"][layer] - cache[x["subject"]]["states"][layer] for x in samples])
    global_response = runner.products(refs, layer, directions)
    beta = meta["layers"][str(layer)]["beta"]
    rows, checks = [], []
    for i, x in enumerate(samples):
        subject = x["subject"]
        c, donor = cache[subject], cache[donors[subject]]
        state, delta = c["states"][layer], directions[i]
        local = runner.products([c], layer, delta[None])[0]
        target, foil = c["top"], donor["top"]
        index = meta["subjects"].index(subject)
        prediction = pack["base"] + beta * pack["products"][layer][index]
        if i < 5:
            with torch.no_grad():
                plus = runner.forward(c["ids"], c["pos"], state_patch=(layer, state + .001*delta))[0]
                minus = runner.forward(c["ids"], c["pos"], state_patch=(layer, state - .001*delta))[0]
                identity = runner.forward(c["ids"], c["pos"], state_patch=(layer, state))[0]
            checks.append({"subject": subject, "finite_difference_relative_error": float(((plus-minus)/.002-local).norm()/local.norm().clamp_min(1e-9)),
                           "identity_max_error": float((identity-c["final"]).abs().max())})
        for t in [.25, .5, .75, 1.]:
            with torch.no_grad():
                actual = runner.forward(c["ids"], c["pos"], state_patch=(layer, state+t*delta))[0]
                local_pred = c["final"] + t*local
                global_pred = c["final"] + t*global_response[i]
                lre_pred = prediction + beta*t*global_response[i]
                change_norm = (actual-c["final"]).norm().clamp_min(1e-9)
            rows.append({"subject": subject, "relation": args.relation, "layer": layer, "t": t,
                "donor": donors[subject], "state_distance": float(delta.norm()),
                "output_change_norm": float(change_norm), "local_relative_residual": float((actual-local_pred).norm()/change_norm),
                "global_relative_residual": float((actual-global_pred).norm()/change_norm),
                "local_global_response_cosine": float(torch.nn.functional.cosine_similarity(local, global_response[i], dim=0)),
                "actual": score_one(runner, actual, target, foil), "local": score_one(runner, local_pred, target, foil),
                "global": score_one(runner, global_pred, target, foil), "lre": score_one(runner, lre_pred, target, foil),
                "clean": score_one(runner, c["final"], target, foil)})
        print(f"nonlinear L{layer} {args.relation} {subject} done", flush=True)
    assert max(c["identity_max_error"] for c in checks) < 1e-4
    assert np.median([c["finite_difference_relative_error"] for c in checks]) < .03, checks
    save_json(stem.with_suffix(".json"), {"complete": True, "provenance": provenance(), "checks": checks, "rows": rows})


def module_for(runner, block, kind):
    return getattr(runner.blocks[block-1], "post_attention_layernorm" if kind == "attention" else "post_feedforward_layernorm")


@torch.no_grad()
def capture_components(runner, c, heads=False, blocks=None):
    blocks = blocks or range(1, len(runner.blocks)+1)
    captured, handles = {}, []
    for block in blocks:
        for kind in (["heads"] if heads else ["attention", "mlp"]):
            if heads:
                module = runner.blocks[block-1].self_attn.o_proj
                def hook(_m, inp, block=block, kind=kind):
                    for group, positions in c["groups"].items():
                        captured[(block, kind, group)] = inp[0][0, positions].detach().clone()
                handles.append(module.register_forward_pre_hook(hook))
            else:
                module = module_for(runner, block, kind)
                def hook(_m, _inp, output, block=block, kind=kind):
                    for group, positions in c["groups"].items():
                        captured[(block, kind, group)] = output[0, positions].detach().clone()
                handles.append(module.register_forward_hook(hook))
    try:
        runner.forward(c["ids"], c["pos"])
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def patched_component(runner, c, block, kind, group, value, head=None):
    positions = c["groups"][group]
    assert len(positions) == len(value)
    if head is None:
        module = module_for(runner, block, kind)
        def hook(_m, _inp, output):
            result = output.clone()
            result[0, positions] = value.to(result.device)
            return result
        handle = module.register_forward_hook(hook)
    else:
        module = runner.blocks[block-1].self_attn.o_proj
        width = runner.dim // runner.model.config.num_attention_heads
        start, end = head*width, (head+1)*width
        def hook(_m, inp):
            result = inp[0].clone()
            result[0, positions, start:end] = value[:, start:end].to(result.device)
            return (result, *inp[1:])
        handle = module.register_forward_pre_hook(hook)
    try:
        final, states, _ = runner.forward(c["ids"], c["pos"], capture=True)
    finally:
        handle.remove()
    return final, states[17]


def routes(runner, args, manifest, heads=False):
    label = "heads" if heads else "routes"
    stem = args.out / f"{label}_{args.relation}_{args.shard}"
    if stem.with_suffix(".json").exists():
        return
    meta, pack = load_baseline(args, args.relation)
    donors = json.loads((args.out / "channels_basis.json").read_text())["donors"]
    plan = manifest["relations"][args.relation]
    cache = pack["cache"]
    selected = json.loads((args.out / "head_selection.json").read_text())["selected"] if heads else None
    samples = (plan["parts"]["test"] if heads else plan["parts"]["validation"] + plan["parts"]["test"])[args.shard::args.nshards]
    validation = {x["subject"] for x in plan["parts"]["validation"]}
    rows, checks, skipped = [], [], []
    beta = meta["layers"]["17"]["beta"]
    for x in samples:
        subject = x["subject"]
        c, donor = cache[subject], cache[donors[subject]]
        block_list = sorted({x["block"] for x in selected}) if heads else None
        donor_comps = capture_components(runner, donor, heads=heads, blocks=block_list)
        clean_comps = capture_components(runner, c, heads=heads, blocks=block_list)
        target, foil = c["top"], donor["top"]
        index = meta["subjects"].index(subject)
        prediction = pack["base"] + beta*pack["products"][17][index]
        clean = score_one(runner, c["final"], target, foil)
        if heads:
            conditions = [(x["block"], "heads", x["group"], h) for x in selected for h in range(runner.model.config.num_attention_heads)]
        else:
            conditions = [(b, k, g, None) for b in range(1, len(runner.blocks)+1) for k in ["attention", "mlp"] for g in GROUPS]
        for block, kind, group, head in conditions:
            value = donor_comps[(block, kind, group)]
            if len(c["groups"][group]) == 0 or len(c["groups"][group]) != len(value):
                skipped.append({"subject": subject, "block": block, "group": group, "kind": kind, "head": head})
                continue
            actual, state17 = patched_component(runner, c, block, kind, group, value, head=head)
            state_error = float((state17-c["states"][17]).abs().max())
            rows.append({"subject": subject, "relation": args.relation, "partition": "validation" if subject in validation else "test",
                "block": block, "kind": kind, "group": group, "head": head, "donor": donors[subject],
                "clean": clean, "patched": score_one(runner, actual, target, foil),
                "state17_max_change": state_error, "lre_input_unchanged": state_error < 1e-5,
                "clean_vector_error": float((c["final"]-prediction).norm()),
                "patched_fixed_lre_vector_error": float((actual-prediction).norm())})
        # One actual identity intervention per subject, in addition to unit tests.
        b, k, g, h = next(cnd for cnd in conditions if len(c["groups"][cnd[2]]) > 0)
        identity, _ = patched_component(runner, c, b, k, g, clean_comps[(b,k,g)], head=h)
        error = float((identity-c["final"]).abs().max())
        assert error < 1e-4, error
        checks.append({"subject": subject, "identity_max_error": error})
        print(f"{label} {args.relation} shard{args.shard} {subject} done ({len(rows)} interventions)", flush=True)
        save_json(stem.with_suffix(".partial.json"), {"complete": False, "rows": rows, "checks": checks, "skipped": skipped})
    save_json(stem.with_suffix(".json"), {"complete": True, "provenance": provenance(), "rows": rows, "checks": checks, "skipped": skipped})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "generated/paired_relation_feasibility_20260920/paired_manifest.json")
    parser.add_argument("--stage", required=True, choices=["baseline", "gradients", "channels", "nonlinear", "routes", "heads"])
    parser.add_argument("--mode", default="controlled", choices=["original", "controlled", "unit"])
    parser.add_argument("--relation", choices=RELATIONS, default=RELATIONS[0])
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    runner = load_runner()
    if args.stage == "baseline": baseline(runner, args, manifest)
    elif args.stage == "gradients": gradients(runner, args, manifest)
    elif args.stage == "channels": channels(runner, args, manifest)
    elif args.stage == "nonlinear": nonlinear(runner, args, manifest)
    else: routes(runner, args, manifest, heads=args.stage == "heads")


if __name__ == "__main__":
    main()
