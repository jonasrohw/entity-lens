"""Held-out mechanistic experiments for neutral Jacobian and relational readouts.

Standalone runner; never overwrites retained paper measurements. See
docs/mechanistic-experiments.md for the frozen protocol and interpretation.
"""

import argparse
import hashlib
import json
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jlens.model import final_logits, resolve_blocks, resolve_final_norm
from jlens.splits import filter_identifying_objects, first_object_token

MODEL = "allenai/Olmo-3-1025-7B"
GLOBAL_SUBJECTS = ["Ada Lovelace", "Isaac Newton", "Tokyo", "Pacific Ocean"]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def make_manifest(data_dir, tokenizer, seed=1909):
    """Subject-disjoint within relation; global references excluded everywhere."""
    datasets = {
        p.stem: json.loads(p.read_text()) for p in sorted(Path(data_dir).glob("*.json"))
    }
    plans = {}
    for name, data in datasets.items():
        trailing = name.startswith("presidents_")
        samples = filter_identifying_objects(data["samples"], tokenizer, trailing)
        unique = {}
        ambiguous = set()
        for x in samples:
            if x["subject"] in unique and unique[x["subject"]]["object"] != x["object"]:
                ambiguous.add(x["subject"])
            unique[x["subject"]] = x
        samples = [
            x
            for s, x in unique.items()
            if s not in ambiguous and s not in GLOBAL_SUBJECTS
        ]
        random.Random(seed).shuffle(samples)
        if len(samples) < 20:
            plans[name] = {
                "skipped": "fewer than 20 uniquely identified samples",
                "n": len(samples),
            }
            continue
        nicl = 8 if len(samples) >= 48 else 4
        remaining = len(samples) - nicl - 4
        ntrain = min(24, max(6, remaining // 2))
        nval = min(8, max(3, (remaining - ntrain) // 3))
        ntest = min(24, remaining - ntrain - nval)
        a, b, c = nicl + 4, nicl + 4 + ntrain, nicl + 4 + ntrain + nval
        parts = {
            "icl": samples[:nicl],
            "references": samples[nicl : nicl + 4],
            "train": samples[a:b],
            "validation": samples[b:c],
            "test": samples[c : c + ntest],
        }
        domain = data["properties"]["domain_name"]
        candidates = [
            k
            for k, d in datasets.items()
            if k != name and d["properties"]["domain_name"] == domain
        ]
        preferences = [
            "country_currency",
            "country_capital_city",
            "person_occupation",
            "person_father",
            "landmark_in_country",
            "landmark_on_continent",
        ]
        control = next(
            (k for k in preferences if k in candidates), next(iter(candidates), None)
        )
        control_context = None
        if control:
            # No held-out entity appears in alternate-relation demonstrations.
            heldout = {x["subject"] for x in parts["validation"] + parts["test"]}
            cs = [
                x for x in datasets[control]["samples"] if x["subject"] not in heldout
            ]
            random.Random(seed).shuffle(cs)
            ct = datasets[control]["prompt_templates"][0]
            control_context = "".join(
                ct.format(x["subject"]) + " " + x["object"] + "\n" for x in cs[:8]
            )
        plans[name] = {
            "parts": parts,
            "template": data["prompt_templates"][0],
            "trailing_space": trailing,
            "control_relation": control,
            "control_template": datasets[control]["prompt_templates"][0]
            if control
            else None,
            "control_context": control_context,
            "n_available": len(samples),
        }
    return {
        "model": MODEL,
        "seed": seed,
        "layers": [5, 11, 17, 23],
        "global_subjects": GLOBAL_SUBJECTS,
        "betas": BETAS,
        "relations": plans,
    }


def attribute_basis(hidden, labels, rank=8):
    """Top left singular vectors of hidden/one-hot cross covariance, train only."""
    _, labels = torch.unique(
        torch.as_tensor(labels, device=hidden.device), return_inverse=True
    )
    onehot = F.one_hot(labels).to(hidden.dtype)
    # Singleton classes only recover subject variation, not a replicated attribute.
    onehot = onehot[:, onehot.sum(0) >= 2]
    if onehot.shape[1] < 2:
        return hidden.new_empty((hidden.shape[1], 0))
    x = hidden - hidden.mean(0)
    y = onehot - onehot.mean(0)
    cross = x.T @ y / len(x)
    u, s, _ = torch.linalg.svd(cross, full_matrices=False)
    k = min(rank, max(0, onehot.shape[1] - 1), int((s > s.max() * 1e-5).sum()))
    return u[:, :k]


def removal(hidden, mean, basis):
    return ((hidden - mean) @ basis) @ basis.T


class Runner:
    def __init__(self, model, tokenizer, layers):
        self.model, self.tokenizer, self.layers = model, tokenizer, layers
        self.blocks = resolve_blocks(model)
        self.norm = resolve_final_norm(model)
        self.input_device = next(self.blocks[0].parameters()).device
        self.unembedding = model.lm_head.weight.detach()
        self.device = self.unembedding.device
        self.dim = self.unembedding.shape[1]

    def encode(self, prompt, subject):
        enc = self.tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
        end = prompt.rindex(subject) + len(subject)
        start = end - len(subject)
        # Last token overlapping the actual subject, not a prefix-retokenization estimate.
        positions = [
            i
            for i, (a, b) in enumerate(enc.offset_mapping[0].tolist())
            if a < end and b > start
        ]
        return enc.input_ids.to(self.input_device), positions[-1]

    def logits(self, hidden):
        return final_logits(hidden, self.norm, self.unembedding)

    def forward(
        self, ids, position, *, capture=False, state_patch=None, component_patch=None
    ):
        """Raw final vector, optional subject states and normalized block updates."""
        states, components, handles, last = {}, {}, [], {}
        if component_patch is not None:
            layer, kind, value = component_patch
            module = getattr(
                self.blocks[layer - 1],
                "post_attention_layernorm"
                if kind == "attention"
                else "post_feedforward_layernorm",
            )

            def patch_component(_m, _i, out, patch_value=value):
                result = out.clone()
                result[0, position] = patch_value.to(out.device)
                return result

            handles.append(module.register_forward_hook(patch_component))
        if state_patch is not None:
            layer, value = state_patch

            def patch_state(_m, _i, out, patch_value=value):
                result = out.clone()
                result[0, position] = patch_value.to(out.device)
                return result

            handles.append(self.blocks[layer - 1].register_forward_hook(patch_state))
        if capture:
            for layer in self.layers:

                def state_hook(_m, _i, out, layer=layer):
                    states[layer] = out[0, position].detach().to(self.device).clone()

                handles.append(self.blocks[layer - 1].register_forward_hook(state_hook))
                for kind, attr in [
                    ("attention", "post_attention_layernorm"),
                    ("mlp", "post_feedforward_layernorm"),
                ]:

                    def comp_hook(_m, _i, out, layer=layer, kind=kind):
                        components[(layer, kind)] = (
                            out[0, position].detach().to(self.device).clone()
                        )

                    handles.append(
                        getattr(self.blocks[layer - 1], attr).register_forward_hook(
                            comp_hook
                        )
                    )

        def final_hook(_m, _i, out):
            last["value"] = out[0, -1]

        handles.append(self.blocks[-1].register_forward_hook(final_hook))
        try:
            self.model.model(input_ids=ids, use_cache=False)
            final = last["value"]
        finally:
            for handle in handles:
                handle.remove()
        return final, states, components

    @torch.no_grad()
    def cache(self, prompt, subject):
        ids, pos = self.encode(prompt, subject)
        final, states, components = self.forward(ids, pos, capture=True)
        logits = self.logits(final)
        return {
            "ids": ids,
            "pos": pos,
            "states": states,
            "components": components,
            "final": final.detach(),
            "top": int(logits.argmax()),
            "logp": float(logits.log_softmax(-1).max()),
        }

    def products(self, refs, layer, directions, *, subtract_reference=False):
        """Exact average JVP, reusing a double-backward graph per reference.

        Reusing the graph changes runtime only, not the estimated operator.
        The cotangent remains zero; its derivative gives J @ direction.
        """
        result = torch.zeros_like(directions)
        for ref in refs:
            hidden = ref["states"][layer].clone().requires_grad_(True)
            final, _, _ = self.forward(
                ref["ids"], ref["pos"], state_patch=(layer, hidden)
            )
            cot = torch.zeros_like(final, requires_grad=True)
            (grad,) = torch.autograd.grad(
                final, hidden, grad_outputs=cot, create_graph=True
            )
            for i, direction in enumerate(directions):
                if subtract_reference:
                    direction = direction - hidden.detach()
                (product,) = torch.autograd.grad(
                    grad,
                    cot,
                    grad_outputs=direction,
                    retain_graph=i < len(directions) - 1,
                )
                result[i] += product.detach() / len(refs)
            del grad, final, hidden, cot
        return result

    @torch.no_grad()
    def score(self, hidden, targets):
        logits = self.logits(hidden)
        targets = torch.as_tensor(targets, device=self.device)
        if logits.ndim == 1:
            logits = logits[None]
        idx = torch.arange(len(targets), device=self.device)
        target_logits = logits[idx, targets]
        top = logits.argmax(-1)
        top2 = logits.topk(2, dim=-1)
        competitor = torch.where(
            top2.indices[:, 0] == targets, top2.values[:, 1], top2.values[:, 0]
        )
        return {
            "top": top.cpu().tolist(),
            "hit": (top == targets).int().cpu().tolist(),
            "logp": logits.log_softmax(-1)[idx, targets].cpu().tolist(),
            "margin": (target_logits - competitor).cpu().tolist(),
            "rank": (logits > target_logits[:, None]).sum(-1).cpu().tolist(),
        }


def cosines(a, b):
    return F.cosine_similarity(a, b, dim=-1).detach().cpu().tolist()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def select_beta(runner, base, products, targets, nval):
    grids = {str(beta): runner.score(base + beta * products, targets) for beta in BETAS}
    # Top-1 validation faithfulness, then target log probability, then smaller beta.
    beta = max(
        BETAS,
        key=lambda b: (
            np.mean(grids[str(b)]["hit"][:nval]),
            np.mean(grids[str(b)]["logp"][:nval]),
            -b,
        ),
    )
    return beta, grids


def run_relation(runner, name, plan, manifest, global_cache, output):
    start = time.time()
    parts = plan["parts"]
    samples = parts["references"] + parts["train"] + parts["validation"] + parts["test"]
    template = plan["template"]
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in parts["icl"]
    )
    cache = {"relation": {}, "neutral": {}, "control": {}}
    for x in samples:
        sub = x["subject"]
        prompts = {
            "relation": context
            + template.format(sub)
            + (" " if plan["trailing_space"] else ""),
            "neutral": f"{sub} is known for",
        }
        if plan["control_relation"] and x in parts["test"]:
            trailing = " " if plan["control_relation"].startswith("presidents_") else ""
            prompts["control"] = (
                plan["control_context"]
                + plan["control_template"].format(sub)
                + trailing
            )
        for kind, prompt in prompts.items():
            cache[kind][sub] = runner.cache(prompt, sub)
    print(
        f"{name}: cached {len(samples)} subjects in {time.time() - start:.1f}s",
        flush=True,
    )
    scored = parts["validation"] + parts["test"]
    nv = len(parts["validation"])
    targets = [cache["relation"][x["subject"]]["top"] for x in scored]
    gold = [
        first_object_token(x["object"], runner.tokenizer, plan["trailing_space"])
        for x in scored
    ]
    labels = [
        first_object_token(x["object"], runner.tokenizer, plan["trailing_space"])
        for x in parts["train"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "relation": name,
        "nval": nv,
        "subjects": scored,
        "targets": targets,
        "gold": gold,
        "model_correct": [int(a == b) for a, b in zip(targets, gold)],
        "plan": plan,
        "model_revision": getattr(runner.model.config, "_commit_hash", None),
        "host": socket.gethostname(),
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "parameter_devices": sorted({str(p.device) for p in runner.model.parameters()}),
        "torch_version": torch.__version__,
        "layers": {},
    }
    saved_tensors = {}

    def stack(kind, part, layer, field="states"):
        return torch.stack([cache[kind][x["subject"]][field][layer] for x in part])

    def finals(kind, part):
        return torch.stack([cache[kind][x["subject"]]["final"] for x in part])

    for layer in manifest["layers"]:
        tick = time.time()
        R, N = stack("relation", scored, layer), stack("neutral", scored, layer)
        Tr, Tn = (
            stack("relation", parts["train"], layer),
            stack("neutral", parts["train"], layer),
        )
        mr, mn = Tr.mean(0), Tn.mean(0)
        fr, fn = (
            finals("relation", parts["train"]).mean(0),
            finals("neutral", parts["train"]).mean(0),
        )
        Q = attribute_basis(Tn, labels)
        k = Q.shape[1]
        generator = torch.Generator(device=runner.device).manual_seed(
            manifest["seed"] + layer
        )
        random_bases = [
            torch.linalg.qr(
                torch.randn(runner.dim, k, device=runner.device, generator=generator)
            ).Q
            for _ in range(3)
        ]
        # Include bases once; linearity supplies each subject-specific erasure response.
        dirs = torch.cat([R - mr, N - mr, Q.T, *(q.T for q in random_bases)])
        refs_r = [cache["relation"][x["subject"]] for x in parts["references"]]
        refs_n = [cache["neutral"][x["subject"]] for x in parts["references"]]
        Jr = runner.products(refs_r, layer, dirs)
        # Neutral maps see identical relation-state directions as well as native neutral ones.
        Jn = runner.products(refs_n, layer, torch.cat([R - mr, N - mn]))
        Jg = runner.products(global_cache, layer, R - mr)
        count = len(scored)
        dr, transplant_dr = Jr[:count], Jr[count : 2 * count]
        basis_products = Jr[2 * count :]
        calibrated_beta, calibrated_grids = select_beta(runner, fr, dr, targets, nv)
        gbeta, global_grids = select_beta(runner, fr, Jg, targets, nv)
        matched_beta, matched_grids = select_beta(runner, fr, Jn[:count], targets, nv)
        offset_r = runner.products(refs_r, layer, mr[None], subtract_reference=True)[0]
        offset_n = runner.products(refs_n, layer, mn[None], subtract_reference=True)[0]
        centered_dr = dr
        dr = dr + offset_r
        transplant_dr = transplant_dr + offset_r
        fr = finals("relation", parts["references"]).mean(0)
        fn = finals("neutral", parts["references"]).mean(0)
        native_dn = Jn[count:] + offset_n
        beta, relation_grids = select_beta(runner, fr, dr, targets, nv)
        nbeta, neutral_grids = select_beta(runner, fn, native_dn, targets, nv)
        native_affine_r, native_affine_n = fr + dr, fn + native_dn
        # Shuffle across subjects to measure background-driven apparent similarity.
        shift = torch.roll(torch.arange(count, device=runner.device), 1)
        comparison = {
            "affine_cosine_beta1": cosines(native_affine_r, native_affine_n),
            "affine_shuffled_cosine_beta1": cosines(
                native_affine_r, native_affine_n[shift]
            ),
            "centered_native_cosine": cosines(centered_dr, Jn[count:]),
            "centered_same_input_cosine": cosines(centered_dr, Jn[:count]),
            "centered_shuffled_cosine": cosines(centered_dr, Jn[count:][shift]),
            "operator_relative_error": (
                (centered_dr - Jn[:count]).norm(dim=-1)
                / centered_dr.norm(dim=-1).clamp_min(1e-9)
            )
            .cpu()
            .tolist(),
            "neutral_native_beta1": runner.score(native_affine_n, targets),
            "relation_beta1": runner.score(native_affine_r, targets),
        }
        # Probe necessity in relation states, using the basis learned in neutral states.
        delta = removal(R, mr, Q)
        coeff = (R - mr) @ Q
        erased_dr = dr - coeff @ basis_products[:k]
        erasure_lre = {"targeted": runner.score(fr + beta * erased_dr, targets)}
        random_deltas = []
        for ri, q in enumerate(random_bases):
            d = removal(R, mr, q)
            scale = delta.norm(dim=-1) / d.norm(dim=-1).clamp_min(1e-9)
            d = d * scale[:, None]
            random_deltas.append(d)
            response = (((R - mr) @ q) * scale[:, None]) @ basis_products[
                (ri + 1) * k : (ri + 2) * k
            ]
            erasure_lre[f"random_{ri}"] = runner.score(
                fr + beta * (dr - response), targets
            )
        interventions = {
            key: []
            for key in [
                "clean",
                "neutral_transplant",
                "neutral_wrong_subject",
                "targeted",
                "restored",
                "random_0",
                "random_1",
                "random_2",
                "attention",
                "mlp",
                "attention_rescue",
                "mlp_rescue",
            ]
        }
        control_interventions = {
            key: [] for key in ["clean", "targeted", "random_0", "random_1", "random_2"]
        }
        ctrl_targets = []
        component_states = {"attention": [], "mlp": []}
        comp_means = {
            kind: torch.stack(
                [
                    cache["relation"][x["subject"]]["components"][(layer, kind)]
                    for x in parts["train"]
                ]
            ).mean(0)
            for kind in ["attention", "mlp"]
        }
        with torch.no_grad():
            for i, x in enumerate(scored[nv:], start=nv):
                sub, h = x["subject"], R[i]
                c = cache["relation"][sub]
                variants = {
                    "clean": h,
                    "neutral_transplant": N[i],
                    "neutral_wrong_subject": N[nv + (i - nv + 1) % (count - nv)],
                    "targeted": h - delta[i],
                    "restored": (h - delta[i]) + delta[i],
                    **{f"random_{ri}": h - d[i] for ri, d in enumerate(random_deltas)},
                }
                for key, val in variants.items():
                    out, _, _ = runner.forward(
                        c["ids"], c["pos"], state_patch=(layer, val)
                    )
                    interventions[key].append(out.detach())
                for kind in ["attention", "mlp"]:
                    out, state, _ = runner.forward(
                        c["ids"],
                        c["pos"],
                        capture=True,
                        component_patch=(layer, kind, comp_means[kind]),
                    )
                    interventions[kind].append(out.detach())
                    component_states[kind].append(state[layer])
                    out, _, _ = runner.forward(
                        c["ids"],
                        c["pos"],
                        component_patch=(layer, kind, comp_means[kind]),
                        state_patch=(layer, h),
                    )
                    interventions[kind + "_rescue"].append(out.detach())
                if sub in cache["control"]:
                    c = cache["control"][sub]
                    hc = c["states"][layer]
                    # Same removed vector, so target/control receive identical perturbations.
                    ctrl_targets.append(c["top"])
                    for key, val in {
                        "clean": hc,
                        "targeted": hc - delta[i],
                        **{
                            f"random_{ri}": hc - d[i]
                            for ri, d in enumerate(random_deltas)
                        },
                    }.items():
                        out, _, _ = runner.forward(
                            c["ids"], c["pos"], state_patch=(layer, val)
                        )
                        control_interventions[key].append(out.detach())
        intervention_scores = {
            key: runner.score(torch.stack(vals), targets[nv:])
            for key, vals in interventions.items()
        }
        control_scores = {
            key: runner.score(torch.stack(vals), ctrl_targets)
            for key, vals in control_interventions.items()
            if vals
        }
        # Exact LRE response to subject states produced by component intervention.
        component_lre = {}
        for kind in component_states:
            new_states = torch.stack(component_states[kind])
            change = runner.products(refs_r, layer, new_states - R[nv:])
            component_lre[kind] = runner.score(
                fr + beta * (dr[nv:] + change), targets[nv:]
            )
        res = {
            "layer": layer,
            "beta": beta,
            "shared_beta": gbeta,
            "neutral_beta": nbeta,
            "calibrated_beta": calibrated_beta,
            "calibrated_relation_grids": calibrated_grids,
            "matched_neutral_beta": matched_beta,
            "attribute_rank": k,
            "relation_grids": relation_grids,
            "shared_grids": global_grids,
            "neutral_grids": neutral_grids,
            "matched_neutral_grids": matched_grids,
            "comparison": comparison,
            "transplant_lre": runner.score(fr + beta * transplant_dr, targets),
            "erasure_lre": erasure_lre,
            "full_model": intervention_scores,
            "control_model": control_scores,
            "component_lre": component_lre,
            "erasure_norm_fraction": (delta.norm(dim=-1) / R.norm(dim=-1))
            .cpu()
            .tolist(),
            "restoration_max_error": float(
                (
                    torch.stack(interventions["clean"])
                    - torch.stack(interventions["restored"])
                )
                .abs()
                .max()
            ),
            "seconds": time.time() - tick,
        }
        meta["layers"][str(layer)] = res
        saved_tensors[layer] = cpu_tree(
            {
                "subject_relation": R,
                "subject_neutral": N,
                "train_relation": Tr,
                "train_neutral": Tn,
                "attribute_basis": Q,
                "relation_products": dr,
                "neutral_products_native": Jn[count:],
                "neutral_products_same_input": Jn[:count],
                "shared_products": Jg,
                "relation_affine": native_affine_r,
                "neutral_affine": native_affine_n,
                "relation_base": fr,
                "neutral_base": fn,
            }
        )
        save_json(output / f"{name}.json", meta)
        torch.save(saved_tensors, output / f"{name}.pt")
        print(
            f"{name} layer {layer}: {time.time() - tick:.1f}s; beta={beta}, faith={np.mean(relation_grids[str(beta)]['hit'][nv:]):.3f}, shared={np.mean(global_grids[str(gbeta)]['hit'][nv:]):.3f}",
            flush=True,
        )
    meta["complete"] = True
    meta["seconds"] = time.time() - start
    save_json(output / f"{name}.json", meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/relations",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=4)
    parser.add_argument("--relations", default="")
    parser.add_argument(
        "--two-gpu",
        action="store_true",
        help="Place half the layers on each of two GPUs; float32 throughout, no CPU offload",
    )
    parser.add_argument(
        "--model-gpus",
        type=int,
        default=1,
        help="Number of GPUs for exact float32 layer placement",
    )
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.manual_seed(1909)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    manifest_path = args.out / "manifest.json"
    if args.prepare:
        manifest = make_manifest(args.data_dir, tokenizer)
        manifest["code_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        save_json(manifest_path, manifest)
        for name, plan in manifest["relations"].items():
            print(
                name,
                {k: len(v) for k, v in plan["parts"].items()}
                if "parts" in plan
                else plan,
            )
        return
    manifest = json.loads(manifest_path.read_text())
    names = [name for name, plan in manifest["relations"].items() if "parts" in plan]
    names = (
        args.relations.split(",")
        if args.relations
        else names[args.shard :: args.nshards]
    )
    names = [
        name
        for name in names
        if not (
            (args.out / f"{name}.json").exists()
            and json.loads((args.out / f"{name}.json").read_text()).get("complete")
        )
    ]
    if not names:
        return
    print(f"Loading {MODEL}; relations={names}", flush=True)
    loading = {}
    model_gpus = 2 if args.two_gpu else args.model_gpus
    if model_gpus > 1:
        config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
        device_map = {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.norm": model_gpus - 1,
            "lm_head": model_gpus - 1,
        }
        device_map.update(
            {
                f"model.layers.{i}": min(
                    model_gpus - 1, i * model_gpus // config.num_hidden_layers
                )
                for i in range(config.num_hidden_layers)
            }
        )
        loading["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        **loading,
    ).eval()
    if model_gpus == 1:
        model.to("cuda:0")
    assert all(
        p.device.type == "cuda" and p.dtype == torch.float32 for p in model.parameters()
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    runner = Runner(model, tokenizer, manifest["layers"])
    global_cache = [
        runner.cache(f"{sub} is known for", sub) for sub in manifest["global_subjects"]
    ]
    for name in names:
        run_relation(
            runner, name, manifest["relations"][name], manifest, global_cache, args.out
        )


if __name__ == "__main__":
    main()
