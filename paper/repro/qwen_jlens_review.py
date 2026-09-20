"""Audit retained Qwen differential lens against an explicit suffix and forward AD."""

import argparse
import hashlib
import json
import random
import sys
from functools import partial
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "code"
sys.path.insert(0, str(BASE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from jlens.model import HiddenStateIntervention, final_logits  # noqa: E402
from torch.func import jvp as forward_jvp  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir", type=Path, default=BASE / "generated/qwen_jlens_review"
)
OUT = parser.parse_args().output_dir
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "Qwen/Qwen3-8B-Base"
REV = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"


def scalar(x):
    return float(x.detach())


def relative(a, b):
    return scalar((a - b).norm() / b.norm().clamp_min(1e-20))


def cos(a, b):
    return scalar(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def layout(model):
    return model.model.layers, model.model.norm


def decode(model, norm, hidden):
    # Use the actual HF module: Gemma2 and Qwen3.8 use (1 + weight) RMSNorm.
    normalized = norm(hidden.to(norm.weight.device))
    head = model.get_output_embeddings()
    logits = head(normalized.to(head.weight.device)).float()
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return cap * torch.tanh(logits / cap) if cap is not None else logits


def cache_prompt(model, tokenizer, prompt, subject, layers):
    blocks, _ = layout(model)
    tokens = tokenizer(prompt, return_tensors="pt")
    ids = tokens.input_ids.to(model.get_input_embeddings().weight.device)
    end = prompt.rindex(subject) + len(subject)
    position = min(len(tokenizer(prompt[:end]).input_ids) - 1, ids.shape[1] - 1)
    captured = {}
    handles = []

    def capture(layer):
        def hook(module, inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            captured[layer] = states[0, position].detach().clone()
            if layer == len(blocks):
                captured["final"] = states[0, -1].detach().clone()

        return hook

    for layer in sorted(set(layers) | {len(blocks)}):
        handles.append(blocks[layer - 1].register_forward_hook(capture(layer)))
    try:
        with torch.no_grad():
            output = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            logits = (
                output.logits[0, -1]
                .detach()
                .to(model.get_output_embeddings().weight.device)
            )
    finally:
        for handle in handles:
            handle.remove()
    return dict(
        ids=ids,
        position=position,
        states=captured,
        target=int(logits.argmax()),
        logits=logits,
    )


def jvps(intervention, reference, directions, ids, position, batch_size=2):
    """Exact double-backward products, reusing a graph across directions."""
    hidden = reference.detach().clone().requires_grad_(True)
    output = intervention.forward(hidden, ids, position)
    cotangent = torch.zeros_like(output, requires_grad=True)
    gradient = torch.autograd.grad(
        output, hidden, grad_outputs=cotangent, create_graph=True
    )[0]
    chunks = []
    for start in range(0, len(directions), batch_size):
        batch = directions[start : start + batch_size]
        keep = start + batch_size < len(directions)
        product = torch.autograd.grad(
            gradient,
            cotangent,
            grad_outputs=batch,
            is_grads_batched=True,
            retain_graph=keep,
        )[0]
        chunks.append(product.detach())
    return torch.cat(chunks)


def aggregate(ranks):
    r = np.asarray(ranks, dtype=float)
    return dict(
        median_rank=float(np.median(r)),
        median_lograng=float(np.median(np.log10(r + 1))),
        agree_at1=float((r == 0).mean()),
        agree_at10=float((r < 10).mean()),
    )


def main():
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REV, local_files_only=True
    )
    print("LOAD", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REV,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            device_map={"": "cuda:0"},
        )
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    blocks, norm = layout(model)
    layer = 19
    original = json.loads(
        (BASE / "docs/qwen3-8b-base-country-language-predictions.json").read_text()
    )
    data = json.loads((BASE / "data/relations/country_language.json").read_text())
    pool = list(data["samples"])
    random.Random(17).shuffle(pool)
    refs, tests = pool[8:16], pool[8:28]
    names = [t["subject"] for t in tests]
    refnames = [t["subject"] for t in refs]
    targets = torch.tensor(
        [r["model_top1_id"] for r in original["lens"]["rows"]], device="cuda"
    )
    assert names == [r["subject"] for r in original["lens"]["rows"]]
    print(
        "CHECK exact no-op, actual decoding, explicit suffix and forward AD", flush=True
    )

    def cache_all(prefix):
        cache = {}
        for s in names:
            prompt = prefix + s + " is known for"
            c = cache_prompt(model, tokenizer, prompt, s, [layer])
            full = []
            h = blocks[layer - 1].register_forward_hook(
                lambda mod, inp, out: full.append(out.detach().clone())
            )
            try:
                with torch.no_grad():
                    model(input_ids=c["ids"])
            finally:
                h.remove()
            c["full"] = full[0]
            offsets = tokenizer(prompt, return_offsets_mapping=True).offset_mapping
            end = prompt.rindex(s) + len(s)
            start = prompt.rindex(s)
            expected = max(
                i for i, (a, b) in enumerate(offsets) if a < end and b > start
            )
            assert c["position"] == expected
            torch.testing.assert_close(
                c["full"][0, c["position"]], c["states"][layer], rtol=0, atol=0
            )
            c["subject_tokens"] = sum(a < end and b > start for a, b in offsets)
            cache[s] = c
        return cache

    def explicit_tail(c):
        # Direct suffix execution: no intervention or output-capture hooks.
        states = c["full"]
        length = states.shape[1]
        position = c["position"]
        pids = torch.arange(length, device=states.device)[None]
        mask = torch.full(
            (length, length), torch.finfo(states.dtype).min, device=states.device
        ).triu(1)[None, None]
        with torch.no_grad():
            pe = model.model.rotary_emb(states, pids)

        def tail(h):
            replaced = torch.cat(
                [states[:, :position], h[None, None, :], states[:, position + 1 :]],
                dim=1,
            )
            for block in blocks[layer:]:
                replaced = block(
                    replaced,
                    attention_mask=mask,
                    position_ids=pids,
                    position_embeddings=pe,
                    use_cache=False,
                )
            return replaced[0, -1]

        return tail

    def compute(prefix, independent=False):
        cache = cache_all(prefix)
        vectors = []
        records = []
        checks = []
        intervention = HiddenStateIntervention(model, blocks[-1])
        for anchor in refnames:
            c = cache[anchor]
            reference = c["states"][layer]
            directions = torch.stack(
                [cache[s]["states"][layer] - reference for s in names]
            )
            h = blocks[layer - 1].register_forward_hook(intervention.hook)
            try:
                with torch.no_grad():
                    noop = intervention.forward(reference, c["ids"], c["position"])
                torch.testing.assert_close(noop, c["states"]["final"], rtol=0, atol=0)
                products = jvps(
                    intervention, reference, directions, c["ids"], c["position"], 2
                )
            finally:
                h.remove()
            vectors.append(products)
            if independent:
                tail = explicit_tail(c)
                with torch.no_grad():
                    tailbase = tail(reference)
                torch.testing.assert_close(
                    tailbase, c["states"]["final"], rtol=2e-5, atol=2e-4
                )
                forward_products = []
                for j, s in enumerate(names):
                    _, value = forward_jvp(tail, (reference,), (directions[j],))
                    forward_products.append(value.detach())
                forward_products = torch.stack(forward_products)
                err = relative(products, forward_products)
                assert err < 0.002, (anchor, err)
                check = dict(
                    anchor=anchor,
                    explicit_forward_error=relative(tailbase, c["states"]["final"]),
                    forward_ad_error=err,
                )
                # Finite differences in actual subject directions, not just random vectors.
                fd = []
                for subject in ["Germany", "Peru"]:
                    j = names.index(subject)
                    direction = directions[j]
                    if not direction.norm():
                        continue
                    vals = {}
                    for eps in [0.01, 0.003, 0.001]:
                        with torch.no_grad():
                            v = (
                                tail(reference + eps * direction)
                                - tail(reference - eps * direction)
                            ) / (2 * eps)
                        vals[str(eps)] = relative(v, products[j])
                    fd.append(dict(subject=subject, errors=vals))
                check["finite_differences"] = fd
                checks.append(check)
            print(
                "REF", repr(prefix), anchor, "norm", float(products.norm()), flush=True
            )
        products = torch.stack(vectors)
        valid = torch.tensor([[s != a for s in names] for a in refnames], device="cuda")
        diff = torch.zeros_like(products[0])
        for i in range(len(refnames)):
            diff += products[i] * valid[i, :, None]
        diff /= valid.sum(0)[:, None]
        base = torch.stack([cache[a]["states"]["final"] for a in refnames])
        affine = ((products + base[:, None, :]) * valid[:, :, None]).sum(0) / valid.sum(
            0
        )[:, None]
        with torch.no_grad():
            logits = decode(model, norm, diff)
            direct = (diff * norm.weight) @ model.lm_head.weight.T
        rawtops = direct.topk(5).indices
        normtops = logits.topk(5).indices
        assert torch.equal(rawtops, normtops)
        for j, s in enumerate(names):
            c = cache[s]
            target = int(targets[j])
            v = diff[j]
            ranks = int((logits[j] > logits[j, target]).sum())
            with torch.no_grad():
                actual = model(input_ids=c["ids"]).logits[0, -1]
            reconstructed = decode(model, norm, c["states"]["final"])
            manual = final_logits(c["states"]["final"], norm, model.lm_head.weight)
            assert relative(actual, reconstructed) < 1e-5
            assert relative(actual, manual) < 1e-5
            records.append(
                dict(
                    subject=s,
                    subject_tokens=c["subject_tokens"],
                    subject_position=c["position"],
                    subject_norm=scalar(c["states"][layer].norm()),
                    final_norm=scalar(c["states"]["final"].norm()),
                    diff_norm=scalar(v.norm()),
                    target_rank=ranks,
                    top_ids=normtops[j].tolist(),
                    top=[tokenizer.decode([i]) for i in normtops[j].tolist()],
                    neutral_top=[
                        tokenizer.decode([i]) for i in actual.topk(5).indices.tolist()
                    ],
                    reference_contributions=[
                        dict(
                            anchor=a,
                            norm=scalar(products[i, j].norm()),
                            cosine_to_mean=cos(products[i, j], v),
                        )
                        for i, a in enumerate(refnames)
                        if s != a
                    ],
                )
            )
        pairwise = F.normalize(diff, dim=-1) @ F.normalize(diff, dim=-1).T
        result = dict(
            prefix=prefix,
            rows=records,
            checks=checks,
            pairwise_cosines=pairwise.tolist(),
            aggregate=aggregate([r["target_rank"] for r in records]),
        )
        torch.save(
            dict(
                diff=diff.cpu(),
                affine=affine.cpu(),
                products=products.cpu(),
                states=torch.stack([cache[s]["states"][layer] for s in names]).cpu(),
                positions=[cache[s]["position"] for s in names],
                names=names,
                references=refnames,
                base=base.cpu(),
            ),
            OUT / ("vectors_base.pt" if not prefix else "vectors_prefix.pt"),
        )
        return result

    baseline = compute("", True)
    baseline["retained_rank_differences"] = [
        dict(
            subject=r["subject"],
            new=r["target_rank"],
            old=o["rank_diff"],
            difference=r["target_rank"] - o["rank_diff"],
        )
        for r, o in zip(baseline["rows"], original["lens"]["rows"])
    ]
    write("baseline.json", baseline)
    print("RANK DIFFERENCES", baseline["retained_rank_differences"], flush=True)
    assert max(abs(r["difference"]) for r in baseline["retained_rank_differences"]) <= 3
    assert [[t.strip() for t in r["top"]] for r in baseline["rows"]] == [
        r["top_diff"] for r in original["lens"]["rows"]
    ]
    write("baseline.json", baseline)
    print("BASELINE REPRODUCED", baseline["aggregate"], flush=True)
    prefixed = compute("Entity: ", False)
    write("prefix.json", prefixed)
    print("PREFIX", prefixed["aggregate"], flush=True)
    newline = compute("\n", False)
    write("newline.json", newline)
    print("NEWLINE", newline["aggregate"], flush=True)
    # Stratify reference set by subject-token position using retained baseline products.
    vectors = torch.load(OUT / "vectors_base.pt", weights_only=True)
    products = vectors["products"].cuda()
    positions = vectors["positions"]
    groups = {
        label: [
            i
            for i, a in enumerate(refnames)
            if (positions[names.index(a)] == 0) == at_zero
        ]
        for label, at_zero in [("first_position", True), ("later_position", False)]
    }
    variants = {}
    for label, indices in groups.items():
        outputs = []
        for j, s in enumerate(names):
            include = [i for i in indices if refnames[i] != s]
            if not include:
                continue
            v = products[include, j].mean(0)
            with torch.no_grad():
                lg = decode(model, norm, v)
            tid = int(targets[j])
            top = lg.topk(5).indices.tolist()
            outputs.append(
                dict(
                    subject=s,
                    n_references=len(include),
                    reference_names=[refnames[i] for i in include],
                    target_rank=int((lg > lg[tid]).sum()),
                    top=[tokenizer.decode([i]) for i in top],
                    vector_norm=scalar(v.norm()),
                )
            )
        variants[label] = outputs
    write("reference_groups.json", variants)
    # Diagnostic ablation of the two observed position-associated coordinates.
    # This is selected after inspection; it is not a proposed replacement lens.
    cache = cache_all("")
    reference_products = vectors["products"].cuda()
    coordinate_products = []
    for anchor in refnames:
        c = cache[anchor]
        reference = c["states"][layer]
        directions = torch.stack([cache[s]["states"][layer] - reference for s in names])
        masked = torch.zeros_like(directions)
        masked[:, [2276, 233]] = directions[:, [2276, 233]]
        intervention = HiddenStateIntervention(model, blocks[-1])
        h = blocks[layer - 1].register_forward_hook(intervention.hook)
        try:
            coordinate_products.append(
                jvps(intervention, reference, masked, c["ids"], c["position"], 2)
            )
        finally:
            h.remove()
    coordinate_products = torch.stack(coordinate_products)
    cleaned = []
    for j, subject in enumerate(names):
        indices = [i for i, a in enumerate(refnames) if a != subject]
        v = (reference_products[indices, j] - coordinate_products[indices, j]).mean(0)
        with torch.no_grad():
            lg = decode(model, norm, v)
        tid = int(targets[j])
        cleaned.append(
            dict(
                subject=subject,
                top=[tokenizer.decode([i]) for i in lg.topk(5).indices.tolist()],
                target_rank=int((lg > lg[tid]).sum()),
            )
        )
    finite_steps = []
    for anchor, subject in [
        ("Peru", "Germany"),
        ("United States", "Germany"),
        ("Australia", "Germany"),
    ]:
        c = cache[anchor]
        reference = c["states"][layer]
        direction = cache[subject]["states"][layer] - reference
        tail = explicit_tail(c)
        jv = reference_products[refnames.index(anchor), names.index(subject)]
        with torch.no_grad():
            change = tail(reference + direction) - tail(reference)
        finite_steps.append(
            dict(
                anchor=anchor,
                subject=subject,
                direction_norm=scalar(direction.norm()),
                reference_norm=scalar(reference.norm()),
                jvp_norm=scalar(jv.norm()),
                full_step_change_norm=scalar(change.norm()),
                relative_linearization_error=relative(jv, change),
                cosine=cos(jv, change),
            )
        )
    write(
        "coordinate_ablation.json",
        dict(
            removed_coordinates=[2276, 233],
            rows=cleaned,
            aggregate=aggregate([r["target_rank"] for r in cleaned]),
            full_step_checks=finite_steps,
        ),
    )
    write(
        "metadata.json",
        dict(
            model=MODEL,
            revision=REV,
            torch=torch.__version__,
            device=torch.cuda.get_device_name(),
            source_sha256={
                str(p.relative_to(BASE.parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in [Path(__file__), BASE / "jlens/model.py"]
            },
        ),
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
