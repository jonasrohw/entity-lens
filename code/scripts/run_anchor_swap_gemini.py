"""Exact paired anchor/subject swaps on the Gemini-generated frozen corpus."""

import argparse
import hashlib
import json
import random
import socket
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/anchor_swap_gemini"
OUT = ROOT / "results/anchor_swap_gemini"
MODEL = "allenai/Olmo-3-1025-7B"
REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"
BETAS = [1.0, 2.0, 3.0, 5.0, 10.0]
LAYER = 17


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    manifest = {
        "seed": 220922,
        "model": MODEL,
        "revision": REVISION,
        "layer": LAYER,
        "primary_beta": 3.0,
        "betas": BETAS,
        "protocol_sha256": sha(DATA / "PROTOCOL.md"),
        "domains": {},
    }
    seen = set()
    for path in sorted(DATA.glob("*_sourced.json")):
        name = path.name.removesuffix("_sourced.json")
        d = json.loads(path.read_text())
        rows = [r for r in d["accepted"] if r["subject"].casefold() not in seen]
        # Common nouns should not acquire different target tokens merely because
        # separate generation batches used title case.
        if name == "athletes":
            for row in rows:
                row["object_a"] = row["object_a"].lower()
                row["object_b"] = row["object_b"].lower()
        seen.update(r["subject"].casefold() for r in rows)
        random.Random(220922).shuffle(rows)
        assert len(rows) >= 64, (name, len(rows))
        for i, r in enumerate(rows):
            r["split"] = (
                "demonstrations"
                if i < 8
                else "references"
                if i < 16
                else "validation"
                if i < 32
                else "test"
            )
        manifest["domains"][name] = {
            "spec": d["spec"],
            "rows": rows,
            "candidates_sha256": sha(path),
        }
    assert len(manifest["domains"]) == 6
    if (DATA / "manifest.json").exists():
        assert json.loads((DATA / "manifest.json").read_text()) == manifest
    else:
        write(DATA / "manifest.json", manifest)
        write(
            DATA / "FREEZE_PROVENANCE.json",
            {
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "manifest_sha256": sha(DATA / "manifest.json"),
                "runner_sha256": sha(Path(__file__)),
                "existing_model_result_files": [
                    p.name for p in sorted(OUT.glob("*.json"))
                ],
            },
        )
    triples = []
    for name, d in manifest["domains"].items():
        for r in d["rows"]:
            for j, rel in enumerate(d["spec"]["relations"]):
                triples.append(
                    {
                        "domain": name,
                        "subject": r["subject"],
                        "relation": rel,
                        "relation_id": f"{name}_{rel}",
                        "object": r["object_a" if j == 0 else "object_b"],
                        "split": r["split"],
                        "region": r["region"],
                        "source_url": r["source_url"],
                        "source_html_sha256": r["source_html_sha256"],
                        "source_status": "automatically_corroborated",
                    }
                )
    (DATA / "triples.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in triples)
    )
    for name, d in manifest["domains"].items():
        for j, relation in enumerate(d["spec"]["relations"]):
            relation_id = f"{name}_{relation}"
            write(
                DATA / "relations" / f"{relation_id}.json",
                {
                    "name": relation_id,
                    "prompt_templates": [d["spec"]["templates"][j]],
                    "properties": {"domain_name": name},
                    "samples": [
                        {
                            "subject": r["subject"],
                            "object": r["object_a" if j == 0 else "object_b"],
                            "split": r["split"],
                        }
                        for r in d["rows"]
                    ],
                    "split_manifest": "../manifest.json",
                },
            )
    print(
        {
            n: {
                s: sum(r["split"] == s for r in d["rows"])
                for s in ["demonstrations", "references", "validation", "test"]
            }
            for n, d in manifest["domains"].items()
        },
        flush=True,
    )


def run(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.mechanistic import Runner
    from jlens.splits import first_object_token

    torch.set_num_threads(2)
    torch.manual_seed(220922)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((DATA / "manifest.json").read_text())
    d = manifest["domains"][name]
    result_path = OUT / f"{name}.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("complete"):
        return
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
        .to("cuda:0")
    )
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    runner = Runner(model, tokenizer, [LAYER])
    demos = [r for r in d["rows"] if r["split"] == "demonstrations"]
    refs = [r for r in d["rows"] if r["split"] == "references"]
    scored = [r for r in d["rows"] if r["split"] in ["validation", "test"]]
    rows = refs + scored
    modes = ["a", "b", "neutral"]
    terms = {}
    targets = {}
    golds = {}
    rawcache = {}
    checks = {}
    contexts = {}
    for mode in modes:
        tick = time.time()
        template = (
            "{} is known for"
            if mode == "neutral"
            else d["spec"]["templates"][0 if mode == "a" else 1]
        )
        context = (
            ""
            if mode == "neutral"
            else "".join(
                template.format(r["subject"]) + " " + r["object_" + mode] + "\n"
                for r in demos
            )
        )
        contexts[mode] = {"template": template, "context": context}
        cache = []
        for row in rows:
            c = runner.cache(context + template.format(row["subject"]), row["subject"])
            c.pop("components")
            cache.append(c)
        rawcache[mode] = cache
        states = torch.stack([c["states"][LAYER] for c in cache[len(refs) :]])
        anchor = torch.stack([c["final"] for c in cache[: len(refs)]]).mean(0)
        # Exact existing implementation, with a progress marker per reference.
        product = torch.zeros_like(states)
        for i, c in enumerate(cache[: len(refs)]):
            product += runner.products(
                [c], LAYER, states, subtract_reference=True
            ) / len(refs)
            print(
                name,
                mode,
                "reference",
                i + 1,
                "/",
                len(refs),
                "seconds",
                round(time.time() - tick),
                flush=True,
            )
        terms[mode] = {"anchor": anchor, "product": product}
        targets[mode] = [c["top"] for c in cache[len(refs) :]]
        if mode != "neutral":
            golds[mode] = [
                first_object_token(r["object_" + mode], tokenizer) for r in scored
            ]
        # Independent central finite difference and zero derivative at one reference.
        c = cache[0]
        h = c["states"][LAYER]
        direction = states[0] - h
        derivative = runner.products(
            [c], LAYER, torch.stack([direction, torch.zeros_like(direction)])
        )
        eps = 0.002
        with torch.no_grad():
            plus = runner.forward(
                c["ids"], c["pos"], state_patch=(LAYER, h + eps * direction)
            )[0]
            minus = runner.forward(
                c["ids"], c["pos"], state_patch=(LAYER, h - eps * direction)
            )[0]
            identity = runner.forward(c["ids"], c["pos"], state_patch=(LAYER, h))[0]
        fd = (plus - minus) / (2 * eps)
        checks[mode] = {
            "fd_relative_error": float(
                (fd - derivative[0]).norm() / derivative[0].norm()
            ),
            "zero_jvp_max": float(derivative[1].abs().max()),
            "identity_max": float((identity - c["final"]).abs().max()),
        }
        assert checks[mode]["zero_jvp_max"] == 0 and checks[mode]["identity_max"] < 1e-5
        assert checks[mode]["fd_relative_error"] < 0.08, checks
    # Targets with a different answer in both relations make an informative donor control.
    donor = []
    cyclic = []
    matched = []
    for i, row in enumerate(scored):
        pool = [
            j for j, r in enumerate(scored) if r["split"] == row["split"] and j != i
        ]
        candidates = [
            j for j in pool if all(targets[m][j] != targets[m][i] for m in ["a", "b"])
        ]
        chosen = min(
            candidates or pool,
            key=lambda j: hashlib.sha256(
                (row["subject"] + "|" + scored[j]["subject"]).encode()
            ).hexdigest(),
        )
        donor.append(chosen)
        matched.append(bool(candidates))
        group = [j for j, r in enumerate(scored) if r["split"] == row["split"]]
        cyclic.append(group[(group.index(i) + 1) % len(group)])
    result = {
        "domain": name,
        "complete": False,
        "rows": scored,
        "targets": targets,
        "gold": golds,
        "donors": donor,
        "cyclic_donors": cyclic,
        "donor_different_both": matched,
        "contexts": contexts,
        "checks": checks,
        "scores": {},
        "provenance": {
            "model": MODEL,
            "revision": REVISION,
            "manifest_sha256": sha(DATA / "manifest.json"),
            "runner_sha256": sha(Path(__file__)),
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(),
            "dtype": "float32",
            "attention": "eager",
            "layer": LAYER,
            "tf32": False,
        },
    }

    @torch.no_grad()
    def score(v, mode):
        # Keep only raw token predictions: outcomes can be independently rebuilt.
        top = runner.logits(v).argmax(-1)
        if top.ndim == 0:
            top = top.repeat(len(scored))
        return {
            "top": top.cpu().tolist(),
            "text": [tokenizer.decode([t]) for t in top.tolist()],
        }

    for mode, other in [("a", "b"), ("b", "a")]:
        anchor = terms[mode]["anchor"]
        own = terms[mode]["product"]
        foreign = terms[other]["product"]
        neutral = terms["neutral"]["product"]
        result["scores"][mode] = {}
        for beta in BETAS:
            conditions = {
                "own": anchor + beta * own,
                "foreign": anchor + beta * foreign,
                "neutral": anchor + beta * neutral,
                "anchor": anchor,
                "neutral_alone": beta * neutral,
                "wrong_foreign": anchor + beta * foreign[donor],
                "wrong_neutral": anchor + beta * neutral[donor],
                "cyclic_foreign": anchor + beta * foreign[cyclic],
                "cyclic_neutral": anchor + beta * neutral[cyclic],
            }
            result["scores"][mode][str(beta)] = {
                k: score(v, mode) for k, v in conditions.items()
            }
    result["tokenization"] = {}
    for mode in ["a", "b"]:
        mapping = {}
        for row in d["rows"]:
            obj = row["object_" + mode]
            token = first_object_token(obj, tokenizer)
            mapping.setdefault(token, set()).add(obj)
        result["tokenization"][mode] = {
            "gold_first_token_unique": [len(mapping[t]) == 1 for t in golds[mode]],
            "model_equals_gold": [t == g for t, g in zip(targets[mode], golds[mode])],
            "gold_text": [tokenizer.decode([t]) for t in golds[mode]],
            "target_text": [tokenizer.decode([t]) for t in targets[mode]],
        }
    result["seconds"] = time.time() - start
    result["complete"] = True
    write(result_path, result)
    tensorpath = ROOT / "generated/anchor_swap_gemini" / f"{name}_terms.pt"
    torch.save(
        {m: {k: v.cpu() for k, v in t.items()} for m, t in terms.items()}, tensorpath
    )
    print("COMPLETE", name, round(result["seconds"]), checks, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--domain")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    if a.domain:
        run(a.domain)


if __name__ == "__main__":
    main()
