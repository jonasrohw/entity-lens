"""Trace Qwen's strange J-lens tokens and export ordinary neutral-prompt outputs.

Reproduces the Germany raw/differential token-score diagnosis, then runs the
unmodified model on all 16 country prompts. Uses pinned cached weights only.
"""

import argparse
import hashlib
import json
import platform
from functools import partial
from pathlib import Path

import torch
import transformers
from jlens.model import HiddenStateIntervention
from last_position_correlation import BASE, cache_prompt, jvps, write
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_vector(model, tokenizer, vector, target):
    with torch.no_grad():
        logits = model.lm_head(model.model.norm(vector))
        top = logits.topk(10).indices.tolist()
    return {
        "top_ids": top,
        "top_tokens": [tokenizer.decode([i]) for i in top],
        "berlin_rank": int((logits > logits[target]).sum()),
        "vector_norm": float(vector.norm()),
    }


def diagnose(model, tokenizer, source, out):
    references = source["specification"]["references"]
    cache = {s: cache_prompt(model, tokenizer, s, 19) for s in references + ["Germany"]}
    blocks, norm, head = model.model.layers, model.model.norm, model.lm_head.weight
    intervention = HiddenStateIntervention(model, blocks[-1])
    handle = blocks[18].register_forward_hook(intervention.hook)
    h = cache["Germany"]["states"]["subject"]
    mask = torch.zeros_like(h)
    mask[[2276, 233]] = 1
    products = []
    try:
        for anchor in references:
            c = cache[anchor]
            reference = c["states"]["subject"]
            with torch.no_grad():
                noop = intervention.forward(
                    reference, c["ids"], c["positions"]["subject"]
                )
            torch.testing.assert_close(noop, c["states"]["final"], rtol=0, atol=0)
            directions = torch.stack([h, h - reference, h * mask, h * (1 - mask)])
            p = jvps(
                intervention,
                reference,
                directions,
                c["ids"],
                c["positions"]["subject"],
                2,
            )
            torch.testing.assert_close(p[0], p[2] + p[3], rtol=1e-3, atol=1e-3)
            products.append(p)
    finally:
        handle.remove()
    products = torch.stack(products)
    vectors = products.mean(0)
    raw, diff = vectors[:2]
    target = next(
        r["model_top1_id"]
        for r in source["specification"]["targets"]
        if r["subject"] == "Germany"
    )
    raw_output = decode_vector(model, tokenizer, raw, target)
    for kind, vector in [("raw", raw), ("diff", diff)]:
        expected = next(
            r
            for r in source["measurement"]["positions"]["subject"]["modes"][kind][
                "rows"
            ]
            if r["subject"] == "Germany"
        )
        actual = decode_vector(model, tokenizer, vector, target)
        assert actual["top_ids"][:5] == expected["top_ids"]
        assert actual["berlin_rank"] == expected["rank"]
    selected_ids = list(dict.fromkeys(raw_output["top_ids"][:6] + [target]))
    # The denominator of RMSNorm is positive and common to all token scores.
    # Its learned diagonal weight is retained in these additive numerators.
    weighted_head = head * norm.weight[None]
    weighted_v = raw * norm.weight
    norms = head.norm(dim=1)
    cosines = torch.nn.functional.cosine_similarity(head, weighted_v[None], dim=1)
    scores = weighted_head @ raw
    with torch.no_grad():
        native = model.lm_head(norm(raw))
    eps = norm.variance_epsilon
    divisor = torch.sqrt(raw.pow(2).mean() + eps)
    torch.testing.assert_close(native, scores / divisor, rtol=1e-3, atol=1e-4)
    assert torch.equal(native.topk(10).indices, scores.topk(10).indices)
    rows = []
    for tid in selected_ids:
        contributions = (products[:, 0, :] * weighted_head[tid]).sum(-1) / len(
            references
        )
        two = float(weighted_head[tid] @ vectors[2])
        rest = float(weighted_head[tid] @ vectors[3])
        torch.testing.assert_close(
            contributions.sum(), scores[tid], rtol=1e-4, atol=1e-3
        )
        assert abs(two + rest - float(scores[tid])) < 0.002
        rows.append(
            {
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "score_before_rms_divisor": float(scores[tid]),
                "native_readout_logit": float(native[tid]),
                "cosine": float(cosines[tid]),
                "cosine_rank": int((cosines > cosines[tid]).sum()),
                "unembedding_norm": float(norms[tid]),
                "unembedding_norm_percentile": float(
                    (norms <= norms[tid]).float().mean() * 100
                ),
                "reference_score_contributions": [
                    {"reference": a, "score": float(v)}
                    for a, v in zip(references, contributions, strict=True)
                ],
                "two_coordinate_score": two,
                "remaining_coordinate_score": rest,
            }
        )
    torch.save(
        {
            "references": references,
            "product_kinds": [
                "raw",
                "diff",
                "two_coordinates",
                "remaining_coordinates",
            ],
            "products": products.cpu(),
            "vectors": vectors.cpu(),
        },
        out / "germany_vectors.pt",
    )
    return {
        "subject": "Germany",
        "relation": "country_capital_city",
        "raw": raw_output,
        "differential": decode_vector(model, tokenizer, diff, target),
        "two_coordinates_only": decode_vector(model, tokenizer, vectors[2], target),
        "two_coordinates_removed": decode_vector(model, tokenizer, vectors[3], target),
        "raw_diff_cosine": float(
            torch.nn.functional.cosine_similarity(raw, diff, dim=0)
        ),
        "two_coordinates_cosine_to_raw": float(
            torch.nn.functional.cosine_similarity(vectors[2], raw, dim=0)
        ),
        "rms_divisor": float(divisor),
        "coordinates": [2276, 233],
        "tokens": rows,
    }


def ordinary_predictions(model, tokenizer, source, max_new_tokens):
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    artifacts = [44092, 89998, 90443, 53321, 84787, 114068]
    rows = []
    for target in source["specification"]["targets"]:
        subject = target["subject"]
        prompt = subject + " is known for"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
        prompt_ids = ids[0].tolist()
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1]
            probabilities = logits.softmax(-1)
            top = logits.topk(10).indices.tolist()
            watched = []
            for tid in artifacts + [target["model_top1_id"]]:
                watched.append(
                    {
                        "id": tid,
                        "token": tokenizer.decode([tid]),
                        "rank": int((logits > logits[tid]).sum()),
                        "probability": float(probabilities[tid]),
                        "logit": float(logits[tid]),
                    }
                )
            generated = []
            for step in range(max_new_tokens):
                tid = int(logits.argmax())
                generated.append(tid)
                if tid in eos_ids or step == max_new_tokens - 1:
                    break
                ids = torch.cat([ids, torch.tensor([[tid]], device=ids.device)], dim=1)
                logits = model(input_ids=ids).logits[0, -1]
        rows.append(
            {
                "subject": subject,
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "top10": [
                    {
                        "id": i,
                        "token": tokenizer.decode([i]),
                        "probability": float(probabilities[i]),
                    }
                    for i in top
                ],
                "top10_probability_mass": float(probabilities[top].sum()),
                "watched_tokens": watched,
                "capital_target_id": target["model_top1_id"],
                "capital_target": target["model_top1"],
                "generated_token_ids": generated,
                "continuation": tokenizer.decode(generated, skip_special_tokens=True),
                "continuation_with_special_tokens": tokenizer.decode(
                    generated, skip_special_tokens=False
                ),
                "stopped_on_eos": generated[-1] in eos_ids,
                "stop_reason": "eos" if generated[-1] in eos_ids else "max_new_tokens",
            }
        )
        assert generated[0] == top[0]
        print("PROMPT", subject, repr(rows[-1]["continuation"]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "generated/qwen_token_readout"
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    assert args.max_new_tokens > 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source_path = BASE / "docs/qwen3-8b-base-country-capital-predictions.json"
    source = json.loads(source_path.read_text())
    model_id, revision = source["provenance"]["model"], source["provenance"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    print("LOAD", model_id, flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
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
    provenance = {
        "model": model_id,
        "revision": revision,
        "state": 19,
        "dtype": "float32",
        "attention": "eager",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "source_snapshot_sha256": sha(source_path),
        "script_sha256": sha(Path(__file__)),
        "helper_sha256": sha(Path(__file__).with_name("last_position_correlation.py")),
        "intervention_sha256": sha(BASE / "jlens/model.py"),
    }
    diagnosis = diagnose(model, tokenizer, source, args.output_dir)
    write(args.output_dir / "diagnosis.json", {"provenance": provenance, **diagnosis})
    ordinary = ordinary_predictions(model, tokenizer, source, args.max_new_tokens)
    diagnosis["normal_neutral_prompt_top10"] = next(
        [t["token"] for t in r["top10"]] for r in ordinary if r["subject"] == "Germany"
    )
    write(args.output_dir / "diagnosis.json", {"provenance": provenance, **diagnosis})
    write(
        args.output_dir / "neutral_predictions.json",
        {
            "provenance": provenance,
            "generation": {
                "method": "greedy argmax",
                "max_new_tokens": args.max_new_tokens,
                "use_cache": False,
                "chat_template": False,
                "icl_demonstrations": False,
                "neutral_prompt": "{} is known for",
                "added_prefix": "",
                "trailing_space": False,
            },
            "rows": ordinary,
        },
    )
    print("DONE", len(ordinary), "ordinary prompts", flush=True)


if __name__ == "__main__":
    main()
