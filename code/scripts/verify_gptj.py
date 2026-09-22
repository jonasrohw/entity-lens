"""Verify pretrained GPT-J math and reproduce representative retained results.

Run from code/: .venv/bin/python scripts/verify_gptj.py
Needs the cached GPT-J checkpoint and a GPU with room for float32 weights/JVPs.
Writes only to generated/gptj_verification/; never overwrites retained results.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import measure_faithfulness, measure_jacobian_lens
from jlens.model import (
    HiddenStateIntervention,
    final_logits,
    forward_capturing_final_block,
    forward_with_raw_hidden_states,
)

MODEL_ID = "EleutherAI/gpt-j-6b"
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def relative_error(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-20))


def manual_tail(model, states, layer):
    length = states.shape[1]
    position_ids = torch.arange(length, device=states.device)[None]
    mask = torch.full(
        (length, length),
        torch.finfo(states.dtype).min,
        device=states.device,
        dtype=states.dtype,
    ).triu(1)[None, None]
    for block in model.transformer.h[layer:]:
        normed = block.ln_1(states)
        attention = block.attn(
            normed, attention_mask=mask, position_ids=position_ids, use_cache=False
        )[0]
        states = attention + block.mlp(normed) + states
    return states[0, -1]


def check_math(model, tokenizer, device):
    data = read(ROOT / "data/relations/country_capital_city.json")
    pool = list(data["samples"])
    random.Random(0).shuffle(pool)
    random.Random(1000).shuffle(pool)
    template = data["prompt_templates"][0]
    context = "".join(
        template.format(x["subject"]) + " " + x["object"] + "\n" for x in pool[:8]
    )
    report = []
    for prompt in ["France is known for", context + template.format("France")]:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        position = len(tokenizer(prompt[: prompt.rindex("France") + 6]).input_ids) - 1
        with torch.no_grad():
            output, raw = forward_capturing_final_block(
                model, model.transformer.h[-1], ids, output_hidden_states=True
            )
            decoded = final_logits(
                raw, model.transformer.ln_f, model.lm_head.weight, model.lm_head.bias
            )
            torch.testing.assert_close(
                decoded, output.logits[0, -1], rtol=1e-4, atol=1e-4
            )
        row = {
            "prompt": prompt,
            "tokens": ids.shape[1],
            "logits_max_abs_error": float((decoded - output.logits[0, -1]).abs().max()),
            "layers": [],
        }
        # Exercise the corrected cache at the final block boundary, both
        # before and at the output token. HF's original tuple stays intact.
        with torch.no_grad():
            cached_output, cached_states = forward_with_raw_hidden_states(model, ids)
            torch.testing.assert_close(
                cached_output.logits, output.logits, rtol=0, atol=0
            )
        intervention = HiddenStateIntervention(model, model.transformer.h[-1])
        handle = model.transformer.h[-1].register_forward_hook(intervention.hook)
        try:
            for pos in (position, ids.shape[1] - 1):
                with torch.no_grad():
                    restored = intervention.forward(cached_states[-1][0, pos], ids, pos)
                    torch.testing.assert_close(restored, raw, rtol=0, atol=0)
        finally:
            handle.remove()
        row["final_layer_cached_noop_max_abs_error"] = 0.0
        for layer in (4, 10, 12, 15, 18, 20):
            states = output.hidden_states[layer].detach()
            reference = states[0, position].clone()
            torch.manual_seed(41 + layer)
            direction = torch.randn_like(reference)
            direction *= reference.norm() / direction.norm()

            def independent(hidden):
                replaced = states.clone()
                replaced[0, position] = hidden
                return manual_tail(model, replaced, layer)

            intervention = HiddenStateIntervention(model, model.transformer.h[-1])
            handle = model.transformer.h[layer - 1].register_forward_hook(
                intervention.hook
            )
            try:
                with torch.no_grad():
                    noop = intervention.forward(reference, ids, position)
                    torch.testing.assert_close(noop, raw, rtol=0, atol=0)
                    explicit = independent(reference)
                    torch.testing.assert_close(explicit, raw, rtol=1e-5, atol=1e-5)
                    shifted = intervention.forward(
                        reference + direction * 0.1, ids, position
                    )
                    torch.testing.assert_close(
                        shifted,
                        independent(reference + direction * 0.1),
                        rtol=1e-5,
                        atol=1e-5,
                    )
                product = intervention.jvp(reference, direction, ids, position)
                fd_errors = {}
                with torch.no_grad():
                    for epsilon in (0.01, 0.003, 0.001):
                        fd = (
                            independent(reference + epsilon * direction)
                            - independent(reference - epsilon * direction)
                        ) / (2 * epsilon)
                        fd_errors[str(epsilon)] = relative_error(fd, product)
                assert min(fd_errors.values()) < 0.005, fd_errors
                # Independent first-order reverse derivative gives the adjoint identity.
                h = reference.clone().requires_grad_(True)
                cotangent = torch.randn_like(h)
                grad = torch.autograd.grad(independent(h), h, grad_outputs=cotangent)[0]
                torch.testing.assert_close(
                    product @ cotangent, grad @ direction, rtol=2e-4, atol=1e-4
                )
                result = {
                    "layer": layer,
                    "noop_max_abs_error": float((noop - raw).abs().max()),
                    "tail_relative_error": relative_error(explicit, raw),
                    "finite_difference_relative_errors": fd_errors,
                    "adjoint_abs_error": float(
                        (product @ cotangent - grad @ direction).abs()
                    ),
                }
                row["layers"].append(result)
                print("MATH", ids.shape[1], result, flush=True)
            finally:
                handle.remove()
        report.append(row)
    return report


def check_saved(model, tokenizer, destination):
    report = {}
    # Reuse exactly the production entry points and the already loaded checkpoint.
    with (
        patch.object(AutoModelForCausalLM, "from_pretrained", return_value=model),
        patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
        patch.object(measure_faithfulness, "MODEL_ID", MODEL_ID),
        patch.object(measure_jacobian_lens, "MODEL_ID", MODEL_ID),
    ):
        relation = "country_capital_city"
        for protocol, extra in [
            ("main", ["--legacy-split"]),
            ("shared_split", ["--distinct-first-token"]),
        ]:
            out_dir = destination / protocol
            argv = [
                "verify",
                "--relation",
                relation,
                "--data-dir",
                str(ROOT / "data/relations"),
                "--out-dir",
                str(out_dir),
                "--layers",
                "4,10,12,15,18,20",
                *extra,
            ]
            with patch.object(sys, "argv", argv):
                measure_faithfulness.main()
            new = read(out_dir / f"faithfulness_{relation}.json")
            old = read(
                ROOT
                / "results/gptj/results"
                / protocol
                / f"faithfulness_{relation}.json"
            )
            for key in (
                "grid",
                "best_layer",
                "best_beta",
                "best_faith",
                "per_subject",
                "split_fingerprint",
                "lm_correct",
            ):
                assert new[key] == old[key], (protocol, key, new[key], old[key])
            report[protocol] = {
                "relation": relation,
                "grid_cells_equal": 36,
                "subject_outcomes_equal": len(new["per_subject"]),
            }
        out = destination / "lens.json"
        argv = [
            "verify",
            "--rels",
            "country_capital_city,city_in_country,person_occupation",
            "--data-dir",
            str(ROOT / "data/relations"),
            "--out",
            str(out),
            "--layer",
            "15",
            "--legacy-split",
            "--betas",
            "1",
        ]
        with patch.object(sys, "argv", argv):
            measure_jacobian_lens.main()
        new = read(out)
        old = read(ROOT / "results/gptj/results/jacobian_lens.json")
        rank_differences = []
        count = 0
        for relation, result in new.items():
            for a, b in zip(result["rows"], old[relation]["rows"], strict=True):
                assert a["subject"] == b["subject"]
                assert a["model_top1"] == b["model_top1"]
                for key in ("rank_diff", "rank_lre"):
                    count += 1
                    if a[key] != b[key]:
                        rank_differences.append(
                            [relation, a["subject"], key, a[key], b[key]]
                        )
                        assert abs(a[key] - b[key]) <= 1, rank_differences[-1]
        report["lens"] = {
            "rank_comparisons": count,
            "rank_differences": rank_differences,
        }
    return report


def check_protocol(tokenizer):
    checks = 0
    for path in sorted((ROOT / "data/relations").glob("*.json")):
        data = read(path)
        for sample in data["samples"]:
            subject = sample["subject"]
            for template in (data["prompt_templates"][0], "{} is known for"):
                prompt = template.format(subject)
                start = prompt.rindex(subject)
                end = start + len(subject)
                encoded = tokenizer(prompt, return_offsets_mapping=True)
                position = min(
                    len(tokenizer(prompt[:end]).input_ids) - 1,
                    len(encoded.input_ids) - 1,
                )
                expected = max(
                    i
                    for i, (a, b) in enumerate(encoded.offset_mapping)
                    if a < end and b > start
                )
                assert position == expected, (path.stem, subject, template)
                checks += 1
    root = ROOT / "results/gptj/results"
    lens = read(root / "jacobian_lens.json")
    years = read(root / "jacobian_lens_years.json")
    correlations = {}
    for variant in ("saved", "original_years", "exclude_years"):
        x, y = [], []
        for relation in sorted(lens):
            is_year = relation.startswith("presidents_")
            if is_year and variant == "exclude_years":
                continue
            corrected = is_year and variant == "saved"
            result = (years if corrected else lens)[relation]
            faith = read(
                root
                / "main"
                / f"faithfulness_{relation}{'_ts' if corrected else ''}.json"
            )
            x.append(-result["agg"]["diff"]["median_lograng"])
            y.append(faith["best_faith"])
        rho, p = spearmanr(x, y)
        correlations[variant] = {"n": len(x), "rho": float(rho), "p": float(p)}
    shared = read(root / "shared_split/jacobian_lens.json")
    x, y = [], []
    subject_checks = 0
    for relation, result in sorted(shared.items()):
        faith = read(root / "shared_split" / f"faithfulness_{relation}.json")
        ranks = read(root / "shared_split" / f"subject_ranks_{relation}.json")
        assert (
            result["split_fingerprint"]
            == faith["split_fingerprint"]
            == ranks["split_fingerprint"]
        )
        assert (
            {r["subject"] for r in result["rows"]}
            == set(faith["per_subject"])
            == set(ranks["ranks"])
        )
        for subject, hit in faith["per_subject"].items():
            assert int(ranks["ranks"][subject]["rank"] == 0) == hit
            subject_checks += 1
        x.append(-result["agg"]["diff"]["median_lograng"])
        y.append(faith["best_faith"])
    rho, p = spearmanr(x, y)
    correlations["shared_split"] = {"n": len(x), "rho": float(rho), "p": float(p)}
    return {
        "subject_position_checks": checks,
        "position_mismatches": 0,
        "shared_subject_outcome_checks": subject_checks,
        "year_tokens": {
            year: [tokenizer.decode([i]) for i in tokenizer.encode(" " + year)]
            for year in ("1743", "1932", "2000")
        },
        "correlations": correlations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "generated/gptj_verification"
    )
    parser.add_argument("--math-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    print("Loading cached float32 GPT-J", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    report = {
        "model_id": MODEL_ID,
        "revision": model.config._commit_hash,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(model.dtype),
    }
    report["protocol"] = check_protocol(tokenizer)
    report["math"] = check_math(model, tokenizer, "cuda:0")
    (args.out_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.math_only:
        report["saved_results"] = check_saved(model, tokenizer, args.out_dir)
    (args.out_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified:", args.out_dir / "verification.json", flush=True)


if __name__ == "__main__":
    main()
