"""Audit the retained Qwen country lens without changing its saved measurements.

Controls preserve the checkpoint, layer, references, and relation target IDs.
The newline control preserves every original token ID after one added token.
"""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep
from jlens.model import HiddenStateIntervention

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/five_model_sweep/qwen3_8b_base/main/country_capital_city.json"
MODEL = "Qwen/Qwen3-8B-Base"
REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes", default="original,next_token,newline,without_outlier_coordinates"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/qwen_country_position_audit"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    original = json.loads(SOURCE.read_text())
    data = json.loads((sweep.DATA / SOURCE.name).read_text())
    _, refs, tests = sweep.partitions(data["samples"], "main", lens=True)
    names = [x["subject"] for x in tests]
    anchors = [x["subject"] for x in refs]
    assert names == [r["subject"] for r in original["lens"]["rows"]]
    layer = original["lens"]["layer"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    print("Loading", MODEL, flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            device_map={"": args.device},
        )
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    blocks, norm = sweep.layout(model)
    verification = sweep.verify(model, tokenizer, layer, 2)
    print("Verified", verification, flush=True)
    provenance = dict(
        model=MODEL,
        revision=REVISION,
        layer=layer,
        dtype="float32",
        attention="eager",
        torch=torch.__version__,
        transformers=transformers.__version__,
        verification=verification,
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                SOURCE,
                Path(__file__),
                Path(sweep.__file__),
                ROOT / "jlens/model.py",
                sweep.DATA / SOURCE.name,
            )
        },
    )
    target_ids = [r["model_top1_id"] for r in original["lens"]["rows"]]

    def readout(vector, target):
        with torch.no_grad():
            logits = sweep.decode(model, norm, vector)
        return dict(
            norm=float(vector.norm()),
            rank=int((logits > logits[target]).sum()),
            top=[tokenizer.decode([i]) for i in logits.topk(5).indices.tolist()],
        )

    for mode in args.modes.split(","):
        assert mode in (
            "original",
            "next_token",
            "newline",
            "without_outlier_coordinates",
        )
        cache, positions = {}, {}
        for subject in names:
            base = subject + " is known for"
            prompt = ("\n" if mode == "newline" else "") + base
            tok = tokenizer(prompt, return_offsets_mapping=True)
            begin = prompt.rindex(subject)
            end = begin + len(subject)
            expected = max(
                i
                for i, (a, b) in enumerate(tok.offset_mapping)
                if a < end and b > begin
            )
            c = sweep.cache_prompt(model, tokenizer, prompt, subject, [layer])
            assert c["position"] == expected, (subject, c["position"], expected)
            if mode == "newline":
                assert tok.input_ids[1:] == tokenizer(base).input_ids
                assert tok.input_ids[0] == tokenizer("\n").input_ids[0]
            if mode == "next_token":
                c["position"] += 1
                captured = {}

                def capture(module, inputs, output):
                    state = output[0] if isinstance(output, tuple) else output
                    captured["state"] = state[0, c["position"]].detach().clone()

                handle = blocks[layer - 1].register_forward_hook(capture)
                try:
                    with torch.no_grad():
                        model(input_ids=c["ids"])
                finally:
                    handle.remove()
                c["states"][layer] = captured["state"]
            state = c["states"][layer]
            largest = state.abs().topk(5).indices.tolist()
            positions[subject] = dict(
                tokens=[tokenizer.decode([i]) for i in tok.input_ids],
                token_ids=tok.input_ids,
                offsets=tok.offset_mapping,
                expected_subject_position=expected,
                selected_position=c["position"],
                previous_position_valid=expected > 0,
                state_norm=float(state.norm()),
                largest_coordinates={str(i): float(state[i]) for i in largest},
            )
            cache[subject] = c
        products = []
        noop_errors = {}
        intervention = HiddenStateIntervention(model, blocks[-1])
        handle = blocks[layer - 1].register_forward_hook(intervention.hook)
        try:
            for anchor in anchors:
                c = cache[anchor]
                ref = c["states"][layer]
                with torch.no_grad():
                    noop = intervention.forward(ref, c["ids"], c["position"])
                error = float((noop - c["states"]["final"]).abs().max())
                assert error == 0, (mode, anchor, error)
                noop_errors[anchor] = error
                directions = torch.stack(
                    [cache[s]["states"][layer] - ref for s in names]
                )
                if mode == "without_outlier_coordinates":
                    # Post-hoc diagnostic: the two largest position-0 coordinates
                    # found in the original run. This is not a proposed default.
                    directions[:, [2276, 233]] = 0
                products.append(
                    sweep.jvps(
                        intervention, ref, directions, c["ids"], c["position"], 2
                    )
                )
                print(mode, "reference", anchor, flush=True)
        finally:
            handle.remove()
        products = torch.stack(products)
        single = {s: len(tokenizer(s).input_ids) == 1 for s in names}
        rows, vectors = [], []
        for j, subject in enumerate(names):
            valid = [i for i, s in enumerate(anchors) if s != subject]
            vector = products[valid, j].mean(0)
            vectors.append(vector)
            row = dict(
                subject=subject,
                original_single_token=single[subject],
                all_references=readout(vector, target_ids[j]),
                by_group={},
            )
            for group, is_single in (
                ("single_token_refs", True),
                ("multitoken_refs", False),
            ):
                selected = [i for i in valid if single[anchors[i]] == is_single]
                row["by_group"][group] = dict(
                    references=[anchors[i] for i in selected],
                    **readout(products[selected, j].mean(0), target_ids[j]),
                )
            row["per_reference"] = {
                anchors[i]: readout(products[i, j], target_ids[j]) for i in valid
            }
            rows.append(row)
        vectors = torch.stack(vectors)
        unit = torch.nn.functional.normalize(vectors, dim=-1)
        result = dict(
            provenance=provenance,
            mode=mode,
            positions=positions,
            reference_names=anchors,
            noop_errors=noop_errors,
            rows=rows,
            differential_cosines=(unit @ unit.T).tolist(),
            ablated_direction_coordinates=[2276, 233]
            if mode == "without_outlier_coordinates"
            else [],
        )
        if mode == "original":
            result["rank_differences_from_retained"] = [
                r["all_references"]["rank"] - old["rank_diff"]
                for r, old in zip(rows, original["lens"]["rows"])
            ]
        sweep.write(args.out / f"{mode}.json", result)
        print(
            "DONE",
            mode,
            [
                (
                    r["subject"],
                    r["all_references"]["rank"],
                    r["all_references"]["top"][0],
                )
                for r in rows
            ],
            flush=True,
        )


if __name__ == "__main__":
    main()
