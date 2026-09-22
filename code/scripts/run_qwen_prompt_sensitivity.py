"""Fixed-layer neutral-prompt comparison against retained Qwen LRE outcomes."""

import argparse
import hashlib
import json
import socket
import time
from functools import partial
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import model_sweep as sweep
from jlens import splits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/five_model_sweep/qwen3_8b_base"
MODEL = "Qwen/Qwen3-8B-Base"
REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
# Frozen before measuring; these contain no relation or answer information.
PROMPTS = {
    "known_for": "{} is known for",
    "bare_subject": "{}",
    "is": "{} is",
    "described_as": "{} can be described as",
    "information_about": "Information about {}:",
    "facts_about": "Facts about {}:",
    "entity_known_for": "Entity: {} is known for",
    "newline_known_for": "\n{} is known for",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT / "results/qwen_prompt_sensitivity"
    )
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--protocols", default="main,shared_split")
    parser.add_argument(
        "--prompts",
        default=",".join(name for name in PROMPTS if name != "newline_known_for"),
        help="comma-separated prompt names; newline_known_for is the position control",
    )
    args = parser.parse_args()
    names = args.prompts.split(",")
    if not names or any(name not in PROMPTS for name in names):
        parser.error(f"Unknown prompt; choose from {list(PROMPTS)}")
    prompts = {name: PROMPTS[name] for name in names}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    print("LOAD", MODEL, flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
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
    verification = sweep.verify(model, tokenizer, 19, args.batch_size)
    metadata = dict(
        model_id=MODEL,
        revision=REVISION,
        layer=19,
        prompts=prompts,
        dtype="float32",
        torch=torch.__version__,
        host=socket.gethostname(),
        device=torch.cuda.get_device_name(),
        batch_size=args.batch_size,
        verification=verification,
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                Path(sweep.__file__),
                ROOT / "jlens/model.py",
                ROOT / "jlens/splits.py",
            ]
        },
    )
    sweep.write(args.root / f"worker_{args.worker}.json", metadata)
    print("VERIFIED", verification, flush=True)
    for protocol in args.protocols.split(","):
        for source in sorted((SOURCE / protocol).glob("*.json")):
            original = json.loads(source.read_text())
            if "agg" not in original.get("lens", {}):
                continue
            assert original["revision"] == REVISION
            data_path = sweep.DATA / source.name
            data = json.loads(data_path.read_text())
            trailing = sweep.trailing_space_needed(
                tokenizer, source.stem, data["samples"]
            )
            assert trailing == original["trailing_space"]
            samples = (
                data["samples"]
                if protocol == "main"
                else splits.filter_identifying_objects(
                    data["samples"], tokenizer, trailing
                )
            )
            for name, prompt in prompts.items():
                target = args.root / protocol / name / source.name
                with sweep.measurement_claim(target) as claimed:
                    if not claimed or target.exists():
                        continue
                    start = time.time()
                    print("START", protocol, source.stem, name, flush=True)
                    # Guard the subject token index for prefix and punctuation variants.
                    demos, refs, tests = sweep.partitions(samples, protocol, lens=True)
                    positions = {}
                    for sample in refs + tests:
                        subject = sample["subject"]
                        text = prompt.format(subject)
                        begin = text.rindex(subject)
                        end = begin + len(subject)
                        tokens = tokenizer(text, return_offsets_mapping=True)
                        if name == "newline_known_for":
                            baseline_ids = tokenizer(
                                PROMPTS["known_for"].format(subject)
                            ).input_ids
                            assert tokens.input_ids == (
                                tokenizer("\n").input_ids + baseline_ids
                            ), (subject, "newline changed original token IDs")
                        expected = max(
                            i
                            for i, (a, b) in enumerate(tokens.offset_mapping)
                            if a < end and b > begin
                        )
                        actual = min(
                            len(tokenizer(text[:end]).input_ids) - 1,
                            len(tokens.input_ids) - 1,
                        )
                        assert actual == expected, (text, actual, expected)
                        if name == "newline_known_for":
                            assert actual > 0, (subject, actual)
                        positions[subject] = actual
                    lens = sweep.measure_lens(
                        model,
                        tokenizer,
                        samples,
                        data["prompt_templates"][0],
                        protocol,
                        19,
                        trailing,
                        args.batch_size,
                        100,
                        neutral_prompt=prompt,
                    )
                    # Require exactly the retained cohort and relation-prompt targets.
                    keys = ("subject", "model_top1_id", "n_references")
                    assert [[r[k] for k in keys] for r in lens["rows"]] == [
                        [r[k] for k in keys] for r in original["lens"]["rows"]
                    ]
                    assert (
                        lens["split_fingerprint"]
                        == original["lens"]["split_fingerprint"]
                    )
                    differences = [
                        r["rank_diff"] - old["rank_diff"]
                        for r, old in zip(lens["rows"], original["lens"]["rows"])
                    ]
                    sweep.write(
                        target,
                        dict(
                            relation=source.stem,
                            protocol=protocol,
                            prompt_name=name,
                            lens=lens,
                            positions=positions,
                            faithfulness=original["faithfulness"]["best_faith"],
                            n_lre=original["faithfulness"]["n_test"],
                            split_fingerprint=splits.split_fingerprint(
                                demos, refs, tests
                            ),
                            baseline_rank_differences=differences
                            if name == "known_for"
                            else None,
                            retained_source=str(source.relative_to(ROOT)),
                            retained_sha256=sha(source),
                            data_sha256=sha(data_path),
                            worker=args.worker,
                            elapsed_seconds=time.time() - start,
                        ),
                    )
                    print(
                        "DONE",
                        protocol,
                        source.stem,
                        name,
                        round(time.time() - start, 1),
                        flush=True,
                    )
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
