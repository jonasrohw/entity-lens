"""Check standalone gold-token labels against tokenization in their prompt."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from jlens import model_sweep as sweep
from jlens import splits

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "generated/cross_model_audit"


def main():
    models = json.loads((BASE / "results/five_model_sweep/models.json").read_text())
    models += [
        dict(name="olmo3_7b", model_id="allenai/Olmo-3-1025-7B"),
        dict(name="gptj", model_id="EleutherAI/gpt-j-6b"),
    ]
    report = {}
    for model in models:
        name = model["name"]
        tokenizer = AutoTokenizer.from_pretrained(
            model["model_id"], local_files_only=True
        )
        rows = []
        for path in sorted(sweep.DATA.glob("*.json")):
            data = json.loads(path.read_text())
            template = data["prompt_templates"][0]
            for protocol in ["main", "shared_split"]:
                samples = data["samples"]
                trailing = sweep.trailing_space_needed(tokenizer, path.stem, samples)
                if (
                    name == "gptj"
                    and protocol == "main"
                    and path.stem.startswith("presidents_")
                ):
                    trailing = True
                if protocol == "shared_split":
                    samples = splits.filter_identifying_objects(
                        samples, tokenizer, trailing
                    )
                demos, _, tests = sweep.partitions(samples, protocol)
                context = "".join(
                    template.format(s["subject"]) + " " + s["object"] + "\n"
                    for s in demos
                )
                for sample in tests:
                    prompt = (
                        context
                        + template.format(sample["subject"])
                        + (" " if trailing else "")
                    )
                    prefix = tokenizer(prompt).input_ids
                    full = tokenizer(
                        prompt + ("" if trailing else " ") + sample["object"]
                    ).input_ids
                    stable = full[: len(prefix)] == prefix
                    actual = (
                        full[len(prefix)]
                        if stable and len(full) > len(prefix)
                        else None
                    )
                    standalone = splits.first_object_token(
                        sample["object"], tokenizer, trailing
                    )
                    rows.append(
                        dict(
                            relation=path.stem,
                            protocol=protocol,
                            subject=sample["subject"],
                            object=sample["object"],
                            prefix_stable=stable,
                            standalone_id=standalone,
                            continuation_id=actual,
                            standalone_token=tokenizer.decode([standalone]),
                            continuation_token=tokenizer.decode([actual])
                            if actual is not None
                            else None,
                        )
                    )
        report[name] = dict(
            checked=len(rows),
            unstable_prefix=[r for r in rows if not r["prefix_stable"]],
            stable_mismatches=[
                r
                for r in rows
                if r["prefix_stable"] and r["standalone_id"] != r["continuation_id"]
            ],
            rows=rows,
        )
        print(
            name,
            "checked",
            len(rows),
            "unstable",
            len(report[name]["unstable_prefix"]),
            "mismatch",
            len(report[name]["stable_mismatches"]),
            flush=True,
        )
        sweep.write(ROOT / "continuation_tokens.json", report)


if __name__ == "__main__":
    main()
