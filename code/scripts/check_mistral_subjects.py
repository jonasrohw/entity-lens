"""Audit subject overlap and prepare frozen-setting runs on cleaned inputs."""

import json
import shutil
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from jlens import grid_search as grid
from jlens import model_sweep as sweep
from jlens import splits
from jlens.grid_confirmation import unique_unambiguous_subjects

BASE = Path(__file__).resolve().parents[1]
GRID = BASE / "generated/mistral_full_grid"


def overlap(samples, protocol, seed):
    parts = grid.partition(samples, protocol, seed=seed)
    names = [[s["subject"] for s in part] for part in parts]
    sets = [set(part) for part in names]
    return dict(
        n_rows=[len(part) for part in names],
        n_unique_subjects=[len(part) for part in sets],
        demo_reference=sorted(sets[0] & sets[1]),
        demo_test=sorted(sets[0] & sets[2]),
        reference_test=sorted(sets[1] & sets[2]),
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.3", local_files_only=True
    )
    audit = {}
    for protocol, original in [
        ("main", GRID),
        ("shared_split", BASE / "generated/mistral_shared_confirmation"),
    ]:
        root = BASE / "generated" / f"mistral_clean_{protocol}"
        root.mkdir(exist_ok=True)
        config = json.loads((original / "confirmation_config.json").read_text())
        config.update(
            protocol=protocol,
            source_grid_root=str(GRID),
            subject_policy="unique_unambiguous_subjects",
            note="Frozen original hyperparameters; collapse identical subject/object rows and exclude multi-object subjects before splitting. Unchanged effective inputs reuse seed-1 confirmation records.",
        )
        sweep.write(root / "confirmation_config.json", config)
        shutil.copy2(GRID / "relation_folds.json", root / "relation_folds.json")
        reused = []
        rerun = []
        for path in sorted(sweep.DATA.glob("*.json")):
            samples = json.loads(path.read_text())["samples"]
            clean = unique_unambiguous_subjects(samples)
            counts = Counter(s["subject"] for s in samples)
            objects = {}
            for sample in samples:
                objects.setdefault(sample["subject"], set()).add(sample["object"])
            audit.setdefault(
                path.stem,
                dict(
                    samples=len(samples),
                    unique_subjects=len(counts),
                    duplicate_subjects={s: n for s, n in counts.items() if n > 1},
                    ambiguous_subjects={
                        s: sorted(o) for s, o in objects.items() if len(o) > 1
                    },
                    cleaned_samples=len(clean),
                    protocols={},
                ),
            )
            before, after = samples, clean
            if protocol == "shared_split":
                trailing = sweep.trailing_space_needed(tokenizer, path.stem, samples)
                before = splits.filter_identifying_objects(samples, tokenizer, trailing)
                after = splits.filter_identifying_objects(clean, tokenizer, trailing)
            audit[path.stem]["protocols"][protocol] = {
                str(seed): dict(
                    original=overlap(before, protocol, seed),
                    cleaned=overlap(after, protocol, seed),
                )
                for seed in [0, 1]
            }
            previous = original / "confirmation" / path.name
            if before == after or path.stem not in config["faith_configs"]:
                row = json.loads(previous.read_text())
                row.update(
                    subject_policy=config["subject_policy"],
                    input_samples=len(samples),
                    cleaned_samples=len(clean),
                    reused_from=str(previous),
                    reuse_reason="identical effective samples or excluded frozen cohort",
                )
                sweep.write(root / "confirmation" / path.name, row)
                reused.append(path.stem)
            else:
                assert len(after) >= 18, (path.stem, protocol, len(after))
                rerun.append(path.stem)
        sweep.write(root / "reuse_manifest.json", dict(reused=reused, rerun=rerun))
        (root / "READY").touch()
        print(protocol, "reused", len(reused), "rerun", rerun, flush=True)
    sweep.write(GRID / "analysis/input_subject_audit.json", audit)


if __name__ == "__main__":
    main()
