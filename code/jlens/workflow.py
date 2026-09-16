"""Portable CPU reproduction and resumable GPU measurement workflows."""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from jlens.config import OUTPUT_DIR, ROOT, YEAR_RELATIONS
from jlens.figure_data import build_figure_data
from jlens.io import read_json, write_json


def command_line(
    root: Path, output_dir: Path, command: str, *arguments: str
) -> list[str]:
    """Use the current uv environment for every child command."""
    return [
        sys.executable,
        "-m",
        "jlens",
        "--root",
        str(root),
        "--output-dir",
        str(output_dir),
        command,
        *map(str, arguments),
    ]


def run_command(
    root: Path,
    output_dir: Path,
    command: str,
    *arguments: str,
    capture: Path | None = None,
) -> None:
    """Fail immediately if a measurement or analysis fails."""
    invocation = command_line(root, output_dir, command, *arguments)
    print(f"Running {command}", flush=True)
    if capture is None:
        subprocess.run(invocation, check=True)
        return
    result = subprocess.run(invocation, check=True, capture_output=True, text=True)
    capture.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")


def reproduce() -> None:
    """Create all paper artifacts without loading a language model."""
    parser = argparse.ArgumentParser(description=reproduce.__doc__)
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help="also generate subject panels, rank plots, and outlier reports",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = build_figure_data(ROOT)
    reference = read_json(ROOT / "results/paper_figure_data.json")
    if figure != reference:
        raise ValueError(
            "Rebuilt figure data differs from results/paper_figure_data.json"
        )
    write_json(OUTPUT_DIR / "paper_figure_data.json", figure)
    for command, filename in [
        ("statistics", "statistics.txt"),
        ("table-relations", "table_relations.tex"),
        ("table-prominence", "table_prominence.tex"),
    ]:
        run_command(ROOT, OUTPUT_DIR, command, capture=OUTPUT_DIR / filename)
    for command in ["plot-agreement", "plot-facts"]:
        run_command(ROOT, OUTPUT_DIR, command)
    if args.supplementary:
        for command in [
            "plot-relation-ranks",
            "plot-subject-ranks",
            "plot-subject-outcomes",
            "rank-outliers",
        ]:
            run_command(ROOT, OUTPUT_DIR, command)
    print((OUTPUT_DIR / "statistics.txt").read_text(encoding="utf-8"), end="")
    print(f"Paper artifacts: {OUTPUT_DIR}")


def measurement_plan(
    root: Path, relations: list[str]
) -> list[tuple[Path, str, list[str]]]:
    """Return ordered, resumable stages with one completion artifact each."""
    results = root / "results"
    data_dir = str(root / "data/relations")
    stages = []
    for relation in relations:
        base = [
            "--relation",
            relation,
            "--data-dir",
            data_dir,
            "--out-dir",
            str(results / "main"),
            "--legacy-split",
        ]
        stages.append(
            (
                results / f"main/faithfulness_{relation}.json",
                "measure-faithfulness",
                base,
            )
        )
        if relation in YEAR_RELATIONS:
            stages.append(
                (
                    results / f"main/faithfulness_{relation}_ts.json",
                    "measure-faithfulness",
                    [*base, "--trailing-space", "--tag", "_ts"],
                )
            )
    for filename, selected, extra in [
        ("jacobian_lens.json", relations, []),
        ("jacobian_lens_years.json", sorted(YEAR_RELATIONS), ["--trailing-space"]),
    ]:
        path = results / filename
        stages.append(
            (
                path,
                "measure-jacobian-lens",
                [
                    "--data-dir",
                    data_dir,
                    "--rels",
                    ",".join(selected),
                    "--legacy-split",
                    "--out",
                    str(path),
                    *extra,
                ],
            )
        )
    for filename, command in [
        ("logit_lens.json", "measure-logit-lens"),
        ("answer_prior.json", "measure-answer-prior"),
        ("prominence.json", "measure-prominence"),
        ("prominence_statistics.json", "prominence-statistics"),
    ]:
        stages.append((results / filename, command, ["--out", str(results / filename)]))
    stages.append(
        (
            results / "paper_figure_data.json",
            "build-figure-data",
            ["--output", str(results / "paper_figure_data.json")],
        )
    )
    for relation in relations:
        stages.append(
            (
                results / f"shared_split/faithfulness_{relation}.json",
                "measure-faithfulness",
                [
                    "--relation",
                    relation,
                    "--data-dir",
                    data_dir,
                    "--out-dir",
                    str(results / "shared_split"),
                    "--distinct-first-token",
                    *(["--trailing-space"] if relation in YEAR_RELATIONS else []),
                ],
            )
        )
    # The identifying-token filter leaves both year relations below the minimum
    # split size. Keep the same uncorrected target behavior as the source runner.
    path = results / "shared_split/jacobian_lens.json"
    stages.append(
        (
            path,
            "measure-jacobian-lens",
            [
                "--data-dir",
                data_dir,
                "--rels",
                ",".join(relations),
                "--distinct-first-token",
                "--out",
                str(path),
            ],
        )
    )
    for relation in relations:
        stages.append(
            (
                results / f"shared_split/subject_ranks_{relation}.json",
                "measure-subject-ranks",
                [
                    "--relation",
                    relation,
                    "--data-dir",
                    data_dir,
                    "--faith-dir",
                    str(results / "shared_split"),
                    "--out-dir",
                    str(results / "shared_split"),
                ],
            )
        )
    return stages


def rerun() -> None:
    """Rerun both paper protocols into an isolated directory; resume completed stages."""
    parser = argparse.ArgumentParser(description=rerun.__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "rerun")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without creating files or loading models",
    )
    args = parser.parse_args()
    run_root = args.run_dir.resolve()
    if run_root == ROOT or ROOT.is_relative_to(run_root):
        parser.error("--run-dir must be separate from the retained repository")
    source_data = ROOT / "data/relations"
    relations = sorted(path.stem for path in source_data.glob("*.json"))
    if len(relations) != 28:
        parser.error(
            f"expected 28 relation inputs in {source_data}; found {len(relations)}"
        )
    output_dir = run_root / "generated"
    if not args.dry_run:
        destination = run_root / "data/relations"
        if destination.exists():
            if {p.name: p.read_bytes() for p in destination.glob("*.json")} != {
                p.name: p.read_bytes() for p in source_data.glob("*.json")
            }:
                parser.error(
                    "existing run inputs differ from the source; use a new --run-dir"
                )
        else:
            shutil.copytree(source_data, destination)
    completion_path = run_root / "completed_stages.json"
    completed = set(read_json(completion_path)) if completion_path.exists() else set()
    for artifact, command, arguments in measurement_plan(run_root, relations):
        stage = str(artifact.relative_to(run_root))
        # A SKIPPED file is a completed faithfulness stage with too few samples.
        relation = artifact.stem.removeprefix("faithfulness_").removeprefix(
            "subject_ranks_"
        )
        skipped = artifact.parent / f"SKIPPED_{relation}.json"
        if (stage in completed and artifact.exists()) or (
            artifact.parent.name == "shared_split" and skipped.exists()
        ):
            print(f"Already complete: {artifact.name}")
            continue
        if args.dry_run:
            print(shlex.join(command_line(run_root, output_dir, command, *arguments)))
        else:
            run_command(run_root, output_dir, command, *arguments)
            completed.add(stage)
            write_json(completion_path, sorted(completed))
    if args.dry_run:
        print(
            shlex.join(
                command_line(run_root, output_dir, "reproduce", "--supplementary")
            )
        )
    else:
        run_command(run_root, output_dir, "reproduce", "--supplementary")
