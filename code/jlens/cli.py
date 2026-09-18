"""Command-line entry point; import each analysis only when requested."""

import argparse
import importlib
import os
import sys
from pathlib import Path

COMMANDS = {
    "corpus-conditionals": (
        "corpus_conditionals",
        "Count directional corpus co-occurrence and correlate with LRE scores",
    ),
    "reproduce": (
        "workflow",
        "Regenerate paper statistics, tables, and figures on CPU",
    ),
    "rerun": ("workflow", "Rerun GPU measurements in a separate directory"),
    "statistics": ("paper_statistics", "Print the paper's numerical results"),
    "build-figure-data": ("figure_data", "Rebuild Figure 1's input from measurements"),
    "table-relations": ("table_relations", "Print the relation table as LaTeX"),
    "table-prominence": ("table_prominence", "Print the prominence table as LaTeX"),
    "plot-agreement": (
        "plot_relation_agreement",
        "Plot relation agreement and faithfulness",
    ),
    "plot-facts": ("plot_fact_ranks", "Plot the paper's four fact-level comparisons"),
    "plot-relation-ranks": ("plot_relation_ranks", "Plot aggregate lens and LRE ranks"),
    "plot-subject-ranks": (
        "plot_subject_ranks",
        "Plot continuous subject ranks by relation",
    ),
    "plot-subject-outcomes": (
        "plot_subject_outcomes",
        "Plot binary subject outcomes by relation",
    ),
    "subject-statistics": (
        "subject_statistics",
        "Compute within-relation AUC statistics",
    ),
    "relation-statistics": (
        "relation_statistics",
        "Run the supplementary domain analysis",
    ),
    "rank-outliers": ("rank_outliers", "Rank relations by regression residuals"),
    "prominence-statistics": (
        "prominence_statistics",
        "Recompute prominence and leakage controls",
    ),
    "measure-faithfulness": (
        "measure_faithfulness",
        "Sweep LRE layers and beta values (GPU)",
    ),
    "measure-jacobian-lens": (
        "measure_jacobian_lens",
        "Score the differential and affine (+zeroth-order) Jacobian lenses (GPU)",
    ),
    "measure-subject-ranks": (
        "measure_subject_ranks",
        "Score the selected LRE per subject (GPU)",
    ),
    "measure-logit-lens": (
        "measure_logit_lens",
        "Score the matched logit-lens baseline (GPU)",
    ),
    "measure-logit-lens-diff": (
        "measure_logit_lens_diff",
        "Score the differential logit-lens baseline, no Jacobian (GPU)",
    ),
    "measure-jacobian-lens-raw": (
        "measure_jacobian_lens_raw",
        "Score the standard Jacobian lens on raw activations, no difference (GPU)",
    ),
    "measure-answer-prior": (
        "measure_answer_prior",
        "Measure neutral answer-token priors (GPU)",
    ),
    "measure-prominence": (
        "measure_prominence",
        "Measure object prominence by generation (GPU)",
    ),
    "probe-names": ("probe_names", "Inspect Jacobian-lens predictions for names (GPU)"),
}

# These commands expose only the common root and output options.
SIMPLE_COMMANDS = {
    "statistics",
    "table-relations",
    "table-prominence",
    "plot-agreement",
    "plot-facts",
    "plot-subject-outcomes",
    "relation-statistics",
}


def main(argv: list[str] | None = None) -> None:
    """Configure paths before loading a command's implementation."""
    parser = argparse.ArgumentParser(
        description="Jacobian-lens paper reproduction and experiments"
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="checkout or rerun directory containing data/ and results/",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="derived artifacts (default: ROOT/generated)"
    )
    parser.add_argument(
        "--model", help="Transformers model id (default: allenai/Olmo-3-1025-7B)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, (_, description) in COMMANDS.items():
        subparsers.add_parser(
            command,
            help=description,
            description=description,
            add_help=command in SIMPLE_COMMANDS,
        )
    args, remaining = parser.parse_known_args(argv)
    if args.command in SIMPLE_COMMANDS and remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if args.root is not None:
        os.environ["JLENS_ROOT"] = str(args.root.resolve())
    if args.output_dir is not None:
        os.environ["JLENS_OUTPUT_DIR"] = str(args.output_dir.resolve())
    if args.model is not None:
        os.environ["JLENS_MODEL"] = args.model
    module_name, _ = COMMANDS[args.command]
    previous_argv = sys.argv
    sys.argv = [f"jlens {args.command}", *remaining]
    try:
        module = importlib.import_module(f"jlens.{module_name}")
        if args.command in {"reproduce", "rerun"}:
            getattr(module, args.command)()
        else:
            module.main()
    except ModuleNotFoundError as error:
        if error.name in {"ahocorasick", "orjson", "zstandard"}:
            parser.exit(
                2,
                "Corpus dependencies are missing. Run: uv sync --extra corpus --inexact\n",
            )
        if error.name in {"torch", "transformers"}:
            parser.exit(2, "GPU dependencies are missing. Run: uv sync --extra gpu\n")
        raise
    finally:
        sys.argv = previous_argv
