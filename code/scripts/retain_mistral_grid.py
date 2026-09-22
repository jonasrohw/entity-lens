"""Publish compact scoring archives, readable results, and diagnostic figures."""

import csv
import gzip
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from jlens import grid_statistics as stats

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "generated/mistral_full_grid"
SHARED_CONFIRMATION = BASE / "generated/mistral_shared_confirmation"
EXTRA_RUNS = {
    "shared_confirmation": SHARED_CONFIRMATION,
    "clean_main": BASE / "generated/mistral_clean_main",
    "clean_shared_split": BASE / "generated/mistral_clean_shared_split",
}
DEST = BASE / "results/mistral_grid_search"


def archive(source):
    relative = next(
        (
            Path(name) / source.name
            for name, root in EXTRA_RUNS.items()
            if source.parent == root / "confirmation"
        ),
        None,
    )
    if relative is None:
        relative = source.relative_to(SOURCE)
    target = DEST / "raw" / (str(relative) + ".gz")
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    temporary = target.with_suffix(".tmp")
    with temporary.open("wb") as output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=output, mtime=0
        ) as compressed:
            compressed.write(data)
    temporary.replace(target)
    return dict(
        path=str(relative),
        sha256=hashlib.sha256(data).hexdigest(),
        raw_bytes=len(data),
        compressed_bytes=target.stat().st_size,
    )


def figures(main, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        json.loads(path.read_text())
        for path in sorted((SOURCE / "main").glob("*.json"))
    ]
    fixed = dict(
        prompt="known_for",
        layer=17,
        references=8,
        readout="diff",
        beta=None,
        metric="median_logrank",
    )
    best = main["analyses"]["coarse_refs8"]["exploratory_top10"][0]["config"]
    x = np.array([stats.lens_value(row, fixed) for row in rows])
    tuned = np.array([stats.lens_value(row, best) for row in rows])
    y = np.array(
        [
            main["analyses"]["coarse_refs8"]["selected_faith"][row["relation"]]["score"]
            for row in rows
        ]
    )
    small = np.array([row["faith"]["n"] < 4 for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    cases = [
        (x, np.ones(len(rows), dtype=bool), "Original settings: all 28"),
        (tuned, np.ones(len(rows), dtype=bool), "Tuned lens: same LRE target"),
        (x, ~small, "Original settings: at least 4 tests"),
    ]
    for ax, (values, include, title) in zip(axes, cases):
        ax.scatter(
            values[include & ~small],
            y[include & ~small],
            s=32,
            color="#2878b5",
            alpha=0.85,
        )
        ax.scatter(
            values[small],
            y[small],
            s=55,
            marker="x",
            color="#cf5b25",
            label="Three-test relations",
        )
        result = stats.correlation(values[include], y[include])
        ax.set_title(f"{title}\nn={result['n']}, rho={result['rho']:.3f}")
        ax.set_xlabel("− median log10(rank + 1)")
        ax.set_ylim(-0.03, 1.04)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Original-grid LRE faithfulness")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[2].text(
        0.03,
        -0.26,
        "Orange points shown for context;\nexcluded only from this panel’s rho.",
        transform=axes[2].transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(DEST / "main_comparison.png", dpi=160)
    fig.savefig(DEST / "main_comparison.pdf")
    plt.close(fig)
    betas = [b for b in rows[0]["betas"] if b > 0]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, prompt, title in zip(
        axes, ["known_for", "bare_subject"], ["“subject is known for”", "Bare subject"]
    ):
        matrix = np.full((len(betas), 33), np.nan)
        for bi, beta in enumerate(betas):
            for layer in range(33):
                config = dict(
                    prompt=prompt,
                    layer=layer,
                    references=8,
                    readout="affine",
                    beta=beta,
                    metric="median_logrank",
                )
                values = np.asarray([stats.lens_value(row, config) for row in rows])
                if np.isfinite(values).all():
                    matrix[bi, layer] = stats.correlation(values, y)["rho"]
        mesh = ax.imshow(
            matrix, origin="lower", aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r"
        )
        ax.set_title(title)
        ax.set_xlabel("State (0 = embedding; 1–32 = block outputs)")
        ax.set_xticks([0, 5, 10, 15, 20, 25, 32])
        ticks = [0, 1, 2, 5, 10, 17, 25, 31, 33, 35, 37, 39]
        ax.set_yticks(ticks, [f"{betas[i]:g}" for i in ticks])
        if prompt == best["prompt"]:
            ax.plot(
                best["layer"],
                betas.index(best["beta"]),
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
    axes[0].set_ylabel("Affine lens beta (sampled grid)")
    fig.suptitle(
        "Exploratory lens grid, original LRE target, 8 lens references, all 28 relations"
    )
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.10, top=0.90, wspace=0.12)
    colorbar_axis = fig.add_axes([0.92, 0.10, 0.015, 0.80])
    fig.colorbar(mesh, cax=colorbar_axis, label="Spearman rho")
    fig.savefig(DEST / "layer_beta_grid.png", dpi=160)
    fig.savefig(DEST / "layer_beta_grid.pdf")
    plt.close(fig)


def readable():
    main = json.loads((SOURCE / "analysis/main.json").read_text())
    diagnostic = json.loads((SOURCE / "analysis/diagnostics_main.json").read_text())
    lines = [
        "# Mistral expanded-grid results",
        "",
        "The expanded search improves LRE fitting. Small test sets, trivial first-token targets, and repeated subjects also affect the correlation. On cleaned seed-1 shared inputs, frozen expanded LRE settings with the original lens reach rho=0.761 versus 0.665 for the original LRE settings. The large gains from tuning the lens do not reproduce consistently. All searched maxima are exploratory.",
        "",
        "| Main protocol, all 28 relations | Spearman rho |",
        "| --- | --- |",
        f"| Original LRE and original lens | {main['analyses']['coarse_refs8']['baseline_lens']['rho']:.3f} |",
        f"| All usable LRE states and continuous positive beta; original lens | {main['analyses']['continuous_refs8']['baseline_lens']['rho']:.3f} |",
        f"| Original LRE; best searched lens | {main['analyses']['coarse_refs8']['exploratory_top10'][0]['rho']:.3f} |",
        f"| Expanded LRE; best searched lens | {main['analyses']['continuous_refs8']['exploratory_top10'][0]['rho']:.3f} |",
        "",
        "The original-target winner is the bare-subject affine lens, state 10, lens beta=30, eight references. The expanded-target winner is bare-subject affine, state 8, lens beta=30, four lens references (eight LRE references). LRE beta and lens beta are separate parameters.",
        "",
        "The three relations `name_gender`, `presidents_birth_year`, and `presidents_election_year` each have only three LRE test subjects. Both year relations have an intercept-only first-token faithfulness of 1. The unchanged original method has rho=0.637 on the 25 relations with at least four tests; this is a changed-cohort sensitivity, not the all-28 result.",
        "",
        "| Minimum LRE test subjects (unchanged original method) | Relations | rho |",
        "| --- | --- | --- |",
    ]
    for minimum, values in diagnostic["min_test_count_sensitivity"].items():
        lines.append(f"| {minimum} | {values['n']} | {values['rho']:.3f} |")
    lines += [
        "",
        "## Selection checks",
        "",
        "Selecting the lens on 14 predefined relations and evaluating on the other 14 gives rho=0.002 with the original LRE target and rho=-0.213 with the expanded target. The configurations selected on the training relations differ from the full-data winners. Model choice itself used earlier full-data results.",
        "",
        f"The 2,000-permutation maximum-over-grid diagnostic gives adjusted p={main['analyses']['coarse_refs8']['maximum_permutation_test']['adjusted_p']:.4f} for the original target and p={main['analyses']['continuous_refs8']['maximum_permutation_test']['adjusted_p']:.4f} for the expanded target. This adjustment is conditional on the chosen model and LRE target; it does not cover every research choice.",
        "",
    ]
    confirmation = SOURCE / "analysis/confirmation.json"
    if confirmation.exists():
        fresh = json.loads(confirmation.read_text())
        lines += [
            "| Frozen configuration, fresh seed-1 operator fit | Original-grid LRE rho | Expanded LRE rho |",
            "| --- | --- | --- |",
        ]
        for name in fresh["results"]["coarse"]:
            lines.append(
                f"| {name} | {fresh['results']['coarse'][name]['all_relations']['rho']:.3f} | {fresh['results']['continuous'][name]['all_relations']['rho']:.3f} |"
            )
        lines += [
            "",
            "Hyperparameters were frozen before this draw. Demonstrations/references changed and operators were refitted. Some subjects overlap across draws; this is a same-dataset robustness check.",
            "The main-protocol lens winner did not improve correlation on this fresh draw. Keeping the original lens while expanding the LRE fit improved correlation on both draws; the much larger in-sample lens gain did not reproduce.",
            "",
        ]
    else:
        lines += ["Fresh-split confirmation is pending.", ""]
    shared = SOURCE / "analysis/shared_split.json"
    if shared.exists():
        values = json.loads(shared.read_text())["analyses"]
        lines += [
            "## Shared-split protocol",
            "",
            f"Original fixed lens/original LRE: rho={values['coarse_refs8']['baseline_lens']['rho']:.3f}. Expanded LRE with the fixed lens: rho={values['continuous_refs8']['baseline_lens']['rho']:.3f}. Best exploratory expanded-target lens: rho={values['continuous_refs8']['exploratory_top10'][0]['rho']:.3f}. These use the shared protocol’s eligible relations and aligned subject/reference sets, not the main protocol’s 28-relation cohort.",
            "",
        ]
    shared_confirmation = SHARED_CONFIRMATION / "analysis/confirmation.json"
    if shared_confirmation.exists():
        fresh = json.loads(shared_confirmation.read_text())
        lines += [
            "| Shared split, frozen settings and fresh seed-1 fit | Original-grid LRE rho | Expanded LRE rho |",
            "| --- | --- | --- |",
        ]
        for name in fresh["results"]["coarse"]:
            lines.append(
                f"| {name} | {fresh['results']['coarse'][name]['all_relations']['rho']:.3f} | {fresh['results']['continuous'][name]['all_relations']['rho']:.3f} |"
            )
        lines += [
            "",
            "This confirmation also freezes the lens and per-relation LRE hyperparameters before refitting on seed 1. It remains a same-dataset check with overlapping subjects. See [shared confirmation](analysis/shared_confirmation.json) for sample counts and all settings.",
            "The shared expanded-target winner also falls below the original lens on this draw (0.585 versus 0.673). Expanded LRE fitting is not uniformly better: the fixed lens with original-grid LRE settings reaches 0.681 on this draw.",
            "",
        ]
    cleaned = {
        name: json.loads((root / "analysis/confirmation.json").read_text())
        for name, root in EXTRA_RUNS.items()
        if name.startswith("clean_") and (root / "analysis/confirmation.json").exists()
    }
    if len(cleaned) == 2:
        lines += [
            "## Subject-duplicate sensitivity",
            "",
            "The input audit found repeated subjects in nine relations, including subjects with multiple gold objects. Row-wise partitioning does not guarantee subject-disjoint partitions: for example, the original main superhero relation shares three subjects between references and tests; the shared product relation shares one. This is separate from the intended main lens/LRE split difference.",
            "",
            "A further seed-1 check collapses identical subject/object rows and removes subjects with multiple gold objects before splitting. All hyperparameters remain frozen. Because cleaning changes the shuffled pool, demonstrations, references, and tests can all change; this is a data-cleaning sensitivity, not an isolated estimate of leakage bias. Identical effective pools reuse their existing seed-1 measurements.",
            "",
            "| Cleaned seed-1 inputs | Original LRE, original lens | Expanded LRE, original lens | Expanded LRE, selected lens |",
            "| --- | --- | --- | --- |",
        ]
        for name, fresh in cleaned.items():
            c = fresh["results"]["coarse"]["baseline"]["all_relations"]
            e = fresh["results"]["continuous"]
            lines.append(
                f"| {fresh['protocol']} (n={c['n']}) | {c['rho']:.3f} | {e['baseline']['all_relations']['rho']:.3f} | {e['exploratory_winner']['all_relations']['rho']:.3f} |"
            )
        lines += [
            "",
            "[Input and overlap audit](analysis/input_subject_audit.json), [clean main results](analysis/clean_main.json), [clean shared results](analysis/clean_shared_split.json). Original protocol results remain above for comparability.",
            "",
        ]
    lines += [
        "## Controls and artifacts",
        "",
        "Subtracting the intercept-only predictor and normalizing by remaining headroom gives the original lens rho=0.719 on 26 relations; the two year relations have zero headroom. This is an exploratory alternative outcome, not the original raw faithfulness metric. The matched-prediction rank/faithfulness control has rho=0.830, but shares predictions and targets and is not independent lens evidence.",
        "",
        "The search covers state 0 and all 32 block outputs, 41 finite beta values, continuous positive LRE beta, 4/8 references, differential/affine readouts, two neutral prompts, both protocols, and five rank summaries. Median log-rank is primary; other summaries are sensitivity analyses. Degenerate constant-direction readouts and beta-zero controls are not promoted as improvements.",
        "",
        "[Protocol and remaining factors](../../docs/mistral-grid-search.md). [Main analysis](analysis/main.json), [factor diagnostics](analysis/diagnostics_main.json), [configuration](config.json), [preassigned relation folds](relation_folds.json). Compressed per-relation scoring records are under `raw/`; the larger derivative-product tensors remain in `generated/mistral_full_grid/` for reuse.",
        "",
        "![Main comparison](main_comparison.png)",
        "",
        "![Layer and beta grid](layer_beta_grid.png)",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(lines))
    with (DEST / "relation_diagnostics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic["diagnostics"][0]))
        writer.writeheader()
        writer.writerows(diagnostic["diagnostics"])
    figures(main, diagnostic)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source_files = (
        [
            BASE / name
            for name in [
                "jlens/model.py",
                "jlens/model_sweep.py",
                "jlens/prefix_intervention.py",
                "jlens/grid_search.py",
                "jlens/grid_confirmation.py",
                "jlens/grid_statistics.py",
                "jlens/beta_intervals.py",
                "jlens/splits.py",
                "pyproject.toml",
                "uv.lock",
            ]
        ]
        + list((BASE / "scripts").glob("*mistral*.py"))
        + list((BASE / "data/relations").glob("*.json"))
    )
    provenance = dict(
        captured_at=datetime.now(timezone.utc).isoformat(),
        note="Sources captured at report generation; the original launch hashes are retained in config.json.",
        packages={
            name: version(name)
            for name in ["torch", "transformers", "numpy", "scipy", "accelerate"]
        },
        sha256={
            str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_files)
        },
    )
    (DEST / "report_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for name in [
        "config.json",
        "relation_folds.json",
        "confirmation_config.json",
        "regression_audit.json",
    ]:
        if (SOURCE / name).exists():
            shutil.copy2(SOURCE / name, DEST / name)
    for folder in ["analysis"]:
        (DEST / folder).mkdir(exist_ok=True)
        for path in (SOURCE / folder).glob("*.json"):
            shutil.copy2(path, DEST / folder / path.name)
    for name, root in EXTRA_RUNS.items():
        for source, target in [
            ("analysis/confirmation.json", f"analysis/{name}.json"),
            ("confirmation_config.json", f"{name}_config.json"),
            ("reuse_manifest.json", f"{name}_reuse.json"),
        ]:
            if (root / source).exists():
                shutil.copy2(root / source, DEST / target)
    (DEST / "verification").mkdir(exist_ok=True)
    for path in SOURCE.glob("*.verification.json"):
        shutil.copy2(path, DEST / "verification" / path.name)
    for root in EXTRA_RUNS.values():
        for path in root.glob("*.verification.json"):
            shutil.copy2(path, DEST / "verification" / path.name)
    if (SOURCE / "targeted_tests.log").exists():
        shutil.copy2(SOURCE / "targeted_tests.log", DEST / "verification/tests.txt")
    paths = [
        p
        for folder in [
            "main",
            "shared_split",
            "continuous/main",
            "continuous/shared_split",
            "confirmation",
        ]
        for p in (SOURCE / folder).glob("*.json")
    ]
    for root in EXTRA_RUNS.values():
        paths += list((root / "confirmation").glob("*.json"))
    manifest_path = DEST / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(archive, paths):
            if result:
                manifest[result["path"]] = result
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    readable()
    print("RETAINED", len(paths), "scoring records", flush=True)


if __name__ == "__main__":
    main()
