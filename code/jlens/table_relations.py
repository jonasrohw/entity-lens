"""Regenerate paper/table_relations.tex from the result JSONs.
Run from the repo root with the project's results/ outputs available, or adapt
the paths.  Kept here so the appendix table is reproducible, not hand-typed."""

import glob

from jlens.config import ROOT, YEAR_RELATIONS
from jlens.io import read_json


def faithfulness(r, faithfulness_results):
    return faithfulness_results[r].get("ts", faithfulness_results[r]["orig"])


def lens_score(r, lens_results, year_results):
    return (year_results if r in YEAR_RELATIONS else lens_results)[r]["agg"]["diff"][
        "median_lograng"
    ]


def row(r, meta, faithfulness_results, lens_results, year_results):
    n, k, nt = meta[r]
    nm = r.replace("_", r"\_")
    mark = r"$^{\dagger}$" if r in YEAR_RELATIONS else (r"$^{*}$" if nt < 10 else "")
    return f"{nm}{mark} & {k:.1f} & {faithfulness(r, faithfulness_results):.2f} & ${-lens_score(r, lens_results, year_results):.2f}$"


def main():
    faithfulness_results = {}
    for f in glob.glob(f"{ROOT}/results/main/faithfulness_*.json"):
        d = read_json(f)
        v = d.get("best_faith")
        r = d.get("relation")
        if isinstance(v, (int, float)):
            faithfulness_results.setdefault(r, {})[
                "ts" if f.endswith("_ts.json") else "orig"
            ] = float(v)
    lens_results = read_json(f"{ROOT}/results/jacobian_lens.json")
    year_results = read_json(f"{ROOT}/results/jacobian_lens_years.json")
    meta = {}
    for f in glob.glob(f"{ROOT}/data/relations/*.json"):
        r = f.split("/")[-1][:-5]
        S = read_json(f)["samples"]
        meta[r] = (
            len(S),
            len(S) / len(set(x["object"] for x in S)),
            max(0, len(S) - 16),
        )
    rels = [
        r for r in lens_results if not r.startswith("_") and r in faithfulness_results
    ]
    order = sorted(rels, key=lambda r: -faithfulness(r, faithfulness_results))
    half = (len(order) + 1) // 2
    left_relations, right_relations = order[:half], order[half:]
    out = [
        r"\begin{table*}[t]",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}lrrr@{\hspace{3em}}lrrr@{}}",
        r"\toprule",
        r"relation & $k$ & faith. & $A_r$ & relation & $k$ & faith. & $A_r$ \\",
        r"\midrule",
    ]
    for i in range(half):
        b = (
            row(
                right_relations[i],
                meta,
                faithfulness_results,
                lens_results,
                year_results,
            )
            if i < len(right_relations)
            else " & & & "
        )
        out.append(
            f"{row(left_relations[i], meta, faithfulness_results, lens_results, year_results)} & {b} \\\\"
        )
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"""\caption{All 28 relations, sorted by LRE faithfulness. $k$ is the mean number
of subjects per distinct object; $A_r$ is the relation-agnostic lens score, higher
meaning closer agreement with the model. $^{\dagger}$rescored for the
numeric-object artifact (\S\ref{sec:artifact}); $^{*}$fewer than ten test items,
so faithfulness is coarsely quantized.}""",
        r"\label{tab:relations}",
        r"\end{table*}",
    ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
