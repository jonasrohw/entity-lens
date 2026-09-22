"""Regenerate paper/table_prominence.tex from results/prominence_statistics.json.

A separate table from `table_relations.tex` because prominence exists for
only 14 of the 28 relations --- as extra columns there, half the cells
would be empty and the pairing the correlation rests on would be hard to
see.  Run `jlens/prominence_statistics.py` first.
"""

from scipy.stats import spearmanr

from jlens.config import ROOT
from jlens.io import read_json


def main():
    rows = read_json(f"{ROOT}/results/prominence_statistics.json")
    rows.sort(key=lambda row: -row["prom"])
    rho, p_value = spearmanr(
        [row["prom"] for row in rows], [row["gold_lens"] for row in rows]
    )

    out = [
        r"\begin{table}[t]",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"relation & prom. & $n$ & gold$\in$lens & $n$ \\",
        r"\midrule",
    ]
    for row in rows:
        name = row["rel"].replace("_", r"\_")
        star = r"$^{*}$" if row["rel"] == "star_constellation" else ""
        out.append(
            f"{name}{star} & {row['prom']:.2f} & {row['n_prom']} & "
            f"{row['gold_lens']:.2f} & {row['n_gl']} \\\\"
        )
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Object prominence, measured by the model's own free"
        "\ngeneration and therefore independently of the lens, against how often"
        "\nthe lens surfaces the gold object in its top five. Spearman"
        f"\n$\\rho = {rho:.3f}$ ($p = {p_value:.3f}$, $n = {len(rows)}$)."
        "\n$^{*}$: star\\_constellation, the outlier discussed in"
        " \\S\\ref{sec:salience}.}",
        r"\label{tab:prominence}",
        r"\end{table}",
    ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
