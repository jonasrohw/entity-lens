"""Regenerate paper/table_prominence.tex from e2/prominence_stats.json.

A separate table from `table_relations.tex` because prominence exists for
only 14 of the 28 relations --- as extra columns there, half the cells
would be empty and the pairing the correlation rests on would be hard to
see.  Run `analysis/prominence_stats.py` first.
"""
import json, sys
from scipy.stats import spearmanr
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mounts/Users/cisintern/hs/auth/linz"
R = json.load(open(f"{ROOT}/e2/prominence_stats.json"))
R.sort(key=lambda r: -r["prom"])
rho, pv = spearmanr([r["prom"] for r in R], [r["gold_lens"] for r in R])

out = [r"\begin{table}[t]", r"\centering\footnotesize",
       r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
       r"relation & prom. & $n$ & gold$\in$lens & $n$ \\", r"\midrule"]
for r in R:
    nm = r["rel"].replace("_", r"\_")
    star = r"$^{*}$" if r["rel"] == "star_constellation" else ""
    out.append(f"{nm}{star} & {r['prom']:.2f} & {r['n_prom']} & "
               f"{r['gold_lens']:.2f} & {r['n_gl']} \\\\")
out += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Object prominence, measured by the model's own free"
        "\ngeneration and therefore independently of the lens, against how often"
        "\nthe lens surfaces the gold object in its top five. Spearman"
        f"\n$\\rho = {rho:.3f}$ ($p = {pv:.3f}$, $n = {len(R)}$)."
        "\n$^{*}$the one relation that disobeys (\\S\\ref{sec:residuals}).}",
        r"\label{tab:prominence}", r"\end{table}"]
print("\n".join(out))
