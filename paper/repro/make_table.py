"""Regenerate paper/table_relations.tex from the result JSONs.
Run from the repo root with the project's e2/ outputs available, or adapt
the paths.  Kept here so the appendix table is reproducible, not hand-typed."""
import json, glob, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mounts/Users/cisintern/hs/auth/linz"
FA = {}
for f in glob.glob(f"{ROOT}/e2/uniform/lrefaithT_*.json"):
    d = json.load(open(f)); v = d.get("best_faith"); r = d.get("relation")
    if isinstance(v, (int, float)):
        FA.setdefault(r, {})["ts" if f.endswith("_ts.json") else "orig"] = float(v)
LN = json.load(open(f"{ROOT}/e2/lens_vs_faith_full.json"))
TS = json.load(open(f"{ROOT}/e2/lens_vs_faith_years_ts.json"))
YR = {"presidents_birth_year", "presidents_election_year"}
meta = {}
for f in glob.glob(f"{ROOT}/data/lre_full/*/*.json"):
    r = f.split("/")[-1][:-5]; S = json.load(open(f))["samples"]
    meta[r] = (len(S), len(S) / len(set(x["object"] for x in S)), max(0, len(S) - 16))
rels = [r for r in LN if not r.startswith("_") and r in FA]
fa = lambda r: FA[r].get("ts", FA[r]["orig"])
le = lambda r: (TS if r in YR else LN)[r]["agg"]["diff"]["median_lograng"]
order = sorted(rels, key=lambda r: -fa(r)); half = (len(order) + 1) // 2
L, R = order[:half], order[half:]
def row(r):
    n, k, nt = meta[r]
    nm = r.replace("_", r"\_")
    mark = r"$^{\dagger}$" if r in YR else (r"$^{*}$" if nt < 10 else "")
    return f"{nm}{mark} & {k:.1f} & {fa(r):.2f} & ${-le(r):.2f}$"
out = [r"\begin{table*}[t]", r"\centering\footnotesize",
       r"\begin{tabular}{@{}lrrr@{\hspace{3em}}lrrr@{}}", r"\toprule",
       r"relation & $k$ & faith. & $A_r$ & relation & $k$ & faith. & $A_r$ \\", r"\midrule"]
for i in range(half):
    b = row(R[i]) if i < len(R) else " & & & "
    out.append(f"{row(L[i])} & {b} \\\\")
out += [r"\bottomrule", r"\end{tabular}",
        r"""\caption{All 28 relations, sorted by LRE faithfulness. $k$ is the mean number
of subjects per distinct object; $A_r$ is the relation-agnostic lens score, higher
meaning closer agreement with the model. $^{\dagger}$rescored for the
numeric-object artifact (\S\ref{sec:artifact}); $^{*}$fewer than ten test items,
so faithfulness is coarsely quantized.}""",
        r"\label{tab:relations}", r"\end{table*}"]
print("\n".join(out))
