"""Paired statistics and validation-only site selection for mechanistic runs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from jlens.mechanistic import save_json

RELATIONS = ["country_capital_city", "country_currency"]
LAYERS = [5, 11, 17, 23]


def paired(values, nboot=10000):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "ci": [None, None], "p": None}
    rng = np.random.default_rng(2009)
    sampled = v[rng.integers(0, len(v), (nboot, len(v)))].mean(1)
    signs = rng.choice([-1, 1], size=(nboot, len(v)))
    p = (1 + np.count_nonzero(np.abs((signs*v).mean(1)) >= abs(v.mean()) - 1e-12)) / (nboot + 1)
    return {"n": len(v), "mean": float(v.mean()), "ci": np.quantile(sampled, [.025, .975]).tolist(), "p": float(p)}


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.
    for i, idx in enumerate(order):
        running = max(running, (len(p)-i)*p[idx])
        adjusted[idx] = min(1., running)
    return adjusted.tolist()


def read_rows(out, prefix):
    rows = []
    files = [p for p in sorted(out.glob(prefix + "*.json")) if not p.name.endswith(".partial.json")]
    for p in files:
        data = json.loads(p.read_text())
        if isinstance(data, dict) and data.get("complete") and "rows" in data:
            rows.extend(data["rows"])
    return rows


def select_heads(out):
    rows = read_rows(out, "routes_")
    by_site = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["partition"] == "validation" and row["kind"] == "attention":
            by_site[(row["block"], row["group"])][row["relation"]][row["subject"]] = row["clean"]["logp"] - row["patched"]["logp"]
    candidates = []
    for (block, group), values in by_site.items():
        a, b = [values[r] for r in RELATIONS]
        subjects = sorted(a.keys() & b.keys())
        # Require every validation subject: unequal-token subject groups cannot
        # win simply through a particularly favorable small subset.
        if len(subjects) != 16:
            continue
        difference = float(np.mean([a[s]-b[s] for s in subjects]))
        candidates.append({"block": block, "group": group, "validation_difference": difference,
                           "validation_n": len(subjects), "selection_score": abs(difference)})
    candidates.sort(key=lambda x: (-x["selection_score"], x["block"], x["group"]))
    selected = []
    for row in candidates:
        if row["block"] not in [x["block"] for x in selected]:
            selected.append(row)
        if len(selected) == 2:
            break
    assert len(selected) == 2
    save_json(out / "head_selection.json", {"selected": selected, "all_candidates": candidates,
                                           "rule": "absolute paired validation logp damage contrast; two distinct attention blocks; n=16"})
    print(json.dumps(selected, indent=2))


def baselines(out):
    from transformers import AutoTokenizer
    from jlens.paired_relation_audit import MODEL, REVISION
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    table, cases = [], []
    for mode in ["original", "controlled", "unit"]:
        packs = [json.loads((out/f"baseline_{mode}_{r}.json").read_text()) for r in RELATIONS]
        assert all(p["complete"] for p in packs)
        assert packs[0]["subjects"] == packs[1]["subjects"]
        nv = packs[0]["nval"]
        correct = np.logical_and(*[np.array(p["model_correct"])[nv:] for p in packs])
        subjects = packs[0]["subjects"][nv:]
        selections = LAYERS + ["selected"]
        for layer in selections:
            chosen = [max(LAYERS, key=lambda l: (np.mean(p["layers"][str(l)]["grid"][str(p["layers"][str(l)]["beta"])]["hit"][:nv]),
                                                   np.mean(p["layers"][str(l)]["grid"][str(p["layers"][str(l)]["beta"])]["logp"][:nv]), -l))
                      if layer == "selected" else layer for p in packs]
            scores = [p["layers"][str(l)] for p,l in zip(packs,chosen)]
            hits = [np.array(s["grid"][str(s["beta"])]["hit"])[nv:] for s in scores]
            wide_hits = [np.array(s["grid"][str(s["wide_beta"])]["hit"])[nv:] for s in scores]
            aonly = int(((hits[0] == 1) & (hits[1] == 0)).sum())
            bonly = int(((hits[0] == 0) & (hits[1] == 1)).sum())
            diff = hits[0] - hits[1]
            entry = {"mode": mode, "layer": layer, "chosen_layers": chosen, "n": len(subjects),
                "faithfulness": [float(h.mean()) for h in hits], "wide_faithfulness": [float(h.mean()) for h in wide_hits],
                "beta": [s["beta"] for s in scores], "wide_beta": [s["wide_beta"] for s in scores],
                "bias": [float(np.mean(s["grid"]["0.0"]["hit"][nv:])) for s in scores],
                "model_correct": [float(np.mean(p["model_correct"][nv:])) for p in packs],
                "paired_capital_minus_currency": paired(diff), "both_correct_n": int(correct.sum()),
                "both_correct_faithfulness": [float(h[correct].mean()) if correct.any() else None for h in hits],
                "both_correct_difference": paired(diff[correct]), "a_only": aonly, "b_only": bonly,
                "mcnemar_p": float(binomtest(aonly, aonly+bonly).pvalue) if aonly+bonly else 1.}
            if layer == 17:
                for i,s in enumerate(subjects):
                    cases.append({"mode": mode, "subject": s, "both_model_correct": bool(correct[i]),
                                  "faithfulness": [int(h[i]) for h in hits],
                                  "model_answers": [p["target_text"][nv+i] for p in packs],
                                  "lre_answers": [tokenizer.decode([score["grid"][str(score["beta"])]["top"][nv+i]]) for score in scores]})
                confidence = np.minimum(*[np.array(p["full"]["logp"])[nv:] for p in packs])
                median = np.median(confidence)
                entry["confidence_strata"] = {
                    "lower": paired(diff[confidence < median]), "higher": paired(diff[confidence >= median])}
                # A descriptive class-balanced contrast, never used to select a model.
                labels = np.array(packs[1]["gold"])[nv:]
                entry["currency_class_balanced_gap"] = float(np.mean([diff[labels == v].mean() for v in np.unique(labels)]))
            table.append(entry)
    return table, cases


def channel_summary(out, correct_subjects):
    rows = read_rows(out, "channels_country_")
    cells = defaultdict(dict)
    for row in rows:
        key = (row["layer"], row["strength"], row["relation"], row["source"], row["subject"])
        cells[key][row["control"]] = row
    summaries, interaction_values = [], {}
    for layer in LAYERS:
        for strength in [.5, 1.]:
            for metric in ["logp", "margin", "hit"]:
                matrix, lre_matrix = {}, {}
                raw_matrix, random_matrix = {}, {}
                for target in RELATIONS:
                    for source in RELATIONS:
                        effects, lre_effects, raw, random = {}, {}, {}, {}
                        for key, controls in cells.items():
                            l, st, t, src, subject = key
                            if (l,st,t,src) != (layer,strength,target,source):
                                continue
                            assert set(controls) == {-1,0,1,2}
                            clean = controls[-1]["clean"][metric]
                            damage = clean - controls[-1]["full"][metric]
                            rand = np.mean([clean-controls[j]["full"][metric] for j in range(3)])
                            effects[subject], raw[subject], random[subject] = damage-rand, damage, rand
                            lclean = controls[-1]["clean_lre"][metric]
                            lre_effects[subject] = (lclean-controls[-1]["lre"][metric]) - np.mean([lclean-controls[j]["lre"][metric] for j in range(3)])
                        matrix[(target,source)], lre_matrix[(target,source)] = effects, lre_effects
                        raw_matrix[(target,source)], random_matrix[(target,source)] = raw, random
                subjects = sorted(set.intersection(*(set(v) for v in matrix.values())))
                a,b = RELATIONS
                # Within each output relation, does its own channel matter more?
                own_a = np.array([matrix[(a,a)][s]-matrix[(a,b)][s] for s in subjects])
                own_b = np.array([matrix[(b,b)][s]-matrix[(b,a)][s] for s in subjects])
                interaction = own_a+own_b
                interaction_values[(layer,strength,metric)] = interaction
                both = np.array([s in correct_subjects for s in subjects])
                summaries.append({"layer": layer, "strength": strength, "metric": metric,
                    "matrix": [[paired(list(matrix[(t,s)].values())) for s in RELATIONS] for t in RELATIONS],
                    "raw_damage": [[paired(list(raw_matrix[(t,s)].values())) for s in RELATIONS] for t in RELATIONS],
                    "random_damage": [[paired(list(random_matrix[(t,s)].values())) for s in RELATIONS] for t in RELATIONS],
                    "lre_matrix": [[paired(list(lre_matrix[(t,s)].values())) for s in RELATIONS] for t in RELATIONS],
                    "capital_own_channel_advantage": paired(own_a), "currency_own_channel_advantage": paired(own_b),
                    "interaction": paired(interaction), "both_correct_interaction": paired(interaction[both]),
                    "both_correct_capital_own_channel_advantage": paired(own_a[both]),
                    "both_correct_currency_own_channel_advantage": paired(own_b[both])})
    for metric in ["logp", "margin", "hit"]:
        primary = [x for x in summaries if x["strength"] == 1 and x["metric"] == metric]
        for field in ["interaction", "both_correct_interaction"]:
            for x,p in zip(primary,holm([x[field]["p"] for x in primary])):
                x[field]["holm_p_four_layers"] = p
        for prefix in ["", "both_correct_"]:
            fields = [prefix+r+"_own_channel_advantage" for r in ["capital", "currency"]]
            corrected = holm([x[f]["p"] for x in primary for f in fields])
            for i,x in enumerate(primary):
                for j,f in enumerate(fields):
                    x[f]["holm_p_eight_directional_tests"] = corrected[2*i+j]
    return summaries


def nonlinear_summary(out, correct_subjects):
    rows = read_rows(out, "nonlinear_")
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        grouped[(r["layer"],r["t"])][r["relation"]][r["subject"]] = r
    result = []
    for (layer,t), byrel in sorted(grouped.items()):
        a,b = [byrel[r] for r in RELATIONS]
        subjects = sorted(a.keys() & b.keys())
        metrics = {}
        for metric in ["local_relative_residual", "global_relative_residual", "local_global_response_cosine", "state_distance", "output_change_norm"]:
            av,bv = [np.array([d[s][metric] for s in subjects]) for d in [a,b]]
            both = np.array([s in correct_subjects for s in subjects])
            metrics[metric] = {"capital": paired(av), "currency": paired(bv),
                               "currency_minus_capital": paired(bv-av),
                               "both_correct_difference": paired((bv-av)[both])}
        token_agreement = {
            kind: [paired([int(v[s][kind]["top"] == v[s]["actual"]["top"]) for s in subjects]) for v in [a,b]]
            for kind in ["local", "global", "lre"]
        }
        result.append({"layer": layer, "t": t, "metrics": metrics,
                       "token_agreement_with_intervened_model": token_agreement,
                       "original_answer_retention": [paired([v[s]["actual"]["hit"] for s in subjects]) for v in [a,b]]})
    for metric in ["local_relative_residual", "global_relative_residual"]:
        for field in ["currency_minus_capital", "both_correct_difference"]:
            for row,p in zip(result,holm([r["metrics"][metric][field]["p"] for r in result])):
                row["metrics"][metric][field]["holm_p_sixteen_conditions"] = p
    return result


def route_summary(out, heads=False, correct_subjects=None):
    rows = read_rows(out, "heads_" if heads else "routes_")
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["block"],r["kind"],r["group"],r["head"],r["partition"])
        grouped[key][r["relation"]][r["subject"]] = r
    summaries = []
    for key, values in grouped.items():
        a,b = [values[r] for r in RELATIONS]
        subjects = sorted(a.keys() & b.keys())
        if not subjects:
            continue
        effect = [np.array([v[s]["clean"]["logp"]-v[s]["patched"]["logp"] for s in subjects]) for v in [a,b]]
        margin_effect = [np.array([v[s]["clean"]["margin"]-v[s]["patched"]["margin"] for s in subjects]) for v in [a,b]]
        entry = dict(zip(["block","kind","group","head","partition"],key))
        entry.update({"capital": paired(effect[0]), "currency": paired(effect[1]), "capital_minus_currency": paired(effect[0]-effect[1]),
                      "margin_damage": [paired(v) for v in margin_effect], "margin_difference": paired(margin_effect[0]-margin_effect[1]),
                      "unchanged_lre_input_fraction": [float(np.mean([v[s]["lre_input_unchanged"] for s in subjects])) for v in [a,b]],
                      "full_model_lre_residual_change": [paired([v[s]["patched_fixed_lre_vector_error"]-v[s]["clean_vector_error"] for s in subjects if v[s]["lre_input_unchanged"]]) for v in [a,b]]})
        if correct_subjects is not None and key[-1] == "test":
            mask = np.array([s in correct_subjects for s in subjects])
            entry["both_correct_difference"] = paired((effect[0]-effect[1])[mask])
            entry["both_correct_margin_difference"] = paired((margin_effect[0]-margin_effect[1])[mask])
        summaries.append(entry)
    if heads:
        for row,p in zip(summaries,holm([x["capital_minus_currency"]["p"] for x in summaries])):
            row["capital_minus_currency"]["holm_p"] = p
        if correct_subjects is not None:
            for row,p in zip(summaries,holm([x["both_correct_difference"]["p"] for x in summaries])):
                row["both_correct_difference"]["holm_p"] = p
    return summaries


def artifacts(out, report):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / "analysis"
    folder.mkdir(exist_ok=True)
    def savefig(fig, name):
        fig.tight_layout()
        fig.savefig(folder/(name+".png"), dpi=180)
        fig.savefig(folder/(name+".pdf"))
        plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax, mode in zip(axes,["original","controlled","unit"]):
        rows = [r for r in report["baselines"] if r["mode"] == mode and r["layer"] != "selected"]
        for i,label in enumerate(["Capital", "Currency"]):
            ax.plot(LAYERS,[100*r["faithfulness"][i] for r in rows], marker="o",label=label)
        ax.set(title=mode, xlabel="Subject-state layer", xticks=LAYERS, ylim=(0,100))
    axes[0].set_ylabel("LRE faithfulness (%)")
    axes[-1].legend()
    savefig(fig,"paired_baselines")

    channels = [x for x in report["channels"] if x["strength"]==1 and x["metric"]=="logp"]
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    arrays=[np.array([[v["mean"] for v in row] for row in x["matrix"]]) for x in channels]
    lim=max(abs(x).max() for x in arrays)
    for ax, row, array in zip(axes,channels,arrays):
        im=ax.imshow(array,vmin=-lim,vmax=lim,cmap="RdBu_r")
        for i in range(2):
            for j in range(2): ax.text(j,i,f"{array[i,j]:.3f}",ha="center",va="center")
        ax.set(title=f"Layer {row['layer']}",xticks=[0,1],xticklabels=["Capital","Currency"],yticks=[0,1],yticklabels=["Capital","Currency"],xlabel="Removed subspace")
    axes[0].set_ylabel("Evaluated relation")
    fig.suptitle("Full-model log-probability damage beyond matched random removal (nats)",y=1.02)
    savefig(fig,"channel_selectivity")

    fig,axes=plt.subplots(1,4,figsize=(12,3.3),sharey=True)
    for ax,layer in zip(axes,LAYERS):
        rows=[r for r in report["nonlinear"] if r["layer"]==layer]
        for relation,color in [("capital","tab:blue"),("currency","tab:orange")]:
            for kind,style in [("local_relative_residual","-"),("global_relative_residual","--")]:
                ax.plot([r["t"] for r in rows],[r["metrics"][kind][relation]["mean"] for r in rows],style,color=color,label=f"{relation}, {'local' if style=='-' else 'global'}")
        ax.set(title=f"Layer {layer}",xlabel="Fraction toward donor state")
    axes[0].set_ylabel("Vector residual / actual output change")
    axes[-1].legend(fontsize=7)
    savefig(fig,"nonlinearity")

    groups=["subject_last","subject_other","suffix","answer"]
    fig,axes=plt.subplots(1,2,figsize=(9,8),sharey=True)
    arrays=[]
    for kind in ["attention","mlp"]:
        array=np.full((32,4),np.nan)
        for row in report["routes"]:
            if row["partition"]=="test" and row["kind"]==kind:
                array[row["block"]-1,groups.index(row["group"])]=row["capital_minus_currency"]["mean"]
        arrays.append(array)
    lim=max(np.nanmax(abs(x)) for x in arrays)
    for ax,kind,array in zip(axes,["Attention","MLP"],arrays):
        im=ax.imshow(array,aspect="auto",cmap="RdBu_r",vmin=-lim,vmax=lim)
        ax.set(title=kind,xticks=range(4),xticklabels=["Subject last","Subject earlier","Suffix","Answer"],yticks=range(0,32,2),yticklabels=range(1,33,2))
        ax.tick_params(axis="x",rotation=30)
    axes[0].set_ylabel("Transformer block (1-based)")
    fig.suptitle("Capital minus currency: full-model log-probability damage\nExploratory test-set map; earlier-subject sample sizes vary")
    savefig(fig,"computation_routes")

    sites=report["head_selection"]["selected"]
    fig,axes=plt.subplots(len(sites),1,figsize=(11,6),squeeze=False)
    for ax,site in zip(axes[:,0],sites):
        rows=sorted([r for r in report["heads"] if r["block"]==site["block"] and r["group"]==site["group"]],key=lambda r:r["head"])
        values=np.array([r["capital_minus_currency"]["mean"] for r in rows])
        ci=np.array([r["capital_minus_currency"]["ci"] for r in rows])
        colors=["tab:red" if r["capital_minus_currency"]["holm_p"]<.05 else "0.6" for r in rows]
        ax.bar([r["head"] for r in rows],values,color=colors)
        ax.errorbar([r["head"] for r in rows],values,yerr=np.maximum(0,np.vstack([values-ci[:,0],ci[:,1]-values])),fmt="none",color="black",lw=.8)
        ax.axhline(0,color="black",lw=.5)
        ax.set(title=f"Block {site['block']}, {site['group']}; red = Holm p < .05",xlabel="Head (0-based)",ylabel="Capital − currency\ndamage (nats)")
    savefig(fig,"attention_heads")

    with (folder/"matched_cases.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=["mode","subject","both_model_correct","capital_faithful","currency_faithful","capital_model","capital_lre","currency_model","currency_lre"])
        writer.writeheader()
        for r in report["cases"]:
            writer.writerow(dict(mode=r["mode"],subject=r["subject"],both_model_correct=r["both_model_correct"],capital_faithful=r["faithfulness"][0],currency_faithful=r["faithfulness"][1],capital_model=r["model_answers"][0],capital_lre=r["lre_answers"][0],currency_model=r["model_answers"][1],currency_lre=r["lre_answers"][1]))

    def interval(r,scale=1):
        if not r["n"]: return "NA"
        return f"{scale*r['mean']:.3f} [{scale*r['ci'][0]:.3f}, {scale*r['ci'][1]:.3f}]"
    lines=["# Paired mechanistic measurements", "", "All intervals bootstrap matched subjects. Primary fixed layer: 17. One model, one split, 53 confirmation subjects. Faithfulness is first-token agreement with the full model.", "", "## Baseline results", "", "| Prompt | Capital | Currency | Capital minus currency, points [95% CI] | Exact McNemar p | Both-model-correct n |", "|---|---:|---:|---:|---:|---:|"]
    for r in report["baselines"]:
        if r["layer"]==17:
            lines.append(f"| {r['mode']} | {100*r['faithfulness'][0]:.1f}% | {100*r['faithfulness'][1]:.1f}% | {interval(r['paired_capital_minus_currency'],100)} | {r['mcnemar_p']:.4f} | {r['both_correct_n']} |")
    lines += ["", "## Channel interventions", "", "Strength one; random-adjusted full-model log-probability damage. Directional advantages compare own versus other subspace within each evaluated relation. A positive interaction alone is not a double dissociation.", "", "| Layer | Capital own-channel advantage [CI] | Currency own-channel advantage [CI] | Interaction [CI] | Holm p (4 layers) |", "|---|---:|---:|---:|---:|"]
    for r in channels:
        lines.append(f"| {r['layer']} | {interval(r['capital_own_channel_advantage'])} | {interval(r['currency_own_channel_advantage'])} | {interval(r['interaction'])} | {r['interaction']['holm_p_four_layers']:.4f} |")
    lines += ["", "## Nonlinearity", "", "At full donor-state displacement (t=1). Values normalize Taylor residual by actual output-change norm; they are not accuracy percentages.", "", "| Layer | Local residual, capital | Local residual, currency | Currency minus capital [CI] | Global residual, capital / currency |", "|---|---:|---:|---:|---:|"]
    for r in report["nonlinear"]:
        if r["t"]==1:
            m=r["metrics"];l=m["local_relative_residual"];g=m["global_relative_residual"]
            lines.append(f"| {r['layer']} | {l['capital']['mean']:.3f} | {l['currency']['mean']:.3f} | {interval(l['currency_minus_capital'])} | {g['capital']['mean']:.3f} / {g['currency']['mean']:.3f} |")
    lines += ["", "## Validation-selected attention sites", "", "| Block | Group | Held-out capital damage | Held-out currency damage | Capital minus currency [CI] |", "|---|---|---:|---:|---:|"]
    for site in sites:
        r=next(r for r in report["routes"] if r['partition']=='test' and r['kind']=='attention' and r['block']==site['block'] and r['group']==site['group'])
        lines.append(f"| {r['block']} | {r['group']} | {r['capital']['mean']:.3f} | {r['currency']['mean']:.3f} | {interval(r['capital_minus_currency'])} |")
    significant=[r for r in report["heads"] if r['capital_minus_currency']['holm_p']<.05]
    lines += ["", f"Head contrasts surviving Holm correction: **{len(significant)} / {len(report['heads'])}**. These are relative sensitivity results under donor corruption, not proof of isolated factual circuits.", "", "## Artifacts", "", "* [Baseline figure](paired_baselines.pdf)", "* [Channel figure](channel_selectivity.pdf)", "* [Nonlinearity figure](nonlinearity.pdf)", "* [Computation routes](computation_routes.pdf)", "* [Attention heads](attention_heads.pdf)", "* [Matched cases](matched_cases.csv)", "* [Complete numerical summary](summary.json)"]
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--select-heads", action="store_true")
    args = parser.parse_args()
    if args.select_heads:
        select_heads(args.out)
        return
    expected = [f"baseline_{m}_{r}.json" for m in ["original","controlled","unit"] for r in RELATIONS]
    expected += ["channels_basis.json", "head_selection.json"]
    expected += [f"{stage}_{r}_{l}.json" for stage in ["channels","nonlinear"] for r in RELATIONS for l in LAYERS]
    expected += [f"{stage}_{r}_{s}.json" for stage in ["routes","heads"] for r in RELATIONS for s in range(4)]
    missing = [name for name in expected if not (args.out/name).exists()]
    assert not missing, missing
    baseline, cases = baselines(args.out)
    correct = {r["subject"] for r in cases if r["mode"] == "controlled" and r["both_model_correct"]}
    report = {"complete": True, "baselines": baseline, "cases": cases,
              "basis": json.loads((args.out/"channels_basis.json").read_text()),
              "channels": channel_summary(args.out, correct), "nonlinear": nonlinear_summary(args.out, correct),
              "routes": route_summary(args.out, correct_subjects=correct), "heads": route_summary(args.out, heads=True, correct_subjects=correct),
              "head_selection": json.loads((args.out/"head_selection.json").read_text())}
    selected_sites=[]
    for site in report["head_selection"]["selected"]:
        selected_sites.append(next(r for r in report["routes"] if r["partition"]=="test" and r["kind"]=="attention" and r["block"]==site["block"] and r["group"]==site["group"]))
    for row,p in zip(selected_sites,holm([r["capital_minus_currency"]["p"] for r in selected_sites])):
        row["capital_minus_currency"]["holm_p_two_selected_sites"]=p
    report["selected_attention_sites"]=selected_sites
    save_json(args.out / "analysis/summary.json", report)
    artifacts(args.out, report)
    print(json.dumps([x for x in baseline if x["layer"] == 17],indent=2))


if __name__ == "__main__":
    main()
