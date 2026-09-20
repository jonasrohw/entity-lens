"""Refit Figure-1 OLS on 26 non-year relations; write report, CSVs and plots.

Run from code/: uv run python ../paper/repro/qwen3_26_relations.py
Only retained measurements are read; no model inference is performed.
"""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "code/docs"
EXCLUDED = {"presidents_birth_year", "presidents_election_year"}
STEM = "qwen3-8b-base-figure1"


def fit(rows):
    x = np.array([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    b, a = np.polyfit(x, y, 1)
    residual = y - (a + b * x)
    # Independent OLS solve and normal-equation checks.
    np.testing.assert_allclose(
        [a, b],
        np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0],
        rtol=0,
        atol=1e-12,
    )
    assert abs(residual.sum()) < 1e-12
    assert abs(x @ residual) < 1e-12
    rho = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)
    return dict(
        n=len(rows),
        a=a,
        b=b,
        rho=rho.statistic,
        p=rho.pvalue,
        pearson=pearson.statistic,
        pearson_p=pearson.pvalue,
    )


def with_residuals(rows, fitted):
    return [
        dict(
            r,
            y_hat=float(fitted["a"] + fitted["b"] * r["x"]),
            residual=float(r["y"] - fitted["a"] - fitted["b"] * r["x"]),
        )
        for r in rows
    ]


def write_csv(name, rows):
    with (DOCS / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (DOCS / f"{STEM}-28-relations.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    q28 = [dict(r, **{k: float(r[k]) for k in ["x", "y"]}) for r in raw]
    assert len(q28) == 28 and len({r["relation"] for r in q28}) == 28
    assert EXCLUDED <= {r["relation"] for r in q28}
    q26 = [r for r in q28 if r["relation"] not in EXCLUDED]
    assert len(q26) == 26 and "name_gender" in {r["relation"] for r in q26}
    olmo_path = ROOT / "code/results/olmo/results/paper_figure_data.json"
    olmo = json.loads(olmo_path.read_text())
    o28 = [
        dict(relation=r, x=-x, y=y)
        for r, x, y in zip(olmo["rels"], olmo["lens"], olmo["faith"], strict=True)
    ]
    assert {r["relation"] for r in o28} == {r["relation"] for r in q28}
    o26 = [r for r in o28 if r["relation"] not in EXCLUDED]
    fits = {
        "Qwen28": fit(q28),
        "Qwen26": fit(q26),
        "OLMo28": fit(o28),
        "OLMo26": fit(o26),
    }
    original = {r["relation"]: r for r in with_residuals(q28, fits["Qwen28"])}
    for r in raw:
        assert abs(float(r["residual"]) - original[r["relation"]]["residual"]) < 1e-12
    q26 = with_residuals(q26, fits["Qwen26"])
    o26 = with_residuals(o26, fits["OLMo26"])
    ob = {r["relation"]: r for r in o26}
    ranking = sorted(q26, key=lambda r: (-abs(r["residual"]), r["relation"]))
    positions = {r["relation"]: i for i, r in enumerate(ranking, 1)}
    # Rank the same 26 retained relations under the old fit to isolate refitting.
    old_order = sorted(q26, key=lambda r: -abs(original[r["relation"]]["residual"]))
    old_positions = {r["relation"]: i for i, r in enumerate(old_order, 1)}
    for r in q26:
        r.update(
            residual_pp=100 * r["residual"],
            abs_residual_rank=positions[r["relation"]],
            y_hat_28=original[r["relation"]]["y_hat"],
            residual_28=original[r["relation"]]["residual"],
            old_fit_rank_among_26=old_positions[r["relation"]],
        )
    write_csv(f"{STEM}.csv", q26)
    comparison = [
        dict(
            relation=q["relation"],
            qwen_x=q["x"],
            qwen_y=q["y"],
            qwen_y_hat=q["y_hat"],
            qwen_residual=q["residual"],
            olmo_x=ob[q["relation"]]["x"],
            olmo_y=ob[q["relation"]]["y"],
            olmo_y_hat=ob[q["relation"]]["y_hat"],
            olmo_residual=ob[q["relation"]]["residual"],
            residual_difference=q["residual"] - ob[q["relation"]]["residual"],
        )
        for q in q26
    ]
    write_csv(f"{STEM}-olmo-comparison.csv", comparison)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.7), sharex=True, sharey=True, layout="constrained"
    )
    xs = np.linspace(-5.4, -0.1, 100)
    for ax, rows, key, label in zip(
        axes,
        [q26, o26],
        ["Qwen26", "OLMo26"],
        ["Qwen3 8B Base", "OLMo 3 7B"],
        strict=True,
    ):
        f = fits[key]
        ax.scatter(
            [r["x"] for r in rows],
            [r["y"] for r in rows],
            s=25,
            color="#24649c",
            label="26 retained relations",
            zorder=3,
        )
        ax.plot(xs, f["a"] + f["b"] * xs, color="#24649c", label="OLS, 26 relations")
        labels = {"name_gender", "company_ceo", "name_religion", "superhero_person"}
        for r in rows:
            if r["relation"] in labels:
                offset = (
                    (5, -13)
                    if r["relation"] in {"name_gender", "name_religion"}
                    else (5, 7)
                )
                if key == "Qwen26" and r["relation"] == "name_religion":
                    offset = (5, -32)
                if key == "OLMo26" and r["relation"] == "company_ceo":
                    offset = (5, -14)
                ax.annotate(
                    r["relation"],
                    (r["x"], r["y"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.set_title(f"{label}\nSpearman rho = {f['rho']:.3f}, p = {f['p']:.3g}")
        ax.set_xlabel("Lens score: -median log10(rank + 1)")
        ax.set_xlim(-5.5, 0.1)
        ax.set_ylim(-0.06, 1.10)
        ax.grid(alpha=0.2)
    removed = [r for r in q28 if r["relation"] in EXCLUDED]
    axes[0].scatter(
        [r["x"] for r in removed],
        [r["y"] for r in removed],
        marker="x",
        s=65,
        color="#b25d21",
        label="Excluded year relations",
    )
    old = fits["Qwen28"]
    axes[0].annotate(
        "2 excluded year relations",
        xy=(np.mean([r["x"] for r in removed]), 1.0),
        xytext=(-4.75, 1.045),
        fontsize=7,
        color="#b25d21",
        arrowprops={"arrowstyle": "-", "color": "#b25d21", "lw": 0.7},
    )
    axes[0].plot(
        xs,
        old["a"] + old["b"] * xs,
        "--",
        color="#888888",
        label="Old OLS, 28 relations",
    )
    axes[0].set_ylabel("LRE faithfulness (first-token agreement)")
    for ax in axes:
        ax.legend(loc="lower right", fontsize=7)
    fig.savefig(DOCS / f"{STEM}-26-relations.png", dpi=180)
    fig.savefig(
        DOCS / f"{STEM}-26-relations.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    f = fits["Qwen26"]
    lines = [
        "# Qwen3 8B Base: Figure 1 mit 26 Relationen",
        "",
        "**Nachtrag zur Interpretation:** Die [Implementierungsprüfung](qwen3-jlens-implementation-review.md)",
        "zeigt einen Positionseffekt in Qwens differentieller Lens: Die Mischung",
        "sehr großer Zustände an Tokenposition 0 mit kleineren späteren Zuständen",
        "erzeugt die wiederkehrenden `-Semit`-Ausgaben bei Länderrelationen.",
        "Die folgenden historischen Zahlen bleiben unverändert; ihr Ausschluss",
        "der beiden Jahresrelationen behebt diesen separaten Effekt nicht.",
        "",
        "Stand: 20. September 2026. Modell: `Qwen/Qwen3-8B-Base`, Revision",
        "`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; `main`-Protokoll.",
        "",
        "**Beide Jahresrelationen sind ausgeschlossen:** `presidents_birth_year` und",
        "`presidents_election_year`. **`name_gender` bleibt enthalten.** Die OLS-Gerade",
        "wurde auf den übrigen 26 Punkten neu gefittet; alle Vorhersagen, Residuen und",
        "Outlier-Rangfolgen unten verwenden diesen neuen Fit. Die gemessenen x-/y-Werte",
        "bleiben unverändert. Die LREs selbst wurden nicht neu trainiert oder ausgewählt.",
        "",
        f"Qwen: **Spearman ρ = {f['rho']:.6f}, p = {f['p']:.6f}**;",
        f"**Pearson r = {f['pearson']:.6f}, p = {f['pearson_p']:.6f}** (zweiseitig).",
        "Der Zusammenhang ist stärker als mit 28 Relationen, erreicht hier aber nicht",
        "das 5-%-Signifikanzniveau. Die deskriptiven Outlier sind keine formal getesteten Anomalien.",
        "",
        "## Warum der Ausschluss die Outlier verändert",
        "",
        "Ein Residuum ist `y − (a + b × x)`. Werden Punkte entfernt, ändern sich",
        "Intercept und Steigung, also auch die Residuen der verbleibenden Punkte.",
        "Nur die zwei Tabellenzeilen zu löschen hätte diesen Effekt nicht erfasst.",
        "",
        "| Modell / Auswahl | n | Intercept a | Steigung b | Spearman ρ | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("Qwen28", "Qwen, bisher"),
        ("Qwen26", "Qwen, ohne Jahre"),
        ("OLMo28", "OLMo, bisher"),
        ("OLMo26", "OLMo, ohne Jahre"),
    ]:
        m = fits[key]
        lines.append(
            f"| {label} | {m['n']} | {m['a']:.6f} | {m['b']:.6f} | {m['rho']:.6f} | {m['p']:.6g} |"
        )
    lines += [
        "",
        "Neue Qwen-Gerade:",
        "",
        "```text",
        f"ŷ = {f['a']:.15f} + {f['b']:.15f} × x",
        "Δ = y − ŷ",
        "```",
        "",
        f"![Qwen und OLMo mit 26 Relationen]({STEM}-26-relations.png)",
        "",
        f"[PDF zum Export]({STEM}-26-relations.pdf). Die orangefarbenen Kreuze zeigen",
        "die zwei ausgeschlossenen Qwen-Punkte nur zur Orientierung. Sie gehen nicht",
        "in den neuen Fit ein; die gestrichelte Gerade ist der alte 28-Punkte-Fit.",
        "",
        "Die Jahresrelationen haben bei Qwen y = 1 trotz sehr schwachem Lens-Score.",
        "Wie der [Tokenisierungs-Audit](tokenization-years-small-relations.md) zeigt,",
        "stimmen Modell und LRE dort jeweils auf dem ersten Token `1` überein; auch",
        "ein konstanter Intercept erreicht diese Faithfulness. OLMo sagt dagegen",
        "unterschiedliche dreistellige Präfixe voraus. Die Aufgaben unterscheiden sich",
        "damit in der Aussagekraft des ersten Tokens.",
        "",
        "## Alle 26 Qwen-Relationen, nach absolutem Residuum sortiert",
        "",
        "**x:** `−median(log10(rank_diff + 1))` der differentiellen Lens (State-Index 19).",
        "**y:** gemessene LRE-Faithfulness, also Top-1-Token-Übereinstimmung mit dem Modell;",
        "höhere Werte sind besser, `0.02` bedeutet 2 % und keine Fehlerquote.",
        "**ŷ:** Vorhersage der OLS-Geraden. **Δ:** `y − ŷ`, hier in Prozentpunkten (PP).",
        "Positiv: höhere Faithfulness als vorhergesagt; negativ: niedrigere.",
        "Die Regression gewichtet jede Relation gleich, unabhängig von den Testzeilenzahlen.",
        "",
        "„Alter Rang“ sortiert dieselben 26 Relationen nach ihren absoluten Residuen",
        "unter der ursprünglichen 28-Punkte-Geraden. Dadurch zeigt die Rangänderung",
        "den Effekt des erneuten Fits zusätzlich zum Wegfall der zwei Relationen.",
        "",
        "| Rang | Relation | n Lens | n LRE | x | y | ŷ neu | Δ neu (PP) | Δ alt (PP) | Alter Rang |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ranking:
        lines.append(
            f"| {r['abs_residual_rank']} | `{r['relation']}` | {r['n_lens']} | {r['n_lre']} | {r['x']:.6f} | {r['y']:.6f} | {r['y_hat']:.6f} | {100 * r['residual']:+.6f} | {100 * r['residual_28']:+.6f} | {r['old_fit_rank_among_26']} |"
        )
    positive = sorted(q26, key=lambda r: -r["residual"])[:3]
    negative = sorted(q26, key=lambda r: r["residual"])[:3]
    lines += [
        "",
        "Die größten positiven Abweichungen: "
        + ", ".join(
            f"`{r['relation']}` ({100 * r['residual']:+.2f} PP)" for r in positive
        )
        + ".",
        "",
        "Die größten negativen Abweichungen: "
        + ", ".join(
            f"`{r['relation']}` ({100 * r['residual']:+.2f} PP)" for r in negative
        )
        + ".",
        "",
        "`person_father` rutscht innerhalb der negativen Ausreißer hinter `person_mother`",
        "und `superhero_person`. `company_ceo` bleibt der größte negative Ausreißer.",
        "",
        "## Welche Relationen unterscheiden Qwen und OLMo?",
        "",
        "Verglichen werden dieselben 26 Relationsnamen, aber jeweils die Messwerte und",
        "die separat neu gefittete Gerade des jeweiligen Modells. Die Tabelle zeigt die",
        "acht größten absoluten Unterschiede `Δ_Qwen − Δ_OLMo`. Diese Differenz beschreibt",
        "eine unterschiedliche Lage relativ zum jeweiligen Trend; sie ist kein kausaler",
        "Nachweis und kein Test auf einen signifikanten Unterschied zwischen Modellen.",
        "",
        "| Relation | x Qwen | x OLMo | y Qwen | y OLMo | Δ Qwen (PP) | Δ OLMo (PP) | Differenz (PP) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(comparison, key=lambda r: -abs(r["residual_difference"]))[:8]:
        lines.append(
            f"| `{r['relation']}` | {r['qwen_x']:.6f} | {r['olmo_x']:.6f} | {r['qwen_y']:.6f} | {r['olmo_y']:.6f} | {100 * r['qwen_residual']:+.6f} | {100 * r['olmo_residual']:+.6f} | {100 * r['residual_difference']:+.6f} |"
        )
    lines += [
        "",
        "Besonders aufschlussreich sind folgende Kandidaten für eine weitere Untersuchung:",
        "",
        "- `name_gender`: sehr schlechter Lens-Score bei beiden Modellen, aber Qwen-Faithfulness",
        "  1.0 gegenüber 1/3 bei OLMo. Die Relation hat nur drei LRE-Testzeilen und bleibt",
        "  eine deutliche Quelle von Stichprobenunsicherheit in dieser 26-Relationen-Auswertung.",
        "- `name_religion`: hohe Faithfulness bei beiden Modellen, aber ein deutlich",
        "  schlechterer Lens-Score bei Qwen. Qwen liegt daher weit über seinem Trend.",
        "- `company_ceo`, `company_hq` und `person_university`: Qwen liegt relativ zum",
        "  eigenen Fit deutlich niedriger als OLMo. Sowohl x als auch y und die",
        "  unterschiedlichen Messprotokolle müssen zur Erklärung betrachtet werden.",
        "",
        "Der Ausschluss der Jahre verkleinert den beobachteten Spearman-Abstand zwischen",
        f"OLMo und Qwen von {fits['OLMo28']['rho'] - fits['Qwen28']['rho']:.6f} auf",
        f"{fits['OLMo26']['rho'] - fits['Qwen26']['rho']:.6f}; er beseitigt ihn nicht.",
        "Eine zusätzliche Entfernung von `name_gender` ergäbe bei Qwen ρ = 0.448336",
        "(25 Relationen). Das ist eine separate Sensitivitätsanalyse und wurde hier",
        "nicht zur Auswahl der 26 Relationen verwendet.",
        "",
        "## Daten, Grenzen und Reproduktion",
        "",
        f"- [26 Qwen-Datenpunkte, ungerundet]({STEM}.csv), einschließlich neuer und alter Residuen.",
        f"- [Qwen-/OLMo-Vergleich für alle 26 Relationen]({STEM}-olmo-comparison.csv).",
        f"- [Archivierter 28-Relationen-Bericht]({STEM}-28-relations.md) und",
        f"  [unveränderter 28-Punkte-CSV-Snapshot]({STEM}-28-relations.csv).",
        "- [Konkrete CEO-Vorhersagen](qwen3-8b-base-company-ceo-predictions.md).",
        "",
        "Der Qwen-Eingabesnapshot enthält die ungerundeten Originalmessungen und die",
        "SHA-256-Prüfsummen der lokalen Rohdateien. Der OLMo-Vergleich verwendet",
        "[`results/olmo/results/paper_figure_data.json`](../results/olmo/results/paper_figure_data.json)",
        "aus dem aktuellen Ergebnissnapshot: OLMo-3-1025-7B, Lens-State-Index 17.",
        "Dieser Snapshot ergibt ρ = 0.770916 über 28 Relationen; er ist nicht der",
        "historische `olmo-old`-Snapshot mit dem im ursprünglichen Paper berichteten ρ ≈ 0.658.",
        "",
        "Unterschiede in Tokenisierung, Vokabular und Modellmessungen bleiben bestehen.",
        "Die `main`-Lens- und LRE-Splits werden unabhängig erzeugt. Die Testzeilenzahlen",
        "sind nicht immer Anzahlen eindeutiger Subjekte. LRE-Layer und β wurden auf den",
        "ausgewerteten Beispielen ausgewählt; die Faithfulness ist daher eine In-Sample-Schätzung.",
        "Die Outlier-Analyse ist deskriptiv und verwendet keine nachträgliche Entfernung weiterer Punkte.",
        "",
        "Regeneration auf CPU aus den versionierten Dateien:",
        "",
        "```sh",
        "cd code",
        "uv run python ../paper/repro/qwen3_26_relations.py",
        "```",
        "",
        "Das Skript prüft die exakte Relationsauswahl, die Reproduktion des alten Fits,",
        "einen unabhängigen OLS-Lösungsweg und die OLS-Residualbedingungen. Es schreibt",
        "den Bericht, beide CSVs sowie die PNG-/PDF-Abbildung neu.",
        "",
    ]
    (DOCS / f"{STEM}.md").write_text("\n".join(lines))
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
