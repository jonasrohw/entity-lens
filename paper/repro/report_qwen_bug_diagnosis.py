"""Publish the Qwen bug diagnosis and ordinary neutral-prompt predictions."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "code"
DOCS = BASE / "docs"
DIAG = "qwen3-8b-base-token-readout-diagnosis"
NEUTRAL = "qwen3-8b-base-neutral-prompt-predictions"
BUG = "qwen3-8b-base-bug-diagnosis"


def token(text):
    # JSON quoting makes leading spaces and line breaks visible.
    return (
        "<code>"
        + json.dumps(text, ensure_ascii=False)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        + "</code>"
    )


def csv_file(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_neutral(data):
    rows = data["rows"]
    low = min(r["top10"][0]["probability"] for r in rows)
    high = max(r["top10"][0]["probability"] for r in rows)
    semit_ranks = [
        next(t["rank"] for t in r["watched_tokens"] if t["id"] == 44092) for r in rows
    ]
    text = [
        "# Qwen3 8B Base: direkte Vorhersagen für `{country} is known for`\n",
        "Dies sind **normale Forward-Passes und Fortsetzungen des unveränderten Modells**, ohne Jacobian-Lens, Subtraktion, Intervention, Chat-Template oder ICL-Beispiele. Ausgewertet werden dieselben 16 Länder wie in der Hauptstadt-/Sprach-Lens-Analyse.\n",
        f"Bei **allen 16 Ländern** ist das nächste Top-1-Token {token(' its')} (ID 1181), mit {100 * low:.2f}–{100 * high:.2f} % Wahrscheinlichkeit. Die sechs untersuchten Artefakt-Tokens erscheinen bei keinem Land in den nächsten Top 10 und in keiner der erzeugten 48-Token-Fortsetzungen. `-Semit` steht im normalen Modell auf den nullbasierten Rängen **{min(semit_ranks)}–{max(semit_ranks)}**.\n",
        "Der Prompt erzeugt grammatische, teils schablonenhafte Länderbeschreibungen. Das macht ihn noch nicht zu einem geeigneten Messprompt für die Lens: Die direkte nächste Fortsetzung ist hier ein Funktionswort, und erst danach folgt inhaltlicher Text. Die [Bug-Diagnose](qwen3-8b-base-bug-diagnosis.md) trennt diese offene Prompt-Frage vom bereits gemessenen Positionsartefakt.\n",
        f"[Alle Top-10-Tokens und Wahrscheinlichkeiten als CSV]({NEUTRAL}.csv) · [Fortsetzungen als CSV]({NEUTRAL}-continuations.csv) · [Vollständiges JSON mit Token-IDs, Artefakt-Rängen und Provenienz]({NEUTRAL}.json).\n",
        "## Exaktes Protokoll\n",
        "- `Qwen/Qwen3-8B-Base`, Revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; float32 und eager attention auf A100 80 GB.",
        "- Exakter Prompt: `{country} is known for`, ohne zusätzliches Präfix, ohne abschließendes Leerzeichen und ohne Frage-/Antwortformatierung.",
        "- Top-10-Wahrscheinlichkeiten stammen aus dem Softmax über das vollständige Vokabular beim ersten nächsten Token; sie sind nicht innerhalb der Top 10 renormalisiert. Token-Zeichenketten sind mit sichtbaren Leerzeichen/Escapes dargestellt.",
        "- Fortsetzung: pro Schritt deterministischer Argmax (greedy), maximal 48 neue Tokens, Stop bei EOS. Alle 16 Fortsetzungen erreichten hier das 48-Token-Limit; Enden können mitten im Satz oder Wort liegen. Die Fortsetzungen sind Modellausgaben, keine geprüften Fakten und keine Wahrscheinlichkeitsverteilung über vollständige Antworten.\n",
        "## Wahrscheinlichste nächste Tokens\n",
        "| Land | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 | Masse der Top 10 |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for r in rows:
        cells = [
            f"{token(t['token'])}: {100 * t['probability']:.2f} %"
            for t in r["top10"][:5]
        ]
        text.append(
            f"| {r['subject']} | "
            + " | ".join(cells)
            + f" | {100 * r['top10_probability_mass']:.2f} % |"
        )
    text += [
        "\n## Vollständige greedy Fortsetzungen bis zum 48-Token-Limit\n",
        "Die Blöcke enthalten den Originalprompt plus die generierte Fortsetzung. Die Markierung des Token-Limits gehört nicht zur Modellausgabe.\n",
    ]
    for r in rows:
        text += [
            f"### {r['subject']}\n",
            "```text\n" + r["prompt"] + r["continuation"] + "\n```\n",
            f"Stopp: {len(r['generated_token_ids'])} neue Tokens, Token-Limit.\n",
        ]
    text += [
        "## Artefakt-Kontrolle im normalen Forward-Pass\n",
        "Die Ränge beziehen sich jeweils auf das erste nächste Token unter dem neutralen Prompt, nicht auf die Hauptstadt-ICL-Abfrage.\n",
        "| Land | Rang `-Semit` | P(`-Semit`) | Rang `(Initialized` | P(`(Initialized`) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        watched = {t["id"]: t for t in r["watched_tokens"]}
        a, b = watched[44092], watched[89998]
        text.append(
            f"| {r['subject']} | {a['rank']} | {a['probability']:.6g} | {b['rank']} | {b['probability']:.6g} |"
        )
    text += [
        "\n## Reproduktion\n",
        "```sh\ncd code\nCUDA_VISIBLE_DEVICES=0 uv run --extra gpu python ../paper/repro/qwen_token_readout.py\nuv run python ../paper/repro/report_qwen_bug_diagnosis.py\n```\n",
        "[GPU-Skript](../../paper/repro/qwen_token_readout.py) · [Export-/Berichtsskript](../../paper/repro/report_qwen_bug_diagnosis.py). Modellrevision, Bibliotheksversionen und Source-Hashes sind im JSON dokumentiert.\n",
    ]
    return "\n".join(text)


def render_bug(diag, neutral):
    german = next(r for r in neutral["rows"] if r["subject"] == "Germany")
    ordinary = {r["id"]: r for r in german["watched_tokens"]}
    by_id = {r["token_id"]: r for r in diag["tokens"]}
    text = [
        "# Qwen3 8B Base: Bug-Diagnose, Token-Artefakt und Prompt-Eignung\n",
        "Diese Übersicht führt alle bisherigen Prüfungen zusammen und ergänzt die bisher nur im Gespräch berichtete Token-Score-Zerlegung für Germany sowie direkte Modellvorhersagen für alle 16 Länder. Die Messungen stammen vom 20. September 2026.\n",
        "**Befund:** Die seltsamen Lens-Tokens sind ein reproduzierbares Artefakt der untersuchten Messmethode. Für falsche Token-IDs, falsche Positionsindizes oder einen falsch berechneten Jacobian-Vector-Product fanden die gezielten Prüfungen keine Evidenz. Das beweist nicht, dass der gesamte Code fehlerfrei ist. Ein separater Normalisierungsbug wurde tatsächlich gefunden und behoben, erklärt aber diese Qwen-Ausgaben nicht.\n",
        "**Die Prompt-Frage bleibt offen:** Im unveränderten Qwen-Modell ist bei allen 16 Ländern das nächste Top-1-Token ` its`; anschließend entstehen Länderbeschreibungen. Die wiederkehrenden Lens-Artefakte fehlen in den normalen Top 10 und den erzeugten Fortsetzungen. Ein syntaktisch funktionierender Prompt muss jedoch nicht für die relation-agnostische Lens geeignet sein.\n",
        "## Was geprüft wurde und was daraus folgt\n",
        "| Frage | Ergebnis | Grenze der Aussage |",
        "| --- | --- | --- |",
        "| Falsche Subjektposition? | Kein Offset-Mismatch in 2.057 gespeicherten Qwen-Zeilen; gewählt wird das letzte Subjekttoken. | Ein korrekt bestimmter Index kann trotzdem eine ungeeignete Repräsentation liefern. |",
        "| Falsche Ableitung oder falscher Layer? | Exakte No-op-Reinsertion; unabhängige Suffix-Ausführung und Forward-AD stimmen mit den JVPs überein (max. relativer Fehler 1,28e−6). | Korrekte lokale Ableitungen garantieren keine gute Näherung für große Zustandswechsel. |",
        "| Falsche Normierung oder Token-Decodierung? | Native Norm/Output-Head und Token-IDs reproduzieren die Ausgaben. Der positive RMS-Divisor verändert die Rangfolge nicht. | Dies validiert den ausgeführten Readout, nicht dessen semantische Interpretation. |",
        "| Tatsächlicher Implementierungsbug? | Der generische `final_logits`-Helper erkannte den Normtyp fälschlich anhand eines Bias. Gemma2 RMSNorm und biaslose LayerNorm wurden korrigiert; Regressionstests schlugen vorher fehl und bestanden danach. | Der Qwen-Sweep nutzte bereits die native Norm. Dieser Bug ist nicht die Ursache des Qwen-Artefakts. |",
        "| Nur die Subtraktion schuld? | Nein: Raw-Lens `mean_i J_i h_e` zeigt dieselben zehn `-Semit`-Fälle in den Länderrelationen. | Subtraktion ist für das Artefakt nicht notwendig. |",
        "| Nur große Normen allgemein problematisch? | Nein: Entscheidend sind die positionsabhängigen Komponenten, die Eingaberichtung und die Reaktion der Referenz-Jacobians. | Größe allein ist weder Bug-Nachweis noch Beweis eines Fehlers der Linearisierung. |",
        "| Erzeugt der normale Prompt die seltsamen Tokens? | Nicht in den hier gemessenen nächsten Top 10 oder den 48-Token-Fortsetzungen. | Dies beweist keine universelle Prompt-Güte und keine Abwesenheit solcher Tokens in beliebig langen Generationen. |\n",
        "## Warum gerade `-Semit`, `(Initialized` und ähnliche Tokens?\n",
        "Bei einteiligen Ländernamen steht das Subjekt an Position 0. Die Qwen-Zustandsnorm liegt dort ungefähr bei 10.106–11.239, gegenüber 128–141 bei den späteren Subjektpositionen. Die Koordinaten 2276 und 233 erklären zusammen etwa 99,80 % der quadrierten Differenz zwischen den Mittelzuständen dieser beiden Gruppen. Die ursprüngliche differentielle Lens ergibt für die zehn Position-0-Länder fast parallele Vektoren (mittlerer Cosinus 0,999885).\n",
        "OLMo verwendet für diese Länder dieselben Subjektpositionen, zeigt aber Normen von etwa 8,90–10,81 an Position 0 und 20,44–22,39 an späteren Positionen. Die enorme Qwen-Skalendifferenz fehlt dort. Das ist ein konkreter Unterschied der Messbedingungen; es erklärt noch nicht sämtliche Unterschiede zwischen den Modellkorrelationen.\n",
        "Auch ohne Subtraktion werden Referenz-Jacobians auf den großen Germany-Zustand angewandt. Bei Germany dominieren die Referenzen Peru und United States. Das lässt sich bis zum Score einzelner Tokens verfolgen.\n",
        "Für Qwens biaslosen Output-Head gilt `logit_t = [w_t · (g ⊙ v)] / RMS(v)`, wobei der RMS-Divisor den Epsilon-Term der nativen Norm enthält. Hier ist g das gelernte finale Normgewicht. Der Zähler ist additiv über Referenzen; der Nenner ist positiv und für alle Tokens gleich. Die folgenden Werte sind diese Zähler, keine Wahrscheinlichkeiten und nicht die Logits des normalen Forward-Passes. Kleine FP32-Rundungsdifferenzen zwischen separat berechneten Komponenten und Summen sind in den numerischen Checks berücksichtigt.\n",
        "| Token | Raw-Score vor RMS-Divisor | Davon Koordinaten 2276/233 | Davon übrige Koordinaten | Raw-Lens-Rang | Rang im normalen neutralen Prompt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in diag["tokens"]:
        rank = (
            diag["raw"]["top_ids"].index(r["token_id"])
            if r["token_id"] in diag["raw"]["top_ids"]
            else diag["raw"]["berlin_rank"]
        )
        text.append(
            f"| {token(r['token'])} | {r['score_before_rms_divisor']:+.6f} | {r['two_coordinate_score']:+.6f} | {r['remaining_coordinate_score']:+.6f} | {rank} | {ordinary[r['token_id']]['rank']} |"
        )
    text += [
        f"\nRMS-Divisor des Raw-Lens-Vektors: **{diag['rms_divisor']:.6f}**. Die Zähler liefern nach dieser Division die nativen Lens-Readout-Logits; die Gleichheit wurde numerisch geprüft. Die ungewöhnlichen Tokens erhalten hohe positive Projektionen auf die rekonstruierte Richtung, während ` Berlin` eine negative Projektion erhält. Es handelt sich auch nicht bloß um besonders große Output-Vektoren: `-Semit` liegt bei deren Norm nur ungefähr im 16. Perzentil.\n",
        "### Beiträge einzelner Referenzen\n",
        "Jede Zelle enthält bereits das Gewicht 1/8. Germany selbst gehört nicht zum Referenzsatz.\n",
        "| Referenz | Beitrag zu `-Semit` | Beitrag zu `(Initialized` | Beitrag zu ` Berlin` |",
        "| --- | ---: | ---: | ---: |",
    ]
    berlin = next(r for r in diag["tokens"] if r["token"].strip() == "Berlin")
    selected = [by_id[44092], by_id[89998], berlin]
    for i, c in enumerate(selected[0]["reference_score_contributions"]):
        text.append(
            f"| {c['reference']} | "
            + " | ".join(
                f"{t['reference_score_contributions'][i]['score']:+.6f}"
                for t in selected
            )
            + " |"
        )
    text += [
        "\nPeru und United States liefern zusammen etwa +202,22 zum `-Semit`-Score; die anderen sechs Referenzen liefern zusammen etwa −4,69. Die Summe ist +197,53. Die gleiche Zerlegung für ` Berlin` erklärt dessen stark negativen Score. Das zeigt, welche Referenzen die konkreten Token-Ausgaben treiben.\n",
        "### Koordinatenkontrolle für Germany\n",
        "Die Jacobians bleiben dabei unverändert; nur die in sie eingespeiste Raw-Richtung wird zerlegt. Die zwei Koordinaten wurden nach Sichtung des Artefakts ausgewählt: Das ist eine Diagnose, kein validiertes Reparaturverfahren.\n",
        "| Eingaberichtung | Top 5 | Rang ` Berlin` | Norm des gemittelten JVP-Vektors |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, label in [
        ("raw", "Vollständiger Raw-Zustand"),
        ("two_coordinates_only", "Nur Koordinaten 2276 und 233"),
        ("two_coordinates_removed", "Beide Koordinaten entfernt"),
    ]:
        r = diag[key]
        text.append(
            f"| {label} | {', '.join(token(t) for t in r['top_tokens'][:5])} | {r['berlin_rank']} | {r['vector_norm']:.6f} |"
        )
    text += [
        f"\nCosinus zwischen Raw-Lens und der Nur-zwei-Koordinaten-Komponente: **{diag['two_coordinates_cosine_to_raw']:.6f}**. Nur diese Komponenten reichen bei Germany aus, um `-Semit` als Top 1 zu reproduzieren. Entfernt man sie, verschwinden diese Tokens, aber es entstehen überwiegend Ellipsen-/Zeilenumbruch-Tokens; Berlin bleibt weit hinten. Das Artefakt zu beseitigen repariert die Lens also noch nicht.\n",
        "## Warum die korrekte Ableitung trotzdem problematisch ist\n",
        "Ein JVP ist eine lokale Antwort-Richtung. Weder `mean_i J_i(h_e − h_i)` noch `mean_i J_i h_e` ist automatisch ein natürlicher finaler Modellzustand. Sein Decodieren setzt voraus, dass die rekonstruierte Richtung semantisch interpretierbar ist. Der Raw-Readout ist auch nicht die affine Taylor-Näherung: Deren Offset wäre `f_i(h_i) − J_i h_i`.\n",
        "Die frühere volle Interventionskontrolle zeigt für Peru → Germany einen Zustandswechsel vom 80,77-Fachen der Referenznorm: JVP-Norm 7.170 gegenüber tatsächlicher nichtlinearer Output-Änderung 389; relativer Vektorfehler 18,26. Bei Australia → Germany beträgt das Verhältnis 0,042 und der Fehler 0,046. Damit ist die schlechte große-Schritt-Näherung für diese konkreten Fälle gemessen, nicht nur aus der großen Norm vermutet.\n",
        "## Was die bisherigen Varianten ergeben\n",
        "| Variante | Gemessener Befund |",
        "| --- | --- |",
        "| Ursprüngliche differentielle Lens, 26 Relationen | Qwen Spearman ρ = 0,298768; OLMo ρ = 0,782758. |",
        "| Cache und Intervention am finalen `for` | Qwen ρ = −0,067077; OLMo ρ = 0,645912. Qwen: 0/491 Targets in Top 10. |",
        "| Raw ohne Subtraktion, ursprüngliche Subjektposition | Qwen ρ = 0,294661; `-Semit` bleibt bei 10/16 Ländern der Hauptstadt-/Sprachrelation Top 1. |",
        "| Raw ohne Subtraktion am finalen `for` | Qwen ρ = 0,364819, p = 0,066881; trotzdem 0/491 Targets in Top 10. |",
        "| Führender Zeilenumbruch, Subjektposition beibehalten | Nur auf den 16 Sprach-Ländern geprüft: medianer Zielrang 75.290,5 → 29; Top 10 von 1/16 → 6/16. Noch kein Ergebnis über alle 26 Relationen. |",
        "| Nur Position-0-Referenzen / entfernte Ausreißerkoordinaten | Wiederkehrendes `-Semit` verschwindet in den bisherigen Kontrollen; Zielränge bleiben oft schlecht. |\n",
        "Die Korrelationen und Rangverteilungen beantworten verschiedene Fragen. Ein höherer Zusammenhang zwischen Relationen bedeutet noch keine guten einzelnen Token-Vorhersagen. Die Varianten wurden explorativ untersucht; p-Werte gegen Null sind keine Tests der Korrelationsdifferenzen.\n",
        "## Ist `{country} is known for` ungeeignet?\n",
        f"Die [direkten Modellvorhersagen]({NEUTRAL}.md) zeigen für alle 16 Länder ` its` als nächsten Gewinner, mit 51,67–74,15 % Wahrscheinlichkeit. Für Germany beginnt die greedy Fortsetzung mit:\n",
        "```text\n" + german["prompt"] + german["continuation"] + "\n```\n",
        "Der Text ist nach 48 neuen Tokens abgeschnitten. Andere Fortsetzungen wechseln in generische Artikelmuster, etwa „In this article …“. Der Prompt ruft also Beschreibungen darüber ab, wofür ein Land bekannt ist; er fragt weder explizit nach der Hauptstadt noch nach der Sprache. Dass unmittelbar zuerst ein Funktionswort folgt, ist ein konkreter Grund, die Eignung für unsere Ein-Token-Lens-Metrik zu hinterfragen. Das allein beweist aber nicht, dass der Prompt die gesamte schlechte Korrelation verursacht. Die Lens soll gerade relationsunabhängig sein; fehlende Hauptstadt-Top-1 im normalen neutralen Prompt ist für sich genommen kein Bug.\n",
        "Es gibt somit zwei getrennte, miteinander vereinbare Befunde: einen numerisch nachgewiesenen Positionseffekt in der Lens und eine offene Frage zur sprachlichen/semantischen Eignung des Messprompts. Ein Präfix oder eine andere Position kann das erste Artefakt entfernen und dennoch schlechte Token-Ränge ergeben.\n",
        "## Noch nicht geklärt\n",
        "- Welche Attention-Heads oder MLPs die enorme Position-0-Aktivierung ursprünglich erzeugen; eine Aufmerksamkeitssenke ist hier keine nachgewiesene Ursache.",
        "- Ob ein vorab festgelegter anderer neutraler Prompt bzw. der Zeilenumbruch-Prompt die Lens über alle 26 Relationen und beide Modelle verbessert. Die 16-Länder-Präfixkontrolle ist keine solche Validierung.",
        "- Welche Repräsentation und welcher Readout zuverlässige semantische Information liefern. Koordinaten einfach zu löschen oder Aktivierungen vorher zu normalisieren ist noch keine begründete Standardkorrektur.\n",
        "## Alle zugehörigen Analysen und Daten\n",
        "| Inhalt | Bericht / Daten |",
        "| --- | --- |",
        "| Vollständige ursprüngliche Implementierungsprüfung, unabhängige AD-Prüfung, Präfix-/Referenz-/Koordinatenkontrollen, OLMo-Normen, Audit der 2.057 Zeilen | [Review](qwen3-jlens-implementation-review.md), [numerische Evidenz](qwen3-jlens-implementation-review.json), [Reproduktion](../../paper/repro/qwen_jlens_review.py) |",
        "| 26-Relationen-Korrelation und neue Residuen nach Entfernen der beiden Jahresrelationen | [Figure-1-Bericht](qwen3-8b-base-figure1.md), [CSV](qwen3-8b-base-figure1.csv) |",
        "| Finale Promptposition, Qwen und OLMo | [Bericht](last-position-lre-correlation.md), [Daten](last-position-lre-correlation.json), [GPU-Skript](../../paper/repro/last_position_correlation.py) |",
        "| Raw-Lens ohne Subtraktion, beide Positionen | [Bericht](qwen3-8b-base-raw-jlens.md), [Daten](qwen3-8b-base-raw-jlens.json), [GPU-Skript](../../paper/repro/qwen_raw_jlens.py) |",
        "| Alle Hauptstadt-Vorhersagen und Referenzbeiträge | [Bericht](qwen3-8b-base-country-capital-predictions.md), [Token-CSV](qwen3-8b-base-country-capital-predictions.csv), [Beiträge](qwen3-8b-base-country-capital-predictions-contributions.csv) |",
        "| Sprachrelation und Abgrenzung zur relationsspezifischen LRE | [Vorhersagen](qwen3-8b-base-country-language-predictions.md) |",
        f"| Neue Token-Score-Zerlegung für Germany | [JSON]({DIAG}.json), [Token-Scores]({DIAG}.csv), [Referenz-Scores]({DIAG}-references.csv), [JVP-Vektoren]({DIAG}-vectors.pt) |",
        f"| Normales Qwen auf allen 16 neutralen Länderprompts | [Vorhersagen und Fortsetzungen]({NEUTRAL}.md), [JSON]({NEUTRAL}.json), [Top-10-CSV]({NEUTRAL}.csv), [Fortsetzungs-CSV]({NEUTRAL}-continuations.csv) |",
        "| Gefundener, unabhängiger Normierungsbug | [Korrigierter Helper](../jlens/model.py), [Regressionstests](../tests/test_readout_norm.py) |\n",
        "### Neue Diagnose reproduzieren\n",
        "```sh\ncd code\nCUDA_VISIBLE_DEVICES=0 uv run --extra gpu python ../paper/repro/qwen_token_readout.py\nuv run python ../paper/repro/report_qwen_bug_diagnosis.py\n```\n",
        "[GPU-Skript](../../paper/repro/qwen_token_readout.py) · [CPU-Berichtsskript](../../paper/repro/report_qwen_bug_diagnosis.py). Die neue Ausführung reproduziert Germanys ursprüngliche Raw-/Differential-Top-5 und Zielränge exakt. Sie prüft außerdem No-op-Interventionen, die Vektorzerlegung in zwei plus übrige Koordinaten, die Addition der Referenz-Scores und die native RMS-/Head-Rekonstruktion.\n",
        "Die Vektordatei enthält `products` in der Reihenfolge `[Referenz, Variante, Hidden-Dimension]` und `vectors` als Referenzmittelwerte; Namen und Variantenreihenfolge sind mitgespeichert. Laden mit `torch.load(path, map_location='cpu', weights_only=True)`. Alle Source-/Input-Hashes und Bibliotheksversionen stehen im JSON.\n",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=BASE / "generated/qwen_token_readout"
    )
    args = parser.parse_args()
    diag = json.loads((args.input_dir / "diagnosis.json").read_text())
    neutral = json.loads((args.input_dir / "neutral_predictions.json").read_text())
    assert diag["provenance"] == neutral["provenance"]
    assert (
        diag["provenance"]["script_sha256"]
        == hashlib.sha256(
            Path(__file__).with_name("qwen_token_readout.py").read_bytes()
        ).hexdigest()
    )
    assert len(neutral["rows"]) == 16
    artifacts = {44092, 89998, 90443, 53321, 84787, 114068}
    predictions, continuations = [], []
    for r in neutral["rows"]:
        assert r["top10"][0]["id"] == 1181 == r["generated_token_ids"][0]
        assert not artifacts.intersection(t["id"] for t in r["top10"])
        assert not artifacts.intersection(r["generated_token_ids"])
        assert len(r["generated_token_ids"]) == 48 and not r["stopped_on_eos"]
        assert (
            abs(sum(t["probability"] for t in r["top10"]) - r["top10_probability_mass"])
            < 1e-6
        )
        for rank, t in enumerate(r["top10"]):
            predictions.append(
                {
                    "subject": r["subject"],
                    "prompt": r["prompt"],
                    "rank": rank,
                    "token_id": t["id"],
                    "token": t["token"],
                    "probability": t["probability"],
                }
            )
        continuations.append(
            {
                "subject": r["subject"],
                "prompt": r["prompt"],
                "continuation": r["continuation"],
                "stop_reason": r["stop_reason"],
                "n_generated_tokens": len(r["generated_token_ids"]),
                "generated_token_ids": json.dumps(r["generated_token_ids"]),
            }
        )
    assert neutral["generation"]["max_new_tokens"] == 48
    token_rows, reference_rows = [], []
    for r in diag["tokens"]:
        token_rows.append(
            {k: v for k, v in r.items() if k != "reference_score_contributions"}
        )
        for c in r["reference_score_contributions"]:
            reference_rows.append({"token_id": r["token_id"], "token": r["token"], **c})
    vector_source = args.input_dir / "germany_vectors.pt"
    diag["vectors_sha256"] = hashlib.sha256(vector_source.read_bytes()).hexdigest()
    shutil.copyfile(vector_source, DOCS / (DIAG + "-vectors.pt"))
    for stem, obj in [(DIAG, diag), (NEUTRAL, neutral)]:
        (DOCS / (stem + ".json")).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
        )
    csv_file(DOCS / (DIAG + ".csv"), token_rows)
    csv_file(DOCS / (DIAG + "-references.csv"), reference_rows)
    csv_file(DOCS / (NEUTRAL + ".csv"), predictions)
    csv_file(DOCS / (NEUTRAL + "-continuations.csv"), continuations)
    (DOCS / (NEUTRAL + ".md")).write_text(render_neutral(neutral))
    (DOCS / (BUG + ".md")).write_text(render_bug(diag, neutral))
    print(
        f"Published diagnosis, {len(predictions)} next-token predictions, and {len(continuations)} continuations."
    )


if __name__ == "__main__":
    main()
