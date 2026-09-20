# Qwen3 8B Base: Bug-Diagnose, Token-Artefakt und Prompt-Eignung

Diese Übersicht führt alle bisherigen Prüfungen zusammen und ergänzt die bisher nur im Gespräch berichtete Token-Score-Zerlegung für Germany sowie direkte Modellvorhersagen für alle 16 Länder. Die Messungen stammen vom 20. September 2026.

**Befund:** Die seltsamen Lens-Tokens sind ein reproduzierbares Artefakt der untersuchten Messmethode. Für falsche Token-IDs, falsche Positionsindizes oder einen falsch berechneten Jacobian-Vector-Product fanden die gezielten Prüfungen keine Evidenz. Das beweist nicht, dass der gesamte Code fehlerfrei ist. Ein separater Normalisierungsbug wurde tatsächlich gefunden und behoben, erklärt aber diese Qwen-Ausgaben nicht.

**Die Prompt-Frage bleibt offen:** Im unveränderten Qwen-Modell ist bei allen 16 Ländern das nächste Top-1-Token ` its`; anschließend entstehen Länderbeschreibungen. Die wiederkehrenden Lens-Artefakte fehlen in den normalen Top 10 und den erzeugten Fortsetzungen. Ein syntaktisch funktionierender Prompt muss jedoch nicht für die relation-agnostische Lens geeignet sein.

## Was geprüft wurde und was daraus folgt

| Frage | Ergebnis | Grenze der Aussage |
| --- | --- | --- |
| Falsche Subjektposition? | Kein Offset-Mismatch in 2.057 gespeicherten Qwen-Zeilen; gewählt wird das letzte Subjekttoken. | Ein korrekt bestimmter Index kann trotzdem eine ungeeignete Repräsentation liefern. |
| Falsche Ableitung oder falscher Layer? | Exakte No-op-Reinsertion; unabhängige Suffix-Ausführung und Forward-AD stimmen mit den JVPs überein (max. relativer Fehler 1,28e−6). | Korrekte lokale Ableitungen garantieren keine gute Näherung für große Zustandswechsel. |
| Falsche Normierung oder Token-Decodierung? | Native Norm/Output-Head und Token-IDs reproduzieren die Ausgaben. Der positive RMS-Divisor verändert die Rangfolge nicht. | Dies validiert den ausgeführten Readout, nicht dessen semantische Interpretation. |
| Tatsächlicher Implementierungsbug? | Der generische `final_logits`-Helper erkannte den Normtyp fälschlich anhand eines Bias. Gemma2 RMSNorm und biaslose LayerNorm wurden korrigiert; Regressionstests schlugen vorher fehl und bestanden danach. | Der Qwen-Sweep nutzte bereits die native Norm. Dieser Bug ist nicht die Ursache des Qwen-Artefakts. |
| Nur die Subtraktion schuld? | Nein: Raw-Lens `mean_i J_i h_e` zeigt dieselben zehn `-Semit`-Fälle in den Länderrelationen. | Subtraktion ist für das Artefakt nicht notwendig. |
| Nur große Normen allgemein problematisch? | Nein: Entscheidend sind die positionsabhängigen Komponenten, die Eingaberichtung und die Reaktion der Referenz-Jacobians. | Größe allein ist weder Bug-Nachweis noch Beweis eines Fehlers der Linearisierung. |
| Erzeugt der normale Prompt die seltsamen Tokens? | Nicht in den hier gemessenen nächsten Top 10 oder den 48-Token-Fortsetzungen. | Dies beweist keine universelle Prompt-Güte und keine Abwesenheit solcher Tokens in beliebig langen Generationen. |

## Warum gerade `-Semit`, `(Initialized` und ähnliche Tokens?

Bei einteiligen Ländernamen steht das Subjekt an Position 0. Die Qwen-Zustandsnorm liegt dort ungefähr bei 10.106–11.239, gegenüber 128–141 bei den späteren Subjektpositionen. Die Koordinaten 2276 und 233 erklären zusammen etwa 99,80 % der quadrierten Differenz zwischen den Mittelzuständen dieser beiden Gruppen. Die ursprüngliche differentielle Lens ergibt für die zehn Position-0-Länder fast parallele Vektoren (mittlerer Cosinus 0,999885).

OLMo verwendet für diese Länder dieselben Subjektpositionen, zeigt aber Normen von etwa 8,90–10,81 an Position 0 und 20,44–22,39 an späteren Positionen. Die enorme Qwen-Skalendifferenz fehlt dort. Das ist ein konkreter Unterschied der Messbedingungen; es erklärt noch nicht sämtliche Unterschiede zwischen den Modellkorrelationen.

Auch ohne Subtraktion werden Referenz-Jacobians auf den großen Germany-Zustand angewandt. Bei Germany dominieren die Referenzen Peru und United States. Das lässt sich bis zum Score einzelner Tokens verfolgen.

Für Qwens biaslosen Output-Head gilt `logit_t = [w_t · (g ⊙ v)] / RMS(v)`, wobei der RMS-Divisor den Epsilon-Term der nativen Norm enthält. Hier ist g das gelernte finale Normgewicht. Der Zähler ist additiv über Referenzen; der Nenner ist positiv und für alle Tokens gleich. Die folgenden Werte sind diese Zähler, keine Wahrscheinlichkeiten und nicht die Logits des normalen Forward-Passes. Kleine FP32-Rundungsdifferenzen zwischen separat berechneten Komponenten und Summen sind in den numerischen Checks berücksichtigt.

| Token | Raw-Score vor RMS-Divisor | Davon Koordinaten 2276/233 | Davon übrige Koordinaten | Raw-Lens-Rang | Rang im normalen neutralen Prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| <code>"-Semit"</code> | +197.531982 | +179.652679 | +17.879204 | 0 | 151837 |
| <code>"(Initialized"</code> | +175.078827 | +174.932510 | +0.146118 | 1 | 151568 |
| <code>" iNdEx"</code> | +169.778915 | +164.997910 | +4.780924 | 2 | 151912 |
| <code>"(MigrationBuilder"</code> | +167.829803 | +179.281815 | -11.452124 | 3 | 151930 |
| <code>" StringWriter"</code> | +167.733963 | +159.105164 | +8.628424 | 4 | 99473 |
| <code>"搜狐首页"</code> | +167.547455 | +156.663483 | +10.883733 | 5 | 151858 |
| <code>" Berlin"</code> | -176.083984 | -173.915405 | -2.168808 | 124913 | 149 |

RMS-Divisor des Raw-Lens-Vektors: **18.023939**. Die Zähler liefern nach dieser Division die nativen Lens-Readout-Logits; die Gleichheit wurde numerisch geprüft. Die ungewöhnlichen Tokens erhalten hohe positive Projektionen auf die rekonstruierte Richtung, während ` Berlin` eine negative Projektion erhält. Es handelt sich auch nicht bloß um besonders große Output-Vektoren: `-Semit` liegt bei deren Norm nur ungefähr im 16. Perzentil.

### Beiträge einzelner Referenzen

Jede Zelle enthält bereits das Gewicht 1/8. Germany selbst gehört nicht zum Referenzsatz.

| Referenz | Beitrag zu `-Semit` | Beitrag zu `(Initialized` | Beitrag zu ` Berlin` |
| --- | ---: | ---: | ---: |
| Australia | -0.587113 | -0.036125 | -1.056742 |
| Argentina | -1.035282 | -0.108413 | -1.101054 |
| Egypt | -0.551975 | +0.080541 | -0.670250 |
| Peru | +110.453468 | +104.474434 | -131.398849 |
| Japan | -0.777711 | -0.236894 | -0.980487 |
| United States | +91.767609 | +71.307510 | -39.146431 |
| China | -0.862178 | -0.258276 | -0.933677 |
| Brazil | -0.874769 | -0.143940 | -0.796467 |

Peru und United States liefern zusammen etwa +202,22 zum `-Semit`-Score; die anderen sechs Referenzen liefern zusammen etwa −4,69. Die Summe ist +197,53. Die gleiche Zerlegung für ` Berlin` erklärt dessen stark negativen Score. Das zeigt, welche Referenzen die konkreten Token-Ausgaben treiben.

### Koordinatenkontrolle für Germany

Die Jacobians bleiben dabei unverändert; nur die in sie eingespeiste Raw-Richtung wird zerlegt. Die zwei Koordinaten wurden nach Sichtung des Artefakts ausgewählt: Das ist eine Diagnose, kein validiertes Reparaturverfahren.

| Eingaberichtung | Top 5 | Rang ` Berlin` | Norm des gemittelten JVP-Vektors |
| --- | --- | ---: | ---: |
| Vollständiger Raw-Zustand | <code>"-Semit"</code>, <code>"(Initialized"</code>, <code>" iNdEx"</code>, <code>"(MigrationBuilder"</code>, <code>" StringWriter"</code> | 124913 | 1153.532104 |
| Nur Koordinaten 2276 und 233 | <code>"-Semit"</code>, <code>"(MigrationBuilder"</code>, <code>"(Initialized"</code>, <code>"南海网"</code>, <code>" iNdEx"</code> | 126604 | 1122.368530 |
| Beide Koordinaten entfernt | <code>"…\n\n\n\n"</code>, <code>"…\n\n\n"</code>, <code>"…\n\n"</code>, <code>" ...\n\n\n\n"</code>, <code>" […]\n\n"</code> | 65488 | 117.477974 |

Cosinus zwischen Raw-Lens und der Nur-zwei-Koordinaten-Komponente: **0.995045**. Nur diese Komponenten reichen bei Germany aus, um `-Semit` als Top 1 zu reproduzieren. Entfernt man sie, verschwinden diese Tokens, aber es entstehen überwiegend Ellipsen-/Zeilenumbruch-Tokens; Berlin bleibt weit hinten. Das Artefakt zu beseitigen repariert die Lens also noch nicht.

## Warum die korrekte Ableitung trotzdem problematisch ist

Ein JVP ist eine lokale Antwort-Richtung. Weder `mean_i J_i(h_e − h_i)` noch `mean_i J_i h_e` ist automatisch ein natürlicher finaler Modellzustand. Sein Decodieren setzt voraus, dass die rekonstruierte Richtung semantisch interpretierbar ist. Der Raw-Readout ist auch nicht die affine Taylor-Näherung: Deren Offset wäre `f_i(h_i) − J_i h_i`.

Die frühere volle Interventionskontrolle zeigt für Peru → Germany einen Zustandswechsel vom 80,77-Fachen der Referenznorm: JVP-Norm 7.170 gegenüber tatsächlicher nichtlinearer Output-Änderung 389; relativer Vektorfehler 18,26. Bei Australia → Germany beträgt das Verhältnis 0,042 und der Fehler 0,046. Damit ist die schlechte große-Schritt-Näherung für diese konkreten Fälle gemessen, nicht nur aus der großen Norm vermutet.

## Was die bisherigen Varianten ergeben

| Variante | Gemessener Befund |
| --- | --- |
| Ursprüngliche differentielle Lens, 26 Relationen | Qwen Spearman ρ = 0,298768; OLMo ρ = 0,782758. |
| Cache und Intervention am finalen `for` | Qwen ρ = −0,067077; OLMo ρ = 0,645912. Qwen: 0/491 Targets in Top 10. |
| Raw ohne Subtraktion, ursprüngliche Subjektposition | Qwen ρ = 0,294661; `-Semit` bleibt bei 10/16 Ländern der Hauptstadt-/Sprachrelation Top 1. |
| Raw ohne Subtraktion am finalen `for` | Qwen ρ = 0,364819, p = 0,066881; trotzdem 0/491 Targets in Top 10. |
| Führender Zeilenumbruch, Subjektposition beibehalten | Nur auf den 16 Sprach-Ländern geprüft: medianer Zielrang 75.290,5 → 29; Top 10 von 1/16 → 6/16. Noch kein Ergebnis über alle 26 Relationen. |
| Nur Position-0-Referenzen / entfernte Ausreißerkoordinaten | Wiederkehrendes `-Semit` verschwindet in den bisherigen Kontrollen; Zielränge bleiben oft schlecht. |

Die Korrelationen und Rangverteilungen beantworten verschiedene Fragen. Ein höherer Zusammenhang zwischen Relationen bedeutet noch keine guten einzelnen Token-Vorhersagen. Die Varianten wurden explorativ untersucht; p-Werte gegen Null sind keine Tests der Korrelationsdifferenzen.

## Ist `{country} is known for` ungeeignet?

Die [direkten Modellvorhersagen](qwen3-8b-base-neutral-prompt-predictions.md) zeigen für alle 16 Länder ` its` als nächsten Gewinner, mit 51,67–74,15 % Wahrscheinlichkeit. Für Germany beginnt die greedy Fortsetzung mit:

```text
Germany is known for its high-quality education system, and the country has a long history of producing world-class scientists and researchers. The German education system is highly regarded for its emphasis on practical skills and hands-on learning, which is reflected in the country's strong engineering
```

Der Text ist nach 48 neuen Tokens abgeschnitten. Andere Fortsetzungen wechseln in generische Artikelmuster, etwa „In this article …“. Der Prompt ruft also Beschreibungen darüber ab, wofür ein Land bekannt ist; er fragt weder explizit nach der Hauptstadt noch nach der Sprache. Dass unmittelbar zuerst ein Funktionswort folgt, ist ein konkreter Grund, die Eignung für unsere Ein-Token-Lens-Metrik zu hinterfragen. Das allein beweist aber nicht, dass der Prompt die gesamte schlechte Korrelation verursacht. Die Lens soll gerade relationsunabhängig sein; fehlende Hauptstadt-Top-1 im normalen neutralen Prompt ist für sich genommen kein Bug.

Es gibt somit zwei getrennte, miteinander vereinbare Befunde: einen numerisch nachgewiesenen Positionseffekt in der Lens und eine offene Frage zur sprachlichen/semantischen Eignung des Messprompts. Ein Präfix oder eine andere Position kann das erste Artefakt entfernen und dennoch schlechte Token-Ränge ergeben.

## Noch nicht geklärt

- Welche Attention-Heads oder MLPs die enorme Position-0-Aktivierung ursprünglich erzeugen; eine Aufmerksamkeitssenke ist hier keine nachgewiesene Ursache.
- Ob ein vorab festgelegter anderer neutraler Prompt bzw. der Zeilenumbruch-Prompt die Lens über alle 26 Relationen und beide Modelle verbessert. Die 16-Länder-Präfixkontrolle ist keine solche Validierung.
- Welche Repräsentation und welcher Readout zuverlässige semantische Information liefern. Koordinaten einfach zu löschen oder Aktivierungen vorher zu normalisieren ist noch keine begründete Standardkorrektur.

## Alle zugehörigen Analysen und Daten

| Inhalt | Bericht / Daten |
| --- | --- |
| Vollständige ursprüngliche Implementierungsprüfung, unabhängige AD-Prüfung, Präfix-/Referenz-/Koordinatenkontrollen, OLMo-Normen, Audit der 2.057 Zeilen | [Review](qwen3-jlens-implementation-review.md), [numerische Evidenz](qwen3-jlens-implementation-review.json), [Reproduktion](../../paper/repro/qwen_jlens_review.py) |
| 26-Relationen-Korrelation und neue Residuen nach Entfernen der beiden Jahresrelationen | [Figure-1-Bericht](qwen3-8b-base-figure1.md), [CSV](qwen3-8b-base-figure1.csv) |
| Finale Promptposition, Qwen und OLMo | [Bericht](last-position-lre-correlation.md), [Daten](last-position-lre-correlation.json), [GPU-Skript](../../paper/repro/last_position_correlation.py) |
| Raw-Lens ohne Subtraktion, beide Positionen | [Bericht](qwen3-8b-base-raw-jlens.md), [Daten](qwen3-8b-base-raw-jlens.json), [GPU-Skript](../../paper/repro/qwen_raw_jlens.py) |
| Alle Hauptstadt-Vorhersagen und Referenzbeiträge | [Bericht](qwen3-8b-base-country-capital-predictions.md), [Token-CSV](qwen3-8b-base-country-capital-predictions.csv), [Beiträge](qwen3-8b-base-country-capital-predictions-contributions.csv) |
| Sprachrelation und Abgrenzung zur relationsspezifischen LRE | [Vorhersagen](qwen3-8b-base-country-language-predictions.md) |
| Neue Token-Score-Zerlegung für Germany | [JSON](qwen3-8b-base-token-readout-diagnosis.json), [Token-Scores](qwen3-8b-base-token-readout-diagnosis.csv), [Referenz-Scores](qwen3-8b-base-token-readout-diagnosis-references.csv), [JVP-Vektoren](qwen3-8b-base-token-readout-diagnosis-vectors.pt) |
| Normales Qwen auf allen 16 neutralen Länderprompts | [Vorhersagen und Fortsetzungen](qwen3-8b-base-neutral-prompt-predictions.md), [JSON](qwen3-8b-base-neutral-prompt-predictions.json), [Top-10-CSV](qwen3-8b-base-neutral-prompt-predictions.csv), [Fortsetzungs-CSV](qwen3-8b-base-neutral-prompt-predictions-continuations.csv) |
| Gefundener, unabhängiger Normierungsbug | [Korrigierter Helper](../jlens/model.py), [Regressionstests](../tests/test_readout_norm.py) |

### Neue Diagnose reproduzieren

```sh
cd code
CUDA_VISIBLE_DEVICES=0 uv run --extra gpu python ../paper/repro/qwen_token_readout.py
uv run python ../paper/repro/report_qwen_bug_diagnosis.py
```

[GPU-Skript](../../paper/repro/qwen_token_readout.py) · [CPU-Berichtsskript](../../paper/repro/report_qwen_bug_diagnosis.py). Die neue Ausführung reproduziert Germanys ursprüngliche Raw-/Differential-Top-5 und Zielränge exakt. Sie prüft außerdem No-op-Interventionen, die Vektorzerlegung in zwei plus übrige Koordinaten, die Addition der Referenz-Scores und die native RMS-/Head-Rekonstruktion.

Die Vektordatei enthält `products` in der Reihenfolge `[Referenz, Variante, Hidden-Dimension]` und `vectors` als Referenzmittelwerte; Namen und Variantenreihenfolge sind mitgespeichert. Laden mit `torch.load(path, map_location='cpu', weights_only=True)`. Alle Source-/Input-Hashes und Bibliotheksversionen stehen im JSON.
