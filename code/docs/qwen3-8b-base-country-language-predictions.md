# Qwen3 8B Base: konkrete Vorhersagen für `country_language`

**Nachtrag zur Interpretation:** Die [Implementierungsprüfung](qwen3-jlens-implementation-review.md)
führt die wiederkehrenden `-Semit`-Ausgaben auf die Mischung sehr großer
Qwen-Zustände an Tokenposition 0 mit deutlich kleineren Zuständen an späteren
Positionen zurück. Die folgenden gespeicherten Ausgaben sind reproduzierbar,
aber als Maß semantischer Salienz durch diesen Positionseffekt beeinträchtigt.

Stand: 20. September 2026; `main`-Protokoll. Diese Tabelle verwendet dieselbe
Darstellung wie die [CEO-Auswertung](qwen3-8b-base-company-ceo-predictions.md).

Die relationsspezifische LRE stimmt bei **6 von 8 Ländern (75 %)** mit dem
ersten Antworttoken des vollständigen Qwen-Modells überein. Die differentielle
Lens erreicht **0/16 Top-1-Treffer** und **1/16 Top-10-Treffer (6,25 %)**.
Der mediane gespeicherte Lens-Rang ist 75.290,5 (nullbasiert).

Im [26-Relationen-Fit](qwen3-8b-base-figure1.md) hat `country_language` die Werte:

| Größe | Wert | Bedeutung |
| --- | ---: | --- |
| n Lens | 16 | Ausgewertete Länder im Lens-Split |
| n LRE | 8 | Ausgewertete Länder im LRE-Split |
| x | −4,876662 | Negativer medianer log10(rank + 1) der differentiellen Lens |
| y | 0,750000 | Gemessene LRE-Faithfulness: 75 % |
| ŷ | 0,493945 | Erwartete Faithfulness laut Regressionsgerade über 26 Relationen |
| y − ŷ | +0,256055 | LRE liegt 25,61 Prozentpunkte über dieser Geraden |

`y` ist eine Trefferquote gegen das Modell, keine Fehlerquote. `ŷ` ist eine
Regressionsvorhersage und keine separat gemessene Modellgenauigkeit.

## Was die Spalten bedeuten

- **Modell im Sprach-Prompt:** Top-1-Fortsetzung von `People in {} speak` nach
  acht Demonstrationsbeispielen. Dieses erste Token ist das Faithfulness-Ziel.
- **LRE:** relationsspezifische Vorhersage aus demselben Prompt-Kontext, mit
  dem gespeicherten State-Index 19 und β = 5. Die konkreten Tokens wurden neu
  ausgelesen; alle acht ursprünglichen Treffer/Fehler wurden exakt reproduziert.
- **Differentielle Lens:** gespeicherte Top-5-Tokens am State-Index 19 unter
  dem neutralen Prompt `{} is known for`. Die Lens bekommt keine Sprachrelation
  als Eingabe. Ihr Zieltoken wird im Sprach-Prompt des Lens-Splits bestimmt.
- **Splits:** LRE und Lens verwenden unabhängig erzeugte Aufteilungen und
  unterschiedliche Demonstrations- und Referenzbeispiele. Fünf Länder kommen
  in beiden Testmengen vor. Die zwei Modellziel-Spalten unten gehören jeweils
  zum eigenen Prompt-Kontext und wurden nicht zwischen den Splits übertragen.
- Alle Ausgaben sind einzelne Tokens. Äußere Leerzeichen sind in den Tabellen
  entfernt. Reine Leerraum-Tokens werden als JSON-Zeichenketten mit Escape-Sequenzen
  dargestellt. Der [JSON-Snapshot](qwen3-8b-base-country-language-predictions.json) enthält die neu decodierten
  LRE-Tokens mit Leerzeichen, Token-IDs, Top-5-Listen und Zielrängen.
- Das Feld `lens.rows[].top_lre` im ursprünglichen Snapshot bezeichnet die
  zusätzliche affine Lens unter dem neutralen Prompt mit β = 1. Es ist nicht
  die relationsspezifische LRE mit der gemessenen Faithfulness 0,75 und wird
  deshalb nicht als LRE-Vorhersage in diesen Tabellen ausgegeben.

## Die fünf gemeinsamen Länder

| Land | Modellziel im LRE-Split | LRE Top-1 | Modellziel im Lens-Split | Differentielle Lens Top-5 |
| --- | --- | --- | --- | --- |
| Egypt | `Arabic` | `Arabic` | `Arabic` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` |
| Brazil | `Portuguese` | `Spanish` | `Portuguese` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` |
| India | `Hindi` | `various` | `Hindi` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` |
| Chile | `Spanish` | `Spanish` | `Spanish` | `Chile`, `Chili`, `Monterey`, `Oregon`, `iod` |
| Germany | `German` | `German` | `German` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` |

## Alle acht LRE-Vorhersagen

„Treffer“ bedeutet Gleichheit mit dem Modellziel im selben Sprach-Prompt.
Das vollständige Modell trifft in diesen acht Fällen auch jeweils das erste
Token der gespeicherten Datensatzantwort (8/8). Die LRE-Auswertung bleibt
jedoch eine Messung der Modellübereinstimmung. Die Länder-Sprach-Zuordnung
ist diejenige dieses Datensatzes und keine vollständige Beschreibung der
Sprachen eines Landes.

| Land | Datensatzantwort | Modellziel | LRE Top-1 | LRE Top-5 | Treffer |
| --- | --- | --- | --- | --- | --- |
| Germany | German | `German` | `German` | `German`, `Spanish`, `Standard`, `French`, `Italian` | ja |
| Chile | Spanish | `Spanish` | `Spanish` | `Spanish`, `"\n"`, `" "`, `Portuguese`, `Russian` | ja |
| India | Hindi | `Hindi` | `various` | `various`, `several`, `multiple`, `different`, `either` | nein |
| Egypt | Arabic | `Arabic` | `Arabic` | `Arabic`, `Standard`, `Modern`, `Egyptian`, `standard` | ja |
| Brazil | Portuguese | `Portuguese` | `Spanish` | `Spanish`, `Portuguese`, `Italian`, `" "`, `"\n"` | nein |
| Colombia | Spanish | `Spanish` | `Spanish` | `Spanish`, `Italian`, `" "`, `"\n"`, `English` | ja |
| Canada | English | `English` | `English` | `English`, `either`, `both`, `French`, `two` | ja |
| Nigeria | English | `English` | `English` | `English`, `two`, `either`, `both`, `several` | ja |

## Alle 16 Lens-Vorhersagen

Der Zielrang ist nullbasiert: 0 wäre ein Top-1-Treffer, 5 bedeutet die sechste
Position. Nur bei **South Korea** liegt das Modellziel `Korean` in den Top 10
(Rang 5); deshalb erscheint es noch nicht in der Top-5-Liste.

| Land | Modellziel im Sprach-Prompt (Lens-Split) | Differentielle Lens Top-5 | Zielrang in der Lens |
| --- | --- | --- | ---: |
| Australia | `English` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 77082 |
| Argentina | `Spanish` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 148524 |
| Egypt | `Arabic` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 76767 |
| Peru | `Spanish` | `Peru`, `Guatemala`, `Honduras`, `Montana`, `Oregon` | 1961 |
| Japan | `Japanese` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 137902 |
| United States | `English` | `Democr`, `Equality`, `Baltimore`, `Attorney`, `Liberals` | 35520 |
| China | `Mandarin` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` | 80402 |
| Brazil | `Portuguese` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` | 95053 |
| Venezuela | `Spanish` | `Venezuela`, `Venezuelan`, `Maduro`, `Venezuel`, `venez` | 8303 |
| India | `Hindi` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 38539 |
| Chile | `Spanish` | `Chile`, `Chili`, `Monterey`, `Oregon`, `iod` | 11129 |
| Saudi Arabia | `Arabic` | `Saudi`, `Riyadh`, `Saudi`, `Saudis`, `沙特` | 100 |
| Germany | `German` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 136572 |
| Spain | `Spanish` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` | 148355 |
| Russia | `Russian` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 73814 |
| South Korea | `Korean` | `Korea`, `Seoul`, `Koreans`, `Samsung`, `Hyundai` | 5 |

## Auffälligkeiten

Die LRE-Fehler betreffen **India** (`various` statt `Hindi`) und **Brazil** (`Spanish` statt `Portuguese`).
Bei nur acht Testländern entspricht ein einzelner Treffer 12,5 Prozentpunkten;
die 75 % sollten deshalb zusammen mit dieser kleinen Stichprobe gelesen werden.

Bei **10 der 16 Lens-Länder** ist das oberste Token identisch: `-Semit`.
Die Top-5-Listen bestehen dort weitgehend aus denselben Tokens, darunter
`(Initialized`, `iNdEx`, `StringWriter` und `搜狐首页`. Bei anderen Ländern
erscheinen eher Länder- oder Assoziationstokens, etwa `Chile`, `Saudi` oder
`Korea`. Diese Ausgaben dokumentieren das schlechte Lens-Ergebnis; aus ihnen
allein lässt sich die Ursache der wiederkehrenden Tokens nicht bestimmen.

## Herkunft und Prüfung

Modell: `Qwen/Qwen3-8B-Base`; Revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; Float32; Eager Attention.
Die ursprünglichen Messungen liegen lokal unter
`code/results/five_model_sweep/qwen3_8b_base/main/country_language.json`.

Die LRE-Tokens wurden auf `beta` erneut berechnet, mit denselben acht
Demonstrationen, acht Referenzbeispielen, acht Testländern und Batch-Size 2.
Es wurde ausschließlich die bereits ausgewählte Kombination aus State-Index 19
und β = 5 verwendet. Es fand keine erneute Hyperparametersuche statt. Die Auswahl
von Layer und β erfolgte im ursprünglichen Lauf auf denselben Testbeispielen;
die Faithfulness ist daher eine In-Sample-Schätzung nach Hyperparameterauswahl.

Geprüft wurden der Split-Fingerprint `d9cfbee9a332`, die Reihenfolge aller acht
Testländer, sämtliche ursprünglichen Treffer/Fehler und alle Modell-Zieltokens
gegen den bestehenden separaten Target-Audit. Die numerischen Prüfungen des
Sweep-Runners wurden vor der Messung ebenfalls ausgeführt. Die 16 Lens-Zeilen
wurden unverändert aus dem ursprünglichen Snapshot übernommen.

Der [JSON-Snapshot](qwen3-8b-base-country-language-predictions.json) enthält die Vorhersagen, Demonstrations-
und Referenzbeispiele, numerischen Prüfwerte sowie SHA-256-Prüfsummen der
Eingabedateien und des Ausleseskripts. Die lokalen Ausleseskripte und das
Laufprotokoll liegen unter `code/generated/qwen3_country_language/`.
