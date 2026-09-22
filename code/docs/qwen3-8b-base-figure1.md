# Qwen3 8B Base: Figure-1-Datenpunkte und Abweichungen

**Update, 21. September 2026:** Die Lens-Werte unten stammen aus dem ursprünglichen
Prompt ohne Präfix und sind durch den diagnostizierten First-Token-Effekt beeinflusst.
Der [vollständige Newline-Rerun](../results/qwen_newline_rerun/README.md) enthält
die korrigierten Lens-Werte und Korrelationen für beide Protokolle. Die LRE-Werte
bleiben unverändert; diese Seite bleibt als ursprünglicher Ergebnissnapshot erhalten.

Stand: 19. September 2026. Modell: `Qwen/Qwen3-8B-Base`, Checkpoint-Revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.

Die Tabelle enthält **alle 28 Relationen des `main`-Protokolls** aus dem Qwen-Sweep,
ausgewertet mit den Achsen und der Residuen-Konvention von Figure 1.
Die im Paper eingebettete Figure 1 zeigt die ursprüngliche Modellauswertung;
dies ist die entsprechende Auswertung für Qwen3 8B Base.

Über alle 28 Relationen beträgt die Spearman-Korrelation **ρ = 0.060390**,
**p = 0.760165** (zweiseitig). In dieser Stichprobe ist damit kein
statistisch signifikanter monotoner Zusammenhang nachgewiesen.
Pearson auf den geplotteten Werten: **r = 0.158995**, **p = 0.419020**.

## Definition der Werte

- **x (Lens-Score):** `−median(log10(rank_diff + 1))`, also das negative
  `lens.agg.diff.median_lograng`. Die gespeicherten Token-Ränge sind nullbasiert;
  höhere x-Werte bedeuten bessere Lens-Ränge. Verwendet wird die differentielle
  Lens am festen State-Index 19, mit dem neutralen Prompt `{} is known for`.
- **y (LRE-Faithfulness):** `faithfulness.best_faith`, der Anteil der LRE-Vorhersagen,
  die dem Top-1-Token des Modells im Relationsprompt mit acht Beispielen entsprechen.
  Dies ist keine Accuracy gegen die Goldantworten.
- **ŷ:** Vorhersage einer gewöhnlichen linearen Regression mit Intercept, über
  alle 28 Punkte gleich gewichtet und auf den ungerundeten x-/y-Werten gefittet.
- **Abweichung Δ:** `y − ŷ`; positiv bedeutet höhere, negativ niedrigere Faithfulness
  als die Gerade vorhersagt. Die letzte Spalte zeigt `100 × Δ` in Prozentpunkten.
  Es handelt sich um den vertikalen Abstand zur OLS-Geraden, entsprechend der
  Residuen-Diskussion im Paper. Ein Spearman-Koeffizient selbst definiert keine
  solche Gerade und keine einzelne „Abweichung von ρ“ pro Relation.

Die Ausgleichsgerade lautet:

```text
ŷ = 0.815096081183934 + 0.046492118027009 × x
Δ = y − ŷ
```

## Alle Relationen

`n Lens` und `n LRE` sind die gespeicherten Anzahlen ausgewerteter Testzeilen;
die beiden Messungen verwenden im `main`-Protokoll unabhängig erzeugte Splits.
Werte sind hier auf sechs Nachkommastellen gerundet. Die
[begleitende CSV](qwen3-8b-base-figure1.csv) enthält die Werte ohne zusätzliche
Rundung sowie die gewählten LRE-Layer, β-Werte und SHA-256-Prüfsummen der Eingabedateien.

| Relation | n Lens | n LRE | x | y | ŷ | Δ | Δ (Prozentpunkte) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `city_in_country` | 19 | 11 | -1.322219 | 0.818182 | 0.753623 | +0.064559 | +6.455851 |
| `company_ceo` | 20 | 100 | -4.347938 | 0.020000 | 0.612951 | -0.592951 | -59.295122 |
| `company_hq` | 20 | 100 | -3.340255 | 0.520000 | 0.659801 | -0.139801 | -13.980056 |
| `country_capital_city` | 16 | 8 | -4.253334 | 0.625000 | 0.617350 | +0.007650 | +0.765044 |
| `country_currency` | 20 | 14 | -4.797193 | 0.285714 | 0.592064 | -0.306350 | -30.635011 |
| `country_language` | 16 | 8 | -4.876662 | 0.750000 | 0.588370 | +0.161630 | +16.163028 |
| `country_largest_city` | 16 | 8 | -4.839913 | 0.750000 | 0.590078 | +0.159922 | +15.992174 |
| `landmark_in_country` | 20 | 100 | -2.705304 | 0.810000 | 0.689321 | +0.120679 | +12.067924 |
| `landmark_on_continent` | 20 | 100 | -3.904485 | 0.950000 | 0.633568 | +0.316432 | +31.643171 |
| `name_birthplace` | 20 | 15 | -1.992489 | 1.000000 | 0.722461 | +0.277539 | +27.753893 |
| `name_gender` | 11 | 3 | -5.151600 | 1.000000 | 0.575587 | +0.424413 | +42.441270 |
| `name_religion` | 20 | 15 | -4.076097 | 1.000000 | 0.625590 | +0.374410 | +37.441032 |
| `person_band_lead_singer` | 13 | 5 | -2.195900 | 0.600000 | 0.713004 | -0.113004 | -11.300406 |
| `person_father` | 20 | 100 | -4.620965 | 0.090000 | 0.600258 | -0.510258 | -51.025763 |
| `person_mother` | 20 | 100 | -4.036062 | 0.130000 | 0.627451 | -0.497451 | -49.745101 |
| `person_native_language` | 20 | 100 | -3.785382 | 0.860000 | 0.639106 | +0.220894 | +22.089435 |
| `person_occupation` | 20 | 100 | -3.449843 | 0.770000 | 0.654706 | +0.115294 | +11.529443 |
| `person_plays_instrument` | 20 | 100 | -3.062620 | 0.870000 | 0.672708 | +0.197292 | +19.729159 |
| `person_plays_position_in_sport` | 20 | 100 | -3.322126 | 0.740000 | 0.660643 | +0.079357 | +7.935659 |
| `person_plays_pro_sport` | 20 | 100 | -1.915230 | 0.970000 | 0.726053 | +0.243947 | +24.394703 |
| `person_university` | 20 | 75 | -4.148308 | 0.546667 | 0.622232 | -0.075566 | -7.556580 |
| `pokemon_evolutions` | 20 | 28 | -4.003584 | 0.285714 | 0.628961 | -0.343247 | -34.324671 |
| `presidents_birth_year` | 11 | 3 | -5.158730 | 1.000000 | 0.575256 | +0.424744 | +42.474422 |
| `presidents_election_year` | 11 | 3 | -5.143287 | 1.000000 | 0.575974 | +0.424026 | +42.402620 |
| `product_by_company` | 20 | 100 | -2.873180 | 0.630000 | 0.681516 | -0.051516 | -5.151585 |
| `star_constellation` | 20 | 100 | -4.361620 | 0.420000 | 0.612315 | -0.192315 | -19.231512 |
| `superhero_archnemesis` | 20 | 80 | -4.200516 | 0.250000 | 0.619805 | -0.369805 | -36.980521 |
| `superhero_person` | 20 | 84 | -2.597462 | 0.273810 | 0.694335 | -0.420525 | -42.052503 |

## Einordnung

Die größten negativen Abweichungen sind `company_ceo` (−59,30 Prozentpunkte),
`person_father` (−51,03) und `person_mother` (−49,75). Die größten positiven
Abweichungen sind `presidents_birth_year` (+42,47), `name_gender` (+42,44) und
`presidents_election_year` (+42,40). Diese drei positiven Ausreißer erreichen
jeweils Faithfulness 1,0 bei nur **drei LRE-Testzeilen**. Ein einzelner Treffer
entspricht dort bereits 33,33 Prozentpunkten.

Entfernt man explorativ genau diese drei Relationen mit weniger als vier
LRE-Testzeilen, steigt Spearman auf **ρ = 0,448336**, **p = 0,024593**, bei
25 Relationen. Das ist eine nachträgliche Sensitivitätsanalyse mit veränderter
Relationsauswahl. Die Tabelle und die Residuen oben bleiben auf allen 28
Relationen basiert; der Fit wurde dafür nicht verändert.

LRE-Layer und β wurden je Relation anhand derselben Testzeilen ausgewählt,
auf denen die Faithfulness berichtet wird (State-Indizes 6, 12, 16, 19, 22, 26;
β ∈ {0,5; 1; 1,5; 2; 3; 5}; acht Referenzbeispiele; ein Trial). Die Werte sind
deshalb In-Sample-Schätzungen nach Hyperparameterauswahl. Die Eingabedaten
enthalten teils doppelte Subjekte; Zeilenzahlen sind nicht notwendigerweise
Anzahlen eindeutiger Subjekte.

Das separate `shared_split`-Protokoll ergibt ρ = 0,267978, p = 0,195273 bei
25 einbezogenen Relationen. Seine Messpunkte werden hier nicht mit `main`
vermischt; Referenz- und Testauswahl unterscheiden sich zwischen den Protokollen.

## Herkunft und Reproduktion

Die Figure-1-Achsen und der Fit folgen
[`plot_relation_agreement.py`](../jlens/plot_relation_agreement.py); die
Residuen-Definition folgt [`rank_outliers.py`](../jlens/rank_outliers.py) und der
Residuen-Diskussion in [`paper/lens.tex`](../../paper/lens.tex).

Die Qwen-Messungen wurden aus dem lokalen Ergebnissnapshot
`code/results/five_model_sweep/qwen3_8b_base/main/<relation>.json` entnommen.
Diese Rohdateien und der lokale Sweep-Runner `code/jlens/model_sweep.py` waren
bei Erstellung noch nicht im Remote versioniert. Dieser Bericht veröffentlicht
daher einen eigenständigen CSV-Snapshot der für die Auswertung benötigten Werte.
Die Spalte `source_sha256` identifiziert die jeweilige lokale Rohdatei;
der CSV-Snapshot genügt zur Reproduktion der Korrelationen und Residuen,
enthält aber nicht sämtliche Token-Ränge oder Modellmessungen.

Bei der Erstellung wurden alle 28 x-Werte aus den gespeicherten einzelnen
`rank_diff`-Werten neu berechnet und gegen die Aggregate geprüft. Außerdem
wurde jeder y-Wert mit dem Maximum seiner gespeicherten LRE-Suchmatrix und
die Spearman-Korrelation mit der vorhandenen Sweep-Zusammenfassung abgeglichen.

Folgendes Skript reproduziert Fit, Korrelationen und Residuen aus der CSV;
es benötigt NumPy und SciPy und wird im Repository-Verzeichnis ausgeführt:

```python
import csv
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

path = Path("code/docs/qwen3-8b-base-figure1.csv")
with path.open() as handle:
    rows = list(csv.DictReader(handle))
x = np.array([float(row["x"]) for row in rows])
y = np.array([float(row["y"]) for row in rows])
b, a = np.polyfit(x, y, 1)
y_hat = a + b * x
residual = y - y_hat
np.testing.assert_allclose(
    y_hat, [float(row["y_hat"]) for row in rows], rtol=0, atol=1e-12
)
np.testing.assert_allclose(
    residual, [float(row["residual"]) for row in rows], rtol=0, atol=1e-12
)
assert abs(residual.sum()) < 1e-12
assert abs(x @ residual) < 1e-12
print("Intercept, slope:", a, b)
print("Spearman:", spearmanr(x, y))
print("Pearson:", pearsonr(x, y))
keep = np.array([int(row["n_lre"]) >= 4 for row in rows])
print("Sensitivity, n =", keep.sum(), spearmanr(x[keep], y[keep]))
```

Die [Detailauswertung für `company_ceo`](qwen3-8b-base-company-ceo-predictions.md) enthält alle 100 LRE- und alle 20 Lens-Ausgaben sowie eine Erklärung der Tabellenwerte.
