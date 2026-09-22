# Qwen3 8B Base: konkrete Vorhersagen für `company_ceo`

**Update, 21. September 2026:** Die Lens-Ausgaben unten verwenden den ursprünglichen
Prompt ohne Präfix. Der [vollständige Newline-Rerun](../results/qwen_newline_rerun/README.md)
ersetzt diese Lens-Messung; die [neuen `company_ceo`-Ausgaben](../results/qwen_newline_rerun/main/newline_known_for/company_ceo.json)
enthalten alle korrigierten Testzeilen. Die LRE-Messung bleibt unverändert.

Die LRE stimmt im Figure-1-`main`-Protokoll bei **2 von 100 Firmen (2 %)**
mit dem ersten Antworttoken des vollständigen Modells überein.
Die differentielle Lens erreicht **0 von 20 Top-1- und Top-10-Treffern**.
Der mediane gespeicherte Lens-Rang ist 23.565,5 (nullbasiert).

Im [Figure-1-Writeup](qwen3-8b-base-figure1.md) bedeutet die Zeile:

| Größe | Wert | Bedeutung |
| --- | ---: | --- |
| n Lens | 20 | Ausgewertete Lens-Testzeilen |
| n LRE | 100 | Ausgewertete LRE-Testzeilen |
| x | −4,347938 | Negativer medianer log10(rank + 1) der Lens |
| y | 0,020000 | Tatsächliche LRE-Faithfulness: 2 % |
| ŷ | 0,612951 | Von der über alle Relationen gefitteten Geraden erwartete Faithfulness |
| y − ŷ | −0,592951 | LRE liegt 59,30 Prozentpunkte unter der Geraden |

**Die 0,612951 sind keine gemessene Modell- oder LRE-Genauigkeit.**
Die LRE extrahiert hier die Modellantworten nur selten; auch die Lens platziert
das Zieltoken weit hinten. Das vollständige Modell selbst trifft auf den
100 LRE-Testzeilen in 49 Fällen das erste Token der im Datensatz gespeicherten
Antwort. Auch das ist keine Prüfung vollständiger CEO-Namen oder ihrer Aktualität.

## Was hier als Vorhersage gezeigt wird

- Alle Ausgaben sind **einzelne Tokens**, keine generierten vollständigen Namen.
  Beispielsweise können `Lak`, `Rein` oder `Sat` den Anfang eines Namens bilden.
- **Modell im CEO-Prompt:** Top-1-Fortsetzung von `{}'s CEO is` nach acht
  Demonstrationsbeispielen. Dieses Token ist das Faithfulness-Ziel.
- **LRE:** Ausgabe der relationsspezifischen LRE im ursprünglichen LRE-Split,
  am ausgewählten State-Index 16 und mit β = 5. Die konkreten Tokens wurden
  erneut berechnet; alle 100 gespeicherten Treffer/Fehler wurden exakt reproduziert.
- **Differentielle Lens:** gespeicherte Top-5-Ausgaben am festen State-Index 19
  unter dem neutralen Prompt `{} is known for`. Ihre 20 Testfirmen stammen
  aus einem anderen Split mit anderen Demonstrations- und Referenzbeispielen.
  Nur acht Firmen kommen in beiden Testmengen vor.
- Die im Rohsnapshot unter `lens.rows[].top_lre` gespeicherte Ausgabe ist eine
  zusätzliche **affine Lens unter dem neutralen Prompt, β = 1**. Sie ist nicht
  die relationsspezifische LRE, deren Faithfulness 0,02 beträgt. Die LRE-Spalten
  in diesem Bericht enthalten deshalb die neu ausgelesene relationsspezifische LRE.
- Die Tabellen entfernen äußere Leerzeichen zur Lesbarkeit. Im
  [JSON-Snapshot](qwen3-8b-base-company-ceo-predictions.json) bleiben die neu decodierten LRE-Tokens einschließlich
  Leerzeichen erhalten; die Lens-Tokens entsprechen dem gespeicherten Snapshot.

## Die acht gemeinsamen Firmen

Die beiden Zieltoken-Spalten gehören zum jeweiligen CEO-Prompt-Kontext.
Die Firmen stimmen überein, die Prompt-Kontexte der beiden Splits unterscheiden sich.

| Firma | Modellziel im LRE-Split | LRE Top-1 | Modellziel im Lens-Split | Differentielle Lens Top-5 |
| --- | --- | --- | --- | --- |
| ArcelorMittal | `Lak` | `J` | `Lak` | `钢铁`, `钢材`, `steel`, `钢厂`, `steel` |
| Aspen Institute | `David` | `Michael` | `David` | `philanth`, `socially`, `social`, `nonprofits`, `humanitarian` |
| Institut de chimie | `Jean` | `R` | `Jean` | `scientific`, `Chemistry`, `chemistry`, `科学家`, `自然科学` |
| Seagate Technology | `Bill` | `Martin` | `Bill` | `aerospace`, `Garmin`, `Lockheed`, `Motorola`, `NASA` |
| Cloudflare | `Matthew` | `Thomas` | `Matthew` | `网络安全`, `黑客`, `.cloudflare`, `Geek`, `Lightning` |
| Infineon Technologies | `Rein` | `J` | `Rein` | `Electronics`, `semiconductor`, `Semiconductor`, `electronics`, `aerospace` |
| Southwest Airlines | `Gary` | `Michael` | `Gary` | `Airlines`, `飞行员`, `employee`, `飞行`, `airline` |
| Qantas | `Alan` | `Adrian` | `Alan` | `航空`, `airline`, `飞行`, `飞机`, `flight` |

## Alle 20 Lens-Vorhersagen

Der Zielrang ist nullbasiert: 0 wäre ein Top-1-Treffer.

| Firma | Modellziel im CEO-Prompt (Lens-Split) | Differentielle Lens Top-5 | Zielrang in der Lens |
| --- | --- | --- | ---: |
| BASF | `Martin` | `chemical`, `chemicals`, `化学`, `化学品`, `Chemical` | 51977 |
| ArcelorMittal | `Lak` | `钢铁`, `钢材`, `steel`, `钢厂`, `steel` | 36490 |
| Nederlandse Spoorwegen | `K` | `Rail`, `运输`, `铁路`, `乘客`, `列车` | 58756 |
| Aspen Institute | `David` | `philanth`, `socially`, `social`, `nonprofits`, `humanitarian` | 13976 |
| Institut de chimie | `Jean` | `scientific`, `Chemistry`, `chemistry`, `科学家`, `自然科学` | 1498 |
| Procter & Gamble | `David` | `洗衣`, `家庭`, `奶粉`, `Coca`, `shampoo` | 119643 |
| Seagate Technology | `Bill` | `aerospace`, `Garmin`, `Lockheed`, `Motorola`, `NASA` | 15743 |
| Deutsche Post AG | `Frank` | `Deutsche`, `Siemens`, `欧元`, `Renault`, `Ernst` | 8456 |
| Bell Resources | `John` | `Petroleum`, `油田`, `开采`, `Mineral`, `油气` | 15889 |
| Cloudflare | `Matthew` | `网络安全`, `黑客`, `.cloudflare`, `Geek`, `Lightning` | 86433 |
| Comcast | `Brian` | `Comcast`, `侵犯`, `Cable`, `电视`, `电费` | 79434 |
| Infineon Technologies | `Rein` | `Electronics`, `semiconductor`, `Semiconductor`, `electronics`, `aerospace` | 2773 |
| Boeing | `David` | `Boeing`, `aircraft`, `aerospace`, `飞机`, `aviation` | 92254 |
| Kroger | `Rodney` | `grocery`, `Grocery`, `超市`, `groceries`, `supermarket` | 12521 |
| Southwest Airlines | `Gary` | `Airlines`, `飞行员`, `employee`, `飞行`, `airline` | 74296 |
| Qantas | `Alan` | `航空`, `airline`, `飞行`, `飞机`, `flight` | 106029 |
| Caterpillar Inc. | `Jim` | `aerospace`, `Boeing`, `工程机械`, `employee`, `Aerospace` | 31242 |
| Koninklijke Philips NV | `Fr` | `Renault`, `欧元`, `Nokia`, `Tesla`, `Tesla` | 3316 |
| Škoda Transportation | `Jan` | `Vehicle`, `vehicle`, `Truck`, `Trucks`, `Rail` | 7670 |
| Meta Platforms | `Mark` | `Facebook`, `Oculus`, `Zuckerberg`, `Snapchat`, `LinkedIn` | 12471 |

## Alle 100 LRE-Vorhersagen

„Treffer“ bedeutet Gleichheit mit dem Modellziel im selben CEO-Prompt.
Die einzigen Treffer sind **Yle** und **SpaceX**. Ein Treffer bezieht sich
auf ein Token und belegt nicht die korrekte Vorhersage des vollständigen Namens.
Der [JSON-Snapshot](qwen3-8b-base-company-ceo-predictions.json) enthält zusätzlich die LRE-Top-5, Token-IDs,
Zielränge, Datensatzantworten sowie die Demonstrations- und Referenzbeispiele.

| Firma | Modellziel im CEO-Prompt | LRE Top-1 | Treffer |
| --- | --- | --- | --- |
| Tesco | `Dave` | `Ren` | nein |
| Czech Airlines | `Jan` | `R` | nein |
| Johnson & Johnson | `Alex` | `Robert` | nein |
| U.S. Steel | `David` | `Gi` | nein |
| University of Leeds | `David` | `R` | nein |
| Kering | `François` | `Philippe` | nein |
| thyssenkrupp AG | `Matthias` | `J` | nein |
| Seagate Technology | `Bill` | `Martin` | nein |
| Yle | `J` | `J` | ja |
| UBS | `Ralph` | `Adrian` | nein |
| ConocoPhillips | `Jim` | `Martin` | nein |
| Cisco Systems | `Chuck` | `Martin` | nein |
| SAP | `Christian` | `Mark` | nein |
| Valeo | `Thi` | `J` | nein |
| Goldman Sachs & Co. (US) | `David` | `R` | nein |
| Sanofi | `Paul` | `J` | nein |
| PepsiCo | `Ram` | `Fr` | nein |
| Warner Music Group | `Edgar` | `J` | nein |
| The Home Depot | `Craig` | `John` | nein |
| GMV Innovating Solutions | `David` | `R` | nein |
| Abbott Laboratories | `Robert` | `Martin` | nein |
| České dráhy | `Jan` | `J` | nein |
| Heineken | `Ar` | `J` | nein |
| Raytheon Company | `Thomas` | `Martin` | nein |
| Alphabet | `Sund` | `Lauren` | nein |
| Bertelsmann | `Thomas` | `Stef` | nein |
| Klinikum Ernst von Bergmann | `Dr` | `R` | nein |
| GlaxoSmithKline | `Emma` | `J` | nein |
| Activision Blizzard | `Bobby` | `Robert` | nein |
| Toyota | `Ak` | `Luca` | nein |
| DuPont | `Ellen` | `Martin` | nein |
| Intel | `Pat` | `Mark` | nein |
| Capgemini | `Arm` | `Philippe` | nein |
| British Airways | `Alex` | `Adrian` | nein |
| Airbus | `Gu` | `Fab` | nein |
| Accenture | `Julie` | `Urs` | nein |
| Colgate-Palmolive Company | `Paul` | `Martin` | nein |
| Carrefour Group | `Alexandre` | `J` | nein |
| Opera Software | `Jon` | `J` | nein |
| AC Sparta Prague | `Milan` | `J` | nein |
| ArcelorMittal | `Lak` | `J` | nein |
| Hewlett-Packard | `Meg` | `Mark` | nein |
| Nokia | `P` | `Marco` | nein |
| Vodafone | `Nick` | `Ren` | nein |
| Valve Corporation | `G` | `T` | nein |
| Sony | `Ken` | `Ren` | nein |
| Audi | `Markus` | `Marco` | nein |
| Southwest Airlines | `Gary` | `Michael` | nein |
| Électricité de France | `Vincent` | `J` | nein |
| Qantas | `Alan` | `Adrian` | nein |
| Bristol-Myers Squibb | `Giovanni` | `Martin` | nein |
| FC Bayern Munich | `Karl` | `J` | nein |
| Gilead Sciences | `John` | `Jan` | nein |
| STMicroelectronics | `Jean` | `R` | nein |
| Nintendo | `Tat` | `Fr` | nein |
| Aspen Institute | `David` | `Michael` | nein |
| PayPal | `Dan` | `Tom` | nein |
| Merck & Co. | `Kenneth` | `R` | nein |
| Rheinmetall AG | `R` | `J` | nein |
| Shell | `Ben` | `Adrian` | nein |
| Airbus Group | `Gu` | `J` | nein |
| McDonald’s | `Chris` | `J` | nein |
| University of Gothenburg | `Jan` | `H` | nein |
| Paramount Global | `Bob` | `Milan` | nein |
| Biogen | `Michel` | `Mark` | nein |
| Aeroflot | `Alexander` | `J` | nein |
| East Japan Railway Company | `T` | `J` | nein |
| NVIDIA | `Jensen` | `Mark` | nein |
| Institut de chimie | `Jean` | `R` | nein |
| SpaceX | `Elon` | `Elon` | ja |
| BNP Paribas | `Jean` | `J` | nein |
| Hyundai Motor Company | `E` | `J` | nein |
| Hasbro | `Brian` | `Frank` | nein |
| Toulouse-Blagnac | `Jean` | `R` | nein |
| Orange | `St` | `J` | nein |
| Radio Free Europe/Radio Liberty | `Michael` | `Mari` | nein |
| Simon & Schuster | `Jonathan` | `Michael` | nein |
| Microsoft | `Sat` | `Jes` | nein |
| Ford Motor Company | `Jim` | `J` | nein |
| Yum! Brands | `Greg` | `R` | nein |
| Berkshire Hathaway | `Warren` | `David` | nein |
| Salesforce | `Marc` | `Mark` | nein |
| Broderbund | `Mark` | `R` | nein |
| IBM | `Ar` | `Mark` | nein |
| Deutsche Telekom | `Tim` | `J` | nein |
| Cloudflare | `Matthew` | `Thomas` | nein |
| Energetický a průmyslový holding | `Jan` | `R` | nein |
| DR | `David` | `K` | nein |
| Chevron Corporation | `Mike` | `T` | nein |
| Eli Lilly and Company | `David` | `Martin` | nein |
| Electrolux | `Peter` | `Hans` | nein |
| Dell Inc. | `Michael` | `R` | nein |
| Yandex | `I` | `Ren` | nein |
| Humana | `Bruce` | `Robert` | nein |
| FC Viktoria Köln | `Andreas` | `J` | nein |
| Verizon | `Hans` | `Martin` | nein |
| BT Group | `Gavin` | `J` | nein |
| Infineon Technologies | `Rein` | `J` | nein |
| Peterson Institute for International Economics | `Bruce` | `S` | nein |
| Publicis | `Arthur` | `Philippe` | nein |

## Herkunft und Prüfung

Modell: `Qwen/Qwen3-8B-Base`; Revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; Float32; Eager Attention.
Die LRE wurde auf `rho1` aus denselben acht Referenzbeispielen mit dem
ursprünglichen Batch-Size-Wert 2 neu berechnet. Es wurde ausschließlich die
gespeicherte ausgewählte Kombination aus Layer 16 und β = 5 ausgewertet;
es fand keine erneute Hyperparametersuche statt.

Der ursprüngliche Snapshot liegt lokal unter
`code/results/five_model_sweep/qwen3_8b_base/main/company_ceo.json`.
Die Ausleseskripte und das Laufprotokoll liegen unter
`code/generated/qwen3_company_ceo/`. Der veröffentlichte JSON-Snapshot
enthält die SHA-256-Prüfsummen der ursprünglichen Messung, der Relationsdaten,
des verwendeten Sweep-Moduls und des Ausleseskripts.

Geprüft wurden der identische Split-Fingerprint `284343630b31`, die Reihenfolge
aller 100 Testfirmen, alle 100 ursprünglichen LRE-Treffer/Fehler, die resultierende
Faithfulness von 0,02 und sämtliche Modell-Zieltokens gegen den vorhandenen
separaten Target-Audit. Die bestehenden numerischen Prüfungen des Sweep-Runners
wurden vor der Messung ebenfalls ausgeführt. Die 20 Lens-Zeilen wurden unverändert
aus dem ursprünglichen Snapshot übernommen.
