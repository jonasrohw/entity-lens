# Qwen3 8B Base: country-capital predictions with and without subtraction

Relation: `country_capital_city`. These are the 16 scored countries from the completed [raw J-lens experiment](qwen3-8b-base-raw-jlens.md), measured on 20 September 2026. This report exports the retained results; it does not rerun or change the experiment.

**Without subtraction at the subject position, 4/16 capital targets are in the top 10, versus 2/16 with subtraction. Neither has a top-1 hit, and both still predict `-Semit` first for the same ten countries.** At the final `for` position, neither variant places a target in the top 10.

The raw subject-position top-10 cases are Peru → `Lima` (rank 2), Chile → `Santiago` (rank 4), Saudi Arabia → `Riyadh` (rank 2), and South Korea → `Seoul` (rank 2). The differential lens has only the latter two in the top 10; its Peru/Chile ranks are 11/10. All ranks below are zero-based: top 10 means rank < 10.

[Prediction CSV: tokens and IDs](qwen3-8b-base-country-capital-predictions.csv) · [Per-reference contributions CSV](qwen3-8b-base-country-capital-predictions-contributions.csv) · [Full relation JSON](qwen3-8b-base-country-capital-predictions.json).

## Protocol and interpretation

- **Differential:** `mean_{i != e} J_i (h_e − h_i)`. **Raw / no subtraction:** `mean_{i != e} J_i h_e`. Both average the same references and exclude the scored country itself. Raw does not add an affine reference output.
- **Model and layer:** `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, state 19 (output of block 18).
- **Lens input:** `{country} is known for`, without a prefix. At each position, both cached input states and Jacobian interventions use that position. Output/readout is at the final prompt token, with the native final normalization and output head.
- **Target:** the model's first answer token under `The capital city of {country} is`, after the eight demonstrations shown below. A target such as `Bras`, `Car`, `Buenos`, or `New` is a token fragment, not a complete capital name. Ranks measure agreement with that model token, not complete-answer correctness.
- **LRE faithfulness:** retained y = 0.625, from the separate relation-specific LRE evaluation. The predictions below are neutral-prompt lenses, not those relation-specific LRE predictions.
- **Token labels:** the retained top-token strings have outer whitespace stripped. Distinct token IDs may therefore have identical displayed text (for example `Saudi`). The CSV and JSON preserve every target and top-five token ID. Dataset answers and ICL strings are retained verbatim, including literal Unicode escape sequences.
- **Same outputs as country-language:** these two relations share the 16 subjects, eight references, neutral prompts, and layer. All four variants' top-five token IDs match the country-language run exactly; only the relation-specific target tokens and their ranks differ. This is expected for a relation-agnostic lens.

References, in their averaging order: `Australia`, `Argentina`, `Egypt`, `Peru`, `Japan`, `United States`, `China`, `Brazil`. Reference countries receive seven contributions; other countries receive eight.

## Relation summaries

The fitted y and residual use the corresponding variant's OLS fit over all 26 relations, not a fit to these 16 countries. x is `−median(log10(rank + 1))`; residual is y − fitted y.

| Position | Readout | Median rank | Top 1 | Top 10 | `-Semit` top 1 | x | Fitted y | Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Last subject token | diff | 18328.5 | 0/16 | 2/16 | 10/16 | -4.253334 | 0.553625 | +0.071375 |
| Last subject token | raw | 24494.5 | 0/16 | 4/16 | 10/16 | -4.388193 | 0.537988 | +0.087012 |
| Final prompt token `for` | diff | 67564.5 | 0/16 | 0/16 | 0/16 | -4.829306 | 0.613049 | +0.011951 |
| Final prompt token `for` | raw | 19900.5 | 0/16 | 0/16 | 0/16 | -4.298522 | 0.577678 | +0.047322 |

## Last subject token: all predictions

Top-five lists are ordered from highest to lowest score.

| Country | Model target | Differential top 5 | Target rank | Raw top 5 (no subtraction) | Target rank |
| --- | --- | --- | ---: | --- | ---: |
| Australia | `Canberra` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 22182 | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 26064 |
| Argentina | `Buenos` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 55413 | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `(MigrationBuilder` | 56842 |
| Egypt | `Cairo` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 14475 | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 15051 |
| Peru | `Lima` | `Peru`, `Guatemala`, `Honduras`, `Montana`, `Oregon` | 11 | `Peru`, `Guatemala`, `Lima`, `peru`, `Ecuador` | 2 |
| Japan | `Tokyo` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 97273 | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 98534 |
| United States | `Washington` | `Democr`, `Equality`, `Baltimore`, `Attorney`, `Liberals` | 43 | `美方`, `Федер`, `Democr`, `Judiciary`, `美军` | 54 |
| China | `Beijing` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` | 77252 | `-Semit`, `(Initialized`, `iNdEx`, `(MigrationBuilder`, `StringWriter` | 79295 |
| Brazil | `Bras` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` | 147047 | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `(MigrationBuilder` | 147247 |
| Venezuela | `Car` | `Venezuela`, `Venezuelan`, `Maduro`, `Venezuel`, `venez` | 11086 | `Venezuela`, `Venezuelan`, `Maduro`, `Venezuel`, `venez` | 22925 |
| India | `New` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 134784 | `-Semit`, `(Initialized`, `iNdEx`, `(MigrationBuilder`, `StringWriter` | 135676 |
| Chile | `Santiago` | `Chile`, `Chili`, `Monterey`, `Oregon`, `iod` | 10 | `Chile`, `Chili`, `Monterey`, `Oregon`, `Santiago` | 4 |
| Saudi Arabia | `Riyadh` | `Saudi`, `Riyadh`, `Saudi`, `Saudis`, `沙特` | 1 | `Saudi`, `Saudi`, `Riyadh`, `Saudis`, `沙特` | 2 |
| Germany | `Berlin` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 123010 | `-Semit`, `(Initialized`, `iNdEx`, `(MigrationBuilder`, `StringWriter` | 124913 |
| Spain | `Madrid` | `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter` | 92610 | `-Semit`, `(Initialized`, `iNdEx`, `(MigrationBuilder`, `StringWriter` | 93412 |
| Russia | `Moscow` | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `搜狐首页` | 9270 | `-Semit`, `(Initialized`, `iNdEx`, `StringWriter`, `(MigrationBuilder` | 10160 |
| South Korea | `Seoul` | `Korea`, `Seoul`, `Koreans`, `Samsung`, `Hyundai` | 1 | `Koreans`, `Korea`, `Seoul`, `Samsung`, `Korean` | 2 |

## Final prompt token `for`: all predictions

Top-five lists are ordered from highest to lowest score.

| Country | Model target | Differential top 5 | Target rank | Raw top 5 (no subtraction) | Target rank |
| --- | --- | --- | ---: | --- | ---: |
| Australia | `Canberra` | `什么时候`, `...`, `...]`, `headaches`, `instability` | 73809 | `Mach`, `...`, `her`, `corruption`, `their` | 77996 |
| Argentina | `Buenos` | `什么`, `:P`, `Hat`, `...]`, `吃什么` | 110313 | `Mach`, `delicious`, `vibrant`, `...`, `what` | 33841 |
| Egypt | `Cairo` | `结合`, `sights`, `购物`, `看点`, `温和` | 103994 | `sights`, `adventure`, `cruise`, `relaxed`, `nightlife` | 96622 |
| Peru | `Lima` | `纯净`, `防护`, `Boutique`, `:null`, `>Lorem` | 105480 | `adventure`, `relaxed`, `cruise`, `sustainability`, `architecture` | 13638 |
| Japan | `Tokyo` | `吃什么`, `which`, `?`, `Which`, `...` | 10732 | `Mach`, `their`, `its`, `her`, `delicious` | 20715 |
| United States | `Washington` | `Hat`, `their`, `Nag`, `M`, `Trev` | 41780 | `their`, `Mach`, `tourism`, `her`, `it` | 19086 |
| China | `Beijing` | `哪`, `哪个`, `；`, `Which`, `（` | 88165 | `her`, `exporting`, `producing`, `cheese`, `peanuts` | 32703 |
| Brazil | `Bras` | `Which`, `哪个`, `哪一个`, `哪`, `哪种` | 61686 | `what`, `which`, `...`, `Mach`, `vibrant` | 52624 |
| Venezuela | `Car` | `安全感`, `购物`, `结合`, `介`, `安全` | 78577 | `relaxed`, `adventure`, `Mach`, `cruise`, `traveler` | 2404 |
| India | `New` | `Hat`, `Trev`, `their`, `peanuts`, `Nag` | 39967 | `her`, `their`, `his`, `Mach`, `producing` | 15597 |
| Chile | `Santiago` | `Hat`, `Trader`, `Trader`, `:Int`, `:P` | 48410 | `growing`, `coffee`, `delicious`, `sustainability`, `sustainable` | 3337 |
| Saudi Arabia | `Riyadh` | `-`, `–`, `??`, `...',`, `-` | 58390 | `Mach`, `corruption`, `corrupt`, `what`, `tourism` | 17597 |
| Germany | `Berlin` | `什么时候`, `...`, `吃什么`, `...]`, `Hat` | 12866 | `Mach`, `her`, `...`, `corruption`, `their` | 11706 |
| Spain | `Madrid` | `:P`, `:Int`, `:*`, `什么时候`, `:` | 64598 | `Mach`, `vibrant`, `delicious`, `adventure`, `tourism` | 9376 |
| Russia | `Moscow` | `peanuts`, `Hat`, `instability`, `Nag`, `Trev` | 92818 | `their`, `producing`, `her`, `cheese`, `its` | 24372 |
| South Korea | `Seoul` | `:T`, `:L`, `:Int`, `:P`, `:null` | 70531 | `Mach`, `adventure`, `relaxed`, `tourism`, `cruise` | 31051 |

## Input positions and norms

The subject position is the last token of the country name, not necessarily position 0. Position indices are zero-based.

| Country | Dataset answer (verbatim) | Model target ID | Subject position | Subject norm | Final position | Final norm | References |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Australia | `Canberra` | 68790 | 0 | 10631.791 | 3 | 109.630 | 7 |
| Argentina | `Buenos Aires` | 68073 | 0 | 10878.408 | 3 | 103.937 | 7 |
| Egypt | `Cairo` | 52550 | 0 | 11176.016 | 3 | 113.424 | 7 |
| Peru | `Lima` | 61953 | 1 | 136.595 | 4 | 111.937 | 7 |
| Japan | `Tokyo` | 26194 | 0 | 11239.146 | 3 | 110.322 | 7 |
| United States | `Washington D.C.` | 6515 | 1 | 128.656 | 4 | 105.011 | 7 |
| China | `Beijing` | 26549 | 0 | 10106.265 | 3 | 107.919 | 7 |
| Brazil | `Bras\u00edlia` | 61124 | 0 | 11013.235 | 3 | 105.456 | 7 |
| Venezuela | `Caracas` | 3261 | 2 | 140.971 | 5 | 112.149 | 8 |
| India | `New Delhi` | 1532 | 0 | 10390.155 | 3 | 110.913 | 8 |
| Chile | `Santiago` | 54578 | 1 | 132.840 | 4 | 105.878 | 8 |
| Saudi Arabia | `Riyadh` | 89790 | 1 | 134.540 | 4 | 105.000 | 8 |
| Germany | `Berlin` | 19846 | 0 | 11070.800 | 3 | 106.869 | 8 |
| Spain | `Madrid` | 24081 | 0 | 11184.160 | 3 | 105.413 | 8 |
| Russia | `Moscow` | 22415 | 0 | 10834.375 | 3 | 108.825 | 8 |
| South Korea | `Seoul` | 50189 | 1 | 128.417 | 4 | 106.925 | 8 |

## Reference contributions for Germany

Germany is not among the references, so each contribution has weight 1/8. These are norms before averaging and decoding; norms themselves are not additive. The Peru and United States subject-position contributions remain very large without subtraction. The downloadable contributions CSV contains the equivalent rows for every scored country and both positions.

| Position | Reference | Reference position | Reference-state norm | Differential JVP norm | Raw JVP norm | J_i h_i norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| subject | Australia | 0 | 10631.791 | 9.060 | 37.661 | 37.365 |
| subject | Argentina | 0 | 10878.408 | 10.088 | 38.873 | 39.795 |
| subject | Egypt | 0 | 11176.016 | 9.934 | 29.635 | 26.835 |
| subject | Peru | 1 | 136.595 | 7170.070 | 7173.538 | 72.497 |
| subject | Japan | 0 | 11239.146 | 7.383 | 36.083 | 37.110 |
| subject | United States | 1 | 128.656 | 4933.006 | 4938.585 | 49.286 |
| subject | China | 0 | 10106.265 | 10.034 | 42.054 | 43.024 |
| subject | Brazil | 0 | 11013.235 | 8.310 | 34.181 | 34.146 |
| last | Australia | 3 | 109.630 | 79.398 | 410.378 | 411.773 |
| last | Argentina | 3 | 103.937 | 172.327 | 446.024 | 378.135 |
| last | Egypt | 3 | 113.424 | 1126.889 | 1115.238 | 567.417 |
| last | Peru | 4 | 111.937 | 1023.706 | 1050.594 | 406.460 |
| last | Japan | 3 | 110.322 | 86.243 | 414.581 | 438.471 |
| last | United States | 4 | 105.011 | 212.689 | 333.066 | 344.871 |
| last | China | 3 | 107.919 | 296.315 | 417.956 | 402.266 |
| last | Brazil | 3 | 105.456 | 157.949 | 413.166 | 382.573 |

## Exact target-prompt context

Append `The capital city of {country} is` to this context to obtain that country's target prompt. Literal backslashes below are part of the retained dataset strings.

```text
The capital city of Nigeria is Abuja
The capital city of Turkey is Ankara
The capital city of Mexico is Mexico City
The capital city of Canada is Ottawa
The capital city of Italy is Rome
The capital city of France is Paris
The capital city of Colombia is Bogot\u00e1
The capital city of Pakistan is Islamabad
```

## Reproduction and checks

The exporter reads only versioned snapshots. It checks all 64 prediction rows against frozen subjects/target IDs/reference counts, recomputes the aggregates from individual ranks, and exports all 240 included reference contributions. It also verifies the shared top-five token IDs against country-language. The underlying GPU run passed raw-JVP finite-difference and batched/scalar checks; its 982 differential control ranks exactly reproduced the preceding position experiment.

```sh
cd code
uv run python ../paper/repro/report_qwen_country_capital.py
```

[Exporter](../../paper/repro/report_qwen_country_capital.py) · [Source measurements](qwen3-8b-base-raw-jlens.json) · [Frozen split/targets](last-position-lre-correlation-inputs.json). Source hashes and the full retained relation measurement are included in the relation JSON.
