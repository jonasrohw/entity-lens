# Qwen3 8B Base: what the J-lens predicts for `name_gender`

**[View the top 100 predictions for every example](name_gender_top100.md)** · [Top-100 CSV](name_gender_top100.csv). The expanded export reproduces all values in this top-five summary exactly.

These are all **11 saved differential J-lens rows** behind the `name_gender` point in the [Figure 1 comparison](README.md), using the latest newline-prompt run. Sofia occurs twice in the saved evaluation and is retained twice here. There are 10 distinct names; no examples were removed or rerun.

[Download the table as CSV](name_gender_predictions.csv) · [Exact measurement snapshot](name_gender_measurement.json)

| Row | Name | Model target | J-lens #1 | #2 | #3 | #4 | #5 | Target rank (0-based) |
| ---: | --- | --- | --- | --- | --- | --- | --- | ---: |
| 1 | Tom | `man` | `萝卜` | `法国` | `**` | `苹果` | `**` | 60,513 |
| 2 | Mia | `woman` | `herself` | `her` | `她的` | `让她` | `她在` | 108,458 |
| 3 | Sofia | `woman` | `Sofia` | `Bulgaria` | `herself` | `Elena` | `her` | 9,365 |
| 4 | Connor | `man` | `Connor` | `Conor` | `Connor` | `Brendan` | `Quinn` | 101,074 |
| 5 | Evan | `man` | `Vermont` | `indie` | `Evan` | `Winnipeg` | `surfing` | 43,448 |
| 6 | Sofia | `woman` | `Sofia` | `Bulgaria` | `herself` | `Elena` | `her` | 9,365 |
| 7 | Natalie | `woman` | `herself` | `her` | `Natalie` | `Hawaiian` | `Newfoundland` | 22,301 |
| 8 | David | `man` | `french` | `kosher` | `Hawaiian` | `Argentine` | `Judaism` | 11,338 |
| 9 | Dylan | `man` | `Dylan` | `Welsh` | `Belfast` | `Swansea` | `Newfoundland` | 110,309 |
| 10 | Emily | `woman` | `Emily` | `Emily` | `她的` | `her` | `herself` | 98,417 |
| 11 | Michael | `man` | `Austrian` | `Michael` | `Wolfgang` | `Jewish` | `beach` | 30,483 |

## How to read the table

- **Model:** `Qwen/Qwen3-8B-Base`, main protocol, differential J-lens at state 19.
- **J-lens prompt:** `\n{name} is known for`, with an actual newline before the name. The J-lens receives this neutral prompt.
- **Model target:** the saved full model’s top-1 token under the relation prompt `{name} is usually a name for a`, with the evaluation’s in-context examples. It is the token whose J-lens rank is measured.
- **J-lens #1–#5:** the five highest-ranked vocabulary tokens from the differential readout (`top_diff`), in order. These are individual decoded tokens, not a generated five-word sentence or a probability distribution.
- **Target rank:** the saved zero-based `rank_diff`; 0 would mean a top-1 match. Decoded strings, including Chinese tokens and repeated strings, are preserved exactly as recorded. The snapshot does not provide token IDs for the top-five predictions, so identical displayed strings should not be deduplicated.

## Why this point is an outlier

| Quantity | Value |
| --- | ---: |
| J-lens top-1 agreement with the model target | 0 / 11 |
| J-lens top-10 agreement with the model target | 0 / 11 |
| Median target rank (0-based) | 43,448 |
| Plotted representation score | −4.6379797854 |
| Plotted LRE faithfulness | 1.0 (3 / 3 LRE test rows) |

The lens sometimes ranks gender-related words highly: `herself` and `her` appear for Mia, Sofia, Natalie, and Emily. However, the evaluated target is the exact token `woman` or `man`, whose rank is much lower. Thus the low exact-token score does not by itself establish that the representation lacks gender-related information.

The LRE faithfulness value comes from a separate three-row evaluation (Benjamin, Lucas, Scarlett), with selected LRE layer 12 and β = 5. It is not LRE accuracy on the 11 J-lens rows above. The two quantities use different test rows in the main protocol.

## Provenance

Extracted from `code/results/all_model_newline/qwen3_8b_base/main/name_gender.json`. The included measurement snapshot is a byte-for-byte copy of that source. Its hash matches the source hash in the plotted relation CSV; the representation score was independently recomputed from all 11 target ranks.

```text
SHA-256: 63a9111b4533010d41c783e1b7951e98918eefd6fbabc6fc70ab1363da3b6153
```
