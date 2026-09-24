# GPT-J and Pythia: full-row newline investigation

Original 28; all 9747 rows/model; 8 demos and 8 refs per method; original continent refs; gender alternatives; years included; neutral newline only; LRE and targets reused.

These are in-sample scores: demonstrations and reference rows are also evaluated. No relation or row was removed. All LRE grids, chosen layers/betas and target IDs were reused unchanged. Only the neutral lens prompt changed from `{} is known for` to `\n{} is known for`.

| Model | Original ρ | Newline ρ | Δρ | Frozen original LRE ρ | Frozen newline LRE ρ |
|---|---:|---:|---:|---:|---:|
| GPT-J 6B | 0.188287 | 0.434592 | +0.246305 | 0.199781 | 0.460865 |
| Pythia 6.9B | 0.174603 | 0.502738 | +0.328135 | 0.134410 | 0.402629 |

The leading newline substantially improves both correlations and removes large subject-state norm spikes at the first token. The strongest lens gains occur in country-language, capital-city and largest-city relations. The evidence supports a beginning-of-prompt sensitivity; it does not isolate the responsible attention heads or fully explain every remaining outlier.

University remains a shared outlier because LRE achieves high faithfulness by predicting the constant first token `University`. Gender remains difficult for the neutral lens even with the accepted wording set. Different LRE/lens demonstration contexts are another source of disagreement, especially for occupation and sports position; those contexts were preserved for this comparison.

## GPT-J 6B

Largest original rank mismatches; rank 1 is best. Median token ranks below are one-based.

| Relation | Rows | LRE faithfulness | Original lens median rank | Newline lens median rank | Original rank gap | Newline rank gap |
|---|---:|---:|---:|---:|---:|---:|
| country_language | 24 | 0.9583 | 4684.0 | 10.0 | +21.0 | +2.0 |
| name_gender | 19 | 1.0000 | 2193.0 | 282.0 | +19.0 | +16.0 |
| person_university | 91 | 0.7912 | 12210.0 | 13611.0 | +18.0 | +18.0 |
| product_by_company | 522 | 0.5153 | 18.0 | 17.0 | -16.0 | -12.0 |
| person_band_lead_singer | 21 | 0.7143 | 16.0 | 39.0 | -14.0 | -8.0 |
| person_father | 991 | 0.1504 | 351.0 | 437.0 | -13.0 | -9.0 |
| landmark_in_country | 836 | 0.7584 | 7.0 | 6.0 | -12.0 | -10.0 |
| star_constellation | 362 | 0.4641 | 153.5 | 174.0 | -12.0 | -6.0 |
| person_plays_position_in_sport | 952 | 0.7983 | 1825.0 | 763.0 | +11.0 | +12.0 |
| superhero_person | 100 | 0.3800 | 266.5 | 124.0 | -11.0 | -10.0 |

University: the selected LRE predicts `University` on every row, scoring 79.1209%; this exactly matches the constant majority-token baseline. This is first-token faithfulness, not full university identification.

The LRE and lens demonstration contexts produce different target tokens on 31.04% of all rows (median across relations 21.89%). This diagnostic does not establish how much correlation would change with matched contexts; that would be a separate protocol experiment.

| Activation norm | Median | 99th percentile | Maximum |
|---|---:|---:|---:|
| original_subject_norm | 109.03 | 4117.40 | 4644.16 |
| newline_subject_norm | 111.98 | 128.39 | 137.39 |
| original_final_norm | 335.77 | 386.51 | 403.36 |
| newline_final_norm | 340.91 | 391.90 | 420.28 |

The overall median hides the first-position spike. The following groups cover every row:

| Original subject endpoint | Rows | Original median subject norm | Newline median subject norm |
|---|---:|---:|---:|
| position_zero | 142 | 4208.75 | 123.21 |
| later_positions | 9605 | 108.90 | 111.80 |

Original subject endpoint at position zero: 142/9747 rows; after newline: 0/9747. Content token IDs changed beyond the added newline: 0/9747. Norm statistics cover every evaluation row (duplicates retained).

Across all 28 relations, the fraction of reference subjects ending at token position zero correlates with the lens-score improvement at Spearman 0.6791. This supports sensitivity to the beginning of the neutral prompt; it does not separately identify a particular attention mechanism.


## Pythia 6.9B

Largest original rank mismatches; rank 1 is best. Median token ranks below are one-based.

| Relation | Rows | LRE faithfulness | Original lens median rank | Newline lens median rank | Original rank gap | Newline rank gap |
|---|---:|---:|---:|---:|---:|---:|
| name_gender | 19 | 0.8947 | 18149.0 | 5640.0 | +20.0 | +14.0 |
| person_university | 91 | 0.8571 | 37103.0 | 27414.0 | +20.0 | +20.0 |
| person_father | 991 | 0.1584 | 455.0 | 821.0 | -19.0 | -13.0 |
| country_language | 24 | 0.9583 | 5562.0 | 10.5 | +15.0 | +5.0 |
| landmark_in_country | 836 | 0.5694 | 18.0 | 120.5 | -15.0 | -10.0 |
| company_hq | 674 | 0.3279 | 856.5 | 2508.0 | -14.0 | -8.0 |
| country_capital_city | 24 | 0.8750 | 6707.5 | 6.5 | +13.0 | -1.0 |
| superhero_person | 100 | 0.4100 | 838.0 | 466.0 | -13.0 | -10.0 |
| city_in_country | 27 | 0.8148 | 3.0 | 3.0 | -10.0 | -9.0 |
| country_largest_city | 24 | 0.8333 | 7132.0 | 4.0 | +10.0 | -6.0 |

University: the selected LRE predicts `University` on every row, scoring 85.7143%; this exactly matches the constant majority-token baseline. This is first-token faithfulness, not full university identification.

The LRE and lens demonstration contexts produce different target tokens on 32.96% of all rows (median across relations 18.01%). This diagnostic does not establish how much correlation would change with matched contexts; that would be a separate protocol experiment.

| Activation norm | Median | 99th percentile | Maximum |
|---|---:|---:|---:|
| original_subject_norm | 234.45 | 282.75 | 2969.54 |
| newline_subject_norm | 237.61 | 281.56 | 297.51 |
| original_final_norm | 367.64 | 437.46 | 452.45 |
| newline_final_norm | 372.13 | 440.48 | 460.07 |

The overall median hides the first-position spike. The following groups cover every row:

| Original subject endpoint | Rows | Original median subject norm | Newline median subject norm |
|---|---:|---:|---:|
| position_zero | 83 | 2875.54 | 281.65 |
| later_positions | 9664 | 234.28 | 237.47 |

Original subject endpoint at position zero: 83/9747 rows; after newline: 0/9747. Content token IDs changed beyond the added newline: 0/9747. Norm statistics cover every evaluation row (duplicates retained).

Across all 28 relations, the fraction of reference subjects ending at token position zero correlates with the lens-score improvement at Spearman 0.5615. This supports sensitivity to the beginning of the neutral prompt; it does not separately identify a particular attention mechanism.

## Validation

All 56 model-relation files and 19,494 evaluated rows verified against immutable baseline hashes, dataset hashes, pinned model revisions, row order, target IDs, reference counts and gender target sets. Aggregates were recomputed from every row; Spearman values were checked with an independent tied-rank implementation. GPU derivative checks and cached-target equivalence test passed.

Artifacts: [correlations](correlations.csv), [all relation diagnostics](relation_diagnostics.csv), [outlier figure](outliers.png), [machine-readable evidence](summary.json).
