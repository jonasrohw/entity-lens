### Table relations (OLMo): k / faith. / A_r

| relation | k | faith. | A_r |
|---|---:|---:|---:|
| country_language | 1.7 | 1.00 | -1.22 |
| country_largest_city | 1.0 | 0.92 | -0.81 |
| landmark_on_continent | 189.4 | 0.92 | -2.43 |
| person_native_language | 30.6 | 0.92 | -1.69 |
| name_birthplace | 3.9 | 0.97 | -1.32 |
| name_religion | 6.2 | 0.97 | -1.40 |
| person_plays_pro_sport | 63.6 | 0.95 | -1.23 |
| person_plays_instrument | 85.5 | 0.89 | -1.72 |
| landmark_in_country | 9.2 | 0.80 | -1.69 |
| city_in_country | 1.3 | 0.89 | -0.48 |
| person_occupation | 26.5 | 0.71 | -2.74 |
| country_capital_city | 1.0 | 0.875 | -0.85 |
| name_gender | 9.5 | 0.53 | -1.04 |
| presidents_birth_year | 1.3 | 0.74 | -4.44 |
| presidents_election_year | 1.1 | 0.79 | -3.45 |
| product_by_company | 17.4 | 0.68 | -1.88 |
| person_plays_position_in_sport | 68.0 | 0.61 | -3.64 |
| person_band_lead_singer | 1.0 | 0.86 | -1.49 |
| person_university | 1.3 | 0.59 | -4.38 |
| company_hq | 4.1 | 0.54 | -4.23 |
| superhero_person | 1.0 | 0.50 | -2.62 |
| star_constellation | 11.7 | 0.47 | -3.50 |
| pokemon_evolutions | 1.0 | 0.32 | -3.45 |
| country_currency | 1.3 | 0.40 | -4.49 |
| superhero_archnemesis | 1.1 | 0.30 | -3.90 |
| person_father | 1.0 | 0.075 | -3.71 |
| company_ceo | 1.0 | 0.05 | -4.50 |
| person_mother | 1.0 | 0.07 | -4.18 |

### Readout table (Spearman ρ)

| model | CEL | NEL | logit | diff. logit | attribute | relations |
|---|---:|---:|---:|---:|---:|---:|
| olmo3_7b | 0.7355 | 0.7650 | 0.2056 | 0.4390 | -0.4016 | 28 |
| gptj | 0.4346 | 0.4196 | 0.4669 | 0.2020 | -0.1035 | 28 |
| mistral7b | 0.6561 | 0.6711 | 0.6581 | 0.5186 | -0.2202 | 26 |
| gemma2_9b | 0.6889 | 0.6868 | 0.4728 | 0.3436 | -0.5515 | 26 |
| qwen3_8b_base | 0.6647 | 0.6232 | 0.7908 | 0.5952 | -0.0557 | 26 |
| qwen3_8_27b | 0.5539 | 0.5717 | 0.3404 | 0.4492 | -0.0099 | 26 |
| pythia6_9b | 0.5027 | 0.5063 | -0.2047 | 0.1845 | 0.0041 | 28 |

### Corpus-frequency table (ρ / p)

| feature | ρ | p |
|---|---:|---:|
| subject | 0.1380 | 0.484 |
| object | 0.5712 | 0.001 |
| joint | 0.4568 | 0.015 |
| mean_log_joint | 0.4191 | 0.026 |
| log_mean_joint | 0.4568 | 0.015 |
| forward | 0.4486 | 0.017 |
| reverse | -0.4149 | 0.028 |
| direction | 0.4826 | 0.009 |
| micro_forward | 0.5504 | 0.002 |
| micro_reverse | -0.1453 | 0.461 |

### Prompt variations (Qwen3 8B, ρ)

| prompt | ρ |
|---|---:|
| `{s} is known for` | 0.3430 |
| `{s}` | 0.2479 |
| `{s} is` | 0.2362 |
| `{s} can be described as` | 0.4297 |
| `Information about {s}:` | 0.6998 |
| `Facts about {s}:` | 0.6725 |
| `Entity: {s} is known for` | 0.7331 |
| `\n{s} is known for` | 0.6647 |

### Prominence table

| relation | prom. | n | gold∈lens | n |
|---|---:|---:|---:|---:|
| city_in_country | 1.00 → 0.96 ✗ | 5 → 24 ✗ | 0.65 → 0.71 ✗ | 17 → 24 ✗ |
| person_plays_pro_sport | 0.60 → 0.69 ✗ | 40 → 318 ✗ | 0.35 → 0.36 ✗ | 20 → 318 ✗ |
| star_constellation | 0.60 → 0.48 ✗ | 40 → 361 ✗ | 0.05 → 0.06 ✗ | 20 → 361 ✗ |
| country_capital_city | 0.50 → 0.29 ✗ | 8 → 24 ✗ | 0.56 → 0.42 ✗ | 16 → 24 ✗ |
| landmark_in_country | 0.47 → 0.56 ✗ | 40 → 836 ✗ | 0.25 → 0.30 ✗ | 20 → 836 ✗ |
| product_by_company | 0.33 → 0.22 ✗ | 40 → 522 ✗ | 0.05 → 0.19 ✗ | 20 → 522 ✗ |
| person_occupation | 0.30 → 0.25 ✗ | 40 → 821 ✗ | 0.10 → 0.12 ✗ | 20 → 821 ✗ |
| country_currency | 0.29 → 0.20 ✗ | 14 → 30 ✗ | 0.00 → 0.03 ✗ | 20 → 30 ✗ |
| person_university | 0.28 → 0.32 ✗ | 29 → 91 ✗ | 0.00 | 20 → 91 ✗ |
| country_largest_city | 0.25 → 0.21 ✗ | 8 → 24 ✗ | 0.25 → 0.42 ✗ | 16 → 24 ✗ |
| person_father | 0.13 → 0.11 ✗ | 39 → 942 ✗ | 0.06 → 0.01 ✗ | 18 → 942 ✗ |
| company_hq | 0.12 → 0.23 ✗ | 40 → 674 ✗ | 0.00 → 0.02 ✗ | 20 → 674 ✗ |
| superhero_archnemesis | 0.05 → 0.06 ✗ | 39 → 94 ✗ | 0.05 → 0.01 ✗ | 19 → 94 ✗ |
| person_mother | 0.03 → 0.10 ✗ | 40 → 986 ✗ | 0.00 → 0.01 ✗ | 20 → 986 ✗ |

### Reference-set table (French subjects)

| reference set | CEL log-rank | NEL log-rank | CEL rank of answer | NEL rank of answer |
|---|---:|---:|---:|---:|
| other | 1.62 → 1.75 ✗ | 1.81 → 1.32 ✗ | 47 → 59 ✗ | 64 → 24 ✗ |
| cross | 1.88 → 1.79 ✗ | 1.72 → 1.58 ✗ | 102 → 74 ✗ | 70 → 46 ✗ |
| mixed | 1.93 → 2.61 ✗ | 1.79 → 1.38 ✗ | 115 → 485 ✗ | 82 → 27 ✗ |
| answer | 2.72 → 2.95 ✗ | 1.71 → 0.95 ✗ | 665 → 1068 ✗ | 55 → 10 ✗ |

### Swap tables (hits)

| relation | own | foreign | NEL | currency | anchor alone | wrong donor |
|---|---:|---:|---:|---:|---:|---:|
| country_capital_city | 12 | 12 | 7 | 6 | 0 | 0 |
| country_largest_city | 12 | 8 | 4 | 5 | 0 | 0 |
| person_plays_position_in_sport | 19 | 21 | 18 | – | 7 | 6 |
| person_plays_pro_sport | 22 | 21 | 21 | – | 6 | 3 |

### Swap resampling table (mean / min / max over anchors)

| arm | capital | largest | position | sport |
|---|---:|---:|---:|---:|
| own | 10.7 → 10.6 ✗ / 10 / 14 | 11.4 / 10 / 12 → 13 ✗ | 18.3 → 18.2 ✗ / 17 / 20 | 22.0 / 21 → 19 ✗ / 23 |
| foreign | 9.7 / 9 / 11 → 12 ✗ | 9.3 / 8 / 10 → 11 ✗ | 17.3 → 17.4 ✗ / 15 / 21 | 20.4 → 20.7 ✗ / 16 / 23 |
| nel | 5.2 → 5.3 ✗ / 1 → 2 ✗ / 8 | 4.7 → 4.6 ✗ / 4 / 5 | 14.7 → 14.6 ✗ / 7 → 8 ✗ / 20 → 19 ✗ | 21.5 → 21.8 ✗ / 16 → 17 ✗ / 24 |
| anchor_alone | 0.0 → 0.1 ✗ / 0 / 1 | 0.0 / 0 / 0 | 7.0 / 7 → 1 ✗ / 7 | 4.5 → 3.7 ✗ / 1 / 7 → 8 ✗ |

### All other values in the text

| location | value | paper | new | |
|---|---|---:|---:|:-:|
| Sec. 5 | `olmo3_7b.cel.p` | 8.24e-06 | 8.24e-06 | ✓ |
| Sec. 5 (k range) | `controls.cardinality_min` | 1.0 | 1.0 | ✓ |
| Sec. 5 (k range) | `controls.cardinality_max` | 189 | 189 | ✓ |
| Sec. 5 | `controls.cardinality_faithfulness.rho` | 0.355 | 0.355 | ✓ |
| Sec. 5 | `controls.cardinality_faithfulness.p` | 0.064 | 0.064 | ✓ |
| Sec. 5 | `controls.cardinality_lens.rho` | 0.086 | 0.086 | ✓ |
| Sec. 5, Sec. 7 | `controls.partial_cardinality.rho` | 0.757 | 0.757 | ✓ |
| Sec. 5 | `controls.partial_cardinality.p` | 4.90e-06 | 4.90e-06 | ✓ |
| Sec. 5 | `controls.prior_faithfulness.rho` | -0.111 | -0.111 | ✓ |
| Sec. 5 | `controls.prior_lens.rho` | 0.098 | 0.098 | ✓ |
| Sec. 5 | `controls.partial_prior.rho` | 0.755 | 0.755 | ✓ |
| Sec. 5 | `controls.partial_both.rho` | 0.784 | 0.784 | ✓ |
| Sec. 5 (eleven person relations) | `domains.person` | 11 | 11 | ✓ |
| Sec. 5 (leave one domain out) | `domains.loo_min` | 0.570 | 0.696 | ✗ |
| Sec. 5 (leave one domain out) | `domains.loo_max` | 0.700 | 0.799 | ✗ |
| Sec. 5 (leave one domain out) | `domains.loo_median` | 0.656 | 0.728 | ✗ |
| Sec. 6.1, Table prominence | `prominence.rho` | 0.614 | 0.570 | ✗ |
| Sec. 6.1, Table prominence | `prominence.p` | 0.019 | 0.033 | ✗ |
| Sec. 6.1 | `prominence.n` | 14 | 14 | ✓ |
| Sec. 6.2 | `refsets.rows` | 100 | 919 | ✗ |
| Sec. 6.2 | `refsets.answer_gold` | 69 | 587 | ✗ |
| Sec. 6.2 | `refsets.answer_target` | 65 | 553 | ✗ |
| Sec. 6.2, Table refsets | `refsets.answer_test` | 61 | 579 | ✗ |
| Sec. 6.2 | `refsets.other_test` | 23 | 324 | ✗ |
| Sec. 6.2 | `refsets.monotone` | 52 | 441 | ✗ |
| Sec. 6.2 (effect reverses) | `refsets.answer.other.cel.log_rank` | 0.78 | 1.54 | ✗ |
| Sec. 6.2 (effect reverses) | `refsets.other.other.cel.log_rank` | 1.97 | 1.67 | ✗ |
| Sec. 6.2 (what is subtracted) | `refsets.answer.subtracted_top` | Cin, La, Une, fils | Bordeaux, France, Marseille, French | ✗ |
| Sec. 7 | `residual.landmark_on_continent` | 0.242 | 0.242 | ✓ |
| Sec. 7 | `residual.name_gender` | -0.374 | -0.374 | ✓ |
| Sec. 7 | `residual.person_native_language` | 0.118 | 0.118 | ✓ |
| Sec. 7 | `residual.person_father` | -0.403 | -0.403 | ✓ |
| Sec. 7 | `residual.person_mother` | -0.330 | -0.330 | ✓ |
| Sec. 7 | `residual.company_ceo` | -0.297 | -0.297 | ✓ |
| Sec. 7 (largest positive residuals) | `residual.most_positive.1` | landmark_on_continent | presidents_birth_year | ✗ |
| Sec. 7 (largest positive residuals) | `residual.most_positive.2` | name_gender | presidents_election_year | ✗ |
| Sec. 7 (largest positive residuals) | `residual.most_positive.3` | person_native_language | landmark_on_continent | ✗ |
| Sec. 7 (largest negative residuals) | `residual.most_negative.1` | person_father | person_father | ✓ |
| Sec. 7 (largest negative residuals) | `residual.most_negative.2` | person_mother | name_gender | ✗ |
| Sec. 7 (largest negative residuals) | `residual.most_negative.3` | company_ceo | person_mother | ✗ |
| Sec. 8 | `facts.min` | -0.401 | -0.401 | ✓ |
| Sec. 8 | `facts.max` | 0.664 | 0.664 | ✓ |
| Sec. 8 | `facts.median` | 0.235 | 0.235 | ✓ |
| Sec. 8.1 | `facts.person_university` | -0.401 | -0.401 | ✓ |
| Sec. 8.1 | `facts.person_occupation` | -0.388 | -0.388 | ✓ |
| Sec. 8.1 | `facts.person_plays_instrument` | -0.035 | -0.035 | ✓ |
| Sec. 8.1 (first lens token) | `competition.person_occupation/Rolf Thommessen.first` | whistlebl | whistlebl | ✓ |
| Sec. 8.1 | `competition.person_occupation/Barry Sobel` | kosher, Jewish, Rabbi | books, selling, Jewish | ✗ |
| Sec. 8.1 | `competition.person_occupation/Vondell Darr` | criminal, murders, felon | DJ, trolling, rap | ✗ |
| Sec. 8.1 | `competition.person_occupation/Simon Gosejohann` | Johann, Hann, German | DJ, Johann, tattoo | ✗ |
| Sec. 8.1 | `competition.star_constellation/rho01 Cnc` | machining, drilled, CNC | machining, CNC, manufacturing | ✗ |
| Sec. 8.1 | `competition.star_constellation/Alnair` | flying, flown, Flight | flying, rescue, flown | ✗ |
| Sec. 8.2 (of 40 relations) | `collapse.relations` | 40 | 47 | ✗ |
| Sec. 8.2 | `collapse.above_threshold` | 6 | 6 | ✓ |
| Sec. 8.2 | `collapse.person_mother` | 0.61 | 0.61 | ✓ |
| Sec. 8.2 | `collapse.person_father` | 0.58 | 0.58 | ✓ |
| Sec. 8.2 | `collapse.person_university` | 0.45 | 0.45 | ✓ |
| Sec. 8.2 (29 of 69 begin with University) | `collapse.weak_prefix_objects` | 29 | 29 | ✓ |
| Sec. 8.2 | `collapse.person_university.objects` | 69 | 69 | ✓ |
| Sec. 8.2 | `collapse.filtered.person_university.rows` | 29 | 45 | ✗ |
| Sec. 8.2 | `collapse.weak_token_targets` | 7 | 19 | ✗ |
| Sec. 8.2 (LRE agrees on every one) | `collapse.weak_token_agreements` | 7 | 19 | ✗ |
| Sec. 8.2 (0.07 -> 0.05) | `collapse.filtered.person_father` | 0.05 | 0.06 | ✗ |
| Sec. 4 (11-20 subjects per relation) | `rows.min` | 11 | 19 | ✗ |
| Sec. 4 (11-20 subjects per relation) | `rows.max` | 20 | 994 | ✗ |
| Limitations (seven relations under ten items) | `rows.relations_below_ten` | 7 | 0 | ✗ |
| Sec. 4 | `dataset.relations` | 47 | 47 | ✓ |
| Sec. 4 | `dataset.entity_relations` | 28 | 28 | ✓ |
| Limitations | `dataset.excluded_relations` | 19 | 19 | ✓ |
| Appendix layer selection | `layers.modal` | 17 | 17 | ✓ |
| Appendix layer selection (10 of 28) | `layers.modal_count` | 10 | 15 | ✗ |
| Appendix numeric objects | `numeric.tokens` | Ġ 174 3 | Ġ 174 3 | ✓ |
| Appendix numeric objects | `numeric.presidents_birth_year.faithfulness` | 0.67 | 0.74 | ✗ |
| Appendix numeric objects | `numeric.presidents_election_year.faithfulness` | 0.67 | 0.79 | ✗ |
| Appendix numeric objects | `numeric.presidents_birth_year.lens` | -4.24 | -4.44 | ✗ |
| Appendix numeric objects | `numeric.presidents_election_year.lens` | -4.17 | -3.45 | ✗ |
| Appendix numeric objects (0.658 -> 0.427) | `numeric.uncorrected.rho` | 0.427 | 0.416 | ✗ |
| Appendix anchored lens | `anchored.train` | 15 | 15 | ✓ |
| Appendix anchored lens | `anchored.test` | 10 | 13 | ✗ |
| Appendix anchored lens | `anchored.held_out_win_rate` | 0.30 | 0.01 | ✗ |
| Sec. 9, Appendix swap | `swap.countries` | 22 | 22 | ✓ |
| Sec. 9, Appendix swap | `swap.athletes` | 24 | 24 | ✓ |
| Appendix swap (31 remaining) | `swap.candidates` | 31 | 31 | ✓ |
| Appendix swap (13 of the 22 added) | `swap.added_passed` | 13 | 13 | ✓ |
| Appendix swap (11 of 12 capitals) | `swap.country_capital_city.foreign.given_own` | 11 | 11 | ✓ |
| Appendix swap (8 of 12) | `swap.country_largest_city.foreign.given_own` | 8 | 8 | ✓ |
| Appendix swap | `swap.country_capital_city.nel.given_own` | 7 | 7 | ✓ |
| Appendix swap | `swap.country_largest_city.nel.given_own` | 4 | 4 | ✓ |
| Appendix swap | `swap.country_capital_city.currency.given_own` | 6 | 6 | ✓ |
| Appendix swap | `swap.country_largest_city.currency.given_own` | 5 | 5 | ✓ |
| Appendix swap (benchmark nine) | `swap.country_capital_city.benchmark_own` | 7 | 7 | ✓ |
| Appendix swap (benchmark nine) | `swap.country_largest_city.benchmark_own` | 8 | 7 | ✗ |
| Appendix swap (13 added) | `swap.country_capital_city.added_own` | 5 | 5 | ✓ |
| Appendix swap (13 added) | `swap.country_largest_city.added_own` | 5 | 5 | ✓ |
| Appendix swap (currency, beta 3) | `swap.currency_own.beta_3` | 6 | 1 | ✗ |
| Appendix swap (currency, beta 5) | `swap.currency_own.beta_5` | 2 | 0 | ✗ |
| Appendix swap (currency, beta 10) | `swap.currency_own.beta_10` | 0 | 0 | ✓ |
| Appendix swap | `swap.shared_father_mother` | 748 | 748 | ✓ |
| Appendix swap | `swap.pool.countries` | 16 | 16 | ✓ |
| Appendix swap | `swap.pool.athletes` | 1206 | 1206 | ✓ |
| Appendix swap | `swap.pool.country_lens` | 30 | 30 | ✓ |
| Appendix swap | `swap.composition_spearman` | -0.31 | -0.14 | ✗ |
| Appendix swap | `swap.variance.country_capital_city.anchor` | 0.36 | 0.40 | ✗ |
| Appendix swap | `swap.variance.country_largest_city.anchor` | 0.26 | 0.25 | ✗ |
| Appendix swap | `swap.variance.country_capital_city.lens` | 0.38 | 0.37 | ✗ |
| Appendix swap | `swap.variance.country_largest_city.lens` | 0.42 | 0.36 | ✗ |
| Appendix swap | `swap.country_capital_city.norm_ratio` | 0.8 | 0.8 | ✓ |
| Table shared-lenses-spearman footnote | `pythia6_9b.empty_attribute_rows` | 2663 | 2663 | ✓ |
| Table shared-lenses-spearman footnote | `gemma2_9b.empty_attribute_rows` | 1 | 1 | ✓ |
| Appendix A (raw larger in five models) | `models.nel_above_cel` | 5 | 4 | ✗ |
| Appendix A (all seven intervals include zero) | `models.intervals_include_zero` | 7 | 7 | ✓ |
| Appendix A.2 (same 26 relations) | `prompt.0.n` | 26 | 26 | ✓ |
| Appendix (9,709 scored rows) | `qwen3_8b_base.scored_rows` | 9709 | 9709 | ✓ |
| Sec. 6.2 (Joseph Laniel, NEL rank of French) | `refsets.example.answer.nel` | 1 | 3 | ✗ |
| Sec. 6.2 (Joseph Laniel, CEL with non-French references) | `refsets.example.other.cel` | 3 | 44 | ✗ |
| Sec. 6.2 (Joseph Laniel, CEL with French references) | `refsets.example.answer.cel` | 623 | 22717 | ✗ |
| Sec. 6.2 (Joseph Laniel, CEL top tokens) | `refsets.example.top` | sculpt, dancing, singing, dance, chore, piano | sculpt, composing, remix, scoring, bass, painting | ✗ |
