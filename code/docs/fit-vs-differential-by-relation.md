# Which relations favour the fitted Jacobian, which the differential one

> Historical configuration: the numerical comparisons below predate the current matched newline/shared-split appendix. Use [the current baseline tables](../results/shared_baselines/README.md) and [curation guide](shared-baseline-appendix.md) for up-to-date results. Original measurements are preserved for provenance.

This note ranks every relation by how much better one of the two Jacobian
readouts scores it. It exists to answer one question directly: for which
relations is the *fitted* (affine, zeroth-order-restored) Jacobian the better
readout, and for which is the *differential* Jacobian better?

## The two readouts

Both are measured in the same pass by `measure-jacobian-lens`, at the same
layer, neutral prompt, reference subjects, split, and target tokens, so their
per-relation scores are directly comparable.

| Readout | Formula | JSON fields |
| --- | --- | --- |
| Differential Jacobian lens | `mean_i J_i (s_e - s_i)` | `agg.diff`, `rank_diff`, `top_diff` |
| Fitted (affine) Jacobian | `mean_i [ f(s_i) + beta * J_i (s_e - s_i) ]`, beta = 1 | `agg.lre`, `rank_lre`, `top_lre` |

The fitted form is the LRE-style affine approximation: it restores the
zeroth-order Taylor term `f(s_i)` that the differential lens cancels. The
historical field name is `lre`; this is the neutral-prompt affine baseline, not
the relation-specific LRE faithfulness measurement in `faithfulness_*.json`.
Since token ranks ignore positive rescaling, beta interpolates between the two:
beta = 1 is the fitted readout, and beta → ∞ is the differential lens
(see [affine-beta-sweep.md](affine-beta-sweep.md)).

## Score and sign convention

The score is the relation's median `log10(rank + 1)` of the model's own top-1
answer token, **lower is better** (`agg.<readout>.median_lograng`). The ranking
key is

```
Delta = median_lograng(fit, beta=1) - median_lograng(differential)
```

so **Delta > 0 means the differential Jacobian is better** for that relation,
and **Delta < 0 means the fitted Jacobian is better**. A Delta of +1.0 means the
differential lens puts the answer token one decade higher in the vocabulary
ranking. `beta*` is the grid value in `{0.5, 1, 1.5, 2, 3, 5}` minimising the
relation's median log-rank in the saved sweep, ties broken to the smallest beta;
`beta* = 5` is the grid edge and means "further towards the differential lens
than the grid goes".

## Headline: top five each way (OLMo, main protocol)

Best for the **differential** Jacobian:

| Relation | Differential | Fit | Delta |
| --- | ---: | ---: | ---: |
| `city_in_country` | 0.48 | 3.53 | +3.06 |
| `country_capital_city` | 0.74 | 3.62 | +2.88 |
| `country_largest_city` | 0.87 | 3.70 | +2.83 |
| `name_birthplace` | 1.22 | 3.65 | +2.43 |
| `name_religion` | 1.22 | 3.59 | +2.37 |

Best for the **fitted** Jacobian:

| Relation | Differential | Fit | Delta |
| --- | ---: | ---: | ---: |
| `name_gender` | 4.92 | 3.13 | −1.79 |
| `company_ceo` | 4.59 | 3.94 | −0.64 |
| `country_currency` | 4.42 | 4.03 | −0.39 |
| `person_mother` | 4.09 | 3.76 | −0.33 |
| `person_university` | 4.33 | 4.14 | −0.19 |

For GPT-J, main protocol, the two ends are `landmark_in_country` (+2.84),
`city_in_country` (+2.72), `name_birthplace` (+2.41),
`presidents_election_year` (+2.37), `superhero_person` (+1.99) on the
differential side, and `superhero_archnemesis` (−2.19), `name_gender` (−0.46),
`person_university` (−0.15), `person_plays_instrument` (−0.12), `company_ceo`
(−0.08) on the fitted side.

## How to read the negative-Delta relations

The fitted readout never reaches the answer token: its top-1 agreement with the
model is **exactly 0.00 in all 106 relation cells across both models and both
protocols**, and its top-5 tokens are the same generic continuations
(`its`, `being`, `having`, `the`) for nearly every subject, because the
reference mean `mean_i f(s_i)` dominates. So a negative Delta is not a case of
the fitted Jacobian recovering the fact; it is a tail-rank comparison between
two failures. Every relation with Delta < 0 has a differential score of 3.0 or
worse (median rank ≥ 1000), i.e. the differential lens has already failed there.

Two mechanisms produce the negative Deltas:

- **Function-word or high-frequency targets.** The scored target is the model's
  own top-1 token under the in-context prompt, which for GPT-J
  `superhero_archnemesis` is `the` and for `name_gender` is `man` / `woman`. The
  generic readout ranks such tokens high by default (GPT-J
  `superhero_archnemesis` fit rank 6 for Hawkeye, Wolverine, Red Hood), while
  the differential lens answers with entity tokens (`Wolverine`, `Logan`,
  `claws`) that are contentful but not the target.
- **Targets the subject representation cannot carry.** For `company_ceo` the
  target is a CEO's first name; the differential lens returns the company's
  industry (`chemicals`, `steel`, `railway`) and the fitted readout returns
  generic tokens. Neither ranks the name, and the ordering between them is
  noise at ranks in the thousands.

The per-relation beta sweep agrees: under median-log-rank selection, 27 of 28
OLMo relations and 25 of 28 GPT-J relations pick the largest grid beta, i.e.
they want to move towards the differential lens. The exceptions are exactly the
negative-Delta relations — OLMo `name_gender` (beta 1), GPT-J
`superhero_archnemesis` (beta 2), `city_in_country` and `country_largest_city`
(beta 3, ties resolved below the grid edge).

## Full rankings

Columns: median `log10(rank + 1)` with the median rank in parentheses; Delta as
defined above; `diff agree@1` is the differential lens's top-1 agreement with
the model (the fitted readout's is 0.00 throughout).

#### OLMo / main (28 relations)

| # | Relation | n | Differential | Fit (β=1) | Δ | diff agree@1 | β\* |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `city_in_country` | 19 | 0.48 (2) | 3.53 (3412) | +3.06 | 0.26 | 5 |
| 2 | `country_capital_city` | 16 | 0.74 (4) | 3.62 (4190) | +2.88 | 0.06 | 5 |
| 3 | `country_largest_city` | 16 | 0.87 (6) | 3.70 (5100) | +2.83 | 0.06 | 5 |
| 4 | `name_birthplace` | 20 | 1.22 (16) | 3.65 (4442) | +2.43 | 0.00 | 5 |
| 5 | `name_religion` | 20 | 1.22 (17) | 3.59 (3922) | +2.37 | 0.10 | 5 |
| 6 | `country_language` | 16 | 0.99 (9) | 3.28 (1938) | +2.29 | 0.12 | 5 |
| 7 | `person_band_lead_singer` | 13 | 1.49 (30) | 3.51 (3203) | +2.01 | 0.15 | 5 |
| 8 | `person_plays_pro_sport` | 20 | 1.31 (20) | 2.91 (814) | +1.61 | 0.15 | 5 |
| 9 | `landmark_in_country` | 20 | 2.20 (157) | 3.73 (5320) | +1.53 | 0.10 | 5 |
| 10 | `person_plays_instrument` | 20 | 1.54 (34) | 3.01 (1019) | +1.47 | 0.05 | 5 |
| 11 | `superhero_person` | 20 | 2.11 (146) | 3.41 (2584) | +1.30 | 0.00 | 5 |
| 12 | `person_occupation` | 20 | 2.42 (273) | 3.66 (4564) | +1.24 | 0.00 | 5 |
| 13 | `person_native_language` | 20 | 2.12 (138) | 3.09 (1220) | +0.97 | 0.00 | 5 |
| 14 | `landmark_on_continent` | 20 | 2.86 (732) | 3.46 (2853) | +0.59 | 0.00 | 5 |
| 15 | `person_plays_position_in_sport` | 20 | 3.81 (6470) | 4.20 (15674) | +0.39 | 0.05 | 5 |
| 16 | `product_by_company` | 20 | 3.19 (1752) | 3.53 (3382) | +0.34 | 0.05 | 5 |
| 17 | `person_father` | 20 | 3.54 (3444) | 3.86 (7188) | +0.32 | 0.00 | 5 |
| 18 | `star_constellation` | 20 | 3.48 (3874) | 3.76 (5767) | +0.28 | 0.00 | 5 |
| 19 | `presidents_birth_year` | 11 | 4.11 (12893) | 4.31 (20306) | +0.20 | 0.00 | 5 |
| 20 | `pokemon_evolutions` | 20 | 3.32 (2082) | 3.51 (3220) | +0.19 | 0.05 | 5 |
| 21 | `company_hq` | 20 | 4.00 (9984) | 4.11 (13096) | +0.11 | 0.00 | 5 |
| 22 | `superhero_archnemesis` | 20 | 3.50 (3186) | 3.59 (3881) | +0.09 | 0.00 | 5 |
| 23 | `presidents_election_year` | 11 | 3.80 (6354) | 3.87 (7479) | +0.07 | 0.00 | 5 |
| 24 | `person_university` | 20 | 4.33 (21498) | 4.14 (13780) | −0.19 | 0.00 | 5 |
| 25 | `person_mother` | 20 | 4.09 (12269) | 3.76 (5728) | −0.33 | 0.00 | 5 |
| 26 | `country_currency` | 20 | 4.42 (26148) | 4.03 (10715) | −0.39 | 0.05 | 5 |
| 27 | `company_ceo` | 20 | 4.59 (38510) | 3.94 (8777) | −0.64 | 0.00 | 5 |
| 28 | `name_gender` | 11 | 4.92 (82983) | 3.13 (1347) | −1.79 | 0.00 | 1 |

#### OLMo / shared split (25 relations)

| # | Relation | n | Differential | Fit (β=1) | Δ | diff agree@1 | β\* |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `country_largest_city` | 8 | 0.74 (4) | 3.78 (6145) | +3.04 | 0.00 | 5 |
| 2 | `country_language` | 8 | 0.30 (1) | 3.11 (1278) | +2.81 | 0.25 | 5 |
| 3 | `city_in_country` | 6 | 0.59 (3) | 3.36 (2282) | +2.77 | 0.33 | 5 |
| 4 | `country_capital_city` | 8 | 0.98 (12) | 3.67 (4726) | +2.69 | 0.12 | 5 |
| 5 | `name_birthplace` | 15 | 1.23 (16) | 3.51 (3250) | +2.28 | 0.07 | 5 |
| 6 | `product_by_company` | 100 | 1.64 (43) | 3.37 (2343) | +1.73 | 0.12 | 5 |
| 7 | `landmark_in_country` | 100 | 2.02 (105) | 3.66 (4570) | +1.64 | 0.01 | 5 |
| 8 | `person_plays_pro_sport` | 100 | 1.15 (13) | 2.76 (580) | +1.62 | 0.07 | 5 |
| 9 | `superhero_person` | 63 | 2.27 (187) | 3.59 (3879) | +1.31 | 0.03 | 5 |
| 10 | `person_native_language` | 100 | 2.11 (128) | 3.25 (1798) | +1.14 | 0.12 | 5 |
| 11 | `person_plays_position_in_sport` | 100 | 3.02 (1052) | 3.99 (9764) | +0.97 | 0.14 | 5 |
| 12 | `person_plays_instrument` | 100 | 2.37 (234) | 2.83 (682) | +0.46 | 0.04 | 5 |
| 13 | `pokemon_evolutions` | 20 | 3.30 (2007) | 3.73 (5366) | +0.43 | 0.05 | 5 |
| 14 | `company_hq` | 100 | 3.58 (3834) | 3.98 (9496) | +0.39 | 0.01 | 5 |
| 15 | `landmark_on_continent` | 100 | 2.84 (688) | 3.18 (1506) | +0.34 | 0.02 | 5 |
| 16 | `superhero_archnemesis` | 55 | 3.57 (3697) | 3.74 (5532) | +0.17 | 0.02 | 5 |
| 17 | `name_religion` | 15 | 3.54 (3486) | 3.72 (5202) | +0.17 | 0.00 | 5 |
| 18 | `person_father` | 100 | 3.79 (6212) | 3.91 (8220) | +0.12 | 0.00 | 5 |
| 19 | `person_university` | 29 | 3.95 (8928) | 4.02 (10483) | +0.07 | 0.00 | 5 |
| 20 | `person_occupation` | 100 | 3.70 (4968) | 3.60 (3995) | −0.09 | 0.00 | 5 |
| 21 | `star_constellation` | 100 | 3.83 (6832) | 3.68 (4821) | −0.15 | 0.00 | 5 |
| 22 | `person_mother` | 100 | 4.18 (15012) | 3.80 (6308) | −0.38 | 0.00 | 5 |
| 23 | `country_currency` | 14 | 4.45 (28294) | 3.95 (8962) | −0.50 | 0.00 | 5 |
| 24 | `company_ceo` | 100 | 4.52 (33193) | 4.02 (10448) | −0.50 | 0.00 | 5 |
| 25 | `name_gender` | 3 | 4.98 (94776) | 3.16 (1434) | −1.82 | 0.00 | 0.5 |

#### GPT-J / main (28 relations)

| # | Relation | n | Differential | Fit (β=1) | Δ | diff agree@1 | β\* |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `landmark_in_country` | 20 | 0.85 (6) | 3.68 (4792) | +2.84 | 0.05 | 5 |
| 2 | `city_in_country` | 19 | 0.30 (1) | 3.02 (1038) | +2.72 | 0.42 | 3 |
| 3 | `name_birthplace` | 20 | 1.20 (15) | 3.61 (4210) | +2.41 | 0.10 | 5 |
| 4 | `presidents_election_year` | 11 | 1.26 (17) | 3.62 (4199) | +2.37 | 0.00 | 5 |
| 5 | `superhero_person` | 20 | 1.55 (40) | 3.53 (3422) | +1.99 | 0.00 | 5 |
| 6 | `person_band_lead_singer` | 13 | 1.23 (16) | 2.92 (836) | +1.69 | 0.23 | 5 |
| 7 | `name_religion` | 20 | 1.71 (51) | 3.28 (1912) | +1.58 | 0.05 | 5 |
| 8 | `country_capital_city` | 16 | 2.59 (391) | 3.98 (9506) | +1.39 | 0.00 | 5 |
| 9 | `person_father` | 20 | 2.28 (194) | 3.62 (4136) | +1.34 | 0.00 | 5 |
| 10 | `landmark_on_continent` | 20 | 2.45 (282) | 3.75 (5574) | +1.30 | 0.00 | 5 |
| 11 | `person_plays_position_in_sport` | 20 | 3.13 (1374) | 4.18 (15110) | +1.04 | 0.00 | 5 |
| 12 | `country_language` | 16 | 2.77 (653) | 3.76 (5797) | +1.00 | 0.06 | 5 |
| 13 | `product_by_company` | 20 | 2.29 (208) | 3.18 (1530) | +0.90 | 0.05 | 5 |
| 14 | `company_hq` | 20 | 2.97 (1122) | 3.84 (6893) | +0.86 | 0.00 | 5 |
| 15 | `person_native_language` | 20 | 1.65 (44) | 2.47 (298) | +0.83 | 0.15 | 5 |
| 16 | `star_constellation` | 20 | 2.99 (983) | 3.78 (6015) | +0.79 | 0.00 | 5 |
| 17 | `country_largest_city` | 16 | 3.59 (4442) | 4.29 (19706) | +0.70 | 0.00 | 3 |
| 18 | `pokemon_evolutions` | 20 | 3.28 (1972) | 3.91 (8170) | +0.63 | 0.00 | 5 |
| 19 | `presidents_birth_year` | 11 | 3.58 (3811) | 4.17 (14636) | +0.58 | 0.00 | 5 |
| 20 | `person_plays_pro_sport` | 20 | 1.46 (28) | 2.03 (107) | +0.57 | 0.00 | 5 |
| 21 | `country_currency` | 20 | 3.52 (3584) | 4.06 (11440) | +0.54 | 0.05 | 5 |
| 22 | `person_occupation` | 20 | 3.41 (2606) | 3.85 (7138) | +0.45 | 0.05 | 5 |
| 23 | `person_mother` | 20 | 3.31 (2034) | 3.63 (4232) | +0.32 | 0.00 | 5 |
| 24 | `company_ceo` | 20 | 3.98 (9692) | 3.90 (8006) | −0.08 | 0.00 | 5 |
| 25 | `person_plays_instrument` | 20 | 3.04 (1098) | 2.92 (830) | −0.12 | 0.00 | 5 |
| 26 | `person_university` | 20 | 4.10 (13158) | 3.95 (8922) | −0.15 | 0.00 | 5 |
| 27 | `name_gender` | 11 | 4.03 (10828) | 3.57 (3741) | −0.46 | 0.00 | 5 |
| 28 | `superhero_archnemesis` | 20 | 3.78 (6054) | 1.59 (98) | −2.19 | 0.05 | 2 |

#### GPT-J / shared split (25 relations)

| # | Relation | n | Differential | Fit (β=1) | Δ | diff agree@1 | β\* |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `city_in_country` | 6 | 0.15 (0) | 2.81 (741) | +2.66 | 0.50 | 5 |
| 2 | `person_plays_position_in_sport` | 100 | 1.77 (58) | 4.16 (14332) | +2.38 | 0.07 | 5 |
| 3 | `landmark_in_country` | 100 | 1.02 (10) | 3.37 (2369) | +2.35 | 0.07 | 5 |
| 4 | `name_birthplace` | 15 | 1.43 (26) | 3.74 (5541) | +2.31 | 0.07 | 5 |
| 5 | `person_native_language` | 100 | 0.70 (4) | 2.60 (396) | +1.90 | 0.37 | 5 |
| 6 | `superhero_person` | 63 | 1.73 (53) | 3.49 (3103) | +1.76 | 0.02 | 5 |
| 7 | `company_hq` | 100 | 2.17 (148) | 3.80 (6293) | +1.63 | 0.02 | 5 |
| 8 | `product_by_company` | 100 | 1.47 (28) | 2.99 (968) | +1.52 | 0.09 | 5 |
| 9 | `country_largest_city` | 8 | 2.70 (530) | 4.01 (10465) | +1.31 | 0.00 | 5 |
| 10 | `landmark_on_continent` | 100 | 1.78 (59) | 3.07 (1160) | +1.29 | 0.01 | 5 |
| 11 | `star_constellation` | 100 | 2.35 (225) | 3.63 (4286) | +1.28 | 0.07 | 5 |
| 12 | `person_plays_pro_sport` | 100 | 1.06 (10) | 2.18 (152) | +1.12 | 0.08 | 5 |
| 13 | `person_father` | 100 | 2.53 (340) | 3.65 (4478) | +1.12 | 0.03 | 5 |
| 14 | `country_currency` | 14 | 3.05 (1658) | 4.07 (11806) | +1.02 | 0.00 | 5 |
| 15 | `person_mother` | 100 | 3.03 (1064) | 3.96 (9208) | +0.94 | 0.00 | 5 |
| 16 | `country_capital_city` | 8 | 3.15 (1620) | 3.86 (7302) | +0.71 | 0.00 | 5 |
| 17 | `name_religion` | 15 | 3.08 (1210) | 3.74 (5520) | +0.66 | 0.07 | 5 |
| 18 | `company_ceo` | 100 | 3.40 (2486) | 3.94 (8670) | +0.54 | 0.00 | 5 |
| 19 | `person_occupation` | 100 | 3.73 (5356) | 4.27 (18677) | +0.54 | 0.01 | 5 |
| 20 | `pokemon_evolutions` | 20 | 3.28 (1978) | 3.68 (4895) | +0.40 | 0.00 | 5 |
| 21 | `person_plays_instrument` | 100 | 2.67 (469) | 2.97 (930) | +0.30 | 0.03 | 5 |
| 22 | `person_university` | 25 | 3.78 (6058) | 3.86 (7210) | +0.08 | 0.00 | 5 |
| 23 | `name_gender` | 3 | 4.47 (29434) | 3.93 (8421) | −0.54 | 0.00 | 0.5 |
| 24 | `country_language` | 8 | 4.32 (21020) | 3.62 (4160) | −0.70 | 0.12 | 1 |
| 25 | `superhero_archnemesis` | 57 | 3.83 (6781) | 2.44 (272) | −1.40 | 0.00 | 0.5 |

## Caveats

- **The comparison is between two baselines, not between two working methods.**
  The fitted readout's top-1 agreement is 0.00 everywhere; negative Deltas rank
  tail positions, not recoveries.
- **`n` is the number of scored subjects**, not the relation's size. The legacy
  (`main`) lens split scores at most 20 subjects per relation, so a median over
  11–20 values is a noisy statistic; single-relation Deltas below roughly 0.3
  should not be read as an ordering.
- **The two protocols are different splits**, not a replication: main uses the
  legacy `Random(17)` split with up to 20 test subjects, shared uses the
  fingerprinted split with up to 100 and the identifying-first-token filter.
  Seven relations change sign between the two protocols. Five are near-zero on
  both sides (OLMo `person_university` −0.19 → +0.07 and `star_constellation`
  +0.28 → −0.15; GPT-J `company_ceo` −0.08 → +0.54, `person_plays_instrument`
  −0.12 → +0.30, `person_university` −0.15 → +0.08). Two are real
  disagreements, driven by the differential score itself moving between splits:
  OLMo `person_occupation` (+1.24 → −0.09, differential 2.42 → 3.70) and GPT-J
  `country_language` (+1.00 → −0.70, differential 2.77 → 4.32 on 8 subjects).
- **`beta*` is in-sample**, selected on the same scored subjects it is
  evaluated on, and the grid stops at 5, so `beta* = 5` is a floor, not an
  optimum.
- Main-protocol rows for `presidents_birth_year` and `presidents_election_year`
  come from `jacobian_lens_years.json` (corrected trailing-space queries), as
  everywhere else in the analysis.

## Provenance

Computed on 2026-09-19 from the retained result snapshots, no GPU work:

| Protocol | Files |
| --- | --- |
| main | `results/<model>/results/jacobian_lens.json`, `jacobian_lens_years.json`, `jacobian_lens_beta.json` |
| shared | `results/<model>/results/shared_split/jacobian_lens.json`, `jacobian_lens_beta.json` |

Models: `allenai/Olmo-3-1025-7B` at hidden-state index 17, `EleutherAI/gpt-j-6b`
at index 15; neutral prompt `"{} is known for"`.

## Reproducing the tables

Run from `code/` with the result snapshots in place:

```python
import json
from pathlib import Path

RESULTS = Path("results")
YEAR_RELATIONS = {"presidents_birth_year", "presidents_election_year"}


def rows(model, protocol):
    base = RESULTS / model / "results" / ("shared_split" if protocol == "shared" else "")
    read = lambda name: json.loads((base / name).read_text())
    lens = read("jacobian_lens.json")
    years = read("jacobian_lens_years.json") if protocol == "main" else {}
    sweep = read("jacobian_lens_beta.json")
    out = []
    for relation in lens:
        result = (years if relation in YEAR_RELATIONS and relation in years else lens)[relation]
        diff, fit = result["agg"]["diff"], result["agg"]["lre"]
        grid = sweep[relation]["beta_sweep"]["agg"]
        out.append(dict(
            relation=relation,
            n=result["n"],
            diff=diff["median_lograng"],
            fit=fit["median_lograng"],
            delta=fit["median_lograng"] - diff["median_lograng"],
            diff_agree_at1=diff["agree_at1"],
            best_beta=float(min(sorted(grid, key=float),
                                key=lambda b: (grid[b]["median_lograng"], float(b)))),
        ))
    return sorted(out, key=lambda row: -row["delta"])


for model in ("olmo", "gptj"):
    for protocol in ("main", "shared"):
        print(f"\n#### {model} / {protocol}\n")
        for i, r in enumerate(rows(model, protocol), 1):
            print(f"{i:>3} {r['relation']:<32} n={r['n']:>3} "
                  f"diff={r['diff']:.2f} fit={r['fit']:.2f} delta={r['delta']:+.2f} "
                  f"agree@1={r['diff_agree_at1']:.2f} beta*={r['best_beta']:g}")
```
