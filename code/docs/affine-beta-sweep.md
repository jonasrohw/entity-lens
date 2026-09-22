# Tuning the zeroth-order term's weight

> Historical configuration: the numerical comparisons below predate the current matched newline/shared-split appendix. Use [the current baseline tables](../results/shared_baselines/README.md) and [curation guide](shared-baseline-appendix.md) for up-to-date results. Original measurements are preserved for provenance.

The affine Jacobian-lens baseline restores the zeroth-order Taylor term that
the differential lens cancels. With beta weighting the first-order term, the
family is

\[
v_e(\beta) = \frac{1}{n_e}\sum_{i\ne e}\Bigl[f(s_i) + \beta\,J_i(s_e-s_i)\Bigr]
= \underbrace{\frac{1}{n_e}\sum_{i\ne e} f(s_i)}_{r_e}
+ \beta\,\underbrace{\frac{1}{n_e}\sum_{i\ne e} J_i(s_e-s_i)}_{d_e},
\]

so \(\beta = 1\) is the affine baseline, and the differential lens is the
\(\beta \to \infty\) limit: token ranks are invariant to positive rescaling,
so once \(\beta d_e\) dominates \(r_e\) the readout is the differential one.
The sweep therefore does not add a new readout — it interpolates between the
two already reported.

## What was measured

`measure-jacobian-lens --betas 0.5,1,1.5,2,3,5` scores every beta from the
same Jacobian-vector products in one pass, so the sweep costs no extra GPU
work beyond the fixed-beta run. `--beta-selection {top1,median-logrank}`
records which rule the run used, and `--vector-dir` caches \(r_e\) and
\(d_e\) per subject so any further beta can be rescored without the model's
backward pass. The grid matches the LRE faithfulness sweep's beta values;
ties select the smallest beta, as there.

Selecting beta per relation uses each relation's own scored targets, so every
tuned number below is **in-sample** — an upper bound on the baseline, not a
held-out result.

Runs on September 18, 2026, legacy split, corrected year targets, OLMo at
hidden-state index 17 and GPT-J at index 15, saved to
`results/<model>/results/jacobian_lens_beta.json`.

## Result: the tuned baseline walks back to the differential lens

Spearman correlation with LRE faithfulness at fixed beta. The large-beta end
approaches the differential lens, which is the check that the family is the one
claimed: the final norm is scale-invariant, so
\(\operatorname{logits}(r_e + \beta d_e) = \operatorname{logits}(r_e/\beta + d_e)\),
and the reference mean is divided away. Both OLMo rows reach a bit-identical
correlation by β = 1000 — not the same per-relation scores, but the same
ordering of relations, which is all Spearman sees. GPT-J is still 0.0005 short
there (0.3429 against 0.3435, one adjacent pair of relations swapped), which is
the useful reminder that the agreement is convergence, not an identity:

| Protocol | 0.5 | 1 | 1.5 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | 1000 | differential |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OLMo, main, 28 rel. | 0.322 | 0.420 | 0.498 | 0.550 | 0.597 | 0.618 | 0.694 | 0.727 | 0.763 | 0.782 | 0.771 | 0.771 |
| OLMo, shared, 25 rel. | 0.441 | 0.585 | 0.659 | 0.703 | **0.711** | 0.708 | 0.686 | 0.666 | 0.656 | 0.648 | 0.624 | 0.624 |
| GPT-J, main, 28 rel. | 0.240 | 0.263 | 0.288 | 0.281 | 0.311 | 0.355 | 0.369 | 0.356 | 0.351 | 0.371 | 0.343 | 0.343 |

Convergence is in the aggregate, not rank by rank. At β = 1000 only 33% of
OLMo's 513 subject ranks equal the differential lens's, rising to 85% at
β = 10⁶; the stragglers are deep-tail near-ties, which move because rescaling
a vector by 10⁶ and renormalizing perturbs float32 logits by about 2 × 10⁻³.
Relation-level median log-ranks at β = 1000 differ from the differential lens
by at most 0.029, and 21 of 28 differ at all — the correlation matches because
none of those shifts reorders a relation.

On the main protocol the OLMo curve rises monotonically to the differential
lens and stops there: the zeroth-order term does not help at any weight, and
the measured grid's best value, β = 5, sits at the grid edge only because the
grid stops there. GPT-J is flat and weak throughout (its differential lens is
ρ = 0.343, p = 0.074), so nothing is distinguishable there either way.

The shared-split protocol is the one exception worth flagging: its curve has a
genuine interior maximum at β ≈ 3 (ρ = 0.711, p = 6.7 × 10⁻⁵) against a
differential lens of 0.624, before declining back to 0.624 at large beta. A
small admixture of the zeroth-order term helps there. With 25 highly dependent
points the gap is not something to lean on, but it is not a grid artifact
either — the peak is bracketed on both sides.

Per-relation tuning, as reported by `statistics`:

| Model | Protocol | Differential ρ | Affine β=1 | Tuned, top-1 | Tuned, median-logrank |
| --- | --- | ---: | ---: | ---: | ---: |
| OLMo | main, 28 relations | 0.771 | 0.420 | 0.322 | 0.605 |
| OLMo | shared split, 25 relations | 0.624 | 0.585 | 0.441 | 0.735 |
| GPT-J | main, 28 relations | 0.343 | 0.263 | 0.297 | 0.339 |
| GPT-J | shared split, 25 relations | 0.155 | 0.085 | 0.079 | 0.173 |

Two things are worth stating plainly:

- **Top-1 selection is degenerate here.** For OLMo the affine readout never
  matches the model's answer: top-1 agreement is 0.00 in all 168 relation-by-
  beta cells of the measured grid, so the rule is decided entirely by the
  smallest-beta tie-break and picks β = 0.5 in all 28 relations. The number it
  produces (0.322) is the β = 0.5 column, not a tuning result. GPT-J has eight
  non-zero cells, every one of them at β = 3 or 5, which is the same story from
  the other side: agreement appears only once the first-order term dominates.
  The LRE sweep's convention transfers to this baseline in form only.
- **Per-relation tuning buys nothing over one good fixed beta.** On the
  extended main-protocol grid, tuning each relation's beta on its own targets
  gives OLMo ρ = 0.711 — below the best fixed β = 100 (0.782) and below the
  differential lens (0.771), even though the tuned rule is in-sample and the
  others are not. On the shared split it gives 0.786 against a best fixed beta
  of 0.711, but that margin is exactly what in-sample selection over 11 betas
  is expected to manufacture. The per-relation choice is mostly fitting rank
  noise; the fixed-beta curve is the honest summary.

## Reproducing

```sh
jlens --root code/results/olmo measure-jacobian-lens --legacy-split --layer 17 \
  --correct-year-targets --betas 0.5,1,1.5,2,3,5 --beta-selection top1 \
  --vector-dir code/generated/affine_beta/olmo/vectors \
  --out code/generated/affine_beta/olmo/jacobian_lens.json
cp code/generated/affine_beta/olmo/jacobian_lens.json \
  code/results/olmo/results/jacobian_lens_beta.json
jlens --root code/results/olmo statistics
```

`rerun` schedules the same two sweeps, one per protocol, so a fresh run folder
reports them without the copy step. `statistics` reports both selection rules
from one saved sweep, because the file retains every beta's aggregate; only the
readout itself needs the GPU.

The extended main-protocol grid above comes from
`generated/affine_beta/extend_grid.py`, which rescores the cached \(r_e\) and
\(d_e\) vectors. It replayed 513 of 513 OLMo beta = 1 ranks and 512 of 513
GPT-J ones exactly, the single GPT-J disagreement being a deep-tail rank that
moved by one between kernel orderings — the replay is the measurement, not an
approximation of it. The extended shared-split row was measured directly, by
rerunning that protocol with `--betas 0.5,1,1.5,2,3,5,10,20,50,100,1000`,
because its standard-grid run cached no vectors.

The fixed-beta files are unchanged by this work. Against the retained
`jacobian_lens.json` and `jacobian_lens_years.json`, over all 513 rows and
both readouts, 10 of 1026 OLMo rank comparisons and 8 of 1026 GPT-J ones
differ, every one of them by exactly one rank deep in the tail. No relation's
median log-rank moves by more than 1.4 × 10⁻⁵, and OLMo's does not move at all.

Implementation: [beta selection](../jlens/affine_beta.py),
[lens scorer](../jlens/measure_jacobian_lens.py), and
[statistics output](../jlens/paper_statistics.py). The fixed-beta baseline is
described in [protocol details](methods.md); the other decomposition of the
same readout is in the
[raw Jacobian baseline](raw-jacobian-baseline.md).
