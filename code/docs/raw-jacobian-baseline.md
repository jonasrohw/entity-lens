The raw Jacobian-lens baseline measures how well the Jacobian readout predicts
LRE faithfulness when applied directly to a subject's hidden representation.
For subject \(e\), its readout is

\[
v_e^{\mathrm{raw}} = \frac{1}{n_e}\sum_{i\ne e} J_i s_e,
\qquad
v_e^{\mathrm{diff}} = \frac{1}{n_e}\sum_{i\ne e} J_i(s_e-s_i).
\]

Here, \(s_e\) and \(s_i\) are the test and reference subjects' hidden states,
and \(J_i\) is evaluated at reference subject \(i\). It maps changes at that
subject's final token in an intermediate layer to the final block's output
at the last prompt position, before final normalization. The rest of each
reference prompt is held fixed. \(n_e\) counts the references remaining after
excluding self-comparisons.

The pipeline implements this comparison as follows:

1. Load the differential-lens results and reconstruct the same legacy split:
   shuffle with seed 17, use the first eight examples as demonstrations, and
   score the next up to 20 subjects. The first up to eight scored subjects are
   the references. A reference that is also the test subject is skipped.
2. Encode each subject under the neutral prompt `"{} is known for"` and extract
   the hidden state at its final token. The saved OLMo run uses hidden-state
   index 17; GPT-J uses index 15. Each raw baseline matches its corresponding
   differential run's layer.
3. At each reference, compute the Jacobian-vector product with direction
   `s_e`, in place of `s_e - s_i`. The code uses double backward and never
   materializes the full Jacobian. Average these products over references.
4. Apply the model's final normalization once, followed by its vocabulary
   output head. Rank the model's own top-1 answer token from the relation's
   eight-shot prompt. The scorer checks that the subjects, layer, template,
   and decoded target answers match the differential results. Both presidential
   year relations use the corrected trailing-space queries.
5. Save per-subject `rank_raw` and `top_raw`, reference counts, and aggregate
   rank and agreement statistics in `results/jacobian_lens_raw.json`. The run
   workflow schedules this stage after the differential-lens measurements.
   Its final statistics report correlates the negative median
   `log10(rank + 1)` for each relation with that relation's LRE faithfulness,
   using Spearman correlation and equal weight for each relation.

The neutral readout uses no relation-specific query or object labels; the
reference subjects are selected within each relation's dataset. The relation
prompt determines the evaluation target. Faithfulness is measured separately
as agreement between the relation-specific LRE and the full model.

This raw baseline also differs from the restored zeroth-order baseline:
it adds back \(\operatorname{mean}_i J_i s_i\) relative to the differential
readout, whereas the affine baseline adds back
\(\operatorname{mean}_i f(s_i)\).

Correlations recomputed from saved measurements on September 18, 2026:

| Measurement set | Relations | Differential ρ | Raw ρ | Raw − differential | Raw p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLMo, corrected raw-baseline measurement | 28 | 0.771 | 0.766 | −0.005 | 2.04 × 10⁻⁶ |
| OLMo, raw file currently in the main run folder | 28 | 0.771 | 0.745 | −0.026 | 5.45 × 10⁻⁶ |
| GPT-J, saved main run | 28 | 0.343 | 0.343 | 0.000 | 0.0735 |

For OLMo, the corrected raw readout has almost the same observed correlation
as the differential readout. For GPT-J, the saved readouts produce exactly
the same Spearman correlation, although their individual token ranks differ.
These point estimates show little observed benefit from reference subtraction
on this relation-level metric; they do not establish statistical equivalence.

The two OLMo rows reflect a documented normalization bug, not two choices of
reference subtraction. The main run's raw file retains the earlier measurement
that differentiated a post-normalization output and then normalized again.
The current implementation differentiates the pre-normalization final-block
output, matching the differential lens. The independent corrected measurement
gives ρ = 0.7658176991, compared with differential ρ = 0.7709162426.
See the [normalization verification](final-norm-fix-verification.md) for the
measurement history.

The corrected OLMo raw results are retained in
[`results/olmo-old/jacobian_lens_raw.json`](../results/olmo-old/jacobian_lens_raw.json)
and are identical to `generated/fixed_pipeline/results/jacobian_lens_raw.json`.
For this write-up, their subjects, decoded targets, and layers were checked
against the current OLMo differential results. The independently verified
pipeline's relation list, differential scores, and faithfulness scores also
match the current OLMo figure data exactly. The corrected row therefore uses
a matched comparison despite the raw file's historical directory name.

The other rows use the raw and differential files under
[`results/olmo/results/`](../results/olmo/results/)
and [`results/gptj/results/`](../results/gptj/results/), respectively, with
their own faithfulness measurements and corrected year targets. The GPT-J row
is a calculation from its saved results; this write-up does not independently
verify its measurement provenance or rerun either model. A normal statistics
report currently reads the raw file in its selected run folder, so the existing
OLMo folder still prints 0.745 until that measurement is refreshed.

Implementation: [raw scorer](../jlens/measure_jacobian_lens_raw.py),
[matched protocol and decoder](../jlens/matched_baseline.py),
[Jacobian-vector products](../jlens/model.py), and
[statistics output](../jlens/paper_statistics.py).
