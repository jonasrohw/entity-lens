# Qwen country lens: position audit

Follow-up: the [complete newline-prefix rerun](../results/qwen_newline_rerun/README.md)
now covers all 28 main and 25 shared-split relations (2,057 subject rows).
Differential Spearman correlations change from 0.0604 to 0.2158 (main) and
from 0.2680 to 0.7291 (shared-split). The original audit below is preserved.

Measured 21 September 2026, `Qwen/Qwen3-8B-Base`, revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, float32, eager attention,
state 19 (output of transformer block 18, zero-based).

**The repeated `-Semit` list is a real experimental-setup artifact. The subject
indices are correct.** The original differential lens mixes first-position
country states with later-position reference states whose activation scales are
radically different. A shared component overwhelms the country-specific output.

The original prompt is `{} is known for`. Qwen's tokenizer adds no initial token
here. The ten single-token countries select position 0. Chile (`Ch`, `ile`),
Peru (`Per`, `u`), Saudi Arabia, South Korea, and United States select position 1;
Venezuela (`V`, `enez`, `uela`) selects position 2. All 16 computed indices match
an independent check against character offsets in the complete prompt.
The code captures and intervenes at the same block output. Moving -1 would be
out of bounds for all ten single-token countries (Python indexing would wrap to
` for`); moving +1 selects ` is` for every country.

At state 19, the single-token states have Euclidean norms **10,106–11,239**;
multitoken states have norms **128–141**. Coordinates 2276 and 233 together account
for **99.82%** of the single-token states' squared norm. Coordinate 2276 alone is
about +10,000 there, versus +25 to +40 at the selected multitoken positions.
These are actual captured activations, not a tokenizer or decoding discrepancy.

The lens averages `J_reference @ (state_country - state_reference)`. Its eight
references are Australia, Argentina, Egypt, Peru, Japan, United States, China,
and Brazil, excluding a country's own reference when applicable. This combines
six position-0 references and two position-1 references. For Germany:

- The mean product using the six single-token references has norm **6.23** and
  starts with `德国`, `武汉市`, `German`, `柏林`, `恩施`. Berlin's zero-based rank is 11.
- The mean product using Peru and United States has norm **4,586.73** and starts
  with `-Semit`, `(Initialized`, `iNdEx`, `搜狐首页`, `StringWriter`.
- Averaging all eight gives the repeated list and puts Berlin at rank **123,010**.

All ten single-token countries' final differential vectors have pairwise cosine
similarities above **0.99980**. Thus the repeated list comes from almost the same
vector direction in every case. Grouping reference contributions directly
isolates where this common component enters. It does not establish that every
same-group readout is accurate or that changing reference groups is a validated
replacement protocol.

A controlled rerun prepends exactly one newline token: `\n{} is known for`.
For **every country**, the complete original token-ID sequence is preserved after
that one token. This avoids the leading-space retokenization that a textual
prefix such as `Entity: ` introduces. The country remains the selected token;
its position increases by one. Selected state norms now range from **119–138**.
The repeated list disappears for all ten affected countries.

The same checkpoint, layer, original eight references, self-exclusions, and
relation-prompt target IDs are retained. These are zero-based ranks of the
model's original capital-prompt top-1 token, not gold-answer accuracy or complete
capital-name predictions. Across all 16 countries, median rank improves from
**18,328.5 to 9**. Some individual ranks worsen; the control does not make every
prediction correct. In particular, India's target is the token `New` and
Venezuela's is `Car`.

| Country | Original position | Original rank | Newline rank | Newline top five tokens |
| --- | ---: | ---: | ---: | --- |
| Australia | 0 | 22,182 | 18 | `澳洲`, `Aussie`, `Australian`, `澳大利亚`, `Australians` |
| Argentina | 0 | 55,413 | 3 | `阿根廷`, `Argentine`, `Argentina`, `Buenos`, `Argentina` |
| Egypt | 0 | 14,475 | 6 | `埃及`, `Egyptian`, `Egypt`, `Egyptians`, `Egypt` |
| Peru | 1 | 11 | 3 | `Peru`, `Ecuador`, `peru`, `Lima`, `Bolivia` |
| Japan | 0 | 97,273 | 1 | `Japanese`, `Tokyo`, `Japanese`, `Japan`, `japon` |
| United States | 1 | 43 | 317 | `NAFTA`, `美军`, `Guantanamo`, `Americans`, `芝加哥` |
| China | 0 | 77,252 | 12 | `Mandarin`, `Alibaba`, `Chinese`, `China`, `对中国` |
| Brazil | 0 | 147,049 | 51 | `巴西`, `Brazilian`, `Brazil`, `Brazil`, `brazil` |
| Venezuela | 2 | 11,086 | 9,397 | `Venezuelan`, `Venezuela`, `Maduro`, `Venezuel`, `venez` |
| India | 0 | 134,784 | 94,538 | `Bollywood`, `Hindu`, `Hindus`, `Indian`, `Indians` |
| Chile | 1 | 10 | 10 | `Chile`, `Chili`, `Oregon`, `copper`, `salmon` |
| Saudi Arabia | 1 | 1 | 7 | `Saudi`, `沙特`, `Saudi`, `沙漠`, `Saudis` |
| Germany | 0 | 123,010 | 11 | `Germany`, `Germans`, `德国`, `Germany`, `ドイツ` |
| Spain | 0 | 92,610 | 7 | `Spain`, `西班牙`, `Catalonia`, `Barcelona`, `Sevilla` |
| Russia | 0 | 9,270 | 8 | `Putin`, `Kremlin`, `Russians`, `中俄`, `普京` |
| South Korea | 1 | 1 | 3 | `是韩国`, `Korean`, `Samsung`, `Seoul`, `Koreans` |

Two further controls help distinguish possible fixes:

- **Deliberate +1 offset:** reads the following ` is` state. The repeated list
  disappears, but this is a different representation: median rank is 213,
  with the original multitoken countries' median worsening from 10.5 to 2,119.5.
  There is no justification for treating this as an indexing correction.
- **Remove the two largest coordinates from each difference vector:** keeping
  original prompts and Jacobians, zeroing differences at coordinates 2276 and
  233 replaces the repeated list with a different repeated list dominated by
  ellipses/newlines. Median rank is 42,471.5. This post-hoc ablation shows that
  simply clipping these two coordinates is insufficient; it is not a fix.

The newline control changes preceding context as well as the country's absolute
position; it is not a pure position-ID intervention. Together with the measured
activation magnitudes and reference-contribution analysis, it supports a
first-position activation mismatch as the explanation. No attention-weight
analysis was performed, so this report does not assign a particular attention
head or claim a fully traced attention-sink mechanism.

For future Qwen measurements, use a fixed neutral prefix consistently for both
targets and references, and rerun the affected lens analyses before interpreting
their relation-level correlations. This audit does not overwrite retained data
or silently change the project's default prompt. The newline is a diagnostic
control, not a prompt selected through held-out validation. The earlier
[prompt-sensitivity experiment](../results/qwen_prompt_sensitivity/README.md)
independently found that `Entity: {} is known for` removes the repeated country
list while preserving the suffix.

Validation used the real checkpoint on an A100 GPU. Reapplying each cached state
at every reference position reproduced the raw final state **exactly**, in all
four controls (32 checks). Decoding the captured final state reproduced model
logits exactly. Batched versus scalar JVP maximum absolute error was 0.000265;
the central finite-difference relative error was **0.0001368**. The unmodified
rerun recovered all original top-five token lists after stripping spaces, with
a maximum target-rank change of two places. This rules out a gross hook/layer
mismatch or broken derivative implementation in these tested paths.

Reproduce from `code/`:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/diagnose_qwen_country_positions.py
```

The [diagnostic script](../scripts/diagnose_qwen_country_positions.py) checks token
alignment and preserves source hashes, token IDs, offsets, activation norms,
largest coordinates, reference-level products, group readouts, cosine matrices,
and derivative checks in these files:

- [Original](../results/qwen_country_position_audit/original.json)
- [Newline control](../results/qwen_country_position_audit/newline.json)
- [+1 offset](../results/qwen_country_position_audit/next_token.json)
- [Two-coordinate ablation](../results/qwen_country_position_audit/without_outlier_coordinates.json)
