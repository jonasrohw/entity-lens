# OLMo versus other models: follow-up audit

Subsequent experiment: [all seven models rerun with one leading newline](../results/all_model_newline/README.md).
That report retains the original checkpoints, splits, layers, and LRE outcomes,
and gives both differential and affine correlations under the identical prefixed
neutral text. This audit below describes the original unprefixed measurements.

The high OLMo main-protocol correlation reproduces in the same implementation
used for the newer models. It is not an artifact of loading a stale figure or
using different normalization code. The apparent cross-model gap nevertheless
mixes several scientifically consequential details: main versus shared
protocols, very small relations, tokenizer-dependent target information, and
agreement with a model's own predictions rather than full factual answers.

[Results, paired uncertainty, and diagnostic controls](../results/cross_model_audit/README.md)
are retained with the raw measurements.

## What was checked again

- Recomputed correlations from individual relation records; the saved OLMo
  figure agrees. The OLMo/GPT-J data copies match the data used for the new
  five-model comparison.
- Re-ran all 28 OLMo relations under both protocols in `model_sweep.py`, the
  same scorer used for the other five models. All 1,944 overlapping LRE grid
  cells match; 4,114 individual ranks were compared. Ninety ranks differ by
  at most two positions with prefix caching and batched float32 arithmetic.
  Both correlations reproduce exactly: main 0.770916, shared 0.624216.
- Every OLMo worker checked reconstructed HF logits, exact unmodified-state
  restoration, scalar/batched derivatives, finite differences, and cached
  versus full-prompt derivatives before scoring. OLMo was added to the tiny
  model regression tests, including short and multi-token cached-prefix cases.
- Inspected installed HF OLMo3 residual equations: attention output is
  normalized before adding its residual, then MLP output is normalized before
  its residual addition. The scorer calls HF blocks directly and intervenes
  after the complete block. It does not replace this with Mistral's pre-norm
  or GPT-J's parallel-residual equation.
- Both runners capture the pre-final-norm block output and decode through one
  actual final normalization and output head. The historical double-norm bug
  is not present. The earlier GPT-J final-input-cache fix is also retained.
- Verified model-specific operator fitting, the six LRE states and beta grid,
  fixed lens state, eight demonstrations/references, full-vocabulary targets,
  exact prompt construction, split fingerprints, self-reference exclusion,
  first-token filtering, and float32/eager execution.
- Recomputed target predictions and intercept-only faithfulness using ordinary
  HF forwards for all seven models (including GPT-J and OLMo). These do not use
  prefix caching, the fast Qwen convolution replacement, or Jacobian products.
  The script asserts the original split, model-correct counts, and selected
  LRE outcomes before comparing controls.
- Audited subject positions against tokenizer offsets and gold-token labels
  against the actual query-plus-answer tokenization. See the saved tokenizer
  and continuation audits for counts and individual cases.
- Refitted OLMo operators on seed 1 with every lens and per-relation LRE setting
  frozen from seed 0. Main correlation remains 0.756538; shared is 0.637414.
  These draws overlap in subjects and are not independent external validation.

## Details that explain much of the gap

**Protocol:** OLMo's 0.771 is a main-protocol result. On the shared protocol,
OLMo is 0.624, Mistral 0.625, and Qwen3.8-27B 0.596. Their paired bootstrap
intervals do not establish an OLMo advantage on that protocol.

**Small relations:** `name_gender` and both presidential-year relations have
only three LRE test subjects, but each receives the same weight in the
relation-level correlation as a relation with 100 tests. Keeping only relations
with at least four tests gives OLMo 0.771, Mistral 0.637, Qwen3.8-27B 0.637,
Gemma2 0.562, GPT-J 0.508, Qwen3-8B 0.448, and Pythia 0.227. This is an
exploratory changed-cohort sensitivity, not a replacement primary analysis.

**Target information differs:** for the three main birth-year test cases, OLMo
predicts distinct first tokens `178`, `188`, and `177`. Mistral, Gemma2, and
both new Qwens predict `1` for each. Intercept-only faithfulness is zero for
OLMo on these cases and one for those four models. Their perfect first-token
agreement does not show recovery of the year, yet it is treated as perfect LRE
faithfulness in the raw metric. This pushes low-lens-score relations to the
top of the outcome ranking for some models.

**The outcome differs more than the lens:** Mistral's and Qwen3.8-27B's lens
rankings correlate 0.839 and 0.851 with OLMo's. Mistral's lens paired with
OLMo's faithfulness gives 0.749; OLMo's lens paired with Mistral's faithfulness
gives 0.316. These crossed calculations are diagnostics, not valid model/LRE
systems, but localize much of the gap to the outcome side.

**Prior-only agreement:** subtracting intercept-only agreement and dividing by
remaining headroom gives OLMo main rho 0.773 and Mistral 0.719. Mistral excludes
two zero-headroom year relations in this calculation, so these are not the
same 28-relation statistic. The report retains all models and protocol results,
as well as conditioning on correctly predicted subjects.

The remaining differences are real in these measurements. Pythia and Qwen3-8B
do not catch up under every control. A shared layer fraction and prompt do not
guarantee equivalent representations, linearity, or relation knowledge across
architectures and training runs. These checks do not identify the causal
architectural/training reason for the residual gap. The earlier exhaustive
Mistral layer/beta search also found that large in-sample lens gains did not
reproduce consistently; simply maximizing correlation is not validation.

## Concrete labeling bug corrected

`first_object_token(..., trailing_space=True)` encoded an answer in isolation
and selected its first token. Mistral's SentencePiece tokenizer can prepend an
empty dummy-prefix token even to a bare year. The query has already supplied
that space, so its next gold token is `1`, not the dummy token. The helper now
skips empty/whitespace prefix tokens in this mode, while retaining content
tokens that include a leading space. New tests cover both cases.

The contextual audit found exactly six stable-prefix mismatches, all Mistral
year test cases. After the correction there are zero stable-prefix mismatches.
The selected subjects and first-token filter membership do not change for this
benchmark. This fixes those gold-accuracy labels, but changes neither the
model-top-1 faithfulness scores nor the correlation. Historical scoring JSON
is preserved; the new analysis applies contextual labels to its auxiliary
model-correct diagnostics and records the corrections.

The six saved GPT-J year examples have a different issue: their trailing-space
query is not a token prefix of the normally tokenized query plus answer. GPT-J
does not need OLMo's trailing-space workaround. This known prompt variant is
explicitly labeled in the comparison; dropping both year relations still
leaves GPT-J's main correlation at 0.361. It does not explain the entire gap.

The common runner also assumed every HF loader exposed `hf_device_map`. OLMo's
single-device loader omitted that metadata attribute; the runner now checks
actual parameter devices when it is absent. The initial attempts stopped before
scoring, and all reported reruns passed the GPU-placement checks.

## Limits and reproduction

The original datasets contain repeated and multi-answer subjects. Row-disjoint
splits are not always subject-disjoint. The separate
[Mistral cleaning study](mistral-grid-search.md) addresses that sensitivity;
cleaned results are not silently substituted into this comparison. Model
families also differ in vocabulary, pretraining, and post-training; Qwen3.8 is
post-trained while the other new checkpoints are base models.

From `code/`:

```sh
.venv/bin/python scripts/report_cross_model_audit.py --retain
.venv/bin/python scripts/audit_continuation_tokens.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_splits.py tests/test_gptj_reference.py tests/test_model_sweep.py \
  tests/test_affine_beta.py tests/test_grid_search.py tests/test_beta_intervals.py -q
```

The GPU worker manifests and checkpoint revisions are retained with the audit.
`audit_model_targets.py` resumes atomic per-relation files and supports multiple
workers through file locks. `run_mistral_grid.py --model` now also permits the
OLMo confirmation runs, without changing the default Mistral behavior.
