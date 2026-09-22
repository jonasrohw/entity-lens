# Frozen before OLMo measurements: 2026-09-22

Question: does the relation anchor retain control of the output when the
subject-dependent term comes from another relation of the same subject?
This is a replication/extension of paper/lens.tex, “What the operator's two
terms carry”. A null or adverse result is retained, not used to replace subjects.

Gemini 3.7 Flash via the existing Vertex billing project proposes 160 paired
subjects in each of six domains: countries, athletes, authors, films, landmarks,
cities. Separate, fresh-context Gemini calls audit every pair in batches of 32,
with a planted invalid control. Uncertain/rejected pairs and duplicates are
excluded before OLMo inference. These are model-generated and model-audited
facts. Before any OLMo inference, fetch each proposed Wikipedia page, extract its
infobox and lead, and run a separate source-backed Gemini audit in batches of 16.
Require both labels to be supported and the entity/relations unambiguous; drop
uncertain, false, missing-source and answer-copying cases. Source HTML hashes,
excerpts and audit verdicts are retained. This extra gate was added before
freezing because the self-audit missed outdated country facts. It is automated
source corroboration, not human adjudication or proof of zero errors.
For authors, supplement excerpts with Wikidata P6886 (writing language), joined
through the article's Wikidata ID, and re-audit with the added evidence. This
avoids inferring an author's language solely from nationality. Preserve the
earlier audit and both source snapshots; no OLMo outcome is consulted.
All exclusions and API responses are retained. This tests generalization across the chosen domains,
not a random sample of all knowledge. Film director/composer is a same-answer-type
stress test; country/currency and sport/position extend earlier cases.
Sport and position common nouns are lowercased consistently before freezing.

Shuffle accepted subjects with fixed seed 220922, independently by domain.
Take 8 demonstrations, 8 Jacobian references, 16 validation subjects, and all
remaining subjects as confirmation. Splits are subject-disjoint; confirmation
subjects never enter demonstrations, references, or tuning. Exclude cross-domain
subject duplicates deterministically before splitting. No OLMo outcome determines
inclusion. Hash and save the manifest before inference. Preserve all subjects even
when their LRE is weak. Strong-operator (both validation faithfulness >= .50)
strata are reported separately and are not selected on confirmation outcomes.

OLMo-3-1025-7B revision a81bae42db3975be1671e27b9c9a56da1a9f980f,
float32, eager attention, TF32 disabled. Layer 17, beta=3 is primary, chosen to
match the existing analysis. Each relation has its own eight demonstration
prompts and exactly the same subject splits. Neutral prompt: '{} is known for',
without demonstrations. Subject is the same string but relation-specific prefixes
can change its hidden state; this does NOT claim identical hidden states.

Compute exact reference-averaged JVPs T_r(e)=mean_i J_ri(s_re-s_ri),
A_r=mean_i f_r(s_ri), and neutral v(e) with the same reference subjects.
All vectors are at the final block before final normalization; normalize once
when decoding. Score own A_r+3*T_r, foreign A_r+3*T_other, neutral A_r+3*v,
anchor only, v alone, and wrong-subject controls for foreign and neutral terms.
Wrong donors are other held-out subjects, deterministically hash-selected with
different full-model target tokens for both relations whenever possible.
Also report all-subject cyclic derangements independent of model/gold targets.
References and demonstration subjects are never wrong donors.

Primary outcome: exact first-token faithfulness to the unmodified model,
with one paired subject contribution averaging the two relation directions.
Report whole cohort and the discriminating subset where the two unmodified
model target tokens differ. Gold first-token accuracy is separate and is NOT
complete-answer factual accuracy. Report the subset whose two gold tokens differ
and the model gets both gold tokens right. Token-prefix ambiguity is audited.

Primary contrasts: foreign minus own; foreign minus wrong-foreign; neutral
minus wrong-neutral; neutral minus anchor. Bootstrap whole paired subjects within
domain (20,000 replicates, seed 220922), then macro-average domains with equal
weight. Give 95% CIs, domain-specific results, and exact paired tests per direction
with Holm correction. Relations/layers/betas are never treated as independent
subjects. Intervals condition on this demonstration/reference split; they do not
cover retraining variance or generalization to untested domains.

“Interchangeable” requires the 90% CI for foreign-minus-own to lie within a
prespecified +/-10 percentage-point equivalence margin, plus a positive 95% CI
against the wrong-subject control. Mere non-significance is insufficient.
Robust universal interchangeability requires domain-level support, not only a
pooled favorable estimate. Validation selects secondary beta from [1,2,3,5,10]
on own-term faithfulness only, ties to smaller beta, applied unchanged to all
compositions for that target relation. All fixed-beta results are retained.

Verification: disjoint partitions; API audit coverage and invalid control;
repeatable decoding; zero direction JVP; finite-difference check; score own equals
diagonal composition; donor indices never self; bootstrap at subject level.
Save all per-subject predictions, uncertainty calculations, provenance and costs
as API token usage (no invented dollar billing estimate).
