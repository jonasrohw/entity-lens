# ICLR 2027 submission

The retained seven-model comparison is in
[appendix_shared_baselines.tex](appendix_shared_baselines.tex), included as
Appendix A. A separately readable [appendix PDF](shared_baselines.pdf) is built
with `latexmk -pdf shared_baselines.tex`. Its tables cover the newline-prefixed
shared split: the first compares differential Jacobian, raw Jacobian,
logit lens, differential logit lens, and attribute rate across seven models;
the second reports only OLMo corpus-frequency results.
The third table has two panels: eight Qwen3 8B prompt templates (legacy and
shared splits) and the newline comparison across all seven models (shared
split). Generate it with
`../code/.venv/bin/python ../code/scripts/format_prompt_variation_table.py`.
Regenerate the first two presentation tables with
`../code/.venv/bin/python ../code/scripts/format_shared_baseline_tables.py`
from this directory. The full numerical report and sensitivity analyses
remain in the result files. The previous expanded appendix source is
preserved under `archive/appendix_shared_baselines_expanded_2026-09-21.tex`.
The main text incorporates collaborator revisions, including the updated headline
figure. The appendix experiments predate the final-normalization correction merged
on 22 September 2026; Gemma2 readouts require a corrected rerun. Their exact
measurement implementation is archived in the result provenance directory.
The appendix explicitly records this limitation rather than presenting these
measurements as validation of the merged implementation.

The [curation and reproduction guide](../code/docs/shared-baseline-appendix.md)
and [complete result tables](../code/results/shared_baselines/README.md) record
the exact protocol, reruns, reusable inputs and provenance.

The submission manuscript is `lens.tex`; the compiled file is `lens.pdf`. The
official anonymous review style is active. Build it directly:

```sh
cd paper
latexmk -pdf lens.tex
```

Requires a TeX Live installation with `latexmk`, pdfLaTeX and BibTeX. The
figure PDFs are committed, so a normal LaTeX build does not need Python or any
scientific packages. To regenerate the three vector figures without loading a
model, use the locked environment in `../code/`:

```sh
cd ../code && uv sync --locked && uv run python ../paper/repro/make_iclr_figures.py
```

`rawmaterial.tex` is a scratch file of unused draft material, not part of the
manuscript; `lens.tex` does not input it.

## Official template provenance

The style files (`iclr2027_conference.sty`, `iclr2027_conference.bst`, plus the
bundled `fancyhdr.sty` and `natbib.sty`) were downloaded unchanged on 2026-09-16
from the archive linked by the
[ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines):
[official style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).

Archive SHA-256:
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.

The archive's example manuscript and its bibliography, along with the earlier
ACL template files and superseded figures, were removed from this branch; they
remain on the `acl` branch if they are ever needed again.

## Historical template-conversion checks

Recorded from the template-conversion check before the new baseline appendix. The checker script
itself has been removed; re-verify by inspection before uploading.

- Nine pages of main text; statements and references follow, then appendices.
- Official ICLR 2027 review header, line numbers, and anonymous title block.
- US Letter pages, embedded fonts, and empty PDF author metadata.
- No missing citations/references, duplicate labels, overfull boxes, or visible
  draft placeholders. Figure labels were visually inspected.
- The official bibliography style is used. Table captions appear above tables.
- All 33 tests in `code/tests/test_reproduction.py` passed. The main, partial,
  shared-split, and fact-level correlations agree with the retained results.
  The reported regression residuals were also reproduced. This verifies the
  retained CPU analyses; it is not a rerun of the original GPU measurements.

That check covered mechanical properties, not scientific correctness or every
possible way an author could be identified. Harmless underfull-page
diagnostics may occur with the official layout; no geometry or font-size
overrides were introduced to force the page count.

## Changes beyond the template

The wide ACL table was converted to one column without changing its numbers.
The missing numeric-token appendix was restored from the draft's existing
editorial note and linked from the figure and table. Visible unfinished prose
was completed using the draft's stated methodological directions; no new
experimental results were added.

The bias is now written as `b_beta = mean_i[f(s_i) - beta J_i s_i]`, matching
the implemented estimator `mean_i[f(s_i) + beta J_i(s - s_i)]`. The previous
equations were inconsistent for beta other than one. The text now acknowledges
in-sample hyperparameter selection and the lack of a layer-sensitivity study,
instead of asserting unbiasedness or invariance of correlations to layer choice.
The relation identifier and rank-axis notation were corrected. The clipped
fact-level figures were reconstructed from the retained matched measurements.

## Items requiring the authors before submission

**This is a mechanically checked manuscript, not certification of a completed
or fully compliant OpenReview submission.**

1. Complete `ai_use.tex` with the full history of AI assistance and the authors'
   actual verification practices. It currently describes only the assistance
   known from this conversion. The same disclosure must be provided in
   OpenReview; see the [2027 AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
   No assertion that the authors reviewed all AI-assisted work has been invented.
2. Review and approve the scientific claims, revised equation, methodological
   qualifications, ethics statement, and editorial changes. Author-only notes
   remain disabled in the TeX source and do not appear in the PDF.
3. Complete OpenReview author information, conflicts, reciprocal-reviewer
   registration or applicable exemptions, submission quotas, and required
   policy acknowledgments. Confirm originality and absence of a prohibited
   simultaneous submission. These cannot be established from this repository.
4. Register the genuine abstract and complete author list by **September 18,
   2026, 23:59 AoE**, and upload the paper by **September 25, 2026, 23:59 AoE**.
   The [official dates](https://iclr.cc/Conferences/2027/Dates) and author guide
   control; recheck for updates before uploading.
5. Upload `lens.pdf` as the paper. If code is supplied separately, prepare an
   anonymous supplement: the full repository contains identifying provenance
   paths, editorial material, and Git history. Do not upload the repository
   verbatim as an anonymous supplement.

No submission has been uploaded to OpenReview.
