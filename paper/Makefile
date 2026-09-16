.PHONY: all check figures

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build lens.tex
	cp build/lens.pdf iclr2027_submission.pdf

check: all
	python3 check_submission.py

figures:
	cd ../code && uv sync --locked && uv run python ../paper/repro/make_iclr_figures.py
