[![Arxiv](https://img.shields.io/badge/Arxiv-2610.27241-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.27241)
[![Python Versions](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

Predicting Linear Relational Decodability Without the Relation
------

> **Abstract:** Linear relational embeddings (LREs) approximate a language model's subject-to-object mapping with an affine function. Their faithfulness varies enormously across relations, from near zero for `person_father` to the ceiling for `country_capital_city`. We investigate the factors that contribute to this variation and demonstrate that it can be significantly predicted from a readout that is not aware of the underlying relation. To do so, we propose a novel form of the Jacobian lens, the *entity lens*. Whereas the Jacobian lens is estimated across all tokens and concepts, the entity lens ELₜ is trained to predict what the model will output for an entity of type t under the neutral prompt "{} is known for"; e.g., we train ELs for countries and persons. After estimating an entity lens, we measure how highly it ranks the model's answer for a relation without using any relation-specific information in estimation and prediction. Across 28 relations of the LRE dataset on Olmo-3-7B this relation-agnostic score correlates with LRE faithfulness at ρ = 0.7355 (p < .01). This suggests that linear decodability is largely a property of what a subject's representation already carries, rather than of the relation being decoded. Moving the unit of analysis from the relation to the individual fact, we conclude that new relation datasets and analysis methods are needed to address several problems in existing datasets, most prominently semantically weak first tokens (e.g., most predicted universities' first token is "University") and reference competition (e.g., the LLM confuses obscure actor names with competing non-actor entities that have the same last name).

This repository contains the code, experimental pipelines, and saved measurements for the paper “Predicting Linear Relational Decodability Without the Relation”.

[CIS, LMU Munich](https://www.cis.lmu.de/) | [UKP Lab, TU Darmstadt](https://www.ukp.tu-darmstadt.de/) | [MCML](https://mcml.ai/)

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager. Use Linux, Python 3.12.12, and uv 0.9.27. Install dependencies and set up the virtual environment:

```bash
uv sync --locked
```

Copy `.env.example` to `.env` to configure local storage. For gated model downloads, accept the upstream licences and export `HF_TOKEN`.

## Run analysis from saved measurements

Verify saved artifacts, download the pinned dataset, and rebuild the paper reports:

```bash
./reproduce.sh analysis
```

This checks outputs against the existing checksum manifest. CPU report generation can require downloading pinned tokenizers.

## Run measurement pipeline

Run data acquisition and all measurement stages:

```bash
./reproduce.sh measure
```

GPU measurements require substantial GPU memory; corpus scans need several terabytes of storage. For a fresh run, set `ENTITY_LENS_ARTIFACTS_DIR` to a new, empty directory.

## Run full reproduction

Run measurements, build reports, and write a new checksum manifest:

```bash
./reproduce.sh all
```

## Run checks

```bash
./reproduce.sh check
uv run --locked entity-lens verify-artifacts
```

See the [reproduction documentation](docs/reproduction.md) for configuration, outputs, resource requirements, and verification scope. List individual pipeline stages with `uv run --locked entity-lens --help`.

## Cite

If you found our data or code helpful, please cite our paper:

```bibtex
@article{rohweder2026predicting,
  title={Predicting Linear Relational Decodability Without the Relation},
  author={Rohweder, Jonas and Mülthaler, Adrian and Mehri, Farid and Gerstner, Sebastian and Schütze, Hinrich},
  journal={arXiv preprint arXiv:2610.27241},
  year={2026},
  url={https://arxiv.org/abs/2610.27241}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
