from __future__ import annotations

from typing import Annotated

import typer

from entity_lens import (
    artifacts,
    data,
    modeling,
    pipeline,
    prominence,
    prompt_variants,
    reference_sets,
    scheduler,
    swap,
)
from entity_lens.config import Project
from entity_lens.records import write as write_record

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False, add_completion=False)
ModelOption = Annotated[
    list[str] | None, typer.Option("--model", help="Model keys; defaults to every configured model")
]


def _models(project: Project, keys: list[str] | None) -> list[str]:
    return keys or [model.key for model in project.models.models]


def _job(name: str, gpus: int, *arguments: str) -> scheduler.Job:
    return scheduler.Job(name=name, command=scheduler.cli_command(*arguments), gpus=gpus)


def _run(project: Project, jobs: list[scheduler.Job]) -> None:
    devices = scheduler.available_devices(project.protocol.runtime.idle_gpu_memory_mib)
    scheduler.run_jobs(jobs, project.artifacts / "logs", devices)


@app.command("download-data")
def download_data() -> None:
    project = Project()
    data.download(project.protocol.dataset, project.data / "relations")


@app.command("download-models")
def download_models(model: ModelOption = None) -> None:
    project = Project()
    for key in _models(project, model):
        print(modeling.download(project.models.get(key), project.protocol.download))


@app.command("evaluate")
def evaluate(model: str) -> None:
    pipeline.evaluate_model(Project(), model)


@app.command("evaluate-all")
def evaluate_all(model: ModelOption = None) -> None:
    project = Project()
    jobs = [_job(f"evaluate_{key}", project.models.get(key).gpus, "evaluate", key) for key in _models(project, model)]
    _run(project, jobs)


@app.command("numeric-objects")
def numeric_objects() -> None:
    project = Project()
    pipeline.evaluate_without_trailing_space(project, project.experiments.numeric_objects.model)


@app.command("prompt-variants")
def run_prompt_variants() -> None:
    prompt_variants.run(Project())


@app.command("prominence-generate")
def prominence_generate(shard: int) -> None:
    prominence.generate(Project(), shard)


@app.command("reference-sets")
def run_reference_sets() -> None:
    reference_sets.run(Project())


@app.command("swap")
def run_swap() -> None:
    swap.run(Project())


@app.command("experiments")
def experiments() -> None:
    project = Project()
    settings = project.experiments
    gpus = project.models.get
    jobs = [
        _job("numeric_objects", gpus(settings.numeric_objects.model).gpus, "numeric-objects"),
        _job("prompt_variants", gpus(settings.prompt_variants.model).gpus, "prompt-variants"),
        _job("reference_sets", gpus(settings.reference_sets.model).gpus, "reference-sets"),
        _job("swap", gpus(settings.swap.model).gpus, "swap"),
    ]
    jobs += [
        _job(f"prominence_{shard}", gpus(settings.prominence.model).gpus, "prominence-generate", str(shard))
        for shard in range(settings.prominence.shards)
    ]
    _run(project, jobs)
    write_record(project.artifacts / "prominence" / "prominence.json", prominence.evaluate(project))


@app.command("corpus-download")
def corpus_download(*, verify_checksums: bool = False) -> None:
    from entity_lens import corpus

    corpus.download(Project(), verify_checksums=verify_checksums)


@app.command("corpus-scan")
def corpus_scan() -> None:
    from entity_lens import corpus

    project = Project()
    corpus.scan(project, project.settings.workers)
    corpus.facts(project)


@app.command("wikipedia")
def wikipedia_retrieval() -> None:
    from entity_lens import wikipedia

    project = Project()
    wikipedia.prepare(project)
    wikipedia.retrieve(project, project.settings.workers)


@app.command("attribute-rates")
def attribute_rates(model: ModelOption = None) -> None:
    from entity_lens import wikipedia

    project = Project()
    for key in _models(project, model):
        wikipedia.score(project, key)


@app.command("report")
def report() -> None:
    from entity_lens import figures, latex, paper_values
    from entity_lens import report as paper_report
    from entity_lens.records import Readout

    project = Project()
    results = paper_report.build(project)
    output = project.artifacts / "paper"
    write_record(output / "results.json", results)
    latex.write_all(project, results, output / "tex")
    primary = next(m for m in results.models if m.model == project.experiments.primary.model)
    headline = primary.correlations[Readout.CEL]
    if headline.rho is not None and headline.p is not None:
        figures.agreement(project, list(primary.relations), headline.rho, headline.p, output / "fig_agreement.pdf")
    _, comparison = paper_values.write(project, results, output)
    print(comparison.read_text().splitlines()[0])


@app.command("manifest")
def manifest() -> None:
    project = Project()
    print(artifacts.write_manifest(project.artifacts))


@app.command("verify-artifacts")
def verify_artifacts() -> None:
    project = Project()
    artifacts.verify_manifest(project.artifacts)
