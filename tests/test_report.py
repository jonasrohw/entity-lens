from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast

from conftest import configure
from entity_lens import data, figures, latex, modeling, paper_values, report
from entity_lens.config import ModelsConfig, Project
from entity_lens.records import Readout
from entity_lens.statistics import Correlation


@pytest.fixture
def reporting(evaluated: Project, tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch) -> Project:
    models = evaluated.models
    evaluated.__dict__["models"] = ModelsConfig(models=(models.get("olmo3_7b"), models.get("qwen3_8b_base")))
    monkeypatch.setattr(modeling, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    configure(
        evaluated,
        "collapse",
        filtered_relations=("country_capital_city",),
        weak_token_relation="country_capital_city",
        weak_token="Paris",
    )
    configure(evaluated, "competition", examples=(("country_capital_city", "France"),))
    configure(evaluated, "anchored", splits=5, train_relations=5)
    configure(evaluated, "bootstrap", resamples=50)
    return evaluated


def write_corpus(project: Project) -> None:
    path = project.artifacts / "corpus" / "facts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relation", "subject", "object", "subject_count", "object_count", "cooccurrence"])
        for relation in data.entity_relations(project.data / "relations", project.protocol.dataset):
            for index, sample in enumerate(dict.fromkeys(relation.samples)):
                writer.writerow([relation.key, sample.subject, sample.object, 10 + index, 5 * index, index])


def test_build_report(reporting: Project) -> None:
    write_corpus(reporting)
    results = report.build(reporting)
    assert [m.model for m in results.models] == ["olmo3_7b", "qwen3_8b_base"]
    primary = results.models[0]
    assert primary.scored_rows == sum(s.n_rows for s in primary.relations)
    assert set(primary.correlations) == set(Readout)
    assert results.controls.domain_counts == {"country": 4, "name": 1, "person": 4}
    assert results.n_dataset_relations == 10
    assert results.n_entity_relations == 9
    assert results.corpus is not None
    assert results.corpus_unobserved_rows is not None
    assert results.corpus_unobserved_rows > 0
    assert "country_capital_city/France" in results.competition
    assert 0 < results.collapse.filtered_rows["country_capital_city"] <= 8
    rate = results.anchored.held_out_win_rate
    assert rate is None or 0 <= rate <= 1
    assert results.numeric_objects is None


def test_tables_and_values(reporting: Project, tmp_path: Path) -> None:
    results = report.build(reporting)
    written = latex.write_all(reporting, results, tmp_path / "tex")
    names = {path.stem for path in written}
    assert {"table_relations", "readout_comparison"} <= names
    relations = (tmp_path / "tex" / "table_relations.tex").read_text()
    assert r"\label{tab:relations}" in relations
    assert "country\\_capital\\_city" in relations
    values_path, comparison = paper_values.write(reporting, results, tmp_path)
    values = json.loads(values_path.read_text())
    assert values["olmo3_7b.cel.rho"] == results.models[0].correlations[Readout.CEL].rho
    assert "printed values match" in comparison.read_text()
    rows = paper_values.comparisons(reporting, results)
    assert {row.status for row in rows} <= {"match", "differs", "missing"}


def test_compare_rules() -> None:
    printed = paper_values.PrintedValue(printed=0.7355, digits=4, location="x")
    assert paper_values.compare(0.73546, printed) == "match"
    assert paper_values.compare(0.7346, printed) == "differs"
    assert paper_values.compare(None, printed) == "missing"
    assert paper_values.compare("a", paper_values.PrintedValue(printed="a", location="x")) == "match"
    assert paper_values.compare(3, paper_values.PrintedValue(printed=3, location="x")) == "match"


def test_latex_formatting() -> None:
    assert latex.signed(0.5, 2) == r"$\phantom{-}0.50$"
    assert latex.signed(-0.5, 2) == "$-0.50$"
    assert latex.p_value(0.0123) == "0.012"
    assert latex.p_value(8.24e-6) == r"$8.24\times10^{-6}$"
    assert latex.rho(None) == "--"
    assert latex.rho(Correlation(rho=None, p=None, n=3)) == "--"
    assert latex.escape("a_b") == r"a\_b"


def test_agreement_figure(reporting: Project, tmp_path: Path) -> None:
    results = report.build(reporting)
    primary = results.models[0]
    output = figures.agreement(reporting, list(primary.relations), 0.5, 1e-3, tmp_path / "figure.pdf")
    assert output.read_bytes().startswith(b"%PDF")
    assert figures.scientific(8.24e-6) == r"8.2 \times 10^{-6}"


def test_subset_comparison() -> None:
    printed = paper_values.PrintedValue(printed="a, c", subset=True, location="x")
    assert paper_values.compare("a, b, c", printed) == "match"
    assert paper_values.compare("a, b", printed) == "differs"
