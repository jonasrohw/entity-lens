from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerFast

from entity_lens.config import ModelSpec, Project, PromptConfig, YearConfig
from entity_lens.data import Relation, RelationProperties, Sample
from entity_lens.tokens import context, first_object_token, query, single_token_ids, trailing_space, year_audit

YEARS = YearConfig(relations=frozenset({"birth_year"}), ignored_first_token="1")


def relation(key: str, samples: list[tuple[str, str]]) -> Relation:
    return Relation(
        key=key,
        name=key,
        prompt_templates=("{} was born in",),
        properties=RelationProperties(
            relation_type="factual", domain_name="person", range_name="year", symmetric=False
        ),
        samples=tuple(Sample(subject=s, object=o) for s, o in samples),
    )


def test_first_object_token_uses_leading_space(tokenizer: PreTrainedTokenizerFast) -> None:
    assert first_object_token(tokenizer, "woman", trailing_space=False) == tokenizer.encode(" woman")[0]


def test_first_object_token_skips_whitespace_after_trailing_space(tokenizer: PreTrainedTokenizerFast) -> None:
    token = first_object_token(tokenizer, " Paris", trailing_space=True)
    assert tokenizer.decode([token]).strip()


def test_trailing_space_only_for_year_relations(tokenizer: PreTrainedTokenizerFast, spec: ModelSpec) -> None:
    other = relation("capital", [("A", "Paris")])
    assert not trailing_space(tokenizer, other, spec, YEARS)
    years = relation("birth_year", [("A", "1743"), ("B", "1905")])
    always = spec.model_copy(update={"year_trailing_space": "always"})
    assert trailing_space(tokenizer, years, always, YEARS)
    assert isinstance(trailing_space(tokenizer, years, spec, YEARS), bool)


def test_context_and_query() -> None:
    prompts = PromptConfig(neutral="{} is known for", newline_prefix="\n", demonstration_separator="\n")
    demos = [Sample(subject="France", object="Paris"), Sample(subject="Spain", object="Madrid")]
    prefix = context("The capital of {} is", demos, prompts)
    assert prefix == "The capital of France is Paris\nThe capital of Spain is Madrid\n"
    assert query(prefix, "The capital of {} is", "Italy", trailing=True).endswith("Italy is ")
    assert query("", "{} is", "Italy", trailing=False) == "Italy is"


def test_year_audit(tokenizer: PreTrainedTokenizerFast) -> None:
    years = relation("birth_year", [("Anna", "1743"), ("Bob", "1905")])
    audit = year_audit(tokenizer, years, "", years.samples, YEARS, trailing=False)
    assert audit is not None
    assert len(audit.rows) == 2
    assert audit.excluded == any(row.first_token == "1" for row in audit.rows)
    assert year_audit(tokenizer, relation("capital", [("A", "Paris")]), "", (), YEARS, trailing=False) is None


def test_single_token_ids(tokenizer: PreTrainedTokenizerFast) -> None:
    ids = single_token_ids(tokenizer, ["woman", "man"])
    assert set(ids) == {"woman", "man"}
    with pytest.raises(ValueError, match="not a single token"):
        single_token_ids(tokenizer, ["Nairobi Santiago"])


def test_gender_terms_are_single_tokens_for_every_configured_tokenizer() -> None:
    project = Project()
    words = [w for group in project.protocol.gender.terms.values() for w in group]
    assert len(words) == 14
