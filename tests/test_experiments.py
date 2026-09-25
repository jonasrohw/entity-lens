from __future__ import annotations

import pytest

from conftest import configure
from entity_lens import data, prominence, prompt_variants, reference_sets, storage
from entity_lens.config import Project
from entity_lens.modeling import LanguageModel
from entity_lens.records import Readout, read, write
from entity_lens.reference_sets import ReferenceSetResult
from entity_lens.statistics import median_log_rank


def test_prompt_variants_reuse_targets(evaluated: Project, loaded: LanguageModel) -> None:
    configure(evaluated, "prompt_variants", prompts=("{} is known for", "Facts about {}:"))
    prompt_variants.run(evaluated)
    measured = prompt_variants.load(evaluated)
    main = storage.load_measurements(evaluated, "qwen3_8b_base")
    same = measured["{} is known for"]
    for relation, record in same.items():
        assert record.ranks == tuple(row.ranks[Readout.CEL] for row in main[relation].lens.rows)
        assert -median_log_rank(record.ranks) == pytest.approx(main[relation].lens.score(Readout.CEL))
    assert set(measured["Facts about {}:"]) == set(same)


def test_reference_sets(evaluated: Project, loaded: LanguageModel) -> None:
    configure(
        evaluated,
        "reference_sets",
        relation="name_gender",
        answer="woman",
        cross_relations=("country_capital_city", "country_largest_city"),
        n_references=2,
    )
    reference_sets.run(evaluated)
    result = read(evaluated.artifacts / "reference_sets" / "olmo3_7b.json", ReferenceSetResult)
    split = evaluated.protocol.split
    start = split.n_demonstrations + split.n_references
    rows = storage.load_measurements(evaluated, "olmo3_7b")["name_gender"].lre.rows[start : start + 100]
    answer = [row for row in rows if row.object == "woman"]
    assert set(result.arms) == {"other", "cross", "mixed", "answer"}
    assert result.n_rows == len(rows)
    assert result.n_answer_gold == len(answer)
    assert result.n_answer_test == len(answer) - 2
    assert result.arms["answer"].groups["other"].n == len(rows) - len(answer) - 2
    assert 0 <= result.monotone_subjects <= result.n_answer_test
    for arm in result.arms.values():
        assert len(arm.references) == 2
        assert len(arm.subtracted_top) == 10


def test_distinctive_words_and_matching() -> None:
    settings = Project().experiments.prominence
    words = prominence.distinctive_words("Johann Georg Kant", "Immanuel Kant", settings)
    assert words == {"johann", "georg"}
    assert prominence.distinctive_words("The Joker", "Batman", settings) == {"joker"}
    assert prominence.mentions(" he fought the joker", {"joker"})
    assert prominence.lens_hit(("Jok", "x"), {"joker"}, settings)
    assert not prominence.lens_hit(("jo", "x"), {"joker"}, settings)


def test_prominence_pipeline(evaluated: Project, loaded: LanguageModel) -> None:
    configure(
        evaluated, "prominence", relations=("country_capital_city",), shards=2, max_new_tokens=3, model="olmo3_7b"
    )
    for shard in range(2):
        prominence.generate(evaluated, shard)
    result = prominence.evaluate(evaluated)
    write(evaluated.artifacts / "prominence" / "prominence.json", result)
    row = result.relations[0]
    relation = data.load_relation(evaluated.data / "relations", "country_capital_city", evaluated.protocol.dataset)
    assert row.prominence_rows == len(relation.samples)
    assert row.lens_rows == len(relation.samples)
    assert 0 <= row.prominence <= 1
