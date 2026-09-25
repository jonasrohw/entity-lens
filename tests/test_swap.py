from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import configure
from entity_lens import data, swap
from entity_lens.config import Project
from entity_lens.modeling import LanguageModel
from entity_lens.records import read

LISTING = """{{short description|None}}
== List ==
{|class="wikitable"
!Country || Capital || Population || Largest city || Population
|-
| {{flag|France}} || '''[[Paris]]''' ||align=right| 1 || [[Lyon]] ||align=right| 2
|-
| {{flag|Spain}} || '''[[Madrid|Madrid]]''' ||align=right| 1 || [[Barcelona<br>City]] ||align=right| 2
|-
|{{flag|Italy}} || '''[[Rome]]''' || 1 || [[Milan]] || 2
|-
| {{flagdeco|Chile}} [[Chile|Republic]] || '''[[Santiago]]''' || 1 || [[Valparaiso]] || 2
|}
== See also ==
"""


def test_fold_and_links() -> None:
    assert swap.normalize_answer("Zürich") == "zurich"
    assert swap.normalize_answer("Bras\\u00edlia") == "brasilia"
    assert swap.normalize_answer("Porto-Novo") == "porto novo"
    assert swap.normalize_answer(" Washington, D.C.") == "washington d c"
    assert swap.wiki_link("[[Santa Cruz de la Sierra|Santa Cruz<br>de la Sierra]]") == "Santa Cruz de la Sierra"
    assert swap.wiki_link("'''[[Washington,&nbsp;D.C.]]'''") == "Washington, D.C."
    with pytest.raises(ValueError, match="No link"):
        swap.wiki_link("plain")


def test_answers_match() -> None:
    settings = Project().experiments.swap.filter
    assert swap.answers_match(" Washington, D.C.\nThe capital", "Washington D.C.", settings)
    assert swap.answers_match(" Washington\n", "Washington D.C.", settings)
    assert swap.answers_match(" New York\nThe largest", "New York City", settings)
    assert not swap.answers_match(" New\n", "New York City", settings)
    assert swap.answers_match(" Bras\\u00edlia\n", "Brasília", settings)
    assert swap.answers_match(" Rangoon", "Yangon", settings)
    assert not swap.answers_match(" Bujumbura\nThe", "Gitega", settings)
    assert not swap.answers_match("\n", "Gitega", settings)


def test_parse_capital_list() -> None:
    assert swap.parse_capital_list(LISTING) == [
        ("France", "Paris", "Lyon"),
        ("Spain", "Madrid", "Barcelona City"),
        ("Italy", "Rome", "Milan"),
        ("Chile", "Santiago", "Valparaiso"),
    ]


def test_two_way_shares_sum_to_one() -> None:
    grid = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0], [0.0, 1.0, 5.0]])
    shares = swap.two_way_shares(grid)
    assert sum(v for v in shares.values() if v is not None) == pytest.approx(1.0)
    additive = np.add.outer([0.0, 1.0], [0.0, 2.0])
    assert swap.two_way_shares(additive)["interaction"] == pytest.approx(0.0, abs=1e-12)
    assert swap.two_way_shares(np.ones((2, 2)))["anchor"] is None


def test_download_checks_hash(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "fetch", lambda url: LISTING.encode())
    with pytest.raises(RuntimeError, match="does not match"):
        swap.download_capital_list(project.experiments.swap)
    configure(project, "swap", wikipedia_sha256=hashlib.sha256(LISTING.encode()).hexdigest())
    assert swap.download_capital_list(project.experiments.swap) == LISTING


def test_swap_run(project: Project, loaded: LanguageModel, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "fetch", lambda url: LISTING.encode())
    configure(
        project,
        "swap",
        wikipedia_sha256=hashlib.sha256(LISTING.encode()).hexdigest(),
        excluded_countries=("Chile",),
        anchor_draws=3,
        anchor_references=2,
        athlete_anchor_pool=2,
        variance_grid=2,
    )
    configure(project, "swap", filter=project.experiments.swap.filter.model_copy(update={"max_new_tokens": 2}))
    original = swap.Experiment.filter_countries

    def accept_all(self: swap.Experiment, candidates: list[swap.Country]) -> list[swap.FilterRow]:
        rows = original(self, candidates)
        return [row.model_copy(update={"passed": True}) for row in rows]

    monkeypatch.setattr(swap.Experiment, "filter_countries", accept_all)
    swap.run(project)
    result = read(project.artifacts / "swap" / "swap.json", swap.SwapResult)
    assert result.countries == ("France", "Spain", "Italy")
    assert result.athletes == ("Anna", "Bob", "Carla", "Dan", "Eva")
    assert result.anchor_pool_sizes == {"countries": 3, "country_lens": 5, "athletes": 2}
    assert result.shared_subjects == 5
    assert [arm.relation for arm in result.arms][:2] == ["country_capital_city", "country_largest_city"]
    for arm in result.arms:
        assert set(arm.hits) >= {"own", "foreign", "nel", "anchor_alone", "wrong_donor"}
        assert all(arm.conditional_on_own[name] <= arm.hits[name] for name in arm.hits)
    for summary in result.anchors:
        assert summary.minimum["own"] <= summary.mean["own"] <= summary.maximum["own"]
    assert set(result.variance_shares) == {"country_capital_city", "country_largest_city"}
    assert set(result.currency_own_hits) == {3.0, 5.0, 10.0}
