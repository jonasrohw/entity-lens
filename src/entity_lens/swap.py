from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from entity_lens import data, modeling, pipeline
from entity_lens.config import Project, RuntimeConfig, SwapConfig, SwapFilterConfig
from entity_lens.data import Relation
from entity_lens.modeling import LanguageModel, PromptCache
from entity_lens.records import Record, write
from entity_lens.statistics import spearman
from entity_lens.tokens import context, first_object_token, query


class Country(Record):
    name: str
    capital: str
    largest_city: str
    benchmark: bool


class FilterRow(Record):
    country: str
    benchmark: bool
    continuations: dict[str, str]
    passed: bool


class ArmHits(Record):
    relation: str
    n: int
    hits: dict[str, int]
    conditional_on_own: dict[str, int]
    benchmark_own_hits: int | None
    added_own_hits: int | None
    norm_ratio: float
    own_predictions: dict[str, str]


class AnchorSummary(Record):
    relation: str
    mean: dict[str, float]
    minimum: dict[str, int]
    maximum: dict[str, int]


class SwapResult(Record):
    candidates: tuple[Country, ...]
    filter: tuple[FilterRow, ...]
    countries: tuple[str, ...]
    athletes: tuple[str, ...]
    arms: tuple[ArmHits, ...]
    currency_own_hits: dict[float, int]
    currency_n: int
    anchors: tuple[AnchorSummary, ...]
    composition_spearman: float | None
    variance_shares: dict[str, dict[str, float | None]]
    anchor_pool_sizes: dict[str, int]
    shared_subjects: int


def normalize_answer(text: str) -> str:
    unescaped = re.sub(r"\\u([0-9a-fA-F]{4})", lambda match: chr(int(match.group(1), 16)), text)
    decomposed = unicodedata.normalize("NFKD", unescaped)
    letters = "".join(char for char in decomposed if not unicodedata.combining(char)).lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", letters).split())


def answers_match(continuation: str, gold: str, settings: SwapFilterConfig) -> bool:
    answer = normalize_answer(continuation.split("\n", maxsplit=1)[0].strip()).split()
    for name in (gold, *settings.aliases.get(gold, ())):
        expected = normalize_answer(name).split()
        width = min(settings.match_words, len(expected))
        if expected and answer[:width] == expected[:width]:
            return True
    return False


def wiki_link(cell: str) -> str:
    match = re.search(r"\[\[([^\]]+)\]\]", cell)
    if match is None:
        message = f"No link in table cell {cell!r}"
        raise ValueError(message)
    label = match.group(1).split("|")[-1]
    return re.sub(r"<br\s*/?>", " ", label).replace("&nbsp;", " ").strip()


def parse_capital_list(text: str) -> list[tuple[str, str, str]]:
    table = text.split("== List ==", 1)[1].split("|}", 1)[0]
    states = []
    for row in table.split("\n|-"):
        if "{{flag" not in row:
            continue
        cells = [cell.strip() for cell in row.strip().lstrip("|").split("||")]
        flag = re.search(r"\{\{flag(?:deco)?\|([^}|]+)", cells[0])
        if flag is None:
            message = f"No country in row {cells[0]!r}"
            raise ValueError(message)
        states.append((flag.group(1).strip(), wiki_link(cells[1]), wiki_link(cells[3])))
    return states


def download_capital_list(config: SwapConfig) -> str:
    payload = data.fetch(config.wikipedia_list)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != config.wikipedia_sha256:
        message = f"Wikipedia list SHA-256 {digest} does not match {config.wikipedia_sha256}"
        raise RuntimeError(message)
    return payload.decode()


def first_objects(relation: Relation) -> dict[str, str]:
    objects: dict[str, str] = {}
    for sample in relation.samples:
        objects.setdefault(sample.subject, sample.object)
    return objects


def candidate_countries(config: SwapConfig, capital: Relation, largest: Relation, listing: str) -> list[Country]:
    capitals, largest_cities = first_objects(capital), first_objects(largest)
    benchmark = [
        Country(name=name, capital=capitals[name], largest_city=largest_cities[name], benchmark=True)
        for name in sorted(capitals)
        if name in largest_cities and capitals[name] != largest_cities[name]
    ]
    known = {country.name for country in benchmark}
    added = [
        Country(name=name, capital=city, largest_city=biggest, benchmark=False)
        for name, city, biggest in parse_capital_list(listing)
        if name not in known and name not in config.excluded_countries
    ]
    return benchmark + added


def demonstrations(relation: Relation, excluded: set[str], count: int) -> list[data.Sample]:
    return [sample for sample in relation.samples if sample.subject not in excluded][:count]


@dataclass
class PromptStates:
    caches: dict[str, PromptCache]
    layer: int

    def state(self, subject: str) -> Tensor:
        return self.caches[subject].states[self.layer]

    def final(self, subject: str) -> Tensor:
        return self.caches[subject].final


def cache_states(
    lm: LanguageModel, prompt_prefix: str, template: str, subjects: Sequence[str], layer: int
) -> PromptStates:
    unique = list(dict.fromkeys(subjects))
    caches = {
        subject: lm.cache_prompt(query(prompt_prefix, template, subject, trailing=False), subject, [layer])
        for subject in unique
    }
    return PromptStates(caches=caches, layer=layer)


def jvp_table(
    lm: LanguageModel, states: PromptStates, references: Sequence[str], subjects: Sequence[str]
) -> dict[str, Tensor]:
    intervention = lm.intervention()
    table = {}
    with lm.hooked(states.layer, intervention):
        for reference in dict.fromkeys(references):
            directions = torch.stack([states.state(subject) for subject in [*subjects, reference]])
            table[reference] = lm.jvps(intervention, states.caches[reference], states.layer, directions).to(
                lm.output_device
            )
    return table


@dataclass
class Cohorts:
    candidates: list[Country]
    filtered: list[FilterRow]
    excluded: set[str]
    countries: list[str]
    benchmark: set[str]
    anchor_pool: list[str]
    lens_pool: list[str]
    athletes: list[str]
    eligible_athletes: list[str]
    athlete_pool: list[str]
    answers: dict[str, dict[str, str]]


class Operator:
    def __init__(self, states: PromptStates, table: dict[str, Tensor], subjects: Sequence[str], beta: float) -> None:
        self.states = states
        self.table = table
        self.subjects = list(subjects)
        self.beta = beta

    def self_term(self, reference: str) -> Tensor:
        return self.table[reference][-1]

    def subject_term(self, subject: str, references: Sequence[str]) -> Tensor:
        index = self.subjects.index(subject)
        return torch.stack([self.table[reference][index] for reference in references]).mean(0)

    def anchor(self, references: Sequence[str]) -> Tensor:
        return torch.stack(
            [
                self.states.final(reference).to(self.self_term(reference).device)
                - self.beta * self.self_term(reference)
                for reference in references
            ]
        ).mean(0)


@dataclass
class Operators:
    relation: dict[str, Operator]
    country_lens: Operator
    athlete_lens: Operator


def leave_one_out(subjects: Sequence[str], subject: str) -> list[str]:
    return [other for other in subjects if other != subject]


def two_way_shares(grid: np.ndarray) -> dict[str, float | None]:
    grand = grid.mean()
    rows, columns = grid.shape
    anchor = columns * float(((grid.mean(1) - grand) ** 2).sum())
    lens = rows * float(((grid.mean(0) - grand) ** 2).sum())
    total = float(((grid - grand) ** 2).sum())
    if total == 0:
        return {"anchor": None, "lens": None, "interaction": None}
    return {"anchor": anchor / total, "lens": lens / total, "interaction": (total - anchor - lens) / total}


class Experiment:
    def __init__(self, project: Project, lm: LanguageModel, config: SwapConfig) -> None:
        self.project = project
        self.lm = lm
        self.config = config
        self.protocol = project.protocol
        self.layer = lm.spec.lens_layer
        self.relations = {
            name: data.load_relation(project.data / "relations", name, self.protocol.dataset)
            for name in {*config.country_relations, *config.person_pair}
        }

    def relation_states(self, name: str, excluded: set[str], subjects: Sequence[str]) -> PromptStates:
        relation = self.relations[name]
        prefix = context(
            relation.template,
            demonstrations(relation, excluded, self.protocol.split.n_demonstrations),
            self.protocol.prompts,
        )
        return cache_states(self.lm, prefix, relation.template, subjects, self.layer)

    def neutral_states(self, subjects: Sequence[str]) -> PromptStates:
        prompt = self.protocol.prompts.lens_prompt(self.lm.spec)
        return cache_states(self.lm, "", prompt, subjects, self.layer)

    def gold_id(self, answer: str) -> int:
        return first_object_token(self.lm.tokenizer, answer, trailing_space=False)

    def predict(self, hidden: Tensor) -> int:
        with torch.no_grad():
            return int(self.lm.decode(hidden[None]).argmax())

    def filter_countries(self, candidates: Sequence[Country]) -> list[FilterRow]:
        settings = self.config.filter
        runtime = RuntimeConfig.model_validate(
            self.protocol.runtime.model_dump() | {"dtype": settings.dtype, "attention": settings.attention}
        )
        generator = modeling.load(self.lm.spec, runtime)
        excluded = {country.name for country in candidates}
        rows = []
        for country in candidates:
            continuations = {}
            passed = True
            for name, answer in zip(self.config.country_pair, (country.capital, country.largest_city), strict=True):
                relation = self.relations[name]
                prefix = context(
                    relation.template,
                    demonstrations(relation, excluded, self.protocol.split.n_demonstrations),
                    self.protocol.prompts,
                )
                text = generator.generate(
                    query(prefix, relation.template, country.name, trailing=False), settings.max_new_tokens
                )
                continuations[name] = text
                passed &= answers_match(text, answer, settings)
            rows.append(
                FilterRow(country=country.name, benchmark=country.benchmark, continuations=continuations, passed=passed)
            )
        del generator
        torch.cuda.empty_cache()
        return rows

    def arms(
        self,
        relation: str,
        subjects: Sequence[str],
        answers: dict[str, str],
        operators: dict[str, Operator],
        benchmark: set[str] | None,
    ) -> ArmHits:
        beta = self.config.beta
        own_operator = operators["own"]
        hits: dict[str, list[bool]] = {arm: [] for arm in [*operators, "anchor_alone", "wrong_donor"]}
        ratios, predictions = [], {}
        for subject in subjects:
            references = leave_one_out(subjects, subject)
            anchor = own_operator.anchor(references)
            gold = self.gold_id(answers[subject])
            for arm, operator in operators.items():
                prediction = self.predict(anchor + beta * operator.subject_term(subject, references))
                hits[arm].append(prediction == gold)
                if arm == "own":
                    predictions[subject] = self.lm.token(prediction)
            donor = subjects[(subjects.index(subject) + 1) % len(subjects)]
            nel = operators["nel"]
            hits["wrong_donor"].append(
                self.predict(anchor + beta * nel.subject_term(donor, leave_one_out(subjects, donor))) == gold
            )
            hits["anchor_alone"].append(self.predict(anchor) == gold)
            ratios.append(float(beta * own_operator.subject_term(subject, references).norm() / anchor.norm()))
        own = hits["own"]
        return ArmHits(
            relation=relation,
            n=len(subjects),
            hits={arm: sum(values) for arm, values in hits.items()},
            conditional_on_own={
                arm: sum(a and b for a, b in zip(own, values, strict=True)) for arm, values in hits.items()
            },
            benchmark_own_hits=None
            if benchmark is None
            else sum(h for s, h in zip(subjects, own, strict=True) if s in benchmark),
            added_own_hits=None
            if benchmark is None
            else sum(h for s, h in zip(subjects, own, strict=True) if s not in benchmark),
            norm_ratio=float(np.mean(ratios)),
            own_predictions=predictions,
        )

    def cohorts(self) -> Cohorts:
        config = self.config
        capital_name, largest_name = config.country_pair
        position_name, sport_name = config.person_pair
        listing = download_capital_list(config)
        candidates = candidate_countries(config, self.relations[capital_name], self.relations[largest_name], listing)
        filtered = self.filter_countries(candidates)
        passed = {row.country for row in filtered if row.passed}
        countries = [c.name for c in candidates if c.name in passed]
        excluded = {country.name for country in candidates}
        country_union = self.subject_union(config.country_relations)
        demo_subjects = self.demonstration_subjects(config.country_pair, excluded)
        athletes = sorted(
            set(first_objects(self.relations[position_name])) & set(first_objects(self.relations[sport_name]))
        )
        athlete_demos = self.demonstration_subjects(config.person_pair, set(athletes))
        eligible = sorted(
            a for a in self.subject_union(config.person_pair) if a not in athletes and a not in athlete_demos
        )
        by_name = {country.name: country for country in candidates}
        return Cohorts(
            candidates=candidates,
            filtered=filtered,
            excluded=excluded,
            countries=countries,
            benchmark={c for c in countries if by_name[c].benchmark},
            anchor_pool=sorted(c for c in country_union if c not in countries and c not in demo_subjects),
            lens_pool=sorted(c for c in country_union if c not in countries),
            athletes=athletes,
            eligible_athletes=eligible,
            athlete_pool=sorted(random.Random(config.seed).sample(eligible, config.athlete_anchor_pool)),
            answers={
                capital_name: {c: by_name[c].capital for c in countries},
                largest_name: {c: by_name[c].largest_city for c in countries},
                position_name: first_objects(self.relations[position_name]),
                sport_name: first_objects(self.relations[sport_name]),
            },
        )

    def subject_union(self, names: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(s.subject for name in names for s in self.relations[name].samples))

    def demonstration_subjects(self, names: Sequence[str], excluded: set[str]) -> set[str]:
        count = self.protocol.split.n_demonstrations
        return {s.subject for name in names for s in demonstrations(self.relations[name], excluded, count)}

    def operator(self, states: PromptStates, references: Sequence[str], subjects: Sequence[str]) -> Operator:
        return Operator(states, jvp_table(self.lm, states, references, subjects), subjects, self.config.beta)

    def operators(self, cohorts: Cohorts) -> Operators:
        config = self.config
        countries, athletes = cohorts.countries, cohorts.athletes
        country_references = [*countries, *cohorts.anchor_pool]
        athlete_references = [*athletes, *cohorts.athlete_pool]
        relation = {
            name: self.operator(
                self.relation_states(name, cohorts.excluded, country_references), country_references, countries
            )
            for name in config.country_pair
        }
        relation[config.currency_relation] = self.operator(
            self.relation_states(config.currency_relation, set(countries), country_references),
            country_references,
            countries,
        )
        relation |= {
            name: self.operator(
                self.relation_states(name, set(athletes), athlete_references), athlete_references, athletes
            )
            for name in config.person_pair
        }
        country_lens_references = [*countries, *cohorts.lens_pool]
        return Operators(
            relation=relation,
            country_lens=self.operator(
                self.neutral_states(country_lens_references), country_lens_references, countries
            ),
            athlete_lens=self.operator(self.neutral_states(athletes), athletes, athletes),
        )

    def recombinations(self, cohorts: Cohorts, operators: Operators) -> list[ArmHits]:
        config = self.config
        capital_name, largest_name = config.country_pair
        position_name, sport_name = config.person_pair
        arms = []
        for relation, other in ((capital_name, largest_name), (largest_name, capital_name)):
            country_arms = {
                "own": operators.relation[relation],
                "foreign": operators.relation[other],
                "nel": operators.country_lens,
                "currency": operators.relation[config.currency_relation],
            }
            arms.append(
                self.arms(relation, cohorts.countries, cohorts.answers[relation], country_arms, cohorts.benchmark)
            )
        for relation, other in ((position_name, sport_name), (sport_name, position_name)):
            athlete_arms = {
                "own": operators.relation[relation],
                "foreign": operators.relation[other],
                "nel": operators.athlete_lens,
            }
            arms.append(self.arms(relation, cohorts.athletes, cohorts.answers[relation], athlete_arms, None))
        return arms

    def currency_hits(self, cohorts: Cohorts, operators: Operators) -> tuple[dict[float, int], int]:
        operator = operators.relation[self.config.currency_relation]
        answers = first_objects(self.relations[self.config.currency_relation])
        subjects = [c for c in cohorts.countries if c in answers]
        hits = {}
        for beta in self.config.currency_betas:
            scaled = Operator(operator.states, operator.table, operator.subjects, beta)
            hits[beta] = sum(
                self.predict(
                    scaled.anchor(leave_one_out(cohorts.countries, subject))
                    + beta * scaled.subject_term(subject, leave_one_out(cohorts.countries, subject))
                )
                == self.gold_id(answers[subject])
                for subject in subjects
            )
        return hits, len(subjects)

    def draw(self, generator: random.Random, pool: Sequence[str]) -> list[str]:
        return sorted(generator.sample(list(pool), self.config.anchor_references))

    def pair_resampling(
        self,
        pair: tuple[str, str],
        pool: Sequence[str],
        lens: Operator,
        subjects: Sequence[str],
        cohorts: Cohorts,
        operators: Operators,
        generator: random.Random,
    ) -> tuple[list[AnchorSummary], list[list[str]], dict[str, list[int]], random.Random]:
        beta = self.config.beta
        draws = [self.draw(generator, pool) for _ in range(self.config.anchor_draws)]
        summaries, lens_hits = [], {}
        for relation, other in (pair, pair[::-1]):
            own, foreign = operators.relation[relation], operators.relation[other]
            gold = {s: self.gold_id(cohorts.answers[relation][s]) for s in subjects}
            terms = {
                arm: {s: op.subject_term(s, leave_one_out(subjects, s)) for s in subjects}
                for arm, op in (("own", own), ("foreign", foreign), ("nel", lens))
            }
            results: dict[str, list[int]] = {arm: [] for arm in [*terms, "anchor_alone"]}
            for references in draws:
                anchor = own.anchor(references)
                for arm, term in terms.items():
                    results[arm].append(sum(self.predict(anchor + beta * term[s]) == gold[s] for s in subjects))
                results["anchor_alone"].append(sum(self.predict(anchor) == gold[s] for s in subjects))
            summaries.append(
                AnchorSummary(
                    relation=relation,
                    mean={arm: float(np.mean(v)) for arm, v in results.items()},
                    minimum={arm: int(np.min(v)) for arm, v in results.items()},
                    maximum={arm: int(np.max(v)) for arm, v in results.items()},
                )
            )
            lens_hits[relation] = results["nel"]
        return summaries, draws, lens_hits, generator

    def variance_shares(
        self, generator: random.Random, cohorts: Cohorts, operators: Operators
    ) -> dict[str, dict[str, float | None]]:
        size = self.config.variance_grid
        anchors = [self.draw(generator, cohorts.anchor_pool) for _ in range(size)]
        lenses = [self.draw(generator, cohorts.lens_pool) for _ in range(size)]
        shares = {}
        for relation in self.config.country_pair:
            gold = {c: self.gold_id(cohorts.answers[relation][c]) for c in cohorts.countries}
            grid = np.zeros((size, size))
            for a, anchor_references in enumerate(anchors):
                anchor = operators.relation[relation].anchor(anchor_references)
                for b, lens_references in enumerate(lenses):
                    grid[a, b] = sum(
                        self.predict(
                            anchor + self.config.beta * operators.country_lens.subject_term(c, lens_references)
                        )
                        == gold[c]
                        for c in cohorts.countries
                    )
            shares[relation] = two_way_shares(grid)
        return shares

    def resampling(
        self, cohorts: Cohorts, operators: Operators
    ) -> tuple[list[AnchorSummary], float | None, dict[str, dict[str, float | None]]]:
        config = self.config
        countries, _, _, generator = self.pair_resampling(
            config.country_pair,
            cohorts.anchor_pool,
            operators.country_lens,
            cohorts.countries,
            cohorts,
            operators,
            random.Random(config.seed),
        )
        shares = self.variance_shares(generator, cohorts, operators)
        athlete_generator = (
            self.benchmark_stream(cohorts) if config.athlete_stream_after_benchmark else random.Random(config.seed)
        )
        athletes, draws, lens_hits, _ = self.pair_resampling(
            config.person_pair,
            cohorts.athlete_pool,
            operators.athlete_lens,
            cohorts.athletes,
            cohorts,
            operators,
            athlete_generator,
        )
        counts = [sum(self.sport(r) == config.composition_object for r in draw) for draw in draws]
        composition = spearman(counts, lens_hits[config.person_pair[0]]).rho
        return [*countries, *athletes], composition, shares

    def sport(self, athlete: str) -> str | None:
        sports = first_objects(self.relations[self.config.composition_relation])
        positions = first_objects(self.relations[self.config.person_pair[0]])
        return sports.get(athlete) or self.config.position_sports.get(positions.get(athlete, ""))

    def benchmark_stream(self, cohorts: Cohorts) -> random.Random:
        generator = random.Random(self.config.seed)
        benchmark = sorted(c.name for c in cohorts.candidates if c.benchmark)
        union = set(self.subject_union(self.config.country_relations))
        demos = self.demonstration_subjects(self.config.country_pair, set(benchmark))
        anchor_pool = sorted(union - set(benchmark) - demos)
        lens_pool = sorted(union - set(benchmark))
        for _ in range(self.config.anchor_draws + self.config.variance_grid):
            self.draw(generator, anchor_pool)
        for _ in range(self.config.variance_grid):
            self.draw(generator, lens_pool)
        return generator

    def shared_subjects(self) -> int:
        first, second = (
            set(first_objects(data.load_relation(self.project.data / "relations", name, self.protocol.dataset)))
            for name in self.config.shared_subject_relations
        )
        return len(first & second)

    def run(self) -> SwapResult:
        cohorts = self.cohorts()
        operators = self.operators(cohorts)
        currency_hits, currency_n = self.currency_hits(cohorts, operators)
        summaries, composition, shares = self.resampling(cohorts, operators)
        return SwapResult(
            candidates=tuple(cohorts.candidates),
            filter=tuple(cohorts.filtered),
            countries=tuple(cohorts.countries),
            athletes=tuple(cohorts.athletes),
            arms=tuple(self.recombinations(cohorts, operators)),
            currency_own_hits=currency_hits,
            currency_n=currency_n,
            anchors=tuple(summaries),
            composition_spearman=composition,
            variance_shares=shares,
            anchor_pool_sizes={
                "countries": len(cohorts.anchor_pool),
                "country_lens": len(cohorts.lens_pool),
                "athletes": len(cohorts.eligible_athletes),
            },
            shared_subjects=self.shared_subjects(),
        )


def run(project: Project) -> None:
    config = project.experiments.swap
    spec = project.models.get(config.model)
    lm = pipeline.load_checked(project, spec, "swap")
    result = Experiment(project, lm, config).run()
    write(project.artifacts / "swap" / "swap.json", result)
