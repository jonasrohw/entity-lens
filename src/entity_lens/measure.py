from __future__ import annotations

import time
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from entity_lens.config import ProtocolConfig
from entity_lens.data import Relation, Sample
from entity_lens.modeling import LanguageModel, PromptCache
from entity_lens.records import (
    GridCell,
    LensMeasurement,
    LensRow,
    LreMeasurement,
    LreRow,
    Readout,
    RelationMeasurement,
)
from entity_lens.splits import Split, full_evaluation_split
from entity_lens.tokens import context, first_object_token, query, single_token_ids, trailing_space, year_audit


def ranks_of(logits: Tensor, targets: Tensor) -> Tensor:
    return (logits > logits.gather(1, targets[:, None])).sum(-1)


def unique_subjects(*groups: Sequence[Sample]) -> list[str]:
    return list(dict.fromkeys(sample.subject for group in groups for sample in group))


def cache_subjects(
    lm: LanguageModel, prefix: str, template: str, subjects: Sequence[str], layers: Sequence[int], *, trailing: bool
) -> dict[str, PromptCache]:
    return {
        subject: lm.cache_prompt(query(prefix, template, subject, trailing=trailing), subject, layers)
        for subject in subjects
    }


def measure_lre(
    lm: LanguageModel, relation: Relation, split: Split, protocol: ProtocolConfig, *, trailing: bool
) -> LreMeasurement:
    layers = lm.spec.lre_layers
    prefix = context(relation.template, split.demonstrations, protocol.prompts)
    cache = cache_subjects(
        lm, prefix, relation.template, unique_subjects(split.references, split.evaluation), layers, trailing=trailing
    )
    rows = split.evaluation
    targets = torch.tensor([cache[row.subject].target for row in rows], device=lm.output_device)
    anchor = torch.stack([cache[ref.subject].final for ref in split.references]).mean(0).to(lm.output_device)
    intervention = lm.intervention()
    predictions: dict[tuple[int, float], Tensor] = {}
    ranks: dict[tuple[int, float], Tensor] = {}
    for layer in layers:
        subject_term = torch.zeros((len(rows), anchor.numel()), device=lm.output_device)
        with lm.hooked(layer, intervention):
            for ref in split.references:
                reference = cache[ref.subject]
                directions = torch.stack([cache[row.subject].states[layer] - reference.states[layer] for row in rows])
                subject_term += lm.jvps(intervention, reference, layer, directions).to(lm.output_device)
        subject_term /= len(split.references)
        for beta in protocol.lre.betas:
            with torch.no_grad():
                logits = lm.decode(anchor[None] + beta * subject_term)
            predictions[layer, beta] = logits.argmax(-1)
            ranks[layer, beta] = ranks_of(logits, targets)
    faithfulness = {cell: int((prediction == targets).sum()) / len(rows) for cell, prediction in predictions.items()}
    best = max(faithfulness, key=faithfulness.__getitem__)
    best_predictions = predictions[best].tolist()
    best_ranks = ranks[best].tolist()
    grid_predictions = torch.stack(list(predictions.values()), dim=1).tolist()
    return LreMeasurement(
        split=split,
        trailing_space=trailing,
        grid=tuple(
            GridCell(layer=layer, beta=beta, faithfulness=value) for (layer, beta), value in faithfulness.items()
        ),
        best_layer=best[0],
        best_beta=best[1],
        faithfulness=faithfulness[best],
        rows=tuple(
            LreRow(
                subject=row.subject,
                object=row.object,
                target_id=cache[row.subject].target,
                target=lm.token(cache[row.subject].target),
                gold_id=first_object_token(lm.tokenizer, row.object, trailing_space=trailing),
                prediction_id=prediction,
                hit=prediction == cache[row.subject].target,
                rank=rank,
                grid_predictions=tuple(cells),
            )
            for row, prediction, rank, cells in zip(rows, best_predictions, best_ranks, grid_predictions, strict=True)
        ),
    )


def lens_targets(
    lm: LanguageModel, relation: Relation, split: Split, protocol: ProtocolConfig, *, trailing: bool
) -> list[int]:
    prefix = context(relation.template, split.demonstrations, protocol.prompts)
    return [
        int(lm.next_token_logits(query(prefix, relation.template, row.subject, trailing=trailing)).argmax())
        for row in split.evaluation
    ]


class LensReadouts(NamedTuple):
    hidden: dict[Readout, Tensor]
    anchor: Tensor
    counts: Tensor


def lens_readouts(lm: LanguageModel, split: Split, prompt: str, layer: int) -> LensReadouts:
    rows = split.evaluation
    names = unique_subjects(split.references, rows)
    cache = {name: lm.cache_prompt(prompt.format(name), name, [layer]) for name in names}
    device = lm.output_device
    states = torch.stack([cache[row.subject].states[layer] for row in rows]).to(device)
    contrastive = torch.zeros((len(rows), cache[names[0]].final.numel()), device=device)
    noncontrastive = torch.zeros_like(contrastive)
    anchor = torch.zeros_like(contrastive)
    centers = torch.zeros_like(states)
    counts = torch.zeros(len(rows), device=device)
    intervention = lm.intervention()
    with lm.hooked(layer, intervention):
        for ref in split.references:
            reference = cache[ref.subject]
            directions = torch.stack([cache[row.subject].states[layer] - reference.states[layer] for row in rows])
            products = lm.jvps(intervention, reference, layer, torch.cat([directions, reference.states[layer][None]]))
            products = products.to(device)
            valid = torch.tensor([row.subject != ref.subject for row in rows], device=device)[:, None]
            contrastive += products[:-1] * valid
            noncontrastive += (products[:-1] + products[-1]) * valid
            anchor += reference.final.to(device) * valid
            centers += reference.states[layer].to(device) * valid
            counts += valid[:, 0]
    if not bool((counts > 0).all()):
        message = "At least one evaluation row has no eligible reference"
        raise ValueError(message)
    denominator = counts[:, None]
    hidden = {
        Readout.CEL: contrastive / denominator,
        Readout.NEL: noncontrastive / denominator,
        Readout.LOGIT_LENS: states,
        Readout.DIFFERENTIAL_LOGIT_LENS: states - centers / denominator,
    }
    return LensReadouts(hidden=hidden, anchor=anchor / denominator, counts=counts)


class TargetRanker:
    def __init__(self, lm: LanguageModel, targets: Sequence[int], alternatives: Sequence[Sequence[int]]) -> None:
        self.lm = lm
        self.targets = torch.tensor(list(targets), device=lm.output_device)
        self.alternatives = alternatives
        self.term_ids = sorted({token for group in alternatives for token in group})

    def literal(self, logits: Tensor) -> list[int]:
        ranks: list[int] = ranks_of(logits, self.targets).tolist()
        return ranks

    def expanded(self, logits: Tensor, literal: Sequence[int]) -> list[int]:
        if not self.term_ids:
            return list(literal)
        term_ranks = {token: ranks_of(logits, torch.full_like(self.targets, token)).tolist() for token in self.term_ids}
        return [
            min([literal[i]] + [term_ranks[token][i] for token in group]) for i, group in enumerate(self.alternatives)
        ]


def gender_alternatives(
    lm: LanguageModel, relation: Relation, protocol: ProtocolConfig, target_words: Sequence[str]
) -> list[list[int]]:
    if relation.key != protocol.gender.relation:
        return [[] for _ in target_words]
    ids = single_token_ids(lm.tokenizer, [word for group in protocol.gender.terms.values() for word in group])
    return [[ids[word] for word in protocol.gender.alternatives(target)] for target in target_words]


def measure_lens(
    lm: LanguageModel,
    relation: Relation,
    split: Split,
    protocol: ProtocolConfig,
    prompt: str,
    targets: Sequence[int],
    prior: Tensor,
) -> LensMeasurement:
    layer = lm.spec.lens_layer
    readouts = lens_readouts(lm, split, prompt, layer)
    target_words = [lm.token(target) for target in targets]
    ranker = TargetRanker(lm, targets, gender_alternatives(lm, relation, protocol, target_words))
    literal: dict[Readout, list[int]] = {}
    expanded: dict[Readout, list[int]] = {}
    tops: dict[Readout, list[list[int]]] = {}
    logit_lens_top: list[list[int]] = []
    with torch.no_grad():
        for readout, values in readouts.hidden.items():
            logits = lm.decode(values)
            literal[readout] = ranker.literal(logits)
            expanded[readout] = ranker.expanded(logits, literal[readout])
            tops[readout] = logits.topk(protocol.readouts.top_k_tokens).indices.tolist()
            if readout is Readout.LOGIT_LENS:
                logit_lens_top = logits.topk(protocol.readouts.logit_lens_top_k_ids).indices.tolist()
        anchored: dict[float, list[int]] = {}
        for beta in protocol.readouts.anchored_betas:
            logits = lm.decode(readouts.anchor + beta * readouts.hidden[Readout.CEL])
            anchored[beta] = ranker.expanded(logits, ranker.literal(logits))
    return LensMeasurement(
        split=split,
        prompt=prompt,
        layer=layer,
        rows=tuple(
            LensRow(
                subject=row.subject,
                object=row.object,
                target_id=targets[i],
                target=target_words[i],
                n_references=int(readouts.counts[i]),
                ranks={readout: values[i] for readout, values in expanded.items()},
                literal_ranks={readout: values[i] for readout, values in literal.items()},
                anchored_ranks={beta: values[i] for beta, values in anchored.items()},
                top_tokens={readout: tuple(lm.token(t) for t in values[i]) for readout, values in tops.items()},
                logit_lens_top_ids=tuple(logit_lens_top[i]),
                answer_prior=float(prior[targets[i]]),
            )
            for i, row in enumerate(split.evaluation)
        ),
    )


def answer_prior(lm: LanguageModel, contexts: Sequence[str]) -> Tensor:
    fallback = lm.tokenizer.bos_token or lm.tokenizer.eos_token
    distributions = [lm.next_token_logits(text or fallback).float().log_softmax(-1) for text in contexts]
    return torch.stack(distributions).mean(0).cpu()


def measure_relation(
    lm: LanguageModel, relation: Relation, protocol: ProtocolConfig, prior: Tensor, *, trailing: bool | None = None
) -> RelationMeasurement:
    start = time.monotonic()
    splits = protocol.split
    lre_split = full_evaluation_split(relation.samples, splits.lre_seeds, splits.n_demonstrations, splits.n_references)
    lens_split = full_evaluation_split(
        relation.samples, splits.lens_seeds, splits.n_demonstrations, splits.n_references
    )
    if trailing is None:
        trailing = trailing_space(lm.tokenizer, relation, lm.spec, protocol.years)
    lre = measure_lre(lm, relation, lre_split, protocol, trailing=trailing)
    targets = lens_targets(lm, relation, lens_split, protocol, trailing=trailing)
    lens = measure_lens(lm, relation, lens_split, protocol, protocol.prompts.lens_prompt(lm.spec), targets, prior)
    audit = year_audit(
        lm.tokenizer,
        relation,
        context(relation.template, lre_split.demonstrations, protocol.prompts),
        lre_split.evaluation,
        protocol.years,
        trailing=trailing,
    )
    return RelationMeasurement(
        model=lm.spec.key,
        relation=relation.key,
        data_sha256=protocol.dataset.sha256[relation.key],
        lre=lre,
        lens=lens,
        year_audit=audit,
        seconds=time.monotonic() - start,
    )
