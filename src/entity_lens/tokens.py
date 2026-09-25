from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from entity_lens.config import ModelSpec, PromptConfig, YearConfig
from entity_lens.data import Relation, Sample
from entity_lens.modeling import decode
from entity_lens.records import YearAudit, YearRow


def first_object_token(tokenizer: PreTrainedTokenizerBase, obj: str, *, trailing_space: bool) -> int:
    ids: list[int] = tokenizer.encode(("" if trailing_space else " ") + obj, add_special_tokens=False)
    index = 0
    if trailing_space:
        while index + 1 < len(ids) and not decode(tokenizer, [ids[index]]).strip():
            index += 1
    return ids[index]


def trailing_space(tokenizer: PreTrainedTokenizerBase, relation: Relation, model: ModelSpec, years: YearConfig) -> bool:
    if relation.key not in years.relations:
        return False
    if model.year_trailing_space == "always":
        return True
    first = [
        decode(tokenizer, [first_object_token(tokenizer, sample.object, trailing_space=False)])
        for sample in relation.samples
    ]
    return all(not token.strip() for token in first)


def context(template: str, demonstrations: Sequence[Sample], prompts: PromptConfig) -> str:
    return "".join(
        template.format(sample.subject) + " " + sample.object + prompts.demonstration_separator
        for sample in demonstrations
    )


def query(prefix: str, template: str, subject: str, *, trailing: bool) -> str:
    return prefix + template.format(subject) + (" " if trailing else "")


def year_audit(
    tokenizer: PreTrainedTokenizerBase,
    relation: Relation,
    prefix: str,
    evaluation: Sequence[Sample],
    years: YearConfig,
    *,
    trailing: bool,
) -> YearAudit | None:
    if relation.key not in years.relations:
        return None
    rows = []
    for sample in evaluation:
        prompt = query(prefix, relation.template, sample.subject, trailing=trailing)
        encoded = tokenizer(
            prompt + ("" if trailing else " ") + sample.object,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        joint_ids = [
            token_id
            for token_id, (_, end) in zip(encoded.input_ids, encoded.offset_mapping, strict=True)
            if end > len(prompt) and decode(tokenizer, [token_id]).strip()
        ]
        first = decode(tokenizer, [first_object_token(tokenizer, sample.object, trailing_space=trailing)]).strip()
        joint = decode(tokenizer, [joint_ids[0]]).strip()
        if (first == years.ignored_first_token) != (joint == years.ignored_first_token):
            message = f"{relation.key}: isolated and in-context tokenization disagree for {sample.object!r}"
            raise AssertionError(message)
        rows.append(
            YearRow(
                subject=sample.subject,
                object=sample.object,
                first_token=first,
                joint_token=joint,
                ignored=first == years.ignored_first_token,
            )
        )
    return YearAudit(rows=tuple(rows))


def single_token_ids(tokenizer: PreTrainedTokenizerBase, words: Sequence[str]) -> dict[str, int]:
    ids = {}
    for word in words:
        encoded = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(encoded) != 1:
            message = f"Gender term {word!r} is not a single token: {encoded}"
            raise ValueError(message)
        ids[word] = encoded[0]
    return ids
