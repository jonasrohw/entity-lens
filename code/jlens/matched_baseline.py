"""Scaffolding shared by the lens ablations matched to the retained legacy runs.

The differential Jacobian lens combines two ingredients: the Jacobian, and the
subtraction of a same-type reference. Dropping one at a time gives a 2x2:

    J, difference      mean_i J_i (s_e - s_i)   measure_jacobian_lens.py
    J, no difference   mean_i J_i s_e           measure_jacobian_lens_raw.py
    no J, difference   s_e - mean_i s_i         measure_logit_lens_diff.py
    no J, no diff      s_e                      measure_logit_lens.py

For the comparison to isolate the readout, every cell must score the same
subjects, at the same layer, against the same model answer tokens. The helpers
here rebuild the legacy split and refuse to score a relation whose template,
layer, subjects, or model answers have drifted from `results/jacobian_lens.json`.
"""

import random

import numpy as np

from jlens.config import ROOT, YEAR_RELATIONS
from jlens.io import read_json

N_ICL = 8
N_REFERENCES = 8
N_SUBJECTS = 20


def load_references(root=ROOT):
    """Retained differential-lens results, years taken from the corrected file."""
    main = read_json(root / "results/jacobian_lens.json")
    years = read_json(root / "results/jacobian_lens_years.json")
    return {
        relation: (years[relation] if relation in YEAR_RELATIONS else main[relation])
        for relation in sorted(main)
        if not relation.startswith("_")
    }


def matched_split(relation, reference, layer, root=ROOT):
    """Rebuild the legacy split and check it against the reference result.

    Returns `(template, context, subjects, anchors)`, where `context` is the
    8-shot demonstration prefix and `anchors` are the reference subjects the
    lens averages over -- under the legacy protocol, the first eight scored
    subjects.
    """
    if reference["layer"] != layer:
        raise ValueError(f"{relation}: reference layer differs from baseline")
    data = read_json(root / f"data/relations/{relation}.json")
    template = data["prompt_templates"][0]
    if template != reference["template"]:
        raise ValueError(f"{relation}: prompt template differs from reference")
    pool = list(data["samples"])
    random.Random(17).shuffle(pool)
    demonstrations = pool[:N_ICL]
    subjects = [row["subject"] for row in pool[N_ICL : N_ICL + N_SUBJECTS]]
    if subjects != [row["subject"] for row in reference["rows"]]:
        raise ValueError(f"{relation}: scored subjects differ from reference")
    context = "".join(
        template.format(row["subject"]) + " " + row["object"] + "\n"
        for row in demonstrations
    )
    return template, context, subjects, subjects[:N_REFERENCES]


def make_decoder(model, tokenizer):
    """Return `rank_of(vector, token_id)`, the scorers' norm-and-unembed readout."""
    import torch

    norm_weight = model.model.norm.weight.detach().float()
    unembed = model.lm_head.weight.detach().float()
    norm_epsilon = float(getattr(model.model.norm, "variance_epsilon", 1e-5))

    def rank_of(vector, token_id):
        vector = vector.float()
        vector = vector / torch.sqrt(vector.pow(2).mean() + norm_epsilon)
        logits = (norm_weight * vector) @ unembed.T
        rank = int((logits > logits[token_id]).sum())
        top = [tokenizer.decode([int(i)]).strip() for i in logits.topk(5).indices]
        return rank, top

    return rank_of


def model_answers(model, tokenizer, device, relation, reference, template, context):
    """The model's own top-1 per subject, verified against the reference rows.

    The year relations are queried with a trailing space, as the corrected
    `jacobian_lens_years.json` run was.
    """
    import torch

    answers = {}
    with torch.no_grad():
        for row in reference["rows"]:
            subject = row["subject"]
            prompt = context + template.format(subject)
            if relation in YEAR_RELATIONS:
                prompt += " "
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
            token_id = int(model(input_ids=input_ids).logits[0, -1].argmax())
            text = tokenizer.decode([token_id]).strip()
            if text != row["model_top1"]:
                raise ValueError(
                    f"{relation}/{subject}: model answer {text!r} differs from "
                    f"reference {row['model_top1']!r}"
                )
            answers[subject] = (token_id, text)
    return answers


def subject_states(model, tokenizer, device, subjects, layer, neutral_prompt):
    """Layer-`layer` hidden state at each subject's final token, neutral prompt.

    Also returns the prompt's token ids and the subject position, which the
    Jacobian ablation needs to re-enter the model at the reference subject.
    """
    import torch

    cache = {}
    with torch.no_grad():
        for subject in list(dict.fromkeys(subjects)):
            prompt = neutral_prompt.format(subject)
            input_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
            end = prompt.rindex(subject) + len(subject)
            position = min(
                len(tokenizer(prompt[:end]).input_ids) - 1, input_ids.shape[1] - 1
            )
            hidden = model(
                input_ids=input_ids, output_hidden_states=True
            ).hidden_states[layer][0, position, :]
            cache[subject] = (input_ids, position, hidden.detach().clone())
    return cache


def aggregate(ranks):
    """The four summary statistics every lens result file reports."""
    ranks = np.asarray(ranks, dtype=float)
    return dict(
        median_rank=float(np.median(ranks)),
        median_lograng=float(np.median(np.log10(ranks + 1))),
        agree_at1=float((ranks == 0).mean()),
        agree_at10=float((ranks < 10).mean()),
    )
