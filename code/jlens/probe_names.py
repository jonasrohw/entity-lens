"""Inspect surname associations with the layer-17 differential Jacobian lens."""

import argparse

from jlens import splits
from jlens.config import MODEL_ID, NEUTRAL_PROMPT, ROOT
from jlens.io import read_json


def subject_state(model, tokenizer, name, device, layer):
    """Cache the unmodified state at the last subject token."""
    import torch

    prompt = NEUTRAL_PROMPT.format(name)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
    end = prompt.rindex(name) + len(name)
    position = min(len(tokenizer(prompt[:end]).input_ids) - 1, input_ids.shape[1] - 1)
    with torch.no_grad():
        output = model(input_ids=input_ids, output_hidden_states=True)
    return (
        input_ids,
        position,
        output.hidden_states[layer][0, position, :].detach().clone(),
    )


def parse_args():
    """Parse this measurement's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="quote names containing spaces")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.model import HiddenStateIntervention

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="eager"
        )
        .to(args.device)
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = read_json(ROOT / "data/relations/person_occupation.json")
    samples = splits.filter_identifying_objects(data["samples"], tokenizer)
    _, references, _ = splits.split_samples(samples)
    anchors = [sample["subject"] for sample in references]
    cache = {
        name: subject_state(model, tokenizer, name, args.device, args.layer)
        for name in dict.fromkeys(anchors + args.names)
    }
    norm_weight = model.model.norm.weight.detach()
    unembedding = model.lm_head.weight.detach().float()
    epsilon = float(getattr(model.model.norm, "variance_epsilon", 1e-5))
    intervention = HiddenStateIntervention(model)
    handle = model.model.layers[args.layer - 1].register_forward_hook(intervention.hook)
    try:
        for name in args.names:
            products = [
                intervention.jvp(
                    cache[anchor][2],
                    cache[name][2] - cache[anchor][2],
                    *cache[anchor][:2],
                )
                for anchor in anchors
                if anchor != name
            ]
            if not products:
                raise ValueError(f"{name}: no reference subjects available")
            # Preserve the original accumulation order.
            vector = sum(products[1:], products[0]) / len(products)
            vector = vector / torch.sqrt(vector.pow(2).mean() + epsilon)
            logits = (norm_weight * vector) @ unembedding.T
            tokens = [
                tokenizer.decode([int(index)]).strip()
                for index in logits.topk(8).indices
            ]
            print(f"{name:26s} -> {tokens}", flush=True)
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
