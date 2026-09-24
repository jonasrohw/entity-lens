"""Reuse a constant prompt prefix with the existing prefix-cache tolerances.

All queries remain scalar. Check every fourth query against a complete ordinary
HF forward, including exact top-1 equality. A rejected check disables this
optional path and the caller recomputes the complete call using full forwards.
"""
import copy
import time

import torch


def run(model, tokenizer, jobs, layers, capture_states):
    from jlens import model_sweep as sweep

    start = time.monotonic()
    device = model.get_input_embeddings().weight.device
    head_device = model.get_output_embeddings().weight.device
    entries = []
    common = None
    for prompt, subject in jobs:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        position = None
        if capture_states:
            end = prompt.rindex(subject) + len(subject)
            position = min(len(tokenizer(prompt[:end]).input_ids) - 1, ids.shape[1] - 1)
        tokens = ids[0].tolist()
        if common is None:
            common = tokens[:-1]
        else:
            limit = min(len(common), len(tokens)-1)
            index = 0
            while index < limit and common[index] == tokens[index]:
                index += 1
            common = common[:index]
        if capture_states:
            common = common[:position]
        entries.append((prompt, subject, ids, position))
    prefix_length = len(common or [])
    if prefix_length < 8:
        return None
    blocks, _ = sweep.layout(model)
    with torch.no_grad():
        prefix = model(input_ids=entries[0][2][:, :prefix_length], use_cache=True, logits_to_keep=1).past_key_values
    assert prefix is not None
    result = []
    errors = dict(states=0.0, logits=0.0)
    checks = 0
    for index, (prompt, subject, ids, position) in enumerate(entries):
        captured = {}
        handles = []

        def capture(layer):
            def hook(module, inputs, output):
                states = output[0] if isinstance(output, tuple) else output
                captured[layer] = states[0, position-prefix_length].detach().clone()
                if layer == len(blocks):
                    captured["final"] = states[0, -1].detach().clone()
            return hook

        if capture_states:
            for layer in sorted(set(layers) | {len(blocks)}):
                handles.append(blocks[layer-1].register_forward_hook(capture(layer)))
        try:
            with torch.no_grad():
                output = model(input_ids=ids[:, prefix_length:], use_cache=True, past_key_values=copy.deepcopy(prefix), logits_to_keep=1)
                logits = output.logits[0, -1].detach().to(head_device)
                del output
        finally:
            for handle in handles:
                handle.remove()
        if index % 4 == 0 or index == len(entries)-1:
            if capture_states:
                scalar = sweep.cache_prompt(model, tokenizer, prompt, subject, layers)
                scalar_logits = scalar["logits"]
                for key, value in captured.items():
                    # Same gate used by model_sweep.enable_prefix_cache.
                    torch.testing.assert_close(value, scalar["states"][key], rtol=1e-3, atol=1e-3)
                    errors["states"] = max(errors["states"], float((value-scalar["states"][key]).abs().max()))
            else:
                with torch.no_grad():
                    scalar_logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].to(head_device)
            torch.testing.assert_close(logits, scalar_logits, rtol=1e-4, atol=1e-3)
            assert int(logits.argmax()) == int(scalar_logits.argmax())
            errors["logits"] = max(errors["logits"], float((logits-scalar_logits).abs().max()))
            checks += 1
        target = int(logits.argmax())
        result.append(dict(ids=ids, position=position, states=captured, target=target, logits=logits) if capture_states else target)
    check = dict(n=len(jobs), state_capture=capture_states, scalar_checks=checks, prefix_length=prefix_length, max_abs_errors=errors, elapsed_seconds=time.monotonic()-start)
    model._sweep_prompt_prefix_checks.append(check)
    print("PROMPT_PREFIX_DONE", check, flush=True)
    return result


def try_run(model, tokenizer, jobs, layers, capture_states):
    if capture_states and getattr(model, "_sweep_prompt_prefix_targets_only", False):
        return None
    if not getattr(model, "_sweep_prompt_prefix", False) or getattr(model, "_sweep_prompt_prefix_disabled", False):
        return None
    try:
        return run(model, tokenizer, jobs, layers, capture_states)
    except AssertionError as exc:
        model._sweep_prompt_prefix_disabled = True
        model._sweep_prompt_prefix_checks.append(dict(fallback="full_forwards", reason=str(exc), n=len(jobs)))
        print("PROMPT_PREFIX_FALLBACK", str(exc), flush=True)
        return None
