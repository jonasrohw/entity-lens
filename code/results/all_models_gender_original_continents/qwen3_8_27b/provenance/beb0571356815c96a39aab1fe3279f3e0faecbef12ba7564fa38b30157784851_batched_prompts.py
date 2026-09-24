"""Optional equal-length prompt batches, gated against scalar HF forwards.

No padding or changes to prompts, positions, rows, or model precision. One
example in every batch is checked against the original scalar implementation.
If any check fails, the whole call is recomputed by the scalar implementation
and batching stays disabled for the remainder of that worker.
"""
from collections import defaultdict
import time

import torch


def _run(model, tokenizer, jobs, layers, capture_states):
    from jlens import model_sweep as sweep

    started = time.monotonic()
    groups = defaultdict(list)
    for index, (prompt, subject) in enumerate(jobs):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        position = None
        if capture_states:
            end = prompt.rindex(subject) + len(subject)
            position = min(len(tokenizer(prompt[:end]).input_ids) - 1, ids.shape[1] - 1)
        groups[ids.shape[1]].append((index, prompt, subject, ids, position))
    result = [None] * len(jobs)
    batch_size = model._sweep_prompt_batch_size
    blocks, _ = sweep.layout(model)
    checks = 0
    errors = dict(logits=0.0, states=0.0)
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            ids = torch.cat([x[3] for x in batch]).to(model.get_input_embeddings().weight.device)
            captured = [{} for _ in batch]
            handles = []

            def capture(layer):
                def hook(module, inputs, output):
                    states = output[0] if isinstance(output, tuple) else output
                    for i, item in enumerate(batch):
                        captured[i][layer] = states[i, item[4]].detach().clone()
                        if layer == len(blocks):
                            captured[i]["final"] = states[i, -1].detach().clone()
                return hook

            if capture_states:
                for layer in sorted(set(layers) | {len(blocks)}):
                    handles.append(blocks[layer - 1].register_forward_hook(capture(layer)))
            try:
                with torch.no_grad():
                    logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].detach().to(model.get_output_embeddings().weight.device)
            finally:
                for handle in handles:
                    handle.remove()
            first = batch[0]
            if capture_states:
                scalar = sweep.cache_prompt(model, tokenizer, first[1], first[2], layers)
                scalar_logits = scalar["logits"]
                for key, value in captured[0].items():
                    torch.testing.assert_close(value, scalar["states"][key], rtol=1e-4, atol=1e-4)
                    errors["states"] = max(errors["states"], float((value - scalar["states"][key]).abs().max()))
            else:
                with torch.no_grad():
                    scalar_logits = model(input_ids=ids[:1], use_cache=False, logits_to_keep=1).logits[0, -1].to(logits.device)
            torch.testing.assert_close(logits[0], scalar_logits, rtol=1e-4, atol=1e-3)
            assert int(logits[0].argmax()) == int(scalar_logits.argmax())
            errors["logits"] = max(errors["logits"], float((logits[0] - scalar_logits).abs().max()))
            checks += 1
            for i, item in enumerate(batch):
                target = int(logits[i].argmax())
                result[item[0]] = (
                    dict(ids=ids[i:i+1].clone(), position=item[4], states=captured[i], target=target, logits=logits[i].clone())
                    if capture_states else target
                )
    check = dict(n=len(jobs), state_capture=capture_states, scalar_checks=checks, max_abs_errors=errors, batch_size=batch_size, elapsed_seconds=time.monotonic()-started)
    model._sweep_prompt_batch_checks.append(check)
    print("PROMPT_BATCH_DONE", check, flush=True)
    return result


def _enabled(model):
    return getattr(model, "_sweep_prompt_batch_size", 1) > 1 and not getattr(model, "_sweep_prompt_batch_disabled", False)


def cache_prompts(model, tokenizer, jobs, layers):
    from jlens import model_sweep as sweep
    if _enabled(model):
        try:
            return _run(model, tokenizer, jobs, layers, True)
        except AssertionError as exc:
            model._sweep_prompt_batch_disabled = True
            model._sweep_prompt_batch_checks.append(dict(fallback="scalar", reason=str(exc), n=len(jobs)))
            print("PROMPT_BATCH_FALLBACK", str(exc), flush=True)
    return [sweep.cache_prompt(model, tokenizer, prompt, subject, layers) for prompt, subject in jobs]


def prompt_targets(model, tokenizer, prompts):
    if _enabled(model):
        try:
            return _run(model, tokenizer, [(p, None) for p in prompts], [], False)
        except AssertionError as exc:
            model._sweep_prompt_batch_disabled = True
            model._sweep_prompt_batch_checks.append(dict(fallback="scalar", reason=str(exc), n=len(prompts)))
            print("PROMPT_BATCH_FALLBACK", str(exc), flush=True)
    targets = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.get_input_embeddings().weight.device)
            targets.append(int(model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].argmax()))
    return targets
