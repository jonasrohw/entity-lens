"""Recover top-100 readouts using the exact earlier Facts about prompt."""

import hashlib
import json
import socket
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens import model_sweep as sweep


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.time()
    torch.set_num_threads(2)
    source = ROOT / 'results/qwen_prompt_sensitivity/main/facts_about/name_gender.json'
    old = json.loads(source.read_text())
    worker_path = ROOT / f"results/qwen_prompt_sensitivity/worker_{old['worker']}.json"
    worker = json.loads(worker_path.read_text())
    prompt = worker['prompts']['facts_about']
    assert prompt == 'Facts about {}:'
    data_path = ROOT / 'data/relations/name_gender.json'
    assert sha(data_path) == old['data_sha256']
    # Verify the actual measurement modules; the historical CLI is not used.
    modules = ['jlens/model_sweep.py', 'jlens/model.py', 'jlens/splits.py']
    for name in modules:
        assert sha(ROOT / name) == worker['source_sha256'][name], name
    assert torch.__version__ == worker['torch']
    newline_path = HERE / 'name_gender_measurement.json'
    newline = json.loads(newline_path.read_text())
    assert newline['data_sha256'] == old['data_sha256']
    print('LOAD', worker['model_id'], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        worker['model_id'], revision=worker['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        worker['model_id'], revision=worker['revision'], local_files_only=True,
        dtype=torch.float32, attn_implementation='eager', device_map={'': 'cuda:0'},
    ).eval().requires_grad_(False)
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    verification = sweep.verify(model, tokenizer, worker['layer'], worker['batch_size'])
    print('VERIFIED', verification, flush=True)
    captured = []
    original_decode = sweep.decode

    def capture_decode(*args, **kwargs):
        logits = original_decode(*args, **kwargs)
        if not captured:
            values, ids = logits.topk(100, dim=-1)
            captured.append((values.cpu().tolist(), ids.cpu().tolist()))
        return logits

    data = json.loads(data_path.read_text())
    sweep.decode = capture_decode
    print('MEASURE', prompt, flush=True)
    try:
        current = sweep.measure_lens(
            model, tokenizer, data['samples'], data['prompt_templates'][0], 'main',
            worker['layer'], newline['trailing_space'], worker['batch_size'], 100,
            neutral_prompt=prompt,
        )
    finally:
        sweep.decode = original_decode
    current = json.loads(json.dumps(current))
    assert current == old['lens'], 'Rerun differs from the earlier Facts about measurement'
    identity = ['subject', 'model_top1_id', 'model_top1', 'n_references']
    assert [[r[k] for k in identity] for r in current['rows']] == [
        [r[k] for k in identity] for r in newline['lens']['rows']]
    assert old['faithfulness'] == newline['faithfulness']['best_faith']
    values, ids = captured[0]
    rows = []
    for i, (r, scores, token_ids) in enumerate(zip(current['rows'], values, ids, strict=True), 1):
        top = [dict(rank=j, token_id=t, token=tokenizer.decode([t]),
                    token_stripped=tokenizer.decode([t]).strip(), score=s)
               for j, (t, s) in enumerate(zip(token_ids, scores, strict=True), 1)]
        assert [t['token_stripped'] for t in top[:5]] == r['top_diff']
        assert len(top) == len({t['token_id'] for t in top}) == 100
        rows.append(dict(row=i, subject=r['subject'], model_target=r['model_top1'],
                         model_target_id=r['model_top1_id'], target_rank_zero_based=r['rank_diff'],
                         n_references=r['n_references'], top100=top))
    result = dict(
        model=worker['model_id'], revision=worker['revision'], protocol='main',
        neutral_prompt=prompt, lens_layer=worker['layer'],
        dtype='float32', attention='eager', batch_size=worker['batch_size'],
        torch=torch.__version__, transformers=transformers.__version__, host=socket.gethostname(),
        device=torch.cuda.get_device_name(), source=str(source.relative_to(ROOT)),
        source_sha256=sha(source), data_sha256=sha(data_path),
        source_code_sha256={name: sha(ROOT / name) for name in modules},
        extraction_script_sha256=sha(Path(__file__)),
        verification=verification, exact_saved_lens_match=True,
        same_row_targets_and_reference_counts_as_newline=True,
        newline_measurement_sha256=sha(newline_path),
        lens_aggregates=current['agg'], elapsed_seconds=time.time()-start, rows=rows,
    )
    (HERE / 'name_gender_facts_about_measurement.json').write_bytes(source.read_bytes())
    out = HERE / 'name_gender_facts_about_top100.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('DONE', out, 'elapsed', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
