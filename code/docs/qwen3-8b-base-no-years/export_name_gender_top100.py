import hashlib
import json
import socket
import time
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
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
    source = ROOT / 'results/all_model_newline/qwen3_8b_base/main/name_gender.json'
    old = json.loads(source.read_text())
    worker_path = source.parent.parent / f"worker_{old['worker']}.json"
    worker = json.loads(worker_path.read_text())
    data_path = ROOT / 'data/relations/name_gender.json'
    assert sha(data_path) == old['data_sha256']
    for name, expected in worker['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert torch.__version__ == worker['torch']
    assert transformers.__version__ == worker['transformers']
    print('LOAD', worker['model_id'], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(worker['model_id'], revision=worker['revision'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        worker['model_id'], revision=worker['revision'], local_files_only=True,
        dtype=torch.float32, attn_implementation='eager', device_map={'': 'cuda:0'},
    ).eval().requires_grad_(False)
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    verification = sweep.verify(model, tokenizer, worker['lens_layer'], worker['batch_size'])
    print('VERIFIED', verification, flush=True)
    captured = []
    original_decode = sweep.decode

    def capture_decode(*args, **kwargs):
        logits = original_decode(*args, **kwargs)
        if not captured:
            values, ids = logits.topk(100, dim=-1)
            captured.append((values.cpu().tolist(), ids.cpu().tolist()))
        return logits

    sweep.decode = capture_decode
    data = json.loads(data_path.read_text())
    print('MEASURE name_gender, 11 rows, differential top 100', flush=True)
    try:
        current = sweep.measure_lens(
            model, tokenizer, data['samples'], data['prompt_templates'][0], 'main',
            worker['lens_layer'], old['trailing_space'], worker['batch_size'], 100,
            neutral_prompt=worker['prompt'],
        )
    finally:
        sweep.decode = original_decode
    # Compare in the saved JSON representation (the beta grid is a tuple in memory).
    current = json.loads(json.dumps(current))
    # Extending the export must not change any of the previous measurement.
    if current != old['lens']:
        Path('/tmp/qwen_name_gender_rerun_debug.json').write_text(json.dumps(current, indent=2))
        def differences(a, b, path='lens'):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in a.keys() | b.keys():
                    differences(a.get(key), b.get(key), path + '.' + key)
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for i, (aa, bb) in enumerate(zip(a, b)):
                    differences(aa, bb, path + f'[{i}]')
            elif a != b:
                print('MISMATCH', path, repr(a), repr(b), flush=True)
        differences(current, old['lens'])
        raise AssertionError('Rerun differs from the saved measurement')
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
        neutral_prompt=worker['prompt'], lens_layer=worker['lens_layer'],
        dtype='float32', attention='eager', batch_size=worker['batch_size'],
        torch=torch.__version__, transformers=transformers.__version__, host=socket.gethostname(),
        device=torch.cuda.get_device_name(), source=str(source.relative_to(ROOT)),
        source_sha256=sha(source), data_sha256=sha(data_path),
        source_code_sha256=worker['source_sha256'], extraction_script_sha256=sha(Path(__file__)),
        verification=verification, exact_saved_lens_match=True,
        elapsed_seconds=time.time()-start, rows=rows,
    )
    out = ROOT / 'docs/qwen3-8b-base-no-years/name_gender_top100.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('DONE', out, 'elapsed', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
