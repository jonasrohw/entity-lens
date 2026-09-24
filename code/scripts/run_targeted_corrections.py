"""Apply requested corrections, reusing complete all-row measurements.

Never changes dataset rows or labels. Every run reserves 8 demonstrations and
8 references and scores all remaining rows, including duplicates. Only affected
relations run by default. Missing unaffected measurements are reported, not run.
"""
import argparse
import collections
import copy
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_all_model_newline import specs, sha
from jlens import model_sweep as sweep
from jlens.full_dataset import partitions, reference_partitions
from jlens.splits import split_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'generated/targeted_corrections'
BASE = ROOT / 'generated/full_dataset'
TERMS = {
    'woman': ['woman', 'her', 'herself', 'she', 'hers', 'female', 'girl'],
    'man': ['man', 'him', 'himself', 'he', 'his', 'male', 'boy'],
}
REFERENCES = ['Bellingshausen Sea', 'Block Bay', 'Rufiji River', 'Niger River',
              'Ismoil Somoni Peak', 'Yangtze', 'Eiger', 'Aconcagua']


def affected(name, relation):
    return name.startswith('qwen') or relation in {'name_gender', 'landmark_on_continent'}


def excluded(name, relation, audit):
    # The current request explicitly removes affected relations, rather than
    # ignoring individual year rows. Mixed 1xxx/2xxx relations are recorded.
    record = audit[name]['relations'].get(relation)
    return bool(record and record['ignored'])


def expanded_ranks(lens, term_ids, term_ranks):
    for i, row in enumerate(lens['rows']):
        target = row['model_top1']
        group = next((words for words in TERMS.values() if target in words), None)
        row['rank_diff_original'] = row['rank_diff']
        row['gender_target_terms'] = list(group or [target])
        row['gender_term_ranks'] = {
            word: dict(token_id=term_ids[word], rank=term_ranks[word][i])
            for word in (group or []) if word in term_ids
        }
        if target in term_ids:
            assert term_ranks[target][i] == row['rank_diff_original']
        row['rank_diff'] = min([row['rank_diff_original']] +
                              [v['rank'] for v in row['gender_term_ranks'].values()])
    lens['agg']['diff_original'] = lens['agg']['diff']
    lens['agg']['diff'] = sweep.aggregate([r['rank_diff'] for r in lens['rows']])
    lens['gender_policy'] = 'Best rank among single-token, leading-space, lowercase terms matching the model target gender; other model targets unchanged. LRE axis retains exact model-top-1 faithfulness.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True, choices=specs())
    p.add_argument('--relations', default='')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--fill-missing', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    spec = specs()[a.name]
    spec.setdefault('layers', [4, 10, 12, 15, 18, 20] if a.name == 'gptj' else [5, 11, 14, 17, 20, 23])
    audit = json.loads((DEST / 'tokenization_audit.json').read_text())
    requested = set(a.relations.split(',')) if a.relations else None
    paths = []
    for path in (ROOT / 'data/full_relations').glob('*.json'):
        if requested is not None and path.stem not in requested:
            continue
        if excluded(a.name, path.stem, audit):
            continue
        old = BASE / a.name / 'relations' / path.name
        if affected(a.name, path.stem) or (a.fill_missing and not old.exists()):
            paths.append(path)
    paths.sort(key=lambda p: (p.stem != 'name_gender', len(json.loads(p.read_text())['samples'])))
    paths = [p for p in paths if not (DEST / a.name / 'relations' / p.name).exists()]
    if not paths:
        print('NOTHING_PENDING', a.name, flush=True)
        return
    tok = AutoTokenizer.from_pretrained(spec['model_id'], revision=spec['revision'], local_files_only=True)
    term_ids, unavailable = {}, {}
    for term in sum(TERMS.values(), []):
        ids = tok.encode(' ' + term, add_special_tokens=False)
        if len(ids) == 1 and tok.decode(ids).strip() == term:
            term_ids[term] = ids[0]
        else:
            unavailable[term] = ids
    kwargs = dict(dtype=torch.float32, attn_implementation='eager', local_files_only=True, device_map={'': 'cuda:0'})
    if torch.cuda.device_count() > 1:
        kwargs.update(device_map='balanced', max_memory={i: f'{int(torch.cuda.get_device_properties(i).total_memory / 2**30) - 11}GiB' for i in range(torch.cuda.device_count())})
    print('LOAD', a.name, len(paths), flush=True)
    model = AutoModelForCausalLM.from_pretrained(spec['model_id'], revision=spec['revision'], **kwargs).eval().requires_grad_(False)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    model.config.use_cache = model.config.get_text_config().use_cache = False
    model.forward = partial(model.forward, use_cache=False, logits_to_keep=1)
    convolution = sweep.enable_fast_qwen_convolution(model, tok, spec['lens_layer'])
    verification = sweep.verify(model, tok, spec['lens_layer'], a.batch_size)
    prefix = sweep.enable_prefix_cache(model, tok, spec['lens_layer'], a.batch_size) if model.config.get_text_config().model_type == 'qwen3_5_text' else None
    prompt = '\n{} is known for' if a.name.startswith('qwen') else '{} is known for'
    source_paths = [Path(__file__), Path(sweep.__file__), ROOT / 'jlens/full_dataset.py', ROOT / 'jlens/splits.py', ROOT / 'jlens/model.py', ROOT / 'jlens/prefix_intervention.py']
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    for source in source_paths:
        dest = DEST / 'provenance' / f'{sha(source)}_{source.name}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): dest.write_bytes(source.read_bytes())
    sweep.write(DEST / a.name / 'metadata.json', dict(**spec, prompt=prompt, dtype='float32', attention='eager', verification=verification, convolution=convolution, prefix=prefix, source_sha256=source_hashes, test_cap=None, collision_filter=False, terms=TERMS, term_ids=term_ids, unavailable_terms=unavailable, fill_missing=a.fill_missing))
    for path in paths:
        output = DEST / a.name / 'relations' / path.name
        with sweep.measurement_claim(output) as acquired:
            if not acquired or output.exists(): continue
            start = time.time()
            data = json.loads(path.read_text()); samples = data['samples']
            split = reference_partitions(samples, REFERENCES) if path.stem == 'landmark_on_continent' else partitions(samples)
            demos, refs, tests = split
            assert collections.Counter(json.dumps(x,sort_keys=True) for part in split for x in part) == collections.Counter(json.dumps(x,sort_keys=True) for x in samples)
            fingerprint = split_fingerprint(*split)
            trailing = sweep.trailing_space_needed(tok, path.stem, samples)
            old_path = BASE / a.name / 'relations' / path.name
            old = json.loads(old_path.read_text()) if old_path.exists() else None
            if old:
                assert old['data_sha256'] == sha(path)
                assert old['n_original'] == len(samples)
            faith = None
            faith_source = None
            if old and old['split_fingerprint'] == fingerprint:
                faith = copy.deepcopy(old['faithfulness'])
                faith_source = dict(path=str(old_path.relative_to(ROOT)), sha256=sha(old_path))
            print('START', path.stem, 'rows', len(tests), 'reuse_faith', faith is not None, flush=True)
            # Refit after reference changes. A gender-only relation with no
            # existing measurement is also within the requested affected set.
            if faith is None and (path.stem in {'name_gender','landmark_on_continent'} or a.fill_missing):
                faith = sweep.measure_faith(model, tok, samples, data['prompt_templates'][0], 'full_dataset', spec['layers'], trailing, a.batch_size, None, split_override=split)
            captured = {}
            original_decode = sweep.decode
            def capture_decode(*args, **kwargs):
                logits = original_decode(*args, **kwargs)
                if not captured:
                    captured.update({word: (logits > logits[:, tid:tid+1]).sum(-1).cpu().tolist() for word, tid in term_ids.items()})
                return logits
            if path.stem == 'name_gender': sweep.decode = capture_decode
            try:
                lens = sweep.measure_lens(model, tok, samples, data['prompt_templates'][0], 'full_dataset', spec['lens_layer'], trailing, a.batch_size, None, neutral_prompt=prompt, split_override=split)
            finally:
                sweep.decode = original_decode
            assert lens['split_fingerprint'] == fingerprint
            assert len(lens['rows']) == len(tests) == len(samples)-16
            if faith:
                assert faith['split_fingerprint'] == fingerprint
                assert [(r['subject'],r['model_top1_id']) for r in faith['rows']] == [(r['subject'],r['model_top1_id']) for r in lens['rows']]
            if path.stem == 'name_gender':
                if old and old['prompt'] == prompt:
                    assert [r['rank_diff'] for r in old['lens']['rows']] == [r['rank_diff'] for r in lens['rows']], 'Unchanged original gender scores failed reproduction'
                expanded_ranks(lens, term_ids, captured)
            sweep.write(output, dict(model=a.name, relation=path.stem, protocol='full_dataset', prompt=prompt, n_original=len(samples), data_sha256=sha(path), split_fingerprint=fingerprint, split=dict(demonstrations=demos,references=refs,tests=tests), reference_category_counts=dict(collections.Counter(x['object'] for x in refs)), trailing_space=trailing, faithfulness=faith, faith_source=faith_source, lens=lens, correction_reasons=(['qwen_newline'] if a.name.startswith('qwen') else []) + (['gender_alternatives'] if path.stem=='name_gender' else []) + (['diverse_continent_references'] if path.stem=='landmark_on_continent' else []), elapsed_seconds=time.time()-start))
            print('DONE', a.name, path.stem, time.time()-start, flush=True)
    print('COMPLETE_AFFECTED', a.name, flush=True)

if __name__ == '__main__': main()
