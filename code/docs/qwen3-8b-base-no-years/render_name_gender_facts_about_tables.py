"""Build the Facts about tables from the exported predictions, without inference."""

import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEM = 'name_gender_facts_about'


def code(text):
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('|', r'\|')
    fence = '`' * (1 + max((len(s) for s in re.findall(r'`+', text)), default=0))
    return fence + ' ' + text + ' ' + fence


def main():
    data = json.loads((HERE / f'{STEM}_top100.json').read_text())
    snapshot_path = HERE / f'{STEM}_measurement.json'
    snapshot = json.loads(snapshot_path.read_text())
    baseline = json.loads((HERE / 'name_gender_top100.json').read_text())
    rows = data['rows']
    assert data['source_sha256'] == hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    assert data['exact_saved_lens_match']
    assert data['neutral_prompt'] == 'Facts about {}:'
    assert len(rows) == len(baseline['rows']) == 11
    for r, old, saved in zip(rows, baseline['rows'], snapshot['lens']['rows'], strict=True):
        for key in ['row', 'subject', 'model_target', 'model_target_id', 'n_references']:
            assert r[key] == old[key]
        assert r['target_rank_zero_based'] == saved['rank_diff']
        assert [t['token_stripped'] for t in r['top100'][:5]] == saved['top_diff']
        assert [t['rank'] for t in r['top100']] == list(range(1, 101))
        assert len({t['token_id'] for t in r['top100']}) == 100
        assert all(a['score'] >= b['score'] for a, b in zip(r['top100'], r['top100'][1:]))
        hits = [t for t in r['top100'] if t['token_id'] == r['model_target_id']]
        if r['target_rank_zero_based'] < 100:
            assert len(hits) == 1 and hits[0]['rank'] == r['target_rank_zero_based'] + 1
        else:
            assert not hits

    fields = ['row', 'subject', 'model_target', 'model_target_id', 'target_rank_zero_based',
              'prediction_rank', 'token_id', 'token_json', 'token_stripped', 'score']
    with (HERE / f'{STEM}_top100.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for r in rows:
            for t in r['top100']:
                writer.writerow({**{k: r[k] for k in fields[:5]},
                                 'prediction_rank': t['rank'], 'token_id': t['token_id'],
                                 'token_json': json.dumps(t['token'], ensure_ascii=False),
                                 'token_stripped': t['token_stripped'], 'score': t['score']})

    lines = [
        '# Qwen3 8B Base: name_gender with the Facts about prompt', '',
        '**Exact prompt: `Facts about {name}:`** — capital F, colon after the name, '
        'and no leading newline. These are 100 differential J-lens predictions '
        'for each of the same 11 saved examples (1,100 entries). Sofia is retained twice.', '',
        f'[CSV: all predictions]({STEM}_top100.csv) · [JSON: scores and provenance]({STEM}_top100.json) · '
        '[Compare the newline-prompt tables](name_gender_top100.md)', '',
        '## Top-five overview and target ranks', '',
        '| Row | Name / top 100 | Model target | J-lens top five | Facts about target rank | Previous newline target rank |',
        '| ---: | --- | --- | --- | ---: | ---: |',
    ]
    for r, old in zip(rows, baseline['rows'], strict=True):
        top5 = ', '.join(code(t['token_stripped']) for t in r['top100'][:5])
        lines.append(f"| {r['row']} | [{r['subject']}](#row-{r['row']}-{r['subject'].lower()}) | "
                     f"{code(r['model_target'])} | {top5} | {r['target_rank_zero_based']:,} | {old['target_rank_zero_based']:,} |")
    lines += ['',
        '**Target ranks above are zero-based** (0 = top-1). The ranked lists below are '
        '**one-based** (1 = the highest-scoring prediction). The target remains the exact '
        'full model top-1 token under the relation prompt `{name} is usually a name for a` '
        'with the original in-context examples.', '',
        '## Summary', '',
        '| Quantity | Facts about | Previous newline |', '| --- | ---: | ---: |',
    ]
    for k in [1, 10, 100]:
        n = sum(r['target_rank_zero_based'] < k for r in rows)
        old_n = sum(r['target_rank_zero_based'] < k for r in baseline['rows'])
        lines.append(f'| Exact target in top {k} | {n} / 11 | {old_n} / 11 |')
    old_lens = json.loads((HERE / 'name_gender_measurement.json').read_text())['lens']
    for label, key, sign in [('Median target rank (zero-based)', 'median_rank', 1),
                             ('Representation score', 'median_lograng', -1)]:
        new_value = sign * data['lens_aggregates']['diff'][key]
        old_value = sign * old_lens['agg']['diff'][key]
        lines.append(f'| {label} | {new_value:,.6f} | {old_value:,.6f} |')
    boys = [(r['subject'], t['rank']) for r in rows for t in r['top100']
            if t['token_stripped'] == '男孩']
    if boys:
        lines += ['', 'A male-related Chinese token appears in this prompt variant: ' +
                  '; '.join(f'**{subject}: 男孩 (boy), prediction #{rank}**'
                            for subject, rank in boys) + ' (one-based).']
    lines += ['',
        'The model, checkpoint, state 19, dataset rows, targets, and reference counts are '
        'unchanged. This rerun uses the earlier Facts about run’s float32/eager attention '
        'and batch size 2. The previous newline export used batch size 8. '
        '**The complete rerun exactly reproduces the earlier saved Facts about measurement**, '
        'including its per-row ranks, top-five tokens, and aggregates. '
        'LRE faithfulness remains 1.0 on its separate three test rows; it was not rerun. '
        'The existing 26-relation plot continues to show the newline results.', '',
        '## Top-100 tables', '',
        'Expand a name for all 100 predictions. Token strings are shown with surrounding '
        'whitespace stripped; IDs distinguish tokens with the same visible text. The CSV '
        '`token_json` field preserves the exact decoded string with JSON escapes; the JSON '
        'also contains unstripped strings and raw readout scores, which are not probabilities. '
        'A check mark identifies an exact match to the model target token ID.', '',
    ]
    for r in rows:
        lines += [f"### Row {r['row']}: {r['subject']}", '',
                  '<details open>' if r['row'] == 1 else '<details>',
                  f"<summary>{r['subject']}: predictions 1–100 · target {r['model_target']} · target rank {r['target_rank_zero_based']:,} (0-based)</summary>", '',
                  '| Prediction rank | J-lens token | Token ID | Exact target |',
                  '| ---: | --- | ---: | :---: |']
        for t in r['top100']:
            token = t['token_stripped'] or json.dumps(t['token'], ensure_ascii=False)
            hit = '✓' if t['token_id'] == r['model_target_id'] else ''
            lines.append(f"| {t['rank']} | {code(token)} | {t['token_id']} | {hit} |")
        lines += ['', '</details>', '']
    lines += ['## Provenance and reproduction', '',
        f'[Original Facts about measurement]({STEM}_measurement.json) · '
        f'[Top-100 extraction script](export_{STEM}_top100.py) · '
        f'[Table rendering script](render_{STEM}_tables.py)', '',
        'Inference requires the original measurement workspace’s `jlens.model_sweep` '
        'modules and cached checkpoint. The exporter checks hashes for the measurement '
        'modules and data, verifies numerical interventions, and verifies equality to '
        'the saved Facts about run. The rendering script needs only the committed JSON files '
        'and Python’s standard library.', '',
        '```sh', f'python code/docs/qwen3-8b-base-no-years/render_{STEM}_tables.py', '```', '']
    (HERE / f'{STEM}_top100.md').write_text('\n'.join(lines))
    with (HERE / f'{STEM}_top100.csv').open() as f:
        exported = list(csv.DictReader(f))
    assert len(exported) == 1100
    for row, token in zip(exported, [t for r in rows for t in r['top100']], strict=True):
        assert json.loads(row['token_json']) == token['token']
    assert sum(1 for line in lines if re.match(r'^\| \d+ \| `', line)) == 1100
    print('Validated 1,100 predictions and rendered the Facts about tables.')


if __name__ == '__main__':
    main()
