import json
from pathlib import Path

from jlens.all_row_protocol import all_row_split,row_multiset

ROOT=Path(__file__).resolve().parents[1]


def test_all_original_relations_have_uncapped_evaluation():
    for path in (ROOT/'data/relations').glob('*.json'):
        samples=json.loads(path.read_text())['samples']
        for lens in [False,True]:
            demos,refs,evaluation=all_row_split(samples,lens=lens,balanced_landmarks=path.stem=='landmark_on_continent')
            assert len(demos)==len(refs)==8
            assert len(evaluation)==len(samples)
            assert row_multiset(evaluation)==row_multiset(samples)
            assert all(x in evaluation for x in demos+refs)


def test_large_relation_has_no_legacy_twenty_or_hundred_row_cap():
    samples=[dict(subject=f's{i}',object='answer') for i in range(1037)]
    for lens in [False,True]:
        assert len(all_row_split(samples,lens=lens)[2])==1037


def test_landmark_references_cover_every_available_category():
    samples=json.loads((ROOT/'data/relations/landmark_on_continent.json').read_text())['samples']
    for lens in [False,True]:
        _,refs,_=all_row_split(samples,lens=lens,balanced_landmarks=True)
        assert {x['object'] for x in refs}=={x['object'] for x in samples}


def test_duplicate_rows_are_preserved_in_evaluation():
    samples=json.loads((ROOT/'data/relations/name_gender.json').read_text())['samples']
    for lens in [False,True]:
        _,_,evaluation=all_row_split(samples,lens=lens)
        assert sum(x['subject']=='Sofia' for x in evaluation)==2
