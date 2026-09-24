import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_targeted_corrections import affected, excluded, expanded_ranks, REFERENCES
from jlens.full_dataset import reference_partitions
import json
from collections import Counter


def test_landmark_partition_covers_original_multiset_and_categories():
    samples=json.loads((ROOT/'data/full_relations/landmark_on_continent.json').read_text())['samples']
    demos,refs,tests=reference_partitions(samples,REFERENCES)
    encode=lambda xs:Counter(json.dumps(x,sort_keys=True) for x in xs)
    assert encode(demos+refs+tests)==encode(samples)
    assert len(tests)==len(samples)-16
    assert {x['object'] for x in refs}=={x['object'] for x in samples}
    assert not ({x['subject'] for x in refs}&{x['subject'] for x in tests})


def test_gender_matching_cannot_accept_opposite_gender_rank():
    lens={'rows':[{'model_top1':'woman','rank_diff':20},{'model_top1':'man','rank_diff':30},{'model_top1':'unknown','rank_diff':40}], 'agg':{'diff':{}}}
    expanded_ranks(lens,{'woman':1,'man':2,'her':3,'him':4}, {'woman':[20,0,0],'man':[0,30,0],'her':[4,0,0],'him':[0,7,0]})
    assert [r['rank_diff'] for r in lens['rows']]==[4,7,40]
    assert lens['agg']['diff']['median_lograng']==np.median(np.log10([5,8,41]))


def test_scope_and_year_rule():
    assert affected('qwen3_8_27b','company_ceo')
    assert not affected('olmo3_7b','company_ceo')
    assert affected('olmo3_7b','name_gender')
    assert affected('gptj','landmark_on_continent')
    audit={'model':{'relations':{'mixed_years':{'ignored':16},'multi_digit_years':{'ignored':0}}}}
    assert excluded('model','mixed_years',audit)
    assert not excluded('model','multi_digit_years',audit)
    assert not excluded('model','name_gender',audit)
