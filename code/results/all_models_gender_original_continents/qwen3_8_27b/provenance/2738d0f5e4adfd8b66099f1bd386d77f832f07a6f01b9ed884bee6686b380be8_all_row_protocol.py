"""Reference overlap protocol: retain every row in both evaluation sets."""
import collections
import json
import random

LANDMARK_REFERENCES = ['Bellingshausen Sea', 'Block Bay', 'Rufiji River', 'Niger River',
                       'Ismoil Somoni Peak', 'Yangtze', 'Eiger', 'Aconcagua']


def all_row_split(samples, *, lens, balanced_landmarks=False):
    """Keep original main-protocol demos/anchors, with uncapped evaluation.

    References and demonstrations are roles, not exclusions from evaluation.
    Preserve repeated source rows and independently reproduce each main shuffle.
    """
    if len(samples) < 16:
        raise ValueError('Eight demonstrations and eight original references required')
    pool = list(samples)
    random.Random(17 if lens else 0).shuffle(pool)
    if not lens:
        random.Random(1000).shuffle(pool)
    demos, refs = pool[:8], pool[8:16]
    if balanced_landmarks:
        refs = []
        for subject in LANDMARK_REFERENCES:
            matches = [x for x in samples if x['subject'] == subject]
            if len(matches) != 1:
                raise ValueError(f'Landmark must identify one source row: {subject}')
            refs.append(matches[0])
        assert {x['object'] for x in refs} == {x['object'] for x in samples}
    assert len(demos) == len(refs) == 8
    assert row_multiset(pool) == row_multiset(samples)
    return demos, refs, pool


def row_multiset(rows):
    return collections.Counter(json.dumps(x, sort_keys=True) for x in rows)
