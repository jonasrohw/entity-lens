"""Deterministic demonstration, reference, and test splits shared by both scorers.

The legacy paper protocols deliberately use independent splits. The shared
protocol uses disjoint references and scored subjects and records their hash."""

import collections
import hashlib
import json
import os
import random

N_ICL = 8
N_REF = 8
N_TEST = 100


def split_samples(samples, seed=0, n_icl=N_ICL, n_ref=N_REF, n_test=N_TEST):
    """Return (icl, ref, test) as disjoint lists of sample dicts.

    `samples` must be the list as loaded from the relation file, not a
    pre-shuffled copy: the split is defined by `seed` alone so that any
    caller reconstructs it identically.
    """
    pool = list(samples)
    random.Random(seed).shuffle(pool)
    demonstrations = pool[:n_icl]
    rest = pool[n_icl:]
    references = rest[:n_ref]
    test_samples = rest[n_ref : n_ref + n_test]
    return demonstrations, references, test_samples


def split_fingerprint(demonstrations, references, test_samples):
    """Short hash of the split, written into every output file.

    Hash the ordered subjects in all three partitions. Historical labels
    (icl, ref, test) are retained so existing measurements still match.
    """
    h = hashlib.sha1()
    for name, part in (
        ("icl", demonstrations),
        ("ref", references),
        ("test", test_samples),
    ):
        h.update(name.encode())
        for x in part:
            h.update(b"\x00")
            h.update(x["subject"].encode())
    return h.hexdigest()[:12]


def first_object_token(obj, tokenizer, trailing_space=False):
    """First token of an object as the scorers see it.

    With `trailing_space` the query already ends in a space, so the object
    is encoded bare and its first token is content; otherwise it is encoded
    with a leading space.
    """
    input_ids = tokenizer.encode(
        ("" if trailing_space else " ") + obj, add_special_tokens=False
    )
    return input_ids[0]


def split_model_correct_samples(
    samples, relation, cache_dir, seed=0, n_icl=N_ICL, n_ref=N_REF, n_test=N_TEST
):
    """(icl, ref, test) restricted to triples the model answers correctly.

    The screen is supplied externally and is not recomputed per caller: the lens
    run and the faithfulness run must select the *same* subjects or their
    split fingerprints diverge and `subject_statistics.py` refuses to run.

    **The ICL set is pinned to the one the screen used**, and is therefore
    drawn from the unfiltered pool.  This matters more than it looks.  A
    subject counts as model-correct only under some particular prompt, and
    the prompt is built from the demonstrations; re-splitting the filtered
    pool draws different demonstrations, and then the set no longer has the
    property it was selected for.  Measured on `person_occupation`, only
    **82%** of the test set survived that change.  Pinning the ICL set makes
    the screening prompt and the scoring prompt identical, so every scored
    subject is model-correct by construction rather than by approximation.

    A side benefit: the demonstrations are then the same as in the
    unfiltered run, so filtered and unfiltered numbers differ in the scored
    subjects alone and not also in the prompt.

    Raises if the cache is missing rather than silently scoring the
    unfiltered relation -- a filter that quietly does nothing is the worst
    outcome here.
    """
    path = f"{cache_dir}/{relation}.json"
    if not os.path.exists(path):
        raise SystemExit(
            f"no model-correct screen for {relation} at {path}; "
            "supply an external screen (see docs/commands.md)"
        )
    with open(path) as fh:
        cache = json.load(fh)

    demonstrations, _, _ = split_samples(samples, seed=seed, n_icl=n_icl)
    if sorted(x["subject"] for x in demonstrations) != sorted(cache["icl_subjects"]):
        raise SystemExit(
            f"{relation}: the ICL set no longer matches the one screened "
            f"(screen_seed={cache['screen_seed']}, n_icl={n_icl}).  The screen "
            "is stale; regenerate the external screen."
        )

    keep = set(cache["correct"])
    pool = [x for x in samples if x["subject"] in keep]
    random.Random(seed).shuffle(pool)
    return demonstrations, pool[:n_ref], pool[n_ref : n_ref + n_test]


def filter_identifying_objects(samples, tokenizer, trailing_space=False):
    """Keep only samples whose object's first token identifies it uniquely.

    An LRE emits ONE vector, hence one token, so it cannot distinguish two
    objects that share a first token -- `University of Chicago` from
    `University of Michigan`.  Where they collide, first-token agreement
    between the LRE and the model measures nothing: both need only say
    `University`.  Measured on this benchmark that inflates faithfulness
    badly (person_university: 29 of 69 objects collapse to `University`,
    and the LRE "agrees" on 100% of them).

    True multi-token scoring is not available -- a second token would have
    to come from the model, not the operator -- so the sound fix is to
    score only the objects the first token does identify.
    """
    objs = {x["object"] for x in samples}
    m = collections.defaultdict(set)
    for o in objs:
        m[first_object_token(o, tokenizer, trailing_space)].add(o)
    keep = {
        o for o in objs if len(m[first_object_token(o, tokenizer, trailing_space)]) == 1
    }
    return [x for x in samples if x["object"] in keep]
