from collections import Counter

import pytest

from jlens.full_dataset import partitions, reference_partitions


def test_references_move_without_dropping_rows_or_changing_demos():
    samples = [dict(subject=str(i), object=str(i % 5)) for i in range(40)]
    samples += [dict(samples[0])]
    demos, old_refs, old_tests = partitions(samples)
    unique = Counter(x["subject"] for x in samples)
    subjects = [x["subject"] for x in old_tests if unique[x["subject"]] == 1][:8]
    actual_demos, refs, tests = reference_partitions(samples, subjects)
    assert actual_demos == demos
    assert [x["subject"] for x in refs] == subjects
    assert tests == [x for x in old_refs + old_tests if x["subject"] not in subjects]
    assert len(tests) == len(old_tests)
    assert Counter(
        (x["subject"], x["object"]) for x in demos + refs + tests
    ) == Counter((x["subject"], x["object"]) for x in samples)


def test_rejects_ambiguous_and_demo_references():
    samples = [dict(subject=str(i), object="x") for i in range(40)]
    demos, refs, _ = partitions(samples)
    subjects = [x["subject"] for x in refs]
    with pytest.raises(ValueError):
        reference_partitions(samples, [demos[0]["subject"]] + subjects[1:])
    with pytest.raises(ValueError):
        reference_partitions(samples, subjects[:7])
    with pytest.raises(ValueError):
        reference_partitions(samples, subjects[:7] + subjects[:1])


def test_both_measurements_use_explicit_references():
    import torch
    from test_model_sweep import Tokenizer, tiny

    from jlens import model_sweep as sweep
    from jlens.splits import split_fingerprint

    torch.set_num_threads(1)
    model, tokenizer = tiny("qwen3"), Tokenizer()
    samples = [dict(subject=f"name{i}", object=chr(65 + i)) for i in range(26)]
    _, _, tests = partitions(samples)
    split = reference_partitions(samples, [x["subject"] for x in tests[:8]])
    faith = sweep.measure_faith(
        model,
        tokenizer,
        samples,
        "{}:",
        "full_dataset",
        [2],
        False,
        2,
        None,
        split_override=split,
    )
    lens = sweep.measure_lens(
        model,
        tokenizer,
        samples,
        "{}:",
        "full_dataset",
        2,
        False,
        2,
        None,
        split_override=split,
    )
    assert (
        faith["split_fingerprint"]
        == lens["split_fingerprint"]
        == split_fingerprint(*split)
    )
    assert [x["subject"] for x in faith["rows"]] == [x["subject"] for x in split[2]]
    assert [x["model_top1_id"] for x in faith["rows"]] == [
        x["model_top1_id"] for x in lens["rows"]
    ]
    assert all(x["n_references"] == 8 for x in lens["rows"])
