"""One explicit full-data cohort, with the sole permitted date exclusion."""

from .splits import split_samples


def partitions(samples):
    if len(samples) <= 16:
        raise ValueError(
            "Full-data evaluation needs 8 demonstrations, 8 references and tests"
        )
    return split_samples(samples, n_test=None)


def reference_partitions(samples, subjects):
    """Replace eight references, preserving demos and every original row.

    New references leave the test partition; displaced references enter it.
    Test order follows the original seed-0 shuffled pool. Reference subjects
    must identify unique rows and may not occur in the demonstration set.
    """
    demos, old_refs, old_tests = partitions(samples)
    if len(subjects) != 8 or len(set(subjects)) != 8:
        raise ValueError("Exactly eight distinct reference subjects are required")
    pool = old_refs + old_tests
    refs = []
    for subject in subjects:
        matches = [x for x in pool if x["subject"] == subject]
        if len(matches) != 1 or any(x["subject"] == subject for x in demos):
            raise ValueError(
                f"Reference must uniquely identify a non-demo row: {subject}"
            )
        refs.append(matches[0])
    tests = [x for x in pool if x["subject"] not in set(subjects)]
    assert len(tests) == len(old_tests)
    return demos, refs, tests


def date_exclusion(relation, samples, tokenizer, trailing_space, faithfulness):
    from .splits import first_object_token

    is_date = relation in {"presidents_birth_year", "presidents_election_year"}
    tokens = (
        [
            tokenizer.decode(
                [first_object_token(x["object"], tokenizer, trailing_space)]
            ).strip()
            for x in samples
        ]
        if is_date
        else []
    )
    rows = faithfulness["rows"]
    # Check raw hits, not a rounded displayed percentage. Require that the
    # trivial digit is also the model target, rather than assuming causation
    # from tokenization and the aggregate alone.
    excluded = bool(
        is_date
        and tokens
        and set(tokens) == {"1"}
        and faithfulness["best_faith"] == 1.0
        and rows
        and all(r["hit"] == 1 and r["model_top1"] == "1" for r in rows)
    )
    return dict(
        excluded=excluded,
        is_date=is_date,
        first_object_tokens=sorted(set(tokens)),
        model_target_tokens=sorted({r["model_top1"] for r in rows}),
        faithfulness=faithfulness["best_faith"],
        reason="date_first_token_1_and_perfect_lre" if excluded else None,
    )
