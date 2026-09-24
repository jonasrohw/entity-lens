import pytest

from jlens.full_dataset import date_exclusion, partitions


class Tokenizer:
    def __init__(self, prefix):
        self.prefix = prefix

    def encode(self, *args, **kwargs):
        return [1]

    def decode(self, ids):
        return self.prefix


def test_no_cap_or_collision_filter_and_preserve_repeats():
    samples = [dict(subject=str(i), object="same") for i in range(250)]
    samples += [dict(samples[0])]
    demos, refs, tests = partitions(samples)
    assert len(demos) == len(refs) == 8
    assert len(tests) == 235
    assert sorted(x["subject"] for x in demos + refs + tests) == sorted(
        x["subject"] for x in samples
    )


@pytest.mark.parametrize(
    "relation,token,faith,target,excluded",
    [
        ("presidents_birth_year", "1", 1.0, "1", True),
        ("presidents_election_year", "1", 1.0, "1", True),
        ("presidents_birth_year", "178", 1.0, "178", False),
        ("presidents_birth_year", "1", 0.9999, "1", False),
        ("presidents_birth_year", "1", 1.0, "2", False),
        ("name_gender", "1", 1.0, "1", False),
        ("person_band_lead_singer", "1", 1.0, "1", False),
    ],
)
def test_only_exact_digit_one_date_collapse_is_excluded(
    relation, token, faith, target, excluded
):
    result = date_exclusion(
        relation,
        [dict(object="1782")],
        Tokenizer(token),
        False,
        dict(best_faith=faith, rows=[dict(hit=1, model_top1=target)]),
    )
    assert result["excluded"] is excluded


def test_full_readouts_match_direct_products(monkeypatch):
    import torch
    from test_model_sweep import Tokenizer as TinyTokenizer
    from test_model_sweep import tiny

    from jlens import model_sweep as sweep

    torch.set_num_threads(1)
    model = tiny("qwen3")
    tokenizer = TinyTokenizer()
    samples = [dict(subject=f"name{i}", object=chr(65 + i)) for i in range(20)]
    _, refs, tests = partitions(samples)
    original = sweep.jvps
    expected_raw = []

    def checked(intervention, reference, directions, ids, position, batch_size):
        products = original(
            intervention, reference, directions, ids, position, batch_size
        )
        # Directly apply J to subject states, independent of the implemented
        # J(s-ref)+J(ref) construction used to share products across readouts.
        direct = original(
            intervention,
            reference,
            directions[:-1] + reference,
            ids,
            position,
            batch_size,
        )
        torch.testing.assert_close(
            products[:-1] + products[-1], direct, rtol=1e-4, atol=1e-5
        )
        expected_raw.append(direct)
        return products

    monkeypatch.setattr(sweep, "jvps", checked)
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
        include_baselines=True,
    )
    assert lens["n"] == len(tests) == 4
    assert len(expected_raw) == len(refs) == 8
    _, norm = sweep.layout(model)
    with torch.no_grad():
        logits = sweep.decode(model, norm, torch.stack(expected_raw).mean(0))
        targets = torch.tensor([r["model_top1_id"] for r in lens["rows"]])
        ranks = (logits > logits.gather(1, targets[:, None])).sum(-1).tolist()
    assert ranks == [r["rank_raw_jacobian"] for r in lens["rows"]]
