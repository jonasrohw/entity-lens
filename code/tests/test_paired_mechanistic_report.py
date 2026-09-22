import json

import numpy as np

from jlens.paired_mechanistic_report import holm, paired, select_heads


def test_holm_and_paired_degenerate_cases():
    np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
    zero = paired([0, 0, 0])
    assert zero["mean"] == 0 and zero["ci"] == [0, 0] and zero["p"] == 1
    assert paired([])["n"] == 0


def test_head_nomination_ignores_confirmation_outcomes(tmp_path):
    rows = []
    for block in [1, 2, 3]:
        for relation in ["country_capital_city", "country_currency"]:
            for partition in ["validation", "test"]:
                for i in range(16):
                    damage = {1: 2., 2: 1., 3: .5}[block] if relation.endswith("city") else 0.
                    if partition == "test" and block == 3:
                        damage = 10000.
                    rows.append(dict(block=block, relation=relation, partition=partition,
                                     kind="attention", group="answer", subject=str(i),
                                     clean={"logp": 0.}, patched={"logp": -damage}))
    (tmp_path/"routes_mock.json").write_text(json.dumps({"complete": True, "rows": rows}))
    (tmp_path/"routes_jobs.json").write_text(json.dumps([{"pid": 123}]))
    (tmp_path/"routes_unfinished.partial.json").write_text(json.dumps({"complete": False, "rows": rows}))
    select_heads(tmp_path)
    selected = json.loads((tmp_path/"head_selection.json").read_text())["selected"]
    assert [r["block"] for r in selected] == [1, 2]
