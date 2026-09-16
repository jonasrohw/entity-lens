"""Protocol invariants independent of a tokenizer download or a GPU."""

import pytest

from jlens import splits
from jlens.io import write_json
from jlens.matching import distinctive, leaky
from jlens.prominence_statistics import leaks


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [text.strip().split()[0]]


def samples(count=30):
    return [
        {"subject": f"subject {index}", "object": f"object {index}"}
        for index in range(count)
    ]


def test_split_is_deterministic_disjoint_and_preserves_input():
    original = samples()
    before = list(original)
    parts = splits.split_samples(original)
    assert parts == splits.split_samples(original)
    assert original == before
    assert [len(part) for part in parts] == [8, 8, 14]
    subjects = [sample["subject"] for part in parts for sample in part]
    assert len(set(subjects)) == len(subjects)
    assert splits.split_fingerprint(*parts) == "e32a3325c3e5"
    changed = splits.split_samples(original, seed=1)
    assert splits.split_fingerprint(*changed) != splits.split_fingerprint(*parts)


def test_identifying_filter_drops_collisions_but_retains_repeated_objects():
    data = [
        {"subject": str(index), "object": value}
        for index, value in enumerate(
            ["University Chicago", "University Michigan", "Paris", "Paris"]
        )
    ]
    assert splits.filter_identifying_objects(data, WordTokenizer()) == data[2:]


def test_model_correct_split_requires_matching_demonstrations(tmp_path):
    data = samples()
    demonstrations, _, _ = splits.split_samples(data)
    write_json(
        tmp_path / "relation.json",
        {
            "icl_subjects": [row["subject"] for row in demonstrations],
            "correct": [row["subject"] for row in data if row not in demonstrations],
            "screen_seed": 0,
        },
    )
    parts = splits.split_model_correct_samples(data, "relation", tmp_path)
    assert parts[0] == demonstrations
    assert not {row["subject"] for row in parts[0]} & {
        row["subject"] for row in parts[1] + parts[2]
    }
    with pytest.raises(SystemExit, match="no longer matches"):
        splits.split_model_correct_samples(data, "relation", tmp_path, seed=1)
    with pytest.raises(SystemExit, match="no model-correct screen"):
        splits.split_model_correct_samples(data, "absent", tmp_path)


def test_lexical_matching_preserves_leakage_distinction():
    assert distinctive("The Joker", "Batman") == {"joker"}
    assert distinctive("Johann Georg Kant", "Immanuel Kant") == {"johann", "georg"}
    assert leaky("Perseus", "Eta Persei")
    assert leaky("Johann Georg Kant", "Immanuel Kant")
    assert not leaks({"subject": "Immanuel Kant", "words": ["johann", "georg"]})
    assert leaks({"subject": "Eta Persei", "words": ["perseus"]})
