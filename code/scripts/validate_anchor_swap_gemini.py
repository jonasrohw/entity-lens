"""Audit corpus split integrity, provenance, controls, and report reconstruction."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/anchor_swap_gemini"
OUT = ROOT / "results/anchor_swap_gemini"
RAW = ROOT / "generated/anchor_swap_gemini"


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    mh = hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest()
    assert (
        manifest["protocol_sha256"]
        == hashlib.sha256((DATA / "PROTOCOL.md").read_bytes()).hexdigest()
    )
    all_subjects = set()
    n_subjects = 0
    n_test = 0
    controls = 0
    max_fd = 0
    for domain, d in manifest["domains"].items():
        rows = d["rows"]
        subjects = [r["subject"].casefold() for r in rows]
        assert len(subjects) == len(set(subjects)) and not all_subjects.intersection(
            subjects
        )
        all_subjects.update(subjects)
        assert all(
            r["audit"]["verdict"] == "accept"
            and r["source_audit"]["verdict"] == "accept"
            and r["source_html_sha256"]
            for r in rows
        )
        assert all(
            sum(r["split"] == s for r in rows) == n
            for s, n in [("demonstrations", 8), ("references", 8), ("validation", 16)]
        )
        n_subjects += len(rows)
        n_test += sum(r["split"] == "test" for r in rows)
        r = json.loads((OUT / f"{domain}.json").read_text())
        assert r["complete"] and r["provenance"]["manifest_sha256"] == mh
        assert r["rows"] == [x for x in rows if x["split"] in ["validation", "test"]]
        demos = {x["subject"] for x in rows if x["split"] == "demonstrations"}
        refs = {x["subject"] for x in rows if x["split"] == "references"}
        for i, x in enumerate(r["rows"]):
            assert x["subject"] not in demos | refs
            for label in ["donors", "cyclic_donors"]:
                j = r[label][i]
                assert j != i and r["rows"][j]["split"] == x["split"]
            if r["donor_different_both"][i]:
                assert all(
                    r["targets"][m][i] != r["targets"][m][r["donors"][i]]
                    for m in ["a", "b"]
                )
        for mode, check in r["checks"].items():
            assert (
                check["identity_max"] < 1e-5
                and check["zero_jvp_max"] == 0
                and check["fd_relative_error"] < 0.08
            )
            max_fd = max(max_fd, check["fd_relative_error"])
        for mode in ["a", "b"]:
            for beta, scores in r["scores"][mode].items():
                assert all(len(x["top"]) == len(r["rows"]) for x in scores.values())
                assert (
                    scores["anchor"]["top"] == r["scores"][mode]["3.0"]["anchor"]["top"]
                )
    for path in (RAW / "api").glob("*audit*.json"):
        if path.name.endswith("_request.json"):
            continue
        data = json.loads(path.read_text())
        rows = data["parsed"]["rows"]
        indices = [x["index"] for x in rows]
        assert len(indices) == len(set(indices)) and set(indices) == set(
            range(len(indices))
        )
        assert max(rows, key=lambda x: x["index"])["verdict"] == "reject"
        controls += 1
    triples = [json.loads(x) for x in (DATA / "triples.jsonl").read_text().splitlines()]
    assert len(triples) == 2 * n_subjects
    summary = {
        "status": "passed",
        "domains": len(manifest["domains"]),
        "subjects": n_subjects,
        "triples": len(triples),
        "test_subjects": n_test,
        "negative_audit_controls_passed": controls,
        "max_fd_relative_error": max_fd,
        "manifest_sha256": mh,
    }
    (OUT / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
