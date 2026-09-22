"""Scientific inference checks: pairing, equivalence, multiplicity, sources."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_uncertainty_cancels_shared_subject_difficulty():
    report = load("report_anchor_swap_gemini")
    # Every condition faces identical easy/hard subjects. Resampling subjects
    # must leave the paired contrast exactly zero despite marginal uncertainty.
    a = np.tile([0.0, 1.0], 40)
    summary, _ = report.calc(
        {k: a.copy() for k in report.CONDITIONS}, np.random.default_rng(4)
    )
    assert summary["rates"]["own"]["ci95"][0] < 0.5
    assert summary["contrasts"]["foreign_minus_own"]["ci95"] == [0.0, 0.0]
    assert summary["equivalent_within_10pp"]
    assert not summary["foreign_beats_wrong"]


def test_equivalence_is_not_merely_positive_control_success():
    report = load("report_anchor_swap_gemini")
    arrays = {k: np.zeros(60) for k in report.CONDITIONS}
    arrays["own"] = np.ones(60)
    arrays["foreign"] = np.full(60, 0.5)
    summary, _ = report.calc(arrays, np.random.default_rng(4))
    assert not summary["equivalent_within_10pp"]
    assert summary["foreign_beats_wrong"]
    assert summary["contrasts"]["foreign_minus_own"]["ci95"] == [-0.5, -0.5]


def test_holm_uses_whole_family_and_original_order():
    report = load("report_anchor_swap_gemini")
    np.testing.assert_allclose(report.holm([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])


def test_source_parser_extracts_labeled_infobox_and_ignores_scripts():
    sys.path.insert(0, str(SCRIPTS))
    try:
        source = load("source_audit_anchor_swap")
    finally:
        sys.path.pop(0)
    parser = source.Evidence()
    parser.feed(
        '<title>Film</title><table class="infobox"><tr><th>Director</th><td>Alice</td></tr></table><script>bad</script><p>A film scored by Bob.</p>'
    )
    assert "Director | Alice" in parser.text()
    assert "scored by Bob" in parser.text()
    assert "bad" not in parser.text()
