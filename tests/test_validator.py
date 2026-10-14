"""The validator's own controls.

Its job is to reject evidence that should not pass. A validator that cannot be
made to fail would wave through a run whose controls did nothing, which is the
only way the headline numbers could be wrong and still look right.
"""

from __future__ import annotations

from typing import Any

from script.validate_evidence import check_controls, check_ordering


def evidence(**over: Any) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {
        "dual-encoder": {"control_shuffled_labels": {"recall_at_20": 0.001}},
        "untrained-control": {"exact_heldout": {"recall_at_20": 0.004}},
        "bm25": {"heldout": {"recall_at_20": 0.65}},
    }
    for key, value in over.items():
        found[key.replace("_", "-")] = value
    return found


def test_a_clean_set_of_controls_passes() -> None:
    assert check_controls(evidence()) == []


def test_a_shuffled_control_that_still_scores_is_rejected() -> None:
    """Relevance found against deranged judgements is relevance that is not there."""
    found = evidence()
    found["dual-encoder"]["control_shuffled_labels"]["recall_at_20"] = 0.4
    assert any("deranged" in p for p in check_controls(found))


def test_an_untrained_encoder_that_scores_well_is_rejected() -> None:
    found = evidence()
    found["untrained-control"]["exact_heldout"]["recall_at_20"] = 0.5
    assert any("untrained" in p for p in check_controls(found))


def test_a_baseline_too_weak_to_mean_anything_is_rejected() -> None:
    found = evidence()
    found["bm25"]["heldout"]["recall_at_20"] = 0.05
    assert any("below" in p for p in check_controls(found))


def test_a_missing_control_is_rejected() -> None:
    found = evidence()
    del found["untrained-control"]
    assert check_controls(found)


def test_missing_evidence_entirely_is_rejected() -> None:
    assert check_controls({})


def test_quantized_search_beating_exact_is_rejected() -> None:
    """Both read the same vectors, so the approximation cannot win.

    If it does, the gap being published as the cost of compression is measuring
    a bug instead.
    """
    summary = {
        "exact_heldout": {"recall_at_20": 0.80},
        "quantized_heldout_rerank10": {"recall_at_20": 0.84},
    }
    assert any("cannot happen" in p for p in check_ordering("run", summary))


def test_quantized_search_below_exact_is_accepted() -> None:
    summary = {
        "exact_heldout": {"recall_at_20": 0.80},
        "quantized_heldout_rerank1": {"recall_at_20": 0.74},
        "quantized_heldout_rerank10": {"recall_at_20": 0.79},
    }
    assert check_ordering("run", summary) == []


def test_matching_exactly_is_accepted() -> None:
    """Lossless is implausible but not a contradiction, so it is not an error."""
    summary = {
        "exact_heldout": {"recall_at_20": 0.80},
        "quantized_heldout_rerank10": {"recall_at_20": 0.80},
    }
    assert check_ordering("run", summary) == []
