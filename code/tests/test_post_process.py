"""Tests for post_process: history-flag merge, calibration toggle, normalisation."""

import main

BASE_VLM = {
    "valid_image": True,
    "evidence_standard_met": True,
    "evidence_standard_met_reason": "ok",
    "risk_flags": [],
    "issue_type": "dent",
    "object_part": "door",
    "claim_status": "supported",
    "claim_status_justification": "visible dent",
    "supporting_image_ids": ["img_1"],
    "severity": "high",
}


def row(user_id="u1", claim="Customer: there is a dent on the door."):
    return {
        "user_id": user_id,
        "image_paths": "images/x/img_1.jpg",
        "user_claim": claim,
        "claim_object": "car",
    }


def test_history_risk_adds_both_flags():
    hist = {"u1": {"history_flags": "user_history_risk"}}
    out = main.post_process(row(), dict(BASE_VLM), hist)
    flags = out["risk_flags"].split(";")
    assert "user_history_risk" in flags and "manual_review_required" in flags


def test_manual_review_only():
    hist = {"u1": {"history_flags": "manual_review_required"}}
    out = main.post_process(row(), dict(BASE_VLM), hist)
    assert out["risk_flags"] == "manual_review_required"


def test_no_history_yields_none():
    assert main.post_process(row(), dict(BASE_VLM), {})["risk_flags"] == "none"


def test_severity_calibrated_down_to_medium():
    assert main.post_process(row(), dict(BASE_VLM), {})["severity"] == "medium"


def test_calibrate_false_keeps_raw_severity():
    assert (
        main.post_process(row(), dict(BASE_VLM), {}, calibrate=False)["severity"]
        == "high"
    )


def test_supporting_ids_joined():
    assert (
        main.post_process(row(), dict(BASE_VLM), {})["supporting_image_ids"] == "img_1"
    )


def test_invalid_issue_defaults_unknown():
    vlm = dict(BASE_VLM, issue_type="explosion", claim_status="contradicted")
    assert main.post_process(row(), vlm, {})["issue_type"] == "unknown"


def test_output_has_all_fields_in_order():
    out = main.post_process(row(), dict(BASE_VLM), {})
    assert list(out.keys()) == main.OUTPUT_FIELDS
