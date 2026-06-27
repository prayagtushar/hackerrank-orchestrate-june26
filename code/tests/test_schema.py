"""Tests for the pydantic VlmPrediction coercion and the output.csv schema."""

import csv
from pathlib import Path

import pytest
import main


def test_vlm_prediction_coerces_messy_input():
    raw = {
        "valid_image": "false",
        "evidence_standard_met": "true",
        "risk_flags": "a;b",
        "issue_type": None,
        "supporting_image_ids": "img_1; img_2",
        "severity": 1,
    }
    out = main.validate_prediction(raw)
    assert out["valid_image"] is False
    assert out["evidence_standard_met"] is True
    assert out["risk_flags"] == ["a", "b"]
    assert out["issue_type"] == ""
    assert out["supporting_image_ids"] == ["img_1", "img_2"]
    assert out["severity"] == "1"


def test_vlm_prediction_defaults_on_empty():
    out = main.validate_prediction({})
    assert out["claim_status"] == "not_enough_information"
    assert out["risk_flags"] == []
    assert out["valid_image"] is True


OUTPUT = Path("output.csv")


def test_output_csv_schema():
    if not OUTPUT.exists():
        pytest.skip("output.csv not generated yet")
    rows = list(csv.DictReader(open(OUTPUT, encoding="utf-8")))
    assert rows, "output.csv is empty"
    assert list(rows[0].keys()) == main.OUTPUT_FIELDS
    for r in rows:
        assert r["claim_status"].lower() in main.ALLOWED_STATUS
        assert r["severity"].lower() in main.ALLOWED_SEVERITY
        assert r["issue_type"].lower() in main.ALLOWED_ISSUES
        assert r["object_part"].lower() in main.ALLOWED_PARTS
        assert r["valid_image"].lower() in {"true", "false"}
        assert r["evidence_standard_met"].lower() in {"true", "false"}
        assert r["claim_status_justification"].strip(), (
            f"empty justification for {r['user_id']}"
        )
        for flag in r["risk_flags"].split(";"):
            assert flag in main.ALLOWED_RISK_FLAGS
