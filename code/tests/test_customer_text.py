"""Tests for customer-turn extraction from the claim transcript."""
import main


def test_extracts_customer_turns_last_first():
    last, allc = main._customer_text(
        "Customer: A scratch. | Support: where? | Customer: front bumper.")
    assert last == "front bumper."
    assert "front bumper" in allc and "a scratch" in allc


def test_excludes_support_turns():
    last, allc = main._customer_text("Customer: torn packaging. | Support: was it crushed?")
    assert "crushed" not in allc
    assert last == "torn packaging."


def test_no_customer_prefix_falls_back_to_whole_string():
    last, allc = main._customer_text("just a plain string")
    assert last == "just a plain string"
    assert allc == "just a plain string"
