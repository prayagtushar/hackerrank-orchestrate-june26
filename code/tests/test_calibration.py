"""Deterministic tests for the issue/severity/object_part calibration layer."""
import main


def cis(claim, issue, status):
    return main.calibrate_issue_and_severity(claim, issue, status)


def test_crack_stays_crack_not_glass_shatter():
    issue, sev = cis("Customer: my windshield has a crack spreading.", "glass_shatter", "supported")
    assert issue == "crack"
    assert sev == "medium"


def test_mirror_not_sitting_is_broken_part():
    issue, _ = cis("Customer: the side mirror got damaged. It is not sitting the way it should.",
                   "glass_shatter", "supported")
    assert issue == "broken_part"


def test_colloquial_shattered_is_crack():
    issue, _ = cis("Customer: the screen looks shattered to me, submitting as screen damage.",
                   "glass_shatter", "supported")
    assert issue == "crack"


def test_explicit_fragmentation_is_glass_shatter():
    issue, sev = cis("Customer: the windshield cracked and is completely shattered into pieces.",
                     "crack", "supported")
    assert issue == "glass_shatter"
    assert sev == "high"


def test_hindi_phati_is_torn_packaging():
    issue, _ = cis("Customer: Seal wali side phati hui thi.", "crushed_packaging", "supported")
    assert issue == "torn_packaging"


def test_crushed_packaging():
    issue, sev = cis("Customer: one corner was crushed in when I received it.",
                     "torn_packaging", "supported")
    assert issue == "crushed_packaging"
    assert sev == "medium"


def test_water_damage_beats_stain():
    issue, _ = cis("Customer: the package looks water damaged.", "stain", "supported")
    assert issue == "water_damage"


def test_stain():
    issue, sev = cis("Customer: it left a stain and some keys feel sticky.", "water_damage", "supported")
    assert issue == "stain"
    assert sev == "medium"


def test_scratch_is_low():
    issue, sev = cis("Customer: front bumper par scratch hai.", "dent", "supported")
    assert issue == "scratch"
    assert sev == "low"


def test_plain_dent_is_medium():
    _, sev = cis("Customer: the back of the car has a dent now.", "dent", "supported")
    assert sev == "medium"


def test_corner_dent_is_low():
    _, sev = cis("Customer: one corner of the laptop has a dent now.", "dent", "supported")
    assert sev == "low"


def test_nei_severity_is_unknown_and_no_anchor():
    issue, sev = cis("Customer: headlight cracked.", "crack", "not_enough_information")
    assert sev == "unknown"
    assert issue == "crack"


def test_contradicted_defers_severity_to_model():
    issue, sev = cis("Customer: the back looks pretty bad.", "dent", "contradicted")
    assert sev is None
    assert issue == "dent"


def cop(claim, obj, part, status):
    return main.calibrate_object_part(claim, obj, part, status)


def test_part_single_family_anchors():
    assert cop("Customer: the package surface looked damaged. The outside has a stain.",
               "package", "box", "supported") == "package_side"


def test_part_ambiguous_keeps_model():
    assert cop("Customer: the hinge area has broken and the screen wobbles.",
               "laptop", "hinge", "supported") == "hinge"


def test_part_front_glass_is_windshield_not_bumper():
    assert cop("Customer: this is only about the front glass.",
               "car", "unknown", "supported") == "windshield"


def test_part_not_supported_keeps_model():
    assert cop("Customer: the rear bumper.", "car", "quarter_panel", "contradicted") == "quarter_panel"
