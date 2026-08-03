"""
tests/test_validator_decision.py

Covers decide_final_status -- the Validator's verdict logic.

This function decides whether a patient can be discharged without a
human ever looking at the case, so the precedence between its signals is
the highest-consequence logic in the system. Each rule is pinned
separately here.

Run:  python -m pytest tests/test_validator_decision.py -q
"""

from agents.validator_agent.nodes import decide_final_status

CLEAN_REPORT = {"risk_level": "low", "discharge_blocked": False}


def finding(rule_id, severity, triggered):
    return {
        "rule_id": rule_id,
        "severity": severity,
        "triggered": triggered,
        "evidence": "",
    }


def test_clean_case_auto_approves():
    status = decide_final_status(
        {"status": "complete"},
        [finding("med_omission_check", "Warning", False)],
        CLEAN_REPORT,
    )
    assert status == "auto_approve"


def test_missing_blocking_field_blocks():
    status = decide_final_status({"status": "blocked"}, [], CLEAN_REPORT)
    assert status == "blocked"


def test_triggered_critical_rule_blocks():
    """allergy_contradiction_check is the P1016 scenario -- must block."""
    status = decide_final_status(
        {"status": "complete"},
        [finding("allergy_contradiction_check", "Critical", True)],
        CLEAN_REPORT,
    )
    assert status == "blocked"


def test_triggered_warning_goes_to_hitl_not_block():
    """med_omission_check is the P1014 scenario -- review, not block."""
    status = decide_final_status(
        {"status": "complete"},
        [finding("med_omission_check", "Warning", True)],
        CLEAN_REPORT,
    )
    assert status == "hitl"


def test_untriggered_critical_rule_does_not_block():
    status = decide_final_status(
        {"status": "complete"},
        [finding("allergy_contradiction_check", "Critical", False)],
        CLEAN_REPORT,
    )
    assert status == "auto_approve"


def test_reporter_blocked_flag_is_honoured():
    """
    The Reporter forces high on conditions no single finding shows --
    low translation confidence, for one. Ignoring its flag would let
    those cases auto-approve.
    """
    status = decide_final_status(
        {"status": "complete"}, [], {"risk_level": "high", "discharge_blocked": True}
    )
    assert status == "blocked"


def test_unresolved_elicitation_goes_to_hitl():
    status = decide_final_status({"status": "unresolved"}, [], CLEAN_REPORT)
    assert status == "hitl"


def test_resolved_elicitation_can_auto_approve():
    """A reviewer who filled the gaps has already been the human in the loop."""
    status = decide_final_status({"status": "resolved"}, [], CLEAN_REPORT)
    assert status == "auto_approve"


def test_medium_risk_goes_to_hitl():
    status = decide_final_status(
        {"status": "complete"}, [], {"risk_level": "medium", "discharge_blocked": False}
    )
    assert status == "hitl"


def test_high_risk_blocks_even_without_a_blocked_flag():
    status = decide_final_status(
        {"status": "complete"}, [], {"risk_level": "high", "discharge_blocked": False}
    )
    assert status == "blocked"


def test_blocking_beats_everything_else():
    """Precedence check: a blocked case must never be downgraded to hitl."""
    status = decide_final_status(
        {"status": "blocked"},
        [finding("med_omission_check", "Warning", True)],
        {"risk_level": "low", "discharge_blocked": False},
    )
    assert status == "blocked"


def test_empty_inputs_do_not_auto_approve_silently():
    """
    An empty report means the Reporter never ran. Approving on missing
    evidence is the worst possible default, so this must not return
    auto_approve on a blocked completeness result.
    """
    assert decide_final_status({"status": "blocked"}, [], {}) == "blocked"
