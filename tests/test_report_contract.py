"""
tests/test_report_contract.py

Pins the contract between the audit report and everything that reads it.

This exists because of a real defect: the Reporter never persisted
`extracted_discharge` / `extracted_bill` / `extracted_lab`, but the
Summary Generator and both dashboard medication tables read them. Three
of the summary's five sections were silently empty, and no test noticed
because each side was only ever tested against its own assumptions.

So these tests assert the two sides against *each other*: every key a
consumer reads must be a key the producer writes.

Run:  python -m pytest tests/test_report_contract.py -q
"""

import pytest

from agents.summary_agent.sections import SECTION_ORDER, section_context
from mcp_primary.tools_reporter import build_json_report

ALL_INPUTS = {
    "extracted_discharge": {
        "patient_id": "P1019",
        "patient_name": "Thomas Wright",
        "discharge_diagnosis": "Type 2 Diabetes Mellitus",
        "discharge_instructions": "Continue diabetic diet.",
        "medications": [
            {
                "sl_no": 1,
                "medicine_name": "Metformin",
                "strength": "500 mg",
                "frequency": "BID",
                "route": "ORAL",
            }
        ],
    },
    "extracted_bill": {
        "patient_id": "P1019",
        "total_amount": 1903.07,
        "payment_status": "PAID",
    },
    "extracted_lab": {
        "patient_id": "P1019",
        "tests": [{"test": "HbA1c", "result": "6.9", "flag": "HIGH"}],
    },
    "completeness_gaps": {"missing_blocking": [], "missing_nonblocking": []},
    "ehr_findings": [
        {
            "rule_id": "med_omission_check",
            "severity": "Warning",
            "triggered": False,
            "evidence": "OK",
            "action": "OK",
        }
    ],
    "translation_confidence": 1.0,
}


@pytest.fixture
def report():
    return build_json_report("P1019", ALL_INPUTS)


# ---------------------------------------------------------------------
# Producer side
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["extracted_discharge", "extracted_bill", "extracted_lab"],
)
def test_report_persists_the_extracted_documents(report, key):
    """The regression this file was written for."""
    assert key in report, f"report is missing '{key}' — consumers read it"
    assert report[key] == ALL_INPUTS[key]


def test_medications_survive_into_the_report(report):
    """
    Dashboard pages 3 and 5 both render from this exact path. When it was
    missing, both tables rendered empty with no error.
    """
    medications = report["extracted_discharge"]["medications"]
    assert medications
    assert medications[0]["medicine_name"] == "Metformin"


def test_report_still_carries_its_audit_fields(report):
    for key in (
        "patient_id", "generated_at", "risk_score", "risk_level",
        "recommendation", "discharge_blocked", "completeness",
        "ehr_findings", "translation_confidence", "rules_version",
    ):
        assert key in report


def test_missing_inputs_produce_empty_dicts_not_none(report):
    """
    Consumers do `(report.get(...) or {}).get(...)`, so None would work --
    but an empty dict keeps the report's shape stable for anything that
    iterates its keys.
    """
    sparse = build_json_report("P9999", {})

    assert sparse["extracted_discharge"] == {}
    assert sparse["extracted_bill"] == {}
    assert sparse["extracted_lab"] == {}


# ---------------------------------------------------------------------
# Consumer side: the summary's sections must find real data
# ---------------------------------------------------------------------
@pytest.mark.parametrize("section", SECTION_ORDER)
def test_every_summary_section_gets_non_empty_context(report, section):
    """
    Each of the five sections must find something to write about. An
    empty context is what produced hollow "your medicines" sections.
    """
    context = section_context(section, report)

    assert context and context.strip() not in ("{}", ""), (
        f"section '{section}' received an empty context"
    )
    # Every value being null is the same failure with extra punctuation.
    assert '"null"' not in context
    assert not all(
        line.strip().endswith("null,") or line.strip().endswith("null")
        for line in context.splitlines()
        if ":" in line
    ), f"section '{section}' context is entirely null"


def test_meds_section_sees_the_medication_list(report):
    context = section_context("meds", report)
    assert "Metformin" in context


def test_labs_section_sees_the_test_results(report):
    context = section_context("labs", report)
    assert "HbA1c" in context


def test_bill_section_sees_amount_and_status(report):
    context = section_context("bill", report)
    assert "1903.07" in context
    assert "PAID" in context


def test_patient_section_sees_identity_and_diagnosis(report):
    context = section_context("patient", report)
    assert "Thomas Wright" in context
    assert "Type 2 Diabetes Mellitus" in context


def test_instructions_section_sees_the_follow_up(report):
    context = section_context("instructions", report)
    assert "diabetic diet" in context


# ---------------------------------------------------------------------
# Pipeline side: lab extraction must actually reach the Validator
# ---------------------------------------------------------------------
def test_pipeline_maps_all_three_doc_types():
    """
    lab_reports used to be extracted and then dropped, so lab data never
    reached the report at all.
    """
    from agents.orchestrator.pipeline import DOC_TYPE_TO_VALIDATOR_KEY, DOC_TYPES

    assert set(DOC_TYPE_TO_VALIDATOR_KEY) == set(DOC_TYPES)
    assert DOC_TYPE_TO_VALIDATOR_KEY["lab_reports"] == "extracted_lab"
