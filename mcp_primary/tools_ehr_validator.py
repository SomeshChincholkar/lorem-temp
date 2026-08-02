"""
tools_ehr_validator.py

The EHR Validation Tool primitive. Implements Table 4's seven
cross-validation rules against the Mock EHR (mock_ehr/main.py, running
on :8050) and rules.yaml.

Field names below match mock_ehr/schemas.py exactly:
  Patient:    patient_id, patient_name, dob, sex, primary_dx, service_line
  MedOrder:   name, dose, frequency
  LabResult:  test, value, abnormal, action_in_ehr
  CarePlan:   followup_required, speciality, window_days
  Guideline:  diagnosis, required_followup, essential_meds
  EHRBundle:  patient, allergies (list[str]), med_orders, labs, care_plan, guidelines
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from mcp.server.fastmcp import Context  # noqa: E402

from common.rules_loader import get_icd10  # noqa: E402

MOCK_EHR_BASE_URL = "http://localhost:8050"


# ---------------------------------------------------------------------
# EHR fetch
# ---------------------------------------------------------------------
async def fetch_ehr_bundle(patient_id: str) -> dict:
    """
    GET /bundle/{patient_id} from the Mock EHR -- one round trip for
    patient, allergies, med_orders, labs, care_plan, and guidelines.
    """
    url = f"{MOCK_EHR_BASE_URL}/bundle/{patient_id}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
    if response.status_code == 404:
        raise ValueError(f"No EHR bundle found for patient_id='{patient_id}'")
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _discharge_med_names(discharge_meds: List[dict]) -> List[str]:
    """
    discharge_meds items follow the "prescription" field schema from
    Table 3 (medicine_name, strength, frequency, route, ...).
    """
    return [_norm(m.get("medicine_name")) for m in (discharge_meds or []) if m.get("medicine_name")]


# ---------------------------------------------------------------------
# Table 4 rule checks
# ---------------------------------------------------------------------
def check_med_omission_check(discharge_meds: List[dict], ehr_bundle: dict) -> Dict:
    """Discharge meds differ from EHR medication history."""
    discharge_names = set(_discharge_med_names(discharge_meds))
    ehr_orders = ehr_bundle.get("med_orders", [])

    omitted = [
        order["name"] for order in ehr_orders
        if _norm(order.get("name")) not in discharge_names
    ]

    if omitted:
        return {"triggered": True, "evidence": f"EHR active medications missing from discharge list: {omitted}"}
    return {"triggered": False, "evidence": "All EHR-active medications are present in the discharge list."}


def check_allergy_contradiction_check(discharge_meds: List[dict], ehr_bundle: dict) -> Dict:
    """Prescribed med conflicts with a known allergy."""
    allergies = [_norm(a) for a in ehr_bundle.get("allergies", [])]
    discharge_names = _discharge_med_names(discharge_meds)

    conflicts = [
        name for name in discharge_names
        if any(allergy and (allergy in name or name in allergy) for allergy in allergies)
    ]

    if conflicts:
        return {"triggered": True, "evidence": f"Discharge medication(s) conflict with documented allergies: {conflicts}"}
    return {"triggered": False, "evidence": "No discharge medications conflict with documented allergies."}


def check_diagnosis_mismatch_check(discharge_dx: Optional[str], ehr_bundle: dict) -> Dict:
    """Discharge diagnosis differs from EHR care plan / primary_dx."""
    primary_dx = ehr_bundle.get("patient", {}).get("primary_dx", [])

    if not discharge_dx:
        return {"triggered": True, "evidence": "No discharge diagnosis provided to compare against EHR."}

    # Try mapping the free-text discharge diagnosis to an ICD-10 code
    # via rules.yaml's icd10_map; fall back to raw text comparison.
    mapped_code = get_icd10(discharge_dx.strip())
    if mapped_code:
        if mapped_code in primary_dx:
            return {"triggered": False, "evidence": f"Discharge diagnosis '{discharge_dx}' ({mapped_code}) matches EHR primary_dx."}
        return {
            "triggered": True,
            "evidence": f"Discharge diagnosis '{discharge_dx}' maps to {mapped_code}, not in EHR primary_dx {primary_dx}.",
        }

    # No ICD mapping available -- can't confidently compare, flag for review.
    return {
        "triggered": True,
        "evidence": f"Could not map discharge diagnosis '{discharge_dx}' to an ICD-10 code to compare against EHR primary_dx {primary_dx}.",
    }


def check_follow_up_missing_check(discharge_followup: Optional[str], ehr_bundle: dict) -> Dict:
    """Follow-up not documented despite care plan requirement."""
    care_plan = ehr_bundle.get("care_plan", {})
    required = care_plan.get("followup_required", False)

    if required and not discharge_followup:
        return {
            "triggered": True,
            "evidence": (
                f"EHR care plan requires follow-up with {care_plan.get('speciality', 'unspecified')} "
                f"within {care_plan.get('window_days', '?')} days, but no follow-up is documented in the discharge."
            ),
        }
    return {"triggered": False, "evidence": "Follow-up requirement satisfied or not required."}


def check_lab_follow_up_mismatch_check(discharge_text: str, ehr_bundle: dict) -> Dict:
    """Abnormal lab values have no documented action in discharge instructions."""
    labs = ehr_bundle.get("labs", [])
    text_norm = _norm(discharge_text)

    unaddressed = [
        lab["test"] for lab in labs
        if lab.get("abnormal") and _norm(lab.get("test")) not in text_norm
    ]

    if unaddressed:
        return {"triggered": True, "evidence": f"Abnormal lab result(s) not referenced in discharge instructions: {unaddressed}"}
    return {"triggered": False, "evidence": "All abnormal lab results are addressed in discharge instructions."}


def check_discharge_approval_check(extracted_fields: dict) -> Dict:
    """Discharge not approved by the treating physician."""
    approved = extracted_fields.get("discharge_approved")
    approved_by = extracted_fields.get("discharge_approved_by")

    if approved and approved_by:
        return {"triggered": False, "evidence": f"Discharge approved by {approved_by}."}
    return {"triggered": True, "evidence": "Discharge is missing physician approval and/or approver name."}


def check_bill_settlement_check(bill_fields: dict) -> Dict:
    """Bill not PAID or lacking an insurance guarantee letter."""
    payment_status = _norm(bill_fields.get("payment_status"))
    has_guarantee_letter = bool(bill_fields.get("guarantee_letter_flag"))

    if payment_status == "paid" or has_guarantee_letter:
        return {"triggered": False, "evidence": f"Bill payment_status='{bill_fields.get('payment_status')}', guarantee_letter={has_guarantee_letter}."}
    return {"triggered": True, "evidence": f"Bill is not settled (payment_status='{bill_fields.get('payment_status')}') and no guarantee letter is on file."}


# ---------------------------------------------------------------------
# Rule registry (Table 4, in order)
# ---------------------------------------------------------------------
RULES = [
    ("med_omission_check", "Warning", "med"),
    ("allergy_contradiction_check", "Critical", "med"),
    ("diagnosis_mismatch_check", "Warning", "dx"),
    ("follow_up_missing_check", "Critical", "followup"),
    ("lab_follow_up_mismatch_check", "Warning", "lab"),
    ("discharge_approval_check", "Critical", "approval"),
    ("bill_settlement_check", "Critical", "bill"),
]

_CHECK_FNS = {
    "med_omission_check": check_med_omission_check,
    "allergy_contradiction_check": check_allergy_contradiction_check,
    "diagnosis_mismatch_check": check_diagnosis_mismatch_check,
    "follow_up_missing_check": check_follow_up_missing_check,
    "lab_follow_up_mismatch_check": check_lab_follow_up_mismatch_check,
    "discharge_approval_check": check_discharge_approval_check,
    "bill_settlement_check": check_bill_settlement_check,
}


# ---------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------
async def ehr_validation_tool(
    ctx: Context,
    patient_id: str,
    extracted_discharge: dict,
    extracted_bill: dict,
) -> List[Dict]:
    """
    Run all 7 Table 4 cross-validation rules for a patient.

    Args:
        ctx: MCP Context (unused directly here -- kept for signature
            consistency with the other tools and in case future rules
            need roots/elicitation).
        patient_id: e.g. "P1019"
        extracted_discharge: dict of discharge_report fields (Table 3
            shape), including a "medications" list and
            "follow_up_appointments" / "discharge_diagnosis" etc.
        extracted_bill: dict of bill fields (Table 3 shape).

    Returns:
        list[{rule_id, severity, triggered, evidence, action}]
    """
    bundle = await fetch_ehr_bundle(patient_id)

    discharge_meds = extracted_discharge.get("medications", [])
    discharge_dx = extracted_discharge.get("discharge_diagnosis")
    discharge_followup = extracted_discharge.get("follow_up_appointments")
    discharge_text = extracted_discharge.get("discharge_instructions", "") or ""

    results = []
    for rule_id, severity, kind in RULES:
        fn = _CHECK_FNS[rule_id]

        if kind == "med":
            outcome = fn(discharge_meds, bundle)
        elif kind == "dx":
            outcome = fn(discharge_dx, bundle)
        elif kind == "followup":
            outcome = fn(discharge_followup, bundle)
        elif kind == "lab":
            outcome = fn(discharge_text, bundle)
        elif kind == "approval":
            outcome = fn(extracted_discharge)
        elif kind == "bill":
            outcome = fn(extracted_bill)
        else:
            raise AssertionError(f"Unhandled rule kind: {kind}")

        triggered = outcome["triggered"]
        if triggered and severity == "Critical":
            action = "Block discharge"
        elif triggered:
            action = "Flag for review"
        else:
            action = "OK"

        results.append({
            "rule_id": rule_id,
            "severity": severity,
            "triggered": triggered,
            "evidence": outcome["evidence"],
            "action": action,
        })

    return results