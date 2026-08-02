"""
mock_ehr/main.py
=================
FastAPI wrapper around the static dicts in data.py, turning "the EHR" into
a network dependency (port 8050) instead of a Python import. Consumed by
the EHR Validation Tool and the Analytics server.

Run:
    uvicorn mock_ehr.main:app --port 8050
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from mock_ehr.data import (
    ALLERGIES,
    CARE_PLANS,
    GUIDELINES,
    LABS,
    MED_ORDERS,
    PATIENTS,
)
from mock_ehr.schemas import (
    CarePlan,
    EHRBundle,
    Guideline,
    LabResult,
    MedOrder,
    Patient,
)

app = FastAPI(title="Mock EHR", version="1.0.0")


@app.get("/patients/{patient_id}", response_model=Patient)
def get_patient(patient_id: str) -> dict:
    patient = PATIENTS.get(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")
    return patient


@app.get("/allergies/{patient_id}", response_model=list[str])
def get_allergies(patient_id: str) -> list[str]:
    return ALLERGIES.get(patient_id, [])


@app.get("/med-orders/{patient_id}", response_model=list[MedOrder])
def get_med_orders(patient_id: str) -> list[dict]:
    return MED_ORDERS.get(patient_id, [])


@app.get("/labs/{patient_id}", response_model=list[LabResult])
def get_labs(patient_id: str) -> list[dict]:
    return LABS.get(patient_id, [])


@app.get("/care-plan/{patient_id}", response_model=CarePlan)
def get_care_plan(patient_id: str) -> dict:
    care_plan = CARE_PLANS.get(patient_id)
    if care_plan is None:
        raise HTTPException(status_code=404, detail="care plan not found")
    return care_plan


@app.get("/guidelines/{icd10_code}", response_model=Guideline)
def get_guideline(icd10_code: str) -> dict:
    guideline = GUIDELINES.get(icd10_code)
    if guideline is None:
        raise HTTPException(status_code=404, detail="guideline not found")
    return guideline


@app.get("/bundle/{patient_id}", response_model=EHRBundle)
def get_full_ehr_bundle(patient_id: str) -> dict:
    """One round trip instead of five: patient, allergies, med orders,
    labs, care plan, and the guidelines relevant to the patient's
    primary_dx codes."""
    patient = PATIENTS.get(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")

    care_plan = CARE_PLANS.get(patient_id)
    if care_plan is None:
        raise HTTPException(status_code=404, detail="care plan not found")

    return {
        "patient": patient,
        "allergies": get_allergies(patient_id),
        "med_orders": get_med_orders(patient_id),
        "labs": get_labs(patient_id),
        "care_plan": care_plan,
        "guidelines": [
            GUIDELINES[code]
            for code in patient["primary_dx"]
            if code in GUIDELINES
        ],
    }