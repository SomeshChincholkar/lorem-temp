"""
mock_ehr/schemas.py
====================
Pydantic response models mirroring the dict shapes in data.py.
Pure typing, no logic.
"""
from __future__ import annotations

from pydantic import BaseModel


class Patient(BaseModel):
    patient_id: str
    patient_name: str
    dob: str
    sex: str
    primary_dx: list[str]
    service_line: str


class Allergy(BaseModel):
    """A single allergy entry. ALLERGIES.get() returns list[str]; this model
    exists for structural completeness / future expansion."""
    name: str


class MedOrder(BaseModel):
    name: str
    dose: str
    frequency: str


class LabResult(BaseModel):
    test: str
    value: str
    abnormal: bool
    action_in_ehr: str


class CarePlan(BaseModel):
    followup_required: bool
    speciality: str
    window_days: int


class Guideline(BaseModel):
    diagnosis: str
    required_followup: str
    essential_meds: list[str]


class EHRBundle(BaseModel):
    patient: Patient
    allergies: list[str]
    med_orders: list[MedOrder]
    labs: list[LabResult]
    care_plan: CarePlan
    guidelines: list[Guideline]