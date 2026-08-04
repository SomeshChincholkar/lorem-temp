"""
agents/validator_agent/state.py

Graph state for the Clinical Validation Agent (LangGraph, A2A port 8101).

Spec 2.4: completeness validation against Table 3 (with Elicitation for
non-blocking gaps) plus the seven Table 4 cross-validation rules against
the Mock EHR, then the audit report.
"""

from typing import Optional, TypedDict


class ValidatorState(TypedDict, total=False):
    patient_id: str

    # Extractor output, one dict per document type (Table 3 shape).
    extracted_discharge: dict
    extracted_bill: dict

    # Normalizer output, used for the low_translation_confidence weight.
    translation_confidence: Optional[float]

    # clinical_rules_engine_tool result:
    # {status: complete|resolved|unresolved|blocked, fields, unresolved_fields}
    completeness_result: dict
    completeness_gaps: dict          # {missing_blocking, missing_nonblocking}

    # ehr_validation_tool result: the seven Table 4 rule outcomes.
    ehr_findings: list

    # Snapshot of resource://clinical-rules/cross-validation, so the
    # audit trail records which policy set the run was judged against.
    cross_validation_policies: dict

    # clinical_insight_reporter_tool result:
    # {json_path, html_path, risk_level, recommendation, discharge_blocked}
    report: dict

    # "auto_approve" | "hitl" | "blocked"
    final_status: str

    trace_id: Optional[str]
    error: Optional[str]
