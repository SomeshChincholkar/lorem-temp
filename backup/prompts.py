"""
prompts.py

Centralizes every LLM prompt template used across the multi-agent
system, so agents fetch them via get_prompt(name, **params) instead of
hardcoding prompt strings in agent code (a grading requirement).

Registration is centralized via register_prompts(mcp), called from
server.py (step 3.9), mirroring resources.py / tools registration.
"""

import sys
from pathlib import Path

import json


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# Table 3 — Completeness validation fields by document type.
# "required" = all fields the extractor should attempt to pull out.
# "blocking" = subset of required fields that, if missing, BLOCK
# auto-generation of the discharge summary and require HITL
# intervention (per section 2.4.1).
DOC_TYPE_FIELD_SCHEMAS = {
    "discharge_report": {
        "required": [
            "patient_id", "patient_name", "age", "gender", "address",
            "admission_date", "discharge_date", "ward", "bed_no",
            "doctors", "discharge_diagnosis", "medications",
            "adr_allergy_info", "follow_up_appointments",
            "discharge_instructions", "discharge_approved_by",
            "discharge_approved",
        ],
        "blocking": [
            "patient_id", "patient_name", "discharge_diagnosis",
            "discharge_approved", "medications",
        ],
    },
    "lab_report": {
        "required": [
            "patient_id", "vendor_name", "lab_name", "report_date", "tests",
        ],
        "blocking": ["patient_id", "tests"],
    },
    "bill": {
        "required": [
            "patient_id", "hospital_name", "billing_date", "line_items",
            "total_amount", "payment_status",
        ],
        "blocking": ["patient_id", "total_amount", "payment_status"],
    },
    "prescription": {
        # per-medication fields (a discharge report's "medications" list
        # is made of objects shaped like this)
        "required": [
            "sl_no", "medicine_name", "strength", "dosage", "frequency",
            "route", "period", "remarks", "total_quantity",
        ],
        "blocking": ["medicine_name", "strength", "frequency", "route"],
    },
}


# ---------------------------------------------------------------------
# Prompt functions
# ---------------------------------------------------------------------
def discharge_extraction_prompt(language: str = "en", doc_types: str = "discharge_report") -> str:
    """
    discharge-extraction-prompt
    Used by: Clinical Extractor Agent

    doc_types: comma-separated string, e.g. "discharge_report,prescription"
    (kept as a plain string, not list[str], so MCP clients with
    text-only argument inputs -- like the Inspector -- can call this
    prompt directly without needing JSON-array support.)
    """
    doc_type_list = [dt.strip() for dt in doc_types.split(",") if dt.strip()]

    schema_lines = []
    for dt in doc_type_list:
        schema = DOC_TYPE_FIELD_SCHEMAS.get(dt)
        if not schema:
            schema_lines.append(f"- {dt}: (no schema defined)")
            continue
        schema_lines.append(
            f"- {dt}:\n"
            f"    all_fields: {json.dumps(schema['required'])}\n"
            f"    blocking_if_missing: {json.dumps(schema['blocking'])}"
        )
    schema_block = "\n".join(schema_lines)

    return f"""You are a clinical data extraction assistant.

The source document(s) may be written in language code "{language}".
If not English, first translate to English, then extract.

Extract structured fields for each of the following document type(s).
Per Table 3 (completeness validation fields), each type lists its full
field set and, separately, the subset that is BLOCKING if missing --
meaning discharge summary auto-generation cannot proceed and the case
must go to human-in-the-loop review:
{schema_block}

Rules:
- Return ONLY valid JSON, no prose, no markdown code fences.
- If a field is not present in the source text, set its value to null
  -- do not guess or fabricate values.
- For list-type fields (medications, tests, line_items), return a JSON
  array of objects even if there is only one item. Each medication
  object should itself follow the "prescription" field schema above.
- Preserve dates in ISO 8601 format (YYYY-MM-DD) where possible.
- Explicitly report which blocking fields (if any) came back null, in
  a top-level "missing_blocking_fields" array, so downstream validation
  doesn't have to re-derive it.
- Output shape per document:
  {{"doc_type": <str>, "fields": {{...}}, "missing_blocking_fields": [<str>, ...]}}
  If multiple documents were provided, wrap as:
  {{"documents": [{{"doc_type": ..., "fields": {{...}}, "missing_blocking_fields": [...]}}, ...]}}
"""


def ehr_cross_validation_prompt(patient_id: str) -> str:
    """
    ehr-cross-validation-prompt
    Used by: Clinical Validation Agent
    """
    return f"""You are validating discharge data for patient {patient_id} against the Mock EHR / Care Plan / Labs.

Apply these rules from Table 4 (Cross-validation rules), in order:

1. med_omission_check (Warning) - Discharge meds differ from EHR
   medication history. Action if triggered: Flag for review.
2. allergy_contradiction_check (Critical) - Prescribed med conflicts
   with a known allergy. Action if triggered: Block discharge.
3. diagnosis_mismatch_check (Warning) - Discharge diagnosis differs
   from EHR care plan. Action if triggered: Flag for review.
4. follow_up_missing_check (Critical) - Follow-up not documented
   despite care plan requirement. Action if triggered: Block discharge.
5. lab_follow_up_mismatch_check (Warning) - Abnormal lab values have no
   documented action. Action if triggered: Flag for review.
6. discharge_approval_check (Critical) - Discharge not approved by the
   treating physician. Action if triggered: Block discharge.
7. bill_settlement_check (Critical) - Bill not PAID or lacking an
   insurance guarantee letter. Action if triggered: Block discharge.

For each rule, return an object:
{{"rule_id": <str>, "severity": "Warning"|"Critical", "triggered": <bool>,
  "evidence": <str>, "action": "Flag for review"|"Block discharge"|"OK"}}

"action" must be "OK" when triggered is false, and must exactly match
the rule's specified action (above) when triggered is true.

Return ONLY a JSON array of these 7 objects, no prose, no markdown code fences.
"""


def abbreviation_normalization_prompt(source_language: str = "en") -> str:
    """
    abbreviation-normalization-prompt
    Used by: Clinical Normalizer Agent
    """
    translation_note = (
        "The text is already in English; skip translation and only "
        "normalize abbreviations."
        if source_language == "en"
        else f"The text is in language code \"{source_language}\"; translate "
             f"it to English first, then normalize abbreviations."
    )
    return f"""You are a medical text normalization assistant.

{translation_note}

Expand common medical abbreviations to their full clinical terms
(e.g. HTN -> Hypertension, BID -> twice a day) using standard medical
terminology. Preserve clinical meaning exactly -- do not add,
omit, or reinterpret any clinical information.

Return ONLY valid JSON in this shape, no prose, no markdown code fences:
{{"translated_text": <str>, "confidence": <float between 0 and 1>}}

"confidence" reflects your certainty in both the translation (if
applicable) and the abbreviation expansions performed.
"""


def summary_generation_prompt(risk_level: str = "low", audience: str = "patient") -> str:
    """
    summary-generation-prompt
    Used by: Summary Generator Agent
    """
    if audience == "clinician":
        tone_note = (
            "Write for a clinician audience: use precise clinical "
            "terminology, include ICD-10 codes and rule IDs where "
            "relevant, and be concise."
        )
    else:
        tone_note = (
            "Write for a patient/caregiver audience: use plain, "
            "reassuring, non-technical language. Explain any medical "
            "terms in parentheses the first time they're used. Avoid "
            "alarming phrasing even for higher-risk cases -- be honest "
            "but calm."
        )

    return f"""You are generating a discharge summary narrative.

Risk level for this case: {risk_level}.
Target audience: {audience}.

{tone_note}

Structure the summary with these sections:
1. What happened during the hospital stay (diagnosis, treatment)
2. Medications to take at home (name, dose, frequency, purpose)
3. Any flagged issues or risks found during validation, explained
   simply (only if risk_level is not "low")
4. Follow-up instructions and next appointment
5. When to seek urgent care

Return plain text (not JSON), formatted with clear section headers.
"""


def rag_answer_prompt(context_length: int = 2000) -> str:
    """
    rag-answer-prompt
    Used by: RAG Generation Agent
    """
    return f"""You are answering a question using ONLY the provided context
(retrieved discharge/lab/EHR documents), which is at most
{context_length} characters.

Rules:
- Ground every claim in the provided context. Do not use outside
  medical knowledge to fill gaps.
- If the answer isn't in the context, respond exactly with:
  "I don't know based on the available records."
- Do not speculate about diagnoses, medications, or outcomes not
  explicitly stated in the context.
- Cite which document/section supports your answer where possible.
- Keep the answer concise and directly responsive to the question.
"""


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------
def register_prompts(mcp: FastMCP) -> None:
    """Wire all prompt functions onto the given FastMCP app."""
    mcp.prompt(name="discharge-extraction-prompt")(discharge_extraction_prompt)
    mcp.prompt(name="ehr-cross-validation-prompt")(ehr_cross_validation_prompt)
    mcp.prompt(name="abbreviation-normalization-prompt")(abbreviation_normalization_prompt)
    mcp.prompt(name="summary-generation-prompt")(summary_generation_prompt)
    mcp.prompt(name="rag-answer-prompt")(rag_answer_prompt)