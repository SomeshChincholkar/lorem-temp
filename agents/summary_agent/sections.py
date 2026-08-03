"""
agents/summary_agent/sections.py

Builds the five sections of a patient-friendly discharge summary, in the
order the spec requires (Table 10):

    patient -> meds -> labs -> bill -> instructions

Each section is generated separately and streamed as its own A2A
artifact update, which is what makes the streaming progressive rather
than "one blob at the end". Section prompts are derived from the
summary-generation-prompt fetched via MCP Prompts, never hardcoded.
"""

import json
from pathlib import Path

from agents.common.llm import get_llm
from agents.common.mcp_client import get_prompt_text

SECTION_ORDER = ("patient", "meds", "labs", "bill", "instructions")

SECTION_BRIEFS = {
    "patient": (
        "Write ONLY the 'Your hospital stay' section: who was treated, "
        "when they were admitted and discharged, and what they were "
        "treated for. 2-4 sentences."
    ),
    "meds": (
        "Write ONLY the 'Your medicines' section: list each discharge "
        "medication with its dose, how often to take it, and what it is "
        "for, in plain English. Expand any abbreviations."
    ),
    "labs": (
        "Write ONLY the 'Your test results' section: explain the lab "
        "results in plain English, calling out anything outside the "
        "normal range and what it means. If there are no lab results, "
        "say so in one sentence."
    ),
    "bill": (
        "Write ONLY the 'Your bill' section: the total amount and "
        "whether it is settled. One short paragraph. If there is no "
        "billing information, say so in one sentence."
    ),
    "instructions": (
        "Write ONLY the 'What to do next' section: follow-up "
        "appointments, self-care instructions, and the warning signs "
        "that mean the patient should seek urgent help."
    ),
}

REPORTS_DIR = Path("Data/reports")


def load_report(patient_id: str) -> dict:
    """
    Load the audit report the Validator's Reporter tool persisted.

    This is the summary's only source of truth -- generating a
    patient-facing document from anything the Validator hasn't already
    judged would let unvalidated content reach a patient.
    """
    path = REPORTS_DIR / f"{patient_id}_report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No validation report at {path}. Run the Validation Agent for "
            f"{patient_id} before generating a summary."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def section_context(section: str, report: dict) -> str:
    """
    Slice the report down to just what this section needs, so each
    generation stays focused and the model can't wander into other
    sections' material.
    """
    if section == "patient":
        data = {
            "patient_id": report.get("patient_id"),
            "patient_name": report.get("patient_name"),
            "discharge_diagnosis": report.get("discharge_diagnosis"),
        }
    elif section == "meds":
        data = {
            "medications": (report.get("extracted_discharge") or {}).get("medications"),
            "medication_findings": [
                f for f in report.get("ehr_findings", [])
                if "med" in f.get("rule_id", "") or "allergy" in f.get("rule_id", "")
            ],
        }
    elif section == "labs":
        data = {
            "lab_findings": [
                f for f in report.get("ehr_findings", [])
                if "lab" in f.get("rule_id", "")
            ],
            "tests": (report.get("extracted_lab") or {}).get("tests"),
        }
    elif section == "bill":
        extracted_bill = report.get("extracted_bill") or {}
        data = {
            "total_amount": extracted_bill.get("total_amount"),
            "payment_status": extracted_bill.get("payment_status"),
            "bill_findings": [
                f for f in report.get("ehr_findings", [])
                if "bill" in f.get("rule_id", "")
            ],
        }
    else:  # instructions
        data = {
            "follow_up_instructions": report.get("follow_up_instructions"),
            "recommendation": report.get("recommendation"),
            "risk_level": report.get("risk_level"),
        }

    return json.dumps(data, indent=2, default=str)


async def build_base_prompt(risk_level: str, audience: str = "patient") -> str:
    """Fetch summary-generation-prompt via MCP Prompts."""
    return await get_prompt_text(
        "summary-generation-prompt", {"risk_level": risk_level, "audience": audience}
    )


async def stream_section(section: str, report: dict, base_prompt: str):
    """
    Async generator yielding text chunks for one section.

    Streams token-by-token off the Bedrock client so the dashboard can
    paint text as it arrives. If the model or transport doesn't support
    streaming, this still yields one final chunk rather than failing --
    a degraded stream beats no summary.
    """
    prompt = (
        f"{base_prompt}\n\n"
        f"SECTION TO WRITE NOW: {section}\n"
        f"{SECTION_BRIEFS[section]}\n\n"
        "Do not write any other section. Do not repeat the section "
        "heading in your output.\n\n"
        f"DATA FOR THIS SECTION:\n{section_context(section, report)}"
    )

    llm = get_llm()

    try:
        async for chunk in llm.astream(prompt):
            text = getattr(chunk, "content", None)
            if isinstance(text, list):
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in text
                )
            if text:
                yield text
    except Exception:
        response = await llm.ainvoke(prompt)
        text = response.content
        if isinstance(text, list):
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )
        yield text or ""
