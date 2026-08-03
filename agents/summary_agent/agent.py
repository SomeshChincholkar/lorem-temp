"""
agents/summary_agent/agent.py

The Discharge Summary Generator's ADK agent definition (spec 6.2,
Google ADK, A2A port 8104, STREAMING).

The streaming path in agent_executor.py drives sections.py directly so
it can emit one A2A artifact update per section. This ADK agent is the
conversational entry point (used by the Host Orchestrator's Gradio UI
and ADK tooling), exposing the same summary generation as a tool.
"""

from google.adk.agents import LlmAgent

from agents.common.adk_runtime import get_adk_model

from .sections import (
    SECTION_ORDER,
    build_base_prompt,
    load_report,
    stream_section,
)


async def generate_discharge_summary(patient_id: str, audience: str = "patient") -> dict:
    """Generate the full discharge summary for a patient.

    Args:
        patient_id: e.g. "P1019". A validation report must already exist.
        audience: "patient" for plain English, "clinician" for clinical
            terminology.

    Returns:
        A dict with a "sections" mapping of section name to text, plus
        the risk_level the summary was written for.
    """
    report = load_report(patient_id)
    risk_level = report.get("risk_level", "low")
    base_prompt = await build_base_prompt(risk_level, audience)

    sections: dict[str, str] = {}
    for section in SECTION_ORDER:
        chunks = [chunk async for chunk in stream_section(section, report, base_prompt)]
        sections[section] = "".join(chunks).strip()

    return {
        "patient_id": patient_id,
        "risk_level": risk_level,
        "audience": audience,
        "sections": sections,
    }


summary_agent = LlmAgent(
    name="discharge_summary_agent",
    model=get_adk_model(),
    description="Writes patient-friendly discharge summaries from a validated audit report.",
    instruction=(
        "You produce discharge summaries for patients leaving hospital.\n"
        "When asked for a summary, call generate_discharge_summary with "
        "the patient ID. Present the returned sections in order: "
        "hospital stay, medicines, test results, bill, what to do next.\n"
        "Never invent clinical details that are not in the tool's "
        "output -- everything a patient reads must trace back to their "
        "validated record."
    ),
    tools=[generate_discharge_summary],
)
