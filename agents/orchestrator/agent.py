"""
agents/orchestrator/agent.py

The Host Orchestrator's ADK agent (spec section 8, Google ADK, :8083).

Unlike every other agent here, this one exposes no clinical tools of its
own -- it is an A2A *client*. Its tools are "call another agent", which
is exactly the coordination role Table 6 assigns it.
"""

import json

from google.adk.agents import LlmAgent

from agents.common.a2a_client import fetch_agent_card, send_message
from agents.common.adk_runtime import get_adk_model

from .pipeline import run_discharge_pipeline


async def process_patient_discharge(patient_id: str) -> dict:
    """Run the full discharge pipeline for one patient.

    Sequences the Monitor, Extractor, Normalizer and Validation agents
    over A2A and returns the verdict.

    Args:
        patient_id: e.g. "P1019".

    Returns:
        Risk level, recommendation, whether discharge is blocked, whether
        a human must review, and a per-step trace of the run.
    """
    return await run_discharge_pipeline(patient_id)


async def ask_about_records(question: str, patient_id: str = "") -> dict:
    """Ask the Clinical RAG Q&A agent a question about patient records.

    Args:
        question: the administrator's question.
        patient_id: optional, restricts retrieval to one patient.

    Returns:
        The grounded answer, its sources, and RAG Triad quality scores.
    """
    result = await send_message(
        "rag",
        {"question": question, "patient_filter": patient_id or None},
    )
    if result["ok"] and result["artifacts"]:
        return result["artifacts"][-1]
    return {"answer": "The Q&A agent could not be reached.", "error": result["error"]}


async def check_agent_health() -> dict:
    """Report which A2A agents are currently reachable.

    Returns:
        A mapping of agent name to its discovered card name, or to null
        when that agent is not responding.
    """
    health = {}
    for name in ("monitor", "extractor", "normalizer", "validator", "summary", "rag"):
        card = await fetch_agent_card(name)
        health[name] = card.get("name") if card else None
    return {"agents": health}


orchestrator_agent = LlmAgent(
    name="host_orchestrator",
    model=get_adk_model(),
    description=(
        "Coordinates the discharge pipeline across every A2A agent and "
        "answers questions about the results."
    ),
    instruction=(
        "You coordinate a hospital discharge review system.\n"
        "- To review a patient, call process_patient_discharge with their ID.\n"
        "- To answer a question about records, call ask_about_records.\n"
        "- To report system status, call check_agent_health.\n"
        "Always state the risk level, the recommendation, and whether a "
        "human reviewer is required. Never tell a user a discharge is "
        "approved when requires_hitl is true."
    ),
    tools=[process_patient_discharge, ask_about_records, check_agent_health],
)


def format_pipeline_result(result: dict) -> str:
    """Human-readable pipeline summary for the Gradio UI."""
    lines = [
        f"### Patient {result['patient_id']}",
        f"- **Risk level**: {result.get('risk_level') or 'n/a'}",
        f"- **Status**: {result.get('final_status') or 'n/a'}",
        f"- **Recommendation**: {result.get('recommendation') or 'n/a'}",
        f"- **Discharge blocked**: {result.get('discharge_blocked')}",
        f"- **Requires human review**: {result.get('requires_hitl')}",
        f"- **Trace ID**: `{result.get('trace_id')}`",
        "",
        "**Pipeline steps**",
    ]
    for step in result.get("steps", []):
        mark = "OK" if step["ok"] else "FAIL"
        detail = f" ({step['detail']})" if step.get("detail") else ""
        error = f" — {step['error']}" if step.get("error") else ""
        lines.append(f"- [{mark}] {step['step']}{detail}{error}")

    findings = [f for f in result.get("ehr_findings", []) if f.get("triggered")]
    if findings:
        lines.append("")
        lines.append("**Triggered cross-validation rules**")
        for finding in findings:
            lines.append(
                f"- `{finding['rule_id']}` ({finding['severity']}): {finding.get('evidence')}"
            )

    if result.get("json_path"):
        lines.append("")
        lines.append(f"Report written to `{result['json_path']}`")

    return "\n".join(lines)


def format_json(value) -> str:
    return json.dumps(value, indent=2, default=str)
