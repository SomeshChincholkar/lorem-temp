"""
agents/validator_agent/nodes.py

Node functions for the Clinical Validation Agent graph:
  completeness -> ehr_cross_validate -> report -> decide

MCP primitives exercised here (spec Table 6: Tools + Elicitation + Resources):
  Tools       : clinical_rules_engine_tool, ehr_validation_tool,
                clinical_insight_reporter_tool
  Elicitation : the Rules Engine tool calls ctx.elicit() back into this
                agent's elicitation callback (see elicitation.py)
  Resources   : clinical-rules/cross-validation, read so the audit trail
                records the rule set the run was judged against
"""

import json

from agents.common.mcp_client import call_tool, read_resource_text

from .elicitation import make_elicitation_callback
from .state import ValidatorState

# Empty gaps, used whenever the Rules Engine returns a status that
# carries no explicit gap lists ("complete"/"resolved").
NO_GAPS = {"missing_blocking": [], "missing_nonblocking": []}


async def node_completeness_check(state: ValidatorState) -> ValidatorState:
    """
    Table 3 completeness check. This is the node that can pause for a
    human: if only non-blocking fields are missing, the Rules Engine
    tool calls ctx.elicit() and this agent's callback parks the request
    for the Streamlit dashboard.

    The callback is built per-run so the parked request carries this
    patient's context -- the MCP elicitation request itself only carries
    a message and a schema.
    """
    callback = make_elicitation_callback(
        patient_id=state.get("patient_id"),
        doc_type="discharge_report",
        trace_id=state.get("trace_id"),
    )

    try:
        result = await call_tool(
            "clinical_rules_engine_tool",
            {
                "doc_type": "discharge_report",
                "extracted_fields": state.get("extracted_discharge") or {},
            },
            elicitation_callback=callback,
        )
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"Rules Engine tool call failed: {exc}"
        state["completeness_result"] = {"status": "blocked", "unresolved_fields": []}
        state["completeness_gaps"] = dict(NO_GAPS)
        return state

    state["completeness_result"] = result

    # If the reviewer supplied values, carry them forward -- the EHR
    # cross-validation below must judge the corrected record, not the
    # original one.
    if result.get("fields"):
        state["extracted_discharge"] = result["fields"]

    unresolved = result.get("unresolved_fields", [])
    if result.get("status") == "blocked":
        state["completeness_gaps"] = {
            "missing_blocking": unresolved,
            "missing_nonblocking": [],
        }
    elif result.get("status") == "unresolved":
        state["completeness_gaps"] = {
            "missing_blocking": [],
            "missing_nonblocking": unresolved,
        }
    else:
        state["completeness_gaps"] = dict(NO_GAPS)

    return state


async def node_ehr_cross_validate(state: ValidatorState) -> ValidatorState:
    """
    Run the seven Table 4 rules against the Mock EHR.

    Also reads resource://clinical-rules/cross-validation so the run is
    stamped with the policy set it was judged against -- an auditor
    needs to know which rules were live, not just which fired.
    """
    try:
        findings = await call_tool(
            "ehr_validation_tool",
            {
                "patient_id": state["patient_id"],
                "extracted_discharge": state.get("extracted_discharge") or {},
                "extracted_bill": state.get("extracted_bill") or {},
            },
        )
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"EHR validation tool call failed: {exc}"
        state["ehr_findings"] = []
        return state

    state["ehr_findings"] = findings if isinstance(findings, list) else []

    try:
        policies = await read_resource_text("resource://clinical-rules/cross-validation")
        state["cross_validation_policies"] = json.loads(policies)
    except Exception:
        # Audit metadata only -- never fail validation because the
        # policy resource was unreachable.
        pass

    return state


async def node_report(state: ValidatorState) -> ValidatorState:
    """
    Build the JSON + HTML audit report. Per the build guide this runs as
    the Validator's final tool call rather than a separate agent hop, so
    a single A2A round trip yields a complete, persisted verdict.
    """
    all_inputs = {
        "extracted_discharge": state.get("extracted_discharge") or {},
        "extracted_bill": state.get("extracted_bill") or {},
        "completeness_gaps": state.get("completeness_gaps") or dict(NO_GAPS),
        "ehr_findings": state.get("ehr_findings") or [],
        "translation_confidence": state.get("translation_confidence"),
    }

    try:
        state["report"] = await call_tool(
            "clinical_insight_reporter_tool",
            {"patient_id": state["patient_id"], "all_inputs": all_inputs},
        )
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"Reporter tool call failed: {exc}"
        state["report"] = {}

    return state


def decide_final_status(
    completeness_result: dict, ehr_findings: list, report: dict
) -> str:
    """
    Collapse the three signals into one verdict.

    Order matters and is deliberate: anything blocking wins over
    anything advisory, and the Reporter's own discharge_blocked flag is
    honoured even when no single finding looks Critical here -- it also
    accounts for forced-high conditions like low translation confidence.
    """
    if completeness_result.get("status") == "blocked":
        return "blocked"

    if any(
        f.get("severity") == "Critical" and f.get("triggered") for f in ehr_findings
    ):
        return "blocked"

    if report.get("discharge_blocked"):
        return "blocked"

    if completeness_result.get("status") == "unresolved":
        return "hitl"

    if any(f.get("triggered") for f in ehr_findings):
        return "hitl"

    if str(report.get("risk_level", "")).lower() == "high":
        return "blocked"
    if str(report.get("risk_level", "")).lower() == "medium":
        return "hitl"

    return "auto_approve"


async def node_decide(state: ValidatorState) -> ValidatorState:
    state["final_status"] = decide_final_status(
        state.get("completeness_result") or {},
        state.get("ehr_findings") or [],
        state.get("report") or {},
    )
    return state
