"""
Dashboard page 3 — HITL Corrections (spec Table 13).

Editable medication table • dynamic schema-driven Elicitation Response
Form • risk label override • approval decision • save feedback • re-run
validation.

This page is the client half of the MCP Elicitation primitive. The
Validation Agent's callback parks a request in the shared store and
blocks; the form below is what unblocks it. Rendering the form from the
schema the server sent -- rather than from a hardcoded field list -- is
what makes this a real Elicitation implementation.
"""

import json
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from agents.common import elicitation_store
from dashboard.common import (
    PROJECT_ROOT,
    require_report,
    risk_badge,
    run_async,
    selected_patient,
)

FEEDBACK_DIR = PROJECT_ROOT / "Data" / "feedback"

st.title("HITL Corrections")

patient_id = selected_patient()
if not patient_id:
    st.stop()

report = require_report(patient_id)
if report is None:
    st.stop()


# =====================================================================
# 1. Elicitation Response Form (dynamic, schema-driven)
# =====================================================================
st.subheader("Elicitation requests")

pending = elicitation_store.list_pending()

if not pending:
    st.caption(
        "No pending elicitation requests. One appears here when the Rules "
        "Engine finds non-blocking missing fields during a validation run."
    )
else:
    st.warning(
        f"{len(pending)} request(s) waiting. The Validation Agent is blocked "
        "until each is answered or times out."
    )

for request in pending:
    header = f"{request.get('patient_id') or 'unknown patient'} — {request['message']}"
    with st.expander(header, expanded=True):
        schema = request.get("schema") or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])

        st.caption(
            f"Request `{request['request_id'][:8]}` · "
            f"doc type: {request.get('doc_type') or 'n/a'} · "
            f"trace: {request.get('trace_id') or 'n/a'}"
        )

        with st.form(f"elicit_{request['request_id']}"):
            values: dict = {}

            # Widget type is driven by the schema the SERVER sent. The
            # MCP spec restricts elicitation schemas to primitives, so
            # these four branches cover every legal case.
            for field, definition in properties.items():
                label = field + (" *" if field in required else "")
                help_text = definition.get("description")
                field_type = definition.get("type", "string")

                if field_type == "boolean":
                    values[field] = st.checkbox(label, help=help_text)
                elif field_type == "integer":
                    values[field] = st.number_input(label, step=1, help=help_text)
                elif field_type == "number":
                    values[field] = st.number_input(label, help=help_text)
                elif definition.get("enum"):
                    values[field] = st.selectbox(label, definition["enum"], help=help_text)
                else:
                    values[field] = st.text_input(label, help=help_text)

            accept_col, decline_col, cancel_col = st.columns(3)
            accepted = accept_col.form_submit_button("Accept", type="primary")
            declined = decline_col.form_submit_button("Decline")
            cancelled = cancel_col.form_submit_button("Cancel")

        # All three MCP elicitation outcomes are reachable from the UI.
        if accepted:
            missing = [f for f in required if not str(values.get(f, "")).strip()]
            if missing:
                st.error("Required field(s) still empty: " + ", ".join(missing))
            else:
                elicitation_store.respond(
                    request["request_id"], elicitation_store.ACCEPTED, values
                )
                st.success("Accepted — the Validation Agent will continue.")
                time.sleep(0.5)
                st.rerun()
        elif declined:
            elicitation_store.respond(request["request_id"], elicitation_store.DECLINED)
            st.info("Declined — fields marked unresolved and flagged for review.")
            time.sleep(0.5)
            st.rerun()
        elif cancelled:
            elicitation_store.respond(request["request_id"], elicitation_store.CANCELLED)
            st.warning("Cancelled — the case is escalated.")
            time.sleep(0.5)
            st.rerun()

st.divider()

# =====================================================================
# 2. Editable medication table
# =====================================================================
st.subheader("Medications")

medications = (report.get("extracted_discharge") or {}).get("medications") or []
if not medications and report.get("medications"):
    medications = report["medications"]

if medications:
    med_frame = pd.DataFrame(medications)
else:
    # An empty editable frame still lets a reviewer add the medications
    # the extractor missed, which is exactly the blocking-field case.
    med_frame = pd.DataFrame(
        columns=[
            "sl_no", "medicine_name", "strength", "dosage",
            "frequency", "route", "period", "remarks", "total_quantity",
        ]
    )
    st.caption("No medications extracted — add them below if the record has them.")

edited_medications = st.data_editor(
    med_frame,
    num_rows="dynamic",
    use_container_width=True,
    key=f"meds_{patient_id}",
)

st.divider()

# =====================================================================
# 3. Reviewer decision
# =====================================================================
st.subheader("Reviewer decision")

current_risk = str(report.get("risk_level") or "low").lower()
st.markdown(f"System assessment: {risk_badge(current_risk)}")

decision_left, decision_right = st.columns(2)

with decision_left:
    risk_options = ["low", "medium", "high"]
    risk_override = st.selectbox(
        "Risk level override",
        risk_options,
        index=risk_options.index(current_risk) if current_risk in risk_options else 0,
    )

with decision_right:
    approval = st.radio(
        "Approval decision",
        ["Approve", "Approve with edits", "Reject"],
        horizontal=False,
    )

reviewer_notes = st.text_area("Reviewer notes", placeholder="Why this decision?")

# The HITL escalation guardrail applies to the reviewer's own override:
# a case cannot be approved out of a blocked state without acknowledging it.
if report.get("discharge_blocked") and approval == "Approve":
    st.error(
        "This discharge is **blocked** by a critical rule. Approving it "
        "overrides a hard guardrail — record why in the notes."
    )

save_col, rerun_col = st.columns(2)
save_clicked = save_col.button("Save feedback", type="primary", use_container_width=True)
rerun_clicked = rerun_col.button("Re-run validation", use_container_width=True)

if save_clicked:
    if report.get("discharge_blocked") and approval == "Approve" and not reviewer_notes.strip():
        st.error("Notes are required when overriding a blocked discharge.")
    else:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        feedback = {
            "patient_id": patient_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "system_risk_level": current_risk,
            "reviewer_risk_level": risk_override,
            "approval_decision": approval,
            "reviewer_notes": reviewer_notes,
            "overrode_block": bool(report.get("discharge_blocked")) and approval == "Approve",
            "corrected_medications": edited_medications.to_dict(orient="records"),
        }
        path = FEEDBACK_DIR / f"{patient_id}_feedback.json"
        path.write_text(json.dumps(feedback, indent=2, default=str), encoding="utf-8")
        st.success(f"Feedback saved to `{path.relative_to(PROJECT_ROOT)}`")

if rerun_clicked:
    from agents.common.a2a_client import send_message

    st.caption("Re-validating with the corrected medication list...")
    corrected = dict(report.get("extracted_discharge") or {})
    corrected["medications"] = edited_medications.to_dict(orient="records")

    with st.spinner("Calling the Validation Agent..."):
        result = run_async(
            send_message(
                "validator",
                {
                    "patient_id": patient_id,
                    "extracted_discharge": corrected,
                    "extracted_bill": report.get("extracted_bill") or {},
                    "translation_confidence": report.get("translation_confidence"),
                },
            )
        )

    if result["ok"] and result["artifacts"]:
        artifact = result["artifacts"][0]
        st.success(
            f"Re-validated — status **{artifact.get('final_status')}**, "
            f"risk **{artifact.get('risk_level')}**."
        )
        st.json(artifact)
    else:
        st.error(f"Re-validation failed: {result['error']}")
