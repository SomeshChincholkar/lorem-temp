"""
Dashboard page 5 — Discharge Summary (spec Table 13).

Patient-friendly summary for auto-approved cases • plain-English
prescription table • colour-coded lab results • export JSON / HTML / PDF
• LangFuse trace link.
"""

import json

import pandas as pd
import streamlit as st

from common.pdf_export import summary_to_pdf_bytes
from dashboard.common import (
    report_html,
    require_report,
    risk_badge,
    run_async,
    selected_patient,
)
from observability import trace_url

st.title("Discharge Summary")

patient_id = selected_patient()
if not patient_id:
    st.stop()

report = require_report(patient_id)
if report is None:
    st.stop()

# ---------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------
blocked = bool(report.get("discharge_blocked"))
risk_level = str(report.get("risk_level") or "low").lower()

st.markdown(f"Risk level: {risk_badge(risk_level)} · Blocked: **{'Yes' if blocked else 'No'}**")

if blocked:
    # Spec Table 13 scopes this page to auto-approved cases, and Table 12
    # row 5 makes human review mandatory here. Generating a reassuring
    # patient-facing document for a case a clinician has not cleared is
    # exactly the failure the guardrail exists to prevent.
    st.error(
        "**This discharge is blocked.** A patient-friendly summary is not "
        "generated for blocked cases. Resolve the findings on the "
        "Validation Report and HITL Corrections pages first."
    )
    triggered = [f for f in report.get("ehr_findings", []) if f.get("triggered")]
    if triggered:
        st.markdown("**Outstanding findings**")
        for finding in triggered:
            st.markdown(
                f"- `{finding['rule_id']}` ({finding['severity']}): {finding.get('evidence')}"
            )
    st.stop()

# ---------------------------------------------------------------------
# Generate (streaming)
# ---------------------------------------------------------------------
audience = st.radio("Audience", ["patient", "clinician"], horizontal=True)

if st.button("Generate summary", type="primary"):
    st.subheader("Summary")
    placeholder = st.empty()

    from agents.common.a2a_client import send_message_streaming, stream_text_from_event

    async def generate():
        text = ""
        final = None
        async for event in send_message_streaming(
            "summary", {"patient_id": patient_id, "audience": audience}
        ):
            if event.get("type") == "error":
                return text, {"error": event["error"]}

            chunk = stream_text_from_event(event)
            if chunk:
                text += chunk
                placeholder.markdown(text)

            result = event.get("result", event)
            artifact = result.get("artifact") or {}
            for part in artifact.get("parts", []):
                if part.get("kind") == "data":
                    final = part["data"]
        return text, final

    with st.spinner("Streaming summary section by section..."):
        streamed, final_payload = run_async(generate())

    if final_payload and final_payload.get("error"):
        st.error(f"Streaming failed: {final_payload['error']}")
    elif final_payload:
        st.session_state[f"summary_{patient_id}"] = final_payload
    elif not streamed:
        placeholder.info("Nothing streamed — is the Summary agent running on :8104?")

summary = st.session_state.get(f"summary_{patient_id}")

if summary:
    st.divider()
    section_titles = {
        "patient": "Your hospital stay",
        "meds": "Your medicines",
        "labs": "Your test results",
        "bill": "Your bill",
        "instructions": "What to do next",
    }
    for key, title in section_titles.items():
        text = (summary.get("sections") or {}).get(key)
        if text:
            st.subheader(title)
            st.write(text)

    if summary.get("guardrail_events"):
        st.warning(
            f"{len(summary['guardrail_events'])} section(s) were modified by the "
            "toxicity guardrail."
        )
        with st.expander("Guardrail events"):
            st.json(summary["guardrail_events"])

st.divider()

# ---------------------------------------------------------------------
# Plain-English prescription table
# ---------------------------------------------------------------------
st.subheader("Your medicines")

medications = (report.get("extracted_discharge") or {}).get("medications") or []
if medications:
    frame = pd.DataFrame(medications)
    rename = {
        "medicine_name": "Medicine",
        "strength": "Strength",
        "dosage": "How much",
        "frequency": "How often",
        "route": "How to take it",
        "period": "For how long",
        "remarks": "Notes",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    display_columns = [c for c in rename.values() if c in frame.columns]
    st.dataframe(
        frame[display_columns] if display_columns else frame,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No medications recorded on this discharge.")

# ---------------------------------------------------------------------
# Colour-coded lab results
# ---------------------------------------------------------------------
st.subheader("Your test results")

lab_findings = [f for f in report.get("ehr_findings", []) if "lab" in f.get("rule_id", "")]
if lab_findings:
    for finding in lab_findings:
        if finding.get("triggered"):
            st.warning(f"{finding['rule_id']}: {finding.get('evidence')}")
        else:
            st.success(f"{finding['rule_id']}: no outstanding action")
else:
    st.caption("No lab findings recorded for this discharge.")

# ---------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------
st.divider()
st.subheader("Export")

json_column, html_column, pdf_column = st.columns(3)

export_payload = {
    "patient_id": patient_id,
    "report": report,
    "summary": summary,
}
json_column.download_button(
    "Download JSON",
    data=json.dumps(export_payload, indent=2, default=str),
    file_name=f"{patient_id}_discharge.json",
    mime="application/json",
    use_container_width=True,
)

html = report_html(patient_id)
html_column.download_button(
    "Download HTML",
    data=html or "<p>No HTML report generated yet.</p>",
    file_name=f"{patient_id}_report.html",
    mime="text/html",
    disabled=html is None,
    use_container_width=True,
)

if summary:
    try:
        pdf_bytes = summary_to_pdf_bytes(patient_id, summary)
        pdf_column.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{patient_id}_discharge_summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:  # noqa: BLE001
        pdf_column.button("Download PDF", disabled=True, use_container_width=True)
        pdf_column.caption(f"PDF unavailable: {exc}")
else:
    pdf_column.button("Download PDF", disabled=True, use_container_width=True)
    pdf_column.caption("Generate the summary first.")

trace_id = st.session_state.get("last_pipeline_result", {}).get("trace_id")
if trace_id:
    url = trace_url(trace_id)
    if url:
        st.caption(f"Trace ID: `{trace_id}` — [open in LangFuse]({url})")
    else:
        st.caption(f"Trace ID: `{trace_id}`")
