"""
Dashboard page 2 — Validation Report (spec Table 13).

Colour-coded completeness score • cross-validation issues table • risk
level badge • recommendation • discharge blocked indicator • LangFuse
trace link.
"""

import pandas as pd
import streamlit as st

from dashboard.common import (
    require_report,
    risk_badge,
    selected_patient,
)
from observability import trace_url

st.title("Validation Report")

patient_id = selected_patient()
if not patient_id:
    st.stop()

report = require_report(patient_id)
if report is None:
    st.stop()

# ---------------------------------------------------------------------
# Headline verdict
# ---------------------------------------------------------------------
blocked = bool(report.get("discharge_blocked"))

if blocked:
    st.error("**DISCHARGE BLOCKED** — this case cannot be auto-released.")
else:
    st.success("Discharge not blocked by validation.")

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"**Risk level**\n\n{risk_badge(report.get('risk_level'))}")
col2.metric("Risk score", report.get("risk_score", "—"))
confidence = report.get("translation_confidence")
col3.metric(
    "Translation confidence",
    f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a",
)
col4.metric("Blocked", "Yes" if blocked else "No")

st.info(f"**Recommendation:** {report.get('recommendation') or 'n/a'}")

# ---------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------
st.subheader("Completeness")

completeness = report.get("completeness") or {}
missing_blocking = completeness.get("missing_blocking", [])
missing_nonblocking = completeness.get("missing_nonblocking", [])

total_missing = len(missing_blocking) + len(missing_nonblocking)
# Scored against Table 3's discharge_report field list (17 fields), so
# the bar means the same thing from one case to the next.
score = max(0.0, 1.0 - total_missing / 17)

st.progress(score, text=f"Completeness {score:.0%} — {total_missing} field(s) missing")

if missing_blocking:
    st.error("**Blocking fields missing:** " + ", ".join(f"`{f}`" for f in missing_blocking))
if missing_nonblocking:
    st.warning(
        "**Non-blocking fields missing:** "
        + ", ".join(f"`{f}`" for f in missing_nonblocking)
        + " — resolvable on the HITL Corrections page."
    )
if not total_missing:
    st.success("All required fields present.")

# ---------------------------------------------------------------------
# Cross-validation findings (Table 4)
# ---------------------------------------------------------------------
st.subheader("Cross-validation against the EHR")

findings = report.get("ehr_findings") or []
if not findings:
    st.info("No cross-validation findings recorded for this case.")
else:
    frame = pd.DataFrame(findings)
    for column in ("rule_id", "severity", "triggered", "evidence", "action"):
        if column not in frame.columns:
            frame[column] = None
    frame = frame[["rule_id", "severity", "triggered", "action", "evidence"]]

    def highlight(row):
        if row["triggered"] and row["severity"] == "Critical":
            return ["background-color: #7f1d1d; color: white"] * len(row)
        if row["triggered"]:
            return ["background-color: #78350f; color: white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        frame.style.apply(highlight, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    triggered = [f for f in findings if f.get("triggered")]
    critical = [f for f in triggered if f.get("severity") == "Critical"]
    st.caption(
        f"{len(triggered)} of {len(findings)} rules triggered "
        f"({len(critical)} critical)."
    )

# ---------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------
st.subheader("Audit trail")

trace_id = st.session_state.get("last_pipeline_result", {}).get("trace_id")
audit_left, audit_right = st.columns(2)
audit_left.text(f"Rules version (SHA-256): {str(report.get('rules_version'))[:16]}...")
audit_left.text(f"Generated at: {report.get('generated_at')}")

if trace_id:
    audit_right.code(trace_id, language=None)
    url = trace_url(trace_id)
    if url:
        audit_right.markdown(f"[Open in LangFuse]({url})")
    else:
        # Showing the raw trace_id is still useful when LangFuse is
        # unconfigured -- it is what correlates this case across every
        # agent's own logs.
        audit_right.caption(
            "Trace ID — set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY for a deep link."
        )
else:
    audit_right.caption("No trace ID in this session — run the pipeline from page 1.")

with st.expander("Raw report JSON"):
    st.json(report)
