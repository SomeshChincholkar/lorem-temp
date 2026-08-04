"""
Dashboard page 1 — Document Viewer (spec Table 13).

Patient selector • tabbed Discharge/Lab/Bill view • language detection
badge • structured data preview • process trigger button.
"""

import json

import streamlit as st

from dashboard.common import (
    DOC_TYPE_LABELS,
    available_doc_types,
    detect_document_language,
    language_badge,
    load_document,
    load_report,
    run_async,
    selected_patient,
)

st.title("Document Viewer")

patient_id = selected_patient()
if not patient_id:
    st.stop()

doc_types = available_doc_types(patient_id)
if not doc_types:
    st.warning(f"No documents on disk for {patient_id}.")
    st.stop()

# ---------------------------------------------------------------------
# Process trigger
# ---------------------------------------------------------------------
left, right = st.columns([3, 1])
with left:
    st.caption(
        f"{len(doc_types)} document(s) for **{patient_id}**: "
        + ", ".join(DOC_TYPE_LABELS[d] for d in doc_types)
    )
with right:
    trigger = st.button("Run pipeline", type="primary", use_container_width=True)

if trigger:
    from agents.orchestrator.pipeline import run_discharge_pipeline

    with st.spinner(f"Running the discharge pipeline for {patient_id}..."):
        result = run_async(run_discharge_pipeline(patient_id))

    st.session_state["last_pipeline_result"] = result

    if result.get("ok"):
        st.success(
            f"Pipeline complete — risk **{result.get('risk_level')}**, "
            f"status **{result.get('final_status')}**. "
            "See the Validation Report page."
        )
    else:
        st.error("Pipeline did not complete. Step detail below.")

    failed = [s for s in result.get("steps", []) if not s["ok"]]
    if failed:
        st.warning("Failed steps: " + ", ".join(
            f"{s['step']}{' (' + s['detail'] + ')' if s.get('detail') else ''}"
            for s in failed
        ))
    with st.expander("Full pipeline trace"):
        st.json(result)

existing = load_report(patient_id)
if existing and not trigger:
    st.caption(
        f"A validation report already exists for {patient_id} "
        f"(risk: {existing.get('risk_level')}). Re-running will overwrite it."
    )

st.divider()

# ---------------------------------------------------------------------
# Tabbed document view
# ---------------------------------------------------------------------
tabs = st.tabs([DOC_TYPE_LABELS[d] for d in doc_types])

for tab, doc_type in zip(tabs, doc_types):
    with tab:
        document = load_document(patient_id, doc_type)

        if document["error"]:
            st.error(f"Could not read this document: {document['error']}")
            continue

        language = detect_document_language(document["text"])

        meta_left, meta_mid, meta_right = st.columns(3)
        meta_left.metric("File", document["filename"])
        meta_mid.metric("Format", document["format"])
        meta_right.markdown(f"**Language**\n\n{language_badge(language)}")

        if language != "en":
            st.info(
                f"This document is in {language_badge(language)} and will be "
                "translated by the Normalizer agent during the pipeline run."
            )

        # Structured preview: JSON documents render as a tree, everything
        # else as raw text.
        st.subheader("Structured preview")
        if document["format"] == ".json":
            try:
                st.json(json.loads(document["text"]))
            except json.JSONDecodeError:
                st.text(document["text"])
        else:
            st.text_area(
                "Extracted text",
                document["text"],
                height=400,
                label_visibility="collapsed",
            )
