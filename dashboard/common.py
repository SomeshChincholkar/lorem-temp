"""
dashboard/common.py

Shared helpers for the Streamlit HITL dashboard (spec section 8).

Streamlit reruns the whole script on every interaction and runs
synchronously, while every agent call is async. run_async() is the
bridge, and the caching here exists to stop each rerun re-reading the
same reports off disk.
"""

import asyncio
import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT, PROJECT_ROOT / "mcp_primary"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mcp_primary.tools_harvester import (  # noqa: E402
    extract_text_any_format,
    find_file_for_patient,
)

INCOMING_DIR = PROJECT_ROOT / "Data" / "incoming"
REPORTS_DIR = PROJECT_ROOT / "Data" / "reports"

DOC_TYPE_LABELS = {
    "doctor_reports": "Discharge Report",
    "lab_reports": "Lab Report",
    "bills": "Bill",
}

RISK_COLORS = {"low": "green", "medium": "orange", "high": "red"}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish",
    "de": "German", "fr": "French", "nl": "Dutch",
}


def run_async(coro):
    """
    Run a coroutine from Streamlit's synchronous script thread.

    asyncio.run() refuses to nest inside a running loop, which can happen
    depending on how Streamlit is hosted -- fall back to a private loop
    in that case rather than crashing the page.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


@st.cache_data(show_spinner=False)
def list_patients() -> list[str]:
    """Every patient with at least one document on disk."""
    patients = set()
    for folder in DOC_TYPE_LABELS:
        directory = INCOMING_DIR / folder
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and "_" in path.name:
                patients.add(path.name.split("_")[0])
    return sorted(patients)


def available_doc_types(patient_id: str) -> list[str]:
    """Which of the three document types this patient actually has."""
    available = []
    for folder in DOC_TYPE_LABELS:
        directory = INCOMING_DIR / folder
        if directory.exists() and any(
            p.name.startswith(f"{patient_id}_") for p in directory.iterdir()
        ):
            available.append(folder)
    return available


@st.cache_data(show_spinner=False)
def load_document(patient_id: str, doc_type: str) -> dict:
    """Extract one document's text, using the same reader the MCP server uses."""
    try:
        path = find_file_for_patient(INCOMING_DIR / doc_type, patient_id)
        return {
            "text": extract_text_any_format(path),
            "filename": path.name,
            "format": path.suffix.lower(),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"text": "", "filename": None, "format": None, "error": str(exc)}


@st.cache_data(show_spinner=False)
def detect_document_language(text: str) -> str:
    """Language badge for page 1. Cheap, offline, never raises."""
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "en"


def load_report(patient_id: str) -> dict | None:
    """
    The validation report written by the Reporter tool.

    Deliberately NOT cached: pages 2, 3 and 5 must see the new verdict
    immediately after a re-run, and a stale cache here would show a
    reviewer the previous decision.
    """
    path = REPORTS_DIR / f"{patient_id}_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def report_html(patient_id: str) -> str | None:
    path = REPORTS_DIR / f"{patient_id}_report.html"
    return path.read_text(encoding="utf-8") if path.exists() else None


def risk_badge(risk_level: str | None) -> str:
    level = str(risk_level or "unknown").lower()
    color = RISK_COLORS.get(level, "gray")
    return f":{color}[**{level.upper()}**]"


def language_badge(code: str) -> str:
    return f"`{code}` {LANGUAGE_NAMES.get(code, 'Unknown')}"


def selected_patient(key: str = "patient_id") -> str | None:
    """
    Patient selector, shared across pages via session state so a reviewer
    doesn't have to re-pick the patient on every page.
    """
    patients = list_patients()
    if not patients:
        st.warning(f"No documents found under {INCOMING_DIR}.")
        return None

    current = st.session_state.get(key)
    index = patients.index(current) if current in patients else 0
    choice = st.sidebar.selectbox("Patient", patients, index=index, key=f"{key}_select")
    st.session_state[key] = choice
    return choice


def require_report(patient_id: str) -> dict | None:
    """Guard used by pages that cannot render without a validation run."""
    report = load_report(patient_id)
    if report is None:
        st.info(
            f"No validation report for {patient_id} yet. "
            "Run the pipeline from the **Document Viewer** page first."
        )
        return None
    return report


def stream_a2a(agent: str, payload: dict, placeholder, prefix: str = "") -> str:
    """
    Consume an A2A stream and repaint a placeholder as text arrives.

    Streamlit's st.write_stream wants a sync iterator, but the A2A client
    is an async generator; collecting it first would defeat the point of
    streaming, so this pumps the async generator manually and writes each
    increment.
    """
    from agents.common.a2a_client import send_message_streaming, stream_text_from_event

    async def pump():
        text = prefix
        async for event in send_message_streaming(agent, payload):
            if event.get("type") == "error":
                text_error = f"{text}\n\n**Streaming failed:** {event['error']}"
                placeholder.markdown(text_error)
                return text_error
            chunk = stream_text_from_event(event)
            if chunk:
                text += chunk
                placeholder.markdown(text)
        return text

    return run_async(pump())
