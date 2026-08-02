"""
agents/extractor_agent/state.py

Graph state for the Clinical Extractor Agent (LangGraph, A2A port 8100).
"""

from typing import Optional, TypedDict


class ExtractorState(TypedDict, total=False):
    patient_id: str
    doc_type: str            # "doctor_reports" | "lab_reports" | "bills"
                              # (matches Clinical Watcher Tool's folder names)
    language: str              # source language -- set ONLY by
                                # node_detect_language from raw_text.
                                # Never supplied by the caller.
    raw_text: str
    prompt: str
    extracted_fields: dict
    trace_id: Optional[str]
    error: Optional[str]