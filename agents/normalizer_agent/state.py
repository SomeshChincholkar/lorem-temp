"""
agents/normalizer_agent/state.py

Graph state for the Clinical Normalizer Agent (LangGraph, A2A port 8102).

Spec 2.3: translates extracted clinical content (Hindi, Spanish, German
-> English) and normalizes medical abbreviations. Translation confidence
must be part of the output.
"""

from typing import Optional, TypedDict


class NormalizerState(TypedDict, total=False):
    patient_id: str

    # Either raw_text is supplied directly (the Orchestrator passes the
    # Extractor's output through), or doc_type is supplied and this
    # agent harvests the text itself via the MCP Harvester tool.
    doc_type: str              # "doctor_reports" | "lab_reports" | "bills"
    raw_text: str

    # Detected by node_detect_language from raw_text -- never taken from
    # the caller, same contract as the Extractor Agent.
    source_language: str

    # Fetched via MCP Prompts (abbreviation-normalization-prompt) and
    # handed to the sampling callback as its instruction block.
    normalization_prompt: str

    # Produced by the Lang Bridge tool via MCP Sampling.
    translated_text: str
    confidence: float
    model_used: Optional[str]

    # Deterministic abbreviation expansion applied on top of the
    # translated text, using resource://medical-abbreviations.
    normalized_text: str
    expanded_abbreviations: list

    # confidence < quality_thresholds.translation_confidence_min.
    # Feeds the low_translation_confidence risk weight downstream.
    low_confidence: bool

    trace_id: Optional[str]
    error: Optional[str]
