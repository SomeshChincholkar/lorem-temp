"""
agents/normalizer_agent/nodes.py

Node functions for the Clinical Normalizer Agent graph:
  harvest -> detect_language -> fetch_prompt -> translate -> normalize_abbrev

MCP primitives exercised here (spec Table 6: Tools + Sampling + Prompts):
  Tools     : clinical_data_harvester_tool, medical_lang_bridge_tool
  Sampling  : the Lang Bridge tool calls back into this agent's
              sampling_callback (see sampling.py)
  Prompts   : abbreviation-normalization-prompt
  Resources : medical-abbreviations (bonus -- makes rules.yaml, not the
              LLM, authoritative for abbreviation expansion)
"""

import json
import re

from agents.common.language_detect import detect_language
from agents.common.mcp_client import call_tool, get_prompt_text, read_resource_text
from common.rules_loader import load_rules
from observability import log_error

from .sampling import make_sampling_callback
from .state import NormalizerState

# Watcher/folder doc_type -> Harvester tool's doc_type argument.
# Same mapping the Extractor Agent uses.
WATCHER_TO_HARVESTER_DOC_TYPE = {
    "doctor_reports": "doctor_report",
    "lab_reports": "lab_report",
    "bills": "bill",
}

DEFAULT_CONFIDENCE_MIN = 0.70


def _confidence_threshold() -> float:
    """quality_thresholds.translation_confidence_min from rules.yaml."""
    rules = load_rules()
    thresholds = rules.get("quality_thresholds", {})
    return float(thresholds.get("translation_confidence_min", DEFAULT_CONFIDENCE_MIN))


async def node_harvest(state: NormalizerState) -> NormalizerState:
    """
    Fetch the document text if the caller didn't supply it.

    In the full pipeline the Orchestrator passes the Extractor's text
    straight through as raw_text, so this is a no-op. Standalone (and in
    tests) only patient_id + doc_type are given, and this agent pulls the
    text itself.
    """
    if state.get("raw_text"):
        return state

    doc_type = state.get("doc_type")
    if not doc_type:
        state["error"] = "Neither raw_text nor doc_type was supplied."
        state["raw_text"] = ""
        return state

    harvester_doc_type = WATCHER_TO_HARVESTER_DOC_TYPE.get(doc_type, doc_type)
    result = await call_tool(
        "clinical_data_harvester_tool",
        {"patient_id": state["patient_id"], "doc_type": harvester_doc_type},
        trace_id=state.get("trace_id"),
    )
    state["raw_text"] = result["raw_text"]
    return state


async def node_detect_language(state: NormalizerState) -> NormalizerState:
    """
    Detect the source language. This drives the model hint the Lang
    Bridge tool sends (English -> command-r-plus, otherwise nova-lite),
    so it must run before translation.
    """
    try:
        state["source_language"] = await detect_language(state.get("raw_text", ""))
    except Exception:
        state["source_language"] = "en"
    return state


async def node_fetch_prompt(state: NormalizerState) -> NormalizerState:
    """
    Fetch abbreviation-normalization-prompt via MCP Prompts.

    This is handed to the sampling callback as its instruction block, so
    the guidance driving translation is server-authored -- not a string
    hardcoded in this agent.
    """
    try:
        state["normalization_prompt"] = await get_prompt_text(
            "abbreviation-normalization-prompt",
            {"source_language": state.get("source_language", "en")},
            trace_id=state.get("trace_id"),
        )
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: the Lang Bridge tool's own sampling message already
        # states the task and the required JSON shape. Losing the prompt
        # costs quality, not correctness.
        state["normalization_prompt"] = ""
        state["error"] = f"Could not fetch normalization prompt: {exc}"
        log_error(
            "mcp.prompt.abbreviation_normalization.failed",
            exc,
            trace_seed=state.get("trace_id"),
            fallback_action="proceeding without server-authored prompt",
        )
    return state


async def node_translate(state: NormalizerState) -> NormalizerState:
    """
    Call medical_lang_bridge_tool -- the Sampling round trip.

    Flow: this agent calls the tool over MCP -> the tool issues
    ctx.session.create_message() back down the same connection -> the
    SDK routes it to the sampling_callback registered on this session ->
    the callback runs Bedrock and returns a CreateMessageResult -> the
    tool parses it and returns {translated_text, confidence, model_used}.

    The callback MUST be registered on the session that makes this call,
    which is why it's passed through call_tool rather than configured
    globally.
    """
    raw_text = state.get("raw_text", "")
    if not raw_text:
        state["translated_text"] = ""
        state["confidence"] = 0.0
        state["model_used"] = None
        return state

    trace_id = state.get("trace_id")
    callback = make_sampling_callback(
        state.get("normalization_prompt") or None, trace_id=trace_id
    )

    try:
        result = await call_tool(
            "medical_lang_bridge_tool",
            {"text": raw_text, "source_language": state.get("source_language", "en")},
            sampling_callback=callback,
            trace_id=trace_id,
        )
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"Lang Bridge tool call failed: {exc}"
        state["translated_text"] = ""
        state["confidence"] = 0.0
        state["model_used"] = None
        log_error(
            "mcp.tool.medical_lang_bridge.failed",
            exc,
            trace_seed=trace_id,
            fallback_action="empty translation, confidence 0.0 (trips low-confidence guardrail)",
        )
        return state

    state["translated_text"] = result.get("translated_text", "")
    state["confidence"] = float(result.get("confidence", 0.0) or 0.0)
    state["model_used"] = result.get("model_used")

    if result.get("parse_warning"):
        state["error"] = result["parse_warning"]

    return state


async def node_normalize_abbrev(state: NormalizerState) -> NormalizerState:
    """
    Deterministic abbreviation expansion on top of the translated text,
    driven by resource://medical-abbreviations (i.e. by rules.yaml).

    The Lang Bridge already asked the LLM to expand abbreviations, but an
    LLM pass is best-effort and non-reproducible. Auditors need
    "BID always became twice daily" to be a fact about the config, not a
    hope about the model -- so rules.yaml gets the final word here.
    """
    text = state.get("translated_text", "")
    if not text:
        state["normalized_text"] = ""
        state["expanded_abbreviations"] = []
        state["low_confidence"] = True
        return state

    try:
        abbrev_json = await read_resource_text(
            "resource://medical-abbreviations", trace_id=state.get("trace_id")
        )
        abbreviations = json.loads(abbrev_json)
    except Exception:
        # Resource unreachable or malformed -- pass the LLM's text
        # through untouched rather than failing the whole run.
        abbreviations = {}

    expanded = []
    for abbrev, expansion in abbreviations.items():
        # Case-sensitive with word boundaries: medical abbreviations are
        # uppercase by convention, and a case-insensitive match would
        # rewrite ordinary words ("Temp" is an abbreviation, "temp" in
        # prose is not; "MI" must not match "mi" in Spanish text).
        pattern = rf"\b{re.escape(abbrev)}\b"
        text, count = re.subn(pattern, expansion, text)
        if count:
            expanded.append({"abbreviation": abbrev, "expansion": expansion, "count": count})

    state["normalized_text"] = text
    state["expanded_abbreviations"] = expanded
    state["low_confidence"] = state.get("confidence", 0.0) < _confidence_threshold()
    return state
