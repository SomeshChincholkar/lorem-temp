"""
agents/extractor_agent/nodes.py

Node functions for the Clinical Extractor Agent graph:
  harvest -> build_prompt -> extract

Calls the Primary MCP server (tool: clinical_data_harvester_tool,
prompt: discharge-extraction-prompt) and the Bedrock LLM.
"""

from agents.common.language_detect import detect_language
from agents.common.llm import get_llm, safe_json_parse
from agents.common.mcp_client import call_tool, get_prompt_text

from .state import ExtractorState

# Watcher/folder doc_type -> Harvester tool's doc_type argument
WATCHER_TO_HARVESTER_DOC_TYPE = {
    "doctor_reports": "doctor_report",
    "lab_reports": "lab_report",
    "bills": "bill",
}

# Harvester doc_type -> Table 3 / prompts.py schema key
HARVESTER_TO_PROMPT_DOC_TYPE = {
    "doctor_report": "discharge_report",
    "lab_report": "lab_report",
    "bill": "bill",
}


async def node_harvest(state: ExtractorState) -> ExtractorState:
    harvester_doc_type = WATCHER_TO_HARVESTER_DOC_TYPE.get(state["doc_type"], state["doc_type"])
    result = await call_tool(
        "clinical_data_harvester_tool",
        {"patient_id": state["patient_id"], "doc_type": harvester_doc_type},
    )
    state["raw_text"] = result["raw_text"]
    return state


async def node_detect_language(state: ExtractorState) -> ExtractorState:
    """
    Detects the source language from the harvested raw text.
    detect_language() is guaranteed to never raise and always return a
    code, but this is belt-and-suspenders in case anything upstream
    changes that contract.
    """
    try:
        state["language"] = await detect_language(state.get("raw_text", ""))
    except Exception:
        state["language"] = "en"
    return state


async def node_build_prompt(state: ExtractorState) -> ExtractorState:
    harvester_doc_type = WATCHER_TO_HARVESTER_DOC_TYPE.get(state["doc_type"], state["doc_type"])
    prompt_doc_type = HARVESTER_TO_PROMPT_DOC_TYPE.get(harvester_doc_type, harvester_doc_type)

    template = await get_prompt_text(
        "discharge-extraction-prompt",
        # note: state["language"] here is the DETECTED language from
        # node_detect_language, not whatever the caller originally sent
        {"language": state.get("language", "en"), "doc_types": prompt_doc_type},
    )
    # discharge-extraction-prompt is instructions-only; the source text
    # itself is appended here so the LLM has something to extract from.
    state["prompt"] = f"{template}\n\nSOURCE DOCUMENT:\n{state['raw_text']}"
    return state


async def node_extract(state: ExtractorState) -> ExtractorState:
    llm = get_llm()
    response = llm.invoke(state["prompt"])
    try:
        state["extracted_fields"] = safe_json_parse(response.content)
    except ValueError as e:
        state["error"] = str(e)
        state["extracted_fields"] = {}
    return state