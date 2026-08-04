"""
agents/monitor_agent/agent.py

The Discharge Monitor Agent's core logic (spec 2.1, Google ADK).

Its whole job is to answer "what new discharge paperwork has arrived?"
by calling the Watcher tool on the Primary MCP server. The interesting
part is the Roots primitive: this agent never passes a filesystem path
as a tool parameter. It registers Data/incoming as a Root when opening
the MCP connection, and the Watcher tool calls ctx.list_roots() to
discover what it's allowed to scan. The server rejects anything outside
that root via Path.relative_to().

Root registration lives in agents/common/mcp_client.py
(_list_roots_callback), which every agent's session already installs.
"""

from google.adk.agents import LlmAgent

from agents.common.adk_runtime import get_adk_model
from agents.common.mcp_client import call_tool

# Subfolders the Watcher understands, mapped to the doc_type the rest of
# the pipeline uses. Kept here so the ADK tool docstring can name them.
DOC_TYPE_FOLDERS = ("doctor_reports", "lab_reports", "bills")


async def scan_for_new_documents(subfolder: str = "") -> dict:
    """Scan the authorized input root for new patient discharge files.

    Args:
        subfolder: Optional subfolder to restrict the scan to. One of
            "doctor_reports", "lab_reports", "bills". Empty string scans
            all three.

    Returns:
        A dict with a "documents" list, each entry holding patient_id,
        filename, doc_type and path.
    """
    arguments = {"subfolder": subfolder} if subfolder else {}
    result = await call_tool("clinical_watcher_tool", arguments)

    documents = result if isinstance(result, list) else result.get("documents", [])
    return {
        "documents": documents,
        "count": len(documents),
        "patient_ids": sorted({d.get("patient_id") for d in documents if d.get("patient_id")}),
    }


# The ADK agent. The LLM layer is genuinely thin here by design -- the
# spec calls this a monitor, and turning "what's new?" into a tool call
# is all the reasoning required. It matters that it IS an ADK agent
# (spec Table 6 assigns this one to Google ADK), not that it reasons.
monitor_agent = LlmAgent(
    name="discharge_monitor_agent",
    model=get_adk_model(),
    description="Detects newly arrived patient discharge documents in the watched folder.",
    instruction=(
        "You monitor a hospital's incoming discharge paperwork folder.\n"
        "When asked what is new, call scan_for_new_documents and report "
        "the patient IDs and document types found. If a specific "
        "subfolder is named (doctor_reports, lab_reports, bills), pass "
        "it as the subfolder argument. Never invent file paths -- the "
        "tool discovers the authorized folder itself."
    ),
    tools=[scan_for_new_documents],
)
