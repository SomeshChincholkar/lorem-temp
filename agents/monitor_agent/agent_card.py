"""
agents/monitor_agent/agent_card.py

AgentCard for the Discharge Monitor Agent (:8103, non-streaming).
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

MONITOR_SKILL = AgentSkill(
    id="scan_new_documents",
    name="Scan for New Discharge Documents",
    description=(
        "Scans the Roots-authorized incoming folder for new discharge "
        "reports, lab reports and bills. The folder is discovered via "
        "the MCP Roots primitive (ctx.list_roots()), never passed in as "
        "a path."
    ),
    tags=["monitoring", "roots", "mcp", "adk"],
    examples=['{"trigger": true}', '{"subfolder": "lab_reports"}'],
)

AGENT_CARD = AgentCard(
    name="Discharge Monitor Agent",
    description=(
        "Google ADK agent that detects newly arrived patient discharge "
        "paperwork within the MCP Roots-scoped workspace."
    ),
    url="http://localhost:8103/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=[MONITOR_SKILL],
)
