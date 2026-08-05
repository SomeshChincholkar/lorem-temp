"""
agents/summary_agent/agent_card.py

AgentCard for the Discharge Summary Generator (:8104, STREAMING).

capabilities.streaming=True is what tells an A2A client it may call
message/stream and consume TaskArtifactUpdateEvents section by section
(spec Table 10).
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

SUMMARY_SKILL = AgentSkill(
    id="generate_discharge_summary",
    name="Generate Discharge Summary",
    description=(
        "Streams a patient-friendly discharge summary section by "
        "section: hospital stay, medicines, test results, bill, and "
        "what to do next. Built from the validated audit report, using "
        "the summary-generation-prompt fetched via MCP Prompts."
    ),
    tags=["summary", "streaming", "patient-friendly", "adk", "mcp"],
    examples=[
        '{"patient_id": "P1019"}',
        '{"patient_id": "P1019", "audience": "clinician"}',
    ],
)

AGENT_CARD = AgentCard(
    name="Discharge Summary Generator",
    description=(
        "Google ADK agent that streams patient-friendly discharge "
        "summaries progressively, one section at a time."
    ),
    url="http://localhost:8104/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json", "text/plain"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=True),
    skills=[SUMMARY_SKILL],
)
