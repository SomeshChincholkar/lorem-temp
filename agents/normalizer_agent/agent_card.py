"""
agents/normalizer_agent/agent_card.py

AgentCard for the Clinical Normalizer Agent, served by
A2AStarletteApplication at GET /.well-known/agent.json.

Non-streaming per spec Table 6/10 -- normalization returns one final
artifact, there is nothing progressive to emit.
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

NORMALIZE_SKILL = AgentSkill(
    id="normalize_clinical_text",
    name="Translate and Normalize Clinical Text",
    description=(
        "Translates clinical document text (Hindi, Spanish, German, "
        "French, Dutch) into English via MCP Sampling, expands medical "
        "abbreviations against rules.yaml, and reports a translation "
        "confidence score."
    ),
    tags=["translation", "normalization", "clinical", "mcp", "sampling"],
    examples=[
        '{"patient_id": "P1015", "doc_type": "doctor_reports"}',
        '{"patient_id": "P1016", "raw_text": "Der Patient wurde entlassen."}',
    ],
)

AGENT_CARD = AgentCard(
    name="Clinical Normalizer Agent",
    description=(
        "Translates extracted clinical content to English and normalizes "
        "medical abbreviations, using the Primary MCP Clinical Tools "
        "Server's Medical Lang Bridge tool via the MCP Sampling primitive."
    ),
    url="http://localhost:8102/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=[NORMALIZE_SKILL],
)
