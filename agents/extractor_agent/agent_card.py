"""
agents/extractor_agent/agent_card.py

Real a2a-sdk AgentCard (a2a.types objects, not a plain dict) for the
Clinical Extractor Agent. A2AStarletteApplication serves this at
GET /.well-known/agent.json.
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

EXTRACT_SKILL = AgentSkill(
    id="extract_clinical_fields",
    name="Extract Clinical Fields",
    description=(
        "Given a patient_id and doc_type, harvests raw document text "
        "and extracts structured JSON fields per Table 3 of the "
        "capstone spec (discharge report / lab report / bill)."
    ),
    tags=["extraction", "clinical", "mcp"],
    examples=[
        '{"patient_id": "P1001", "doc_type": "doctor_reports", "language": "en"}'
    ],
)

AGENT_CARD = AgentCard(
    name="Clinical Extractor Agent",
    description=(
        "Extracts structured clinical fields from discharge reports, "
        "lab reports, and bills using the Primary MCP Clinical Tools Server."
    ),
    url="http://localhost:8100/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=True),
    skills=[EXTRACT_SKILL],
)