"""
agents/validator_agent/agent_card.py

AgentCard for the Clinical Validation Agent (:8101, non-streaming).
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

VALIDATE_SKILL = AgentSkill(
    id="validate_discharge",
    name="Validate Discharge Record",
    description=(
        "Runs Table 3 completeness validation (eliciting missing "
        "non-blocking fields from a human reviewer), the seven Table 4 "
        "cross-validation rules against the Mock EHR, and produces the "
        "JSON + HTML audit report with a risk level and recommendation."
    ),
    tags=["validation", "clinical", "ehr", "mcp", "elicitation"],
    examples=[
        '{"patient_id": "P1019", "extracted_discharge": {...}, "extracted_bill": {...}}'
    ],
)

AGENT_CARD = AgentCard(
    name="Clinical Validation Agent",
    description=(
        "Validates discharge completeness and cross-checks the record "
        "against the Mock EHR, then generates the discharge audit report."
    ),
    url="http://localhost:8101/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    capabilities=AgentCapabilities(streaming=False, push_notifications=True),
    skills=[VALIDATE_SKILL],
)
