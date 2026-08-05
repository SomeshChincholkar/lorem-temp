"""
agents/rag_agent/agent_card.py

AgentCard for the Clinical RAG Q&A Agent (:8105, STREAMING).
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

RAG_SKILL = AgentSkill(
    id="clinical_qa",
    name="Clinical Question Answering",
    description=(
        "Answers questions about ONE patient's discharge records using a "
        "five-role Agentic RAG pipeline (Indexing, Retrieval, "
        "Augmentation, Generation, Reflection) over that patient's own "
        "FAISS index, streaming the answer token by token and reporting "
        "RAG Triad quality scores. 'patient_id' is required."
    ),
    tags=["rag", "qa", "streaming", "agno", "faiss", "mcp", "per-patient"],
    examples=[
        '{"question": "What medications was this patient discharged on?", "patient_id": "P1019"}',
        '{"question": "Why was this discharge blocked?", "patient_id": "P1016"}',
    ],
)

AGENT_CARD = AgentCard(
    name="Clinical RAG Q&A Agent",
    description=(
        "Agno agent providing grounded question answering over a single "
        "patient's discharge records. Each patient has a dedicated FAISS "
        "index, so answers cannot draw on another patient's data. Uses "
        "MultiMCPTools across both MCP servers and SQLite-backed session "
        "memory."
    ),
    url="http://localhost:8105/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json", "text/plain"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=True),
    skills=[RAG_SKILL],
)
