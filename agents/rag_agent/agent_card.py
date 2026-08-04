"""
agents/rag_agent/agent_card.py

AgentCard for the Clinical RAG Q&A Agent (:8105, STREAMING).
"""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

RAG_SKILL = AgentSkill(
    id="clinical_qa",
    name="Clinical Question Answering",
    description=(
        "Answers hospital administrators' questions about patient "
        "discharge records using a five-role Agentic RAG pipeline "
        "(Indexing, Retrieval, Augmentation, Generation, Reflection) "
        "over a FAISS index, streaming the answer token by token and "
        "reporting RAG Triad quality scores."
    ),
    tags=["rag", "qa", "streaming", "agno", "faiss", "mcp"],
    examples=[
        '{"question": "What medications was P1019 discharged on?"}',
        '{"question": "Why was this discharge blocked?", "patient_filter": "P1016"}',
    ],
)

AGENT_CARD = AgentCard(
    name="Clinical RAG Q&A Agent",
    description=(
        "Agno agent providing grounded question answering over patient "
        "discharge records, with MultiMCPTools across both MCP servers "
        "and SQLite-backed session memory."
    ),
    url="http://localhost:8105/",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["application/json", "text/plain"],
    capabilities=AgentCapabilities(streaming=True, push_notifications=False),
    skills=[RAG_SKILL],
)
