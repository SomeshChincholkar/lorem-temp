"""
agents/rag_agent/server.py

A2A server for the Clinical RAG Q&A Agent (port 8105, STREAMING).

Run:  python -m agents.rag_agent.server

Prerequisites:
  - Primary MCP Clinical Tools Server   :8200 (rag-answer-prompt)
  - Secondary MCP Analytics Server      :8201 (MultiMCPTools)
  - Live Bedrock credentials in .env

Each patient has their OWN FAISS index under data/vector_db/{patient_id}/,
so every question must name a patient and no answer can draw on another
patient's records. A missing patient's index builds itself on the first
question about them; that first request also downloads the
all-MiniLM-L6-v2 weights (~90MB) and will be noticeably slower. Build
them ahead of time with:

    python -m agents.rag_agent.build_index
"""

import uvicorn

from agents.common.a2a_server import build_a2a_app

from .agent_card import AGENT_CARD
from .agent_executor import RagAgentExecutor

app = build_a2a_app(AGENT_CARD, RagAgentExecutor())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)
